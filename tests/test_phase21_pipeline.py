from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from pathline_template_matching.arc_length_primitives import ArcLengthScaleTable
from pathline_template_matching.encoder import IndependentFMT3DConfig
from pathline_template_matching.netcdf_io import FlowWindow3D
from pathline_template_matching.phase21_pipeline import (
    EXPERIMENT,
    METHODS,
    Phase21Plan,
    _choose_two_class_candidate_rows,
    _metric_values,
    _validate_strict_cuda_workspace,
    balanced_scale_assignment,
    build_phase21_cache_slice,
    cache_summary_payload,
    fit_streaming_covariance_pca,
    load_cache_summary_sidecar,
    load_phase21_plan,
    recover_phase21_cache_summary,
    run_phase21_from_resolvers,
    write_cache_summary_sidecar,
)
from pathline_template_matching.portable_flow import canonical_json_sha256
from pathline_template_matching.pca import DeterministicPCA


ROOT = Path(__file__).resolve().parents[1]


def test_phase21_config_freezes_split_scales_and_balanced_assignment():
    plan = load_phase21_plan(ROOT / "config/mainExp_TemplateMatching_2.1.yaml")
    assert plan.train_datasets == (
        "cylinder3d",
        "halfcylinderRe640",
        "halfcylinderRe6400",
        "deltaWing_resampled",
        "deltaWing_LBM",
        "f22raptor",
        "channel",
        "boeing747",
    )
    assert plan.test_datasets == ("tangaroa", "smokeBuoyancy")
    assert plan.config["library"]["sampling_rule"] == (
        "one_global_generator_draws_negative_then_positive_only_for_each_"
        "two_class_nonempty_stratum"
    )
    assert len(plan.scale_table) == 1000
    assert plan.scale_table.scale_id.dtype == np.int32
    assert np.isclose(plan.scale_table.dx_grid_scale[0], 0.25)
    assert np.isclose(plan.scale_table.ds_frame_scale[10], 0.144444444444)
    assert np.isclose(plan.scale_table.arc_length_grid_scale[999], 12.0)
    first = balanced_scale_assignment(64_000, 1000, 15068)
    second = balanced_scale_assignment(64_000, 1000, 15068)
    assert np.array_equal(first, second)
    assert np.array_equal(np.bincount(first, minlength=1000), np.full(1000, 64))
    for frame_count in (16, 20, 47, 76, 151, 159, 160, 171, 201, 213):
        maximum = frame_count - 13
        assert plan.source_indices(frame_count) == (
            0,
            maximum // 3,
            (2 * maximum) // 3,
            maximum,
        )
    expected_workspace = plan.config["execution"]["cublas_workspace_config"]
    previous_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = "wrong"
        try:
            _validate_strict_cuda_workspace(
                plan, selected_device="cuda", strict_protocol=True
            )
        except RuntimeError as error:
            assert "CUBLAS_WORKSPACE_CONFIG" in str(error)
        else:
            raise AssertionError("strict CUDA accepted the wrong cuBLAS workspace")
        _validate_strict_cuda_workspace(
            plan, selected_device="cpu", strict_protocol=True
        )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected_workspace
        _validate_strict_cuda_workspace(
            plan, selected_device="cuda", strict_protocol=True
        )
    finally:
        if previous_workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_workspace


def test_phase21_skipped_library_strata_consume_no_random_draws():
    skipped_then_selected = np.random.Generator(np.random.PCG64(15068))
    assert _choose_two_class_candidate_rows(
        skipped_then_selected,
        (np.asarray([2, 4, 6]), np.empty(0, dtype=np.int64)),
    ) == (None, None)
    after_skip = _choose_two_class_candidate_rows(
        skipped_then_selected,
        (np.asarray([10, 11, 12]), np.asarray([20, 21, 22, 23])),
    )

    fresh = np.random.Generator(np.random.PCG64(15068))
    direct = _choose_two_class_candidate_rows(
        fresh,
        (np.asarray([10, 11, 12]), np.asarray([20, 21, 22, 23])),
    )
    assert after_skip == direct


def test_phase21_single_class_ranking_metrics_are_null_but_accuracy_is_retained():
    result = _metric_values(
        np.ones(3, dtype=bool),
        np.asarray([1, 0, 1], dtype=bool),
        np.asarray([0.8, 0.2, 0.7]),
    )
    assert result["single_class_group"] is True
    assert np.isnan(result["average_precision"])
    assert np.isnan(result["auroc"])
    assert np.isnan(result["balanced_accuracy"])
    assert np.isclose(result["accuracy"], 2.0 / 3.0)


def test_phase21_streaming_covariance_pca_matches_full_svd_without_concatenation():
    generator = np.random.default_rng(25068)
    values = generator.normal(size=(97, 12)).astype(np.float32)
    values *= np.linspace(0.5, 3.0, 12, dtype=np.float32)
    values[:, 3] += 0.2 * values[:, 0]

    def blocks():
        return iter((values[:17], values[17:63], values[63:]))

    streaming = fit_streaming_covariance_pca(
        blocks, input_width=12, components=6
    )
    full = DeterministicPCA.fit(values, 6)
    assert streaming.sample_count == len(values)
    assert streaming.input_width == values.shape[1]
    assert np.allclose(streaming.mean, full.mean, rtol=1e-6, atol=1e-7)
    assert np.allclose(
        streaming.singular_values, full.singular_values, rtol=1e-6, atol=1e-7
    )
    assert np.allclose(
        streaming.components.T @ streaming.components,
        full.components.T @ full.components,
        rtol=1e-5,
        atol=1e-5,
    )
    assert np.allclose(
        streaming.transform(values[:9]),
        full.transform(values[:9]),
        rtol=2e-5,
        atol=2e-5,
    )


def test_phase21_portable_root_reads_dataset_windows_and_relative_paths():
    spec = importlib.util.spec_from_file_location(
        "phase21_cli_test_module",
        ROOT / "scripts/run_mainexp_template_matching_2_1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    index_class = module.PortableManifestIndex
    plan = load_phase21_plan(ROOT / "config/mainExp_TemplateMatching_2.1.yaml")
    builder_commit = "4" * 40
    registry_sha = plan.dataset_registry_sha256
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for dataset_ordinal, dataset in enumerate(plan.datasets):
            dataset_dir = root / dataset
            dataset_dir.mkdir()
            total_frames = 16 + dataset_ordinal
            indices = plan.source_indices(total_frames)
            windows = []
            for ordinal, source_index in enumerate(indices):
                name = f"window_{ordinal}.npz"
                (dataset_dir / name).write_bytes(b"placeholder")
                windows.append(
                    {
                        "relative_path": name,
                        "file_sha256": "5" * 64,
                        "file_size": len(b"placeholder"),
                        "dataset": dataset,
                        "experiment": plan.experiment,
                        "config_sha256": plan.config_sha256,
                        "physical_family": plan.family_by_dataset[dataset],
                        "split": plan.split_for(dataset),
                        "dataset_registry_sha256": registry_sha,
                        "builder_git_commit": builder_commit,
                        "source_kind": "netcdf",
                        "source_file": "source.nc",
                        "source_file_size": 123,
                        "source_file_sha256": "6" * 64,
                        "source_total_frames": total_frames,
                        "source_ordinal": ordinal,
                        "source_start_index": source_index,
                        "frame_count": 13,
                    }
                )
            manifest = {
                "schema": "pathline_template_matching.portable_flow_dataset_manifest.v1",
                "experiment": plan.experiment,
                "config_sha256": plan.config_sha256,
                "dataset": dataset,
                "physical_family": plan.family_by_dataset[dataset],
                "split": plan.split_for(dataset),
                "dataset_registry_sha256": registry_sha,
                "builder_git_commit": builder_commit,
                "source_kind": "netcdf",
                "source_file": "source.nc",
                "source_file_size": 123,
                "source_file_sha256": "6" * 64,
                "source_total_frames": total_frames,
                "selected_source_indices": list(indices),
                "window_count": 4,
                "windows": windows,
            }
            manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
            (dataset_dir / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
        index = index_class(
            None,
            portable_root=root,
            expected_experiment=plan.experiment,
            expected_config_sha256=plan.config_sha256,
            expected_dataset_registry_sha256=plan.dataset_registry_sha256,
            expected_datasets=plan.datasets,
            expected_family_by_dataset=plan.family_by_dataset,
            expected_split_by_dataset={
                dataset: plan.split_for(dataset) for dataset in plan.datasets
            },
            expected_builder_git_commit=builder_commit,
        )
        first = index.dataset_rows(plan.datasets[0])[0]
        assert index.resolve_path(first) == (
            root / plan.datasets[0] / "window_0.npz"
        ).resolve()


def _synthetic_plan(config_path: Path) -> Phase21Plan:
    payload = config_path.read_bytes()
    train = tuple(f"train_{index}" for index in range(8))
    test = ("test_0", "test_1")
    datasets = train + test
    return Phase21Plan(
        config_path=config_path,
        config_sha256=hashlib.sha256(payload).hexdigest(),
        dataset_registry_path=ROOT / "config/datasets.yaml",
        dataset_registry_sha256=hashlib.sha256(
            (ROOT / "config/datasets.yaml").read_bytes()
        ).hexdigest(),
        config={"experiment": EXPERIMENT, "profile": "synthetic_test_only"},
        experiment=EXPERIMENT,
        output_root="synthetic",
        train_datasets=train,
        test_datasets=test,
        family_by_dataset={name: f"family_{name}" for name in datasets},
        source_count=1,
        window_frame_count=13,
        seed_shape_xyz=(9, 2, 3),
        scale_table=ArcLengthScaleTable(
            scale_id=np.asarray([0], dtype=np.int32),
            dx_grid_scale=np.asarray([1.0]),
            ds_frame_scale=np.asarray([0.125]),
            arc_length_grid_scale=np.asarray([0.05]),
        ),
        assignment_seed=15068,
        library_seed=15068,
        pca_components=161,
        bootstrap_seed=25068,
        bootstrap_replicates=20,
        descriptor_config=IndependentFMT3DConfig(),
        method_ids=METHODS,
        required_outputs=(),
    )


def _synthetic_window(dataset: str) -> FlowWindow3D:
    coordinate = np.linspace(0.0, 2.0, 11, dtype=np.float64)
    time = np.arange(13, dtype=np.float64)
    velocity = np.zeros((13, 11, 11, 11, 3), dtype=np.float32)
    velocity[..., 2] = (
        0.2 + 0.02 * np.cos(np.pi * coordinate)
    )[None, None, None, :]
    return FlowWindow3D(
        velocity=velocity,
        coordinates_xyz=(coordinate, coordinate, coordinate),
        time=time,
        source_path=f"synthetic://{dataset}",
        source_start_index=0,
        spatial_strides={"x": 1, "y": 1, "z": 1},
        components=("u", "v", "w"),
        coordinate_sources={"x": "x", "y": "y", "z": "z", "t": "time"},
    )


def test_phase21_atomic_cache_can_recover_a_missing_sidecar_without_reintegration():
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        config_path = temporary / "synthetic_phase21.yaml"
        config_path.write_text(
            "experiment: mainExp_TemplateMatching_2.1\nprofile: synthetic_test_only\n",
            encoding="utf-8",
        )
        plan = _synthetic_plan(config_path)
        cache_path = temporary / "cache.npz"
        original = build_phase21_cache_slice(
            plan,
            dataset="train_0",
            source_ordinal=0,
            source_index=0,
            resolved_input=_synthetic_window("train_0"),
            cache_path=cache_path,
            strict_evidence=False,
            cache_builder_git_commit="synthetic-test",
        )
        with np.load(cache_path, allow_pickle=False) as archive:
            assert archive["seeds_xyz"].dtype == np.float64
            stored = {name: np.array(archive[name], copy=True) for name in archive.files}
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            stored_names = set(archive.files) - {"metadata_json"}
            assert set(metadata["array_sha256"]) == stored_names
            assert archive["ivd_volume"].dtype == np.float32
            assert archive["ivd_volume"].shape == (11, 11, 11)
            assert archive["center_sample_time"].dtype == np.float32
            assert archive["center_sample_time"].shape[1:] == (32,)
            assert np.all(np.diff(archive["center_sample_time"], axis=1) >= -1e-7)
        recovered = recover_phase21_cache_summary(
            plan,
            cache_path=cache_path,
            dataset="train_0",
            source_ordinal=0,
            source_index=0,
            cache_builder_git_commit="synthetic-test",
            strict_evidence=False,
        )
        assert cache_summary_payload(recovered) == cache_summary_payload(original)
        recovered_with_current_provenance = recover_phase21_cache_summary(
            plan,
            cache_path=cache_path,
            dataset="train_0",
            source_ordinal=0,
            source_index=0,
            cache_builder_git_commit="synthetic-test",
            strict_evidence=False,
            expected_window_provenance=metadata["window_provenance"],
        )
        assert cache_summary_payload(recovered_with_current_provenance) == (
            cache_summary_payload(original)
        )
        wrong_provenance = dict(metadata["window_provenance"])
        wrong_provenance["source_path"] = "synthetic://different-source"
        try:
            recover_phase21_cache_summary(
                plan,
                cache_path=cache_path,
                dataset="train_0",
                source_ordinal=0,
                source_index=0,
                cache_builder_git_commit="synthetic-test",
                strict_evidence=False,
                expected_window_provenance=wrong_provenance,
            )
        except ValueError as error:
            assert "portable-window provenance" in str(error)
        else:
            raise AssertionError("cache recovery accepted different input provenance")
        sidecar = temporary / "cache.summary.json"
        write_cache_summary_sidecar(recovered, sidecar)
        assert cache_summary_payload(load_cache_summary_sidecar(sidecar)) == (
            cache_summary_payload(original)
        )
        diagnostic_names = (
            "valid_scale_id",
            "center_sample_time",
            "ivd_volume",
            "line_steps",
            "line_travel",
            "line_end_time",
            "line_reached_target",
        )
        for diagnostic_name in diagnostic_names:
            corrupted = {name: np.array(values, copy=True) for name, values in stored.items()}
            values = corrupted[diagnostic_name]
            if values.dtype == np.bool_:
                values.flat[0] = not bool(values.flat[0])
            else:
                values.flat[0] += 1
            damaged_path = temporary / f"damaged_{diagnostic_name}.npz"
            np.savez_compressed(damaged_path, **corrupted)
            try:
                recover_phase21_cache_summary(
                    plan,
                    cache_path=damaged_path,
                    dataset="train_0",
                    source_ordinal=0,
                    source_index=0,
                    cache_builder_git_commit="synthetic-test",
                    strict_evidence=False,
                )
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"recovery accepted corrupted diagnostic array {diagnostic_name}"
                )


def test_phase21_tiny_synthetic_eight_two_end_to_end():
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        config_path = temporary / "synthetic_phase21.yaml"
        config_path.write_text(
            "experiment: mainExp_TemplateMatching_2.1\nprofile: synthetic_test_only\n",
            encoding="utf-8",
        )
        plan = _synthetic_plan(config_path)
        run_dir = temporary / "run"
        result = run_phase21_from_resolvers(
            plan,
            run_dir=run_dir,
            frame_count_resolver=lambda _dataset: 13,
            window_resolver=lambda dataset, _source, _frames: _synthetic_window(dataset),
            git_commit="synthetic-test",
            device="cpu",
            strict_protocol=False,
            integration_chunk_size=128,
            encoding_chunk_size=128,
            verify_cache_hashes=True,
        )
        assert result["status"] == "development_completed_confirmation_not_run"
        assert result["assigned_test_query_count"] == 108
        assert 0 < result["valid_test_query_count"] < 108
        assert (run_dir / "RUN_COMPLETE.json").is_file()
        cache_manifest = json.loads(
            (run_dir / "cache_manifest.json").read_text(encoding="utf-8")
        )
        assert cache_manifest["row_count"] == 10
        with (run_dir / "main_table.csv").open(encoding="utf-8", newline="") as source:
            main_rows = list(csv.DictReader(source))
        assert {row["method"] for row in main_rows} == set(METHODS)
        assert all("accuracy" in row and "coverage" in row for row in main_rows)
        query_text = (run_dir / "per_query_matches.csv").read_text(encoding="utf-8")
        assert "test_0" in query_text and "test_1" in query_text
        assert "train_0" in query_text  # matched-template provenance is retained

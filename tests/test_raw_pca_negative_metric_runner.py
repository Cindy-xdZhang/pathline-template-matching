from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from pathline_template_matching.per_scale_negative_metric import (
    PerScaleNegativeTailModel,
)
from pathline_template_matching.portable_flow import (
    canonical_array_sha256,
    sha256_file,
)
from pathline_template_matching.raw_pca_representation import (
    RAW_INPUT_WIDTH,
    RAW_OUTPUT_WIDTH,
    RawPCARepresentation,
)
from scripts import run_verify_raw_pca_negative_metric_1_1 as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "Verify_RawPCANegativeMetric_1.1.yaml"


def _expect_error(error_types, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error_types:
        return
    raise AssertionError(f"expected {error_types}")


def _pca(sample_count: int = 2) -> RawPCARepresentation:
    components = np.zeros(
        (RAW_OUTPUT_WIDTH, RAW_INPUT_WIDTH), dtype=np.float32
    )
    components[np.arange(RAW_OUTPUT_WIDTH), np.arange(RAW_OUTPUT_WIDTH)] = 1.0
    singular = np.linspace(2.0, 1.0, RAW_OUTPUT_WIDTH, dtype=np.float64)
    explained = np.linspace(2.0, 1.0, RAW_OUTPUT_WIDTH, dtype=np.float64)
    explained /= 1000.0 * explained.sum(dtype=np.float64)
    return RawPCARepresentation(
        mean=np.zeros(RAW_INPUT_WIDTH, dtype=np.float32),
        components=components,
        singular_values=singular,
        explained_variance_ratio=explained,
        sample_count=sample_count,
    )


def _cache_arrays(count: int = 4, *, poison_labels: bool = False):
    raw = np.arange(count * RAW_INPUT_WIDTH, dtype=np.float32).reshape(
        count, RAW_INPUT_WIDTH
    )
    scales = np.asarray([0, 1, 1000, 1001][:count], dtype=np.int32)
    centers = np.arange(count, dtype=np.int64)
    blocks = (scales >= 1000).astype(np.int8)
    assigned = blocks.astype(np.int64) * 64000 + centers
    labels = (
        np.asarray({"reference_poison": True}, dtype=object)
        if poison_labels
        else np.asarray([False, True, False, True][:count], dtype=bool)
    )
    return raw, scales, centers, blocks, assigned, labels


def _write_cache(
    path: Path,
    plan: runner.Plan,
    *,
    dataset: str,
    family: str,
    source_ordinal: int,
    source_index: int,
    poison_labels: bool = False,
) -> runner.CacheRow:
    raw, scales, centers, blocks, assigned, labels = _cache_arrays(
        poison_labels=poison_labels
    )
    arrays = {
        "raw_features": raw,
        "valid_scale_id": scales,
        "valid_center_seed_index": centers,
        "valid_scale_block_index": blocks,
        "valid_assigned_row_index": assigned,
    }
    if not poison_labels:
        arrays["valid_labels"] = labels
    metadata = {
        "schema": plan.cache_schema,
        "experiment": "mainExp_TemplateMatching_3.1",
        "split": "train",
        "dataset": dataset,
        "physical_family": family,
        "source_ordinal": source_ordinal,
        "source_index": source_index,
        "config_sha256": plan.parent_config_sha256,
        "cache_builder_git_commit": plan.cache_commit,
        "valid_count": len(raw),
        "array_sha256": {
            name: canonical_array_sha256(values) for name, values in arrays.items()
        },
    }
    np.savez_compressed(
        path,
        raw_features=raw,
        valid_scale_id=scales,
        valid_center_seed_index=centers,
        valid_scale_block_index=blocks,
        valid_assigned_row_index=assigned,
        valid_labels=labels,
        # These object members prove that allow_pickle=False is safe when the
        # strict loader never requests them.
        fmt_features=np.asarray({"fmt_poison": True}, dtype=object),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return runner.CacheRow(
        dataset=dataset,
        family=family,
        source_ordinal=source_ordinal,
        source_index=source_index,
        path=path,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


def test_raw_pca_runner_plan_candidate_grid_schedule_and_output_contract():
    plan = runner.load_plan(CONFIG)
    candidates = runner.candidate_specs(plan)
    assert plan.sha256 == runner.EXPECTED_CONFIG_SHA256
    assert len(candidates) == 1020
    assert len({candidate.candidate_id for candidate in candidates}) == 1020
    assert {candidate.representation for candidate in candidates} == {"raw_pca161"}
    assert {candidate.decision_rule for candidate in candidates} == {
        "fixed_top_fraction",
        "calibrated_tail_anomaly_threshold",
    }
    schedule = runner.nested_pca_fit_schedule(plan, "half_cylinder")
    assert len(schedule) == 5
    assert all(len(families) == 3 for _, families in schedule[:4])
    assert schedule[-1] == (
        "final",
        ("delta_wing", "f22_raptor", "channel", "boeing_747"),
    )
    assert len(plan.required_fold_files) == 17
    assert len(runner.result_artifact_names(plan)) == 15
    assert "final_pca.npz" in plan.required_fold_files
    assert "final_pca_manifest.json" in plan.required_fold_files


def test_strict_raw_loader_never_opens_fmt_or_poisoned_labels():
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        row = _write_cache(
            root / "poisoned.npz",
            plan,
            dataset="deltaWing_resampled",
            family="delta_wing",
            source_ordinal=0,
            source_index=0,
            poison_labels=True,
        )
        raw, audit = runner.load_pca_fit_block(plan, row)
        assert raw.shape == (4, 672)
        assert audit["opened_members"] == ("raw_features", "metadata_json")
        assert audit["valid_labels_opened"] is False
        assert audit["fmt_features_opened"] is False
        projection = runner.load_raw_projection(plan, row, include_labels=False)
        assert projection.labels is None
        assert not projection.metadata
        assert "fmt_features" not in projection.opened_members
        assert "valid_labels" not in projection.opened_members
        assert "metadata_json" not in projection.opened_members
        _expect_error(
            ValueError,
            runner._open_cache_members,
            row,
            ("fmt_features",),
        )


def test_pca_two_pass_gate_survives_poisoned_labels_and_records_both_passes():
    plan = runner.load_plan(CONFIG)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        rows = []
        identities = (
            ("deltaWing_resampled", "delta_wing"),
            ("deltaWing_LBM", "delta_wing"),
            ("f22raptor", "f22_raptor"),
            ("channel", "channel"),
        )
        source_index = 0
        for dataset, family in identities:
            for source_ordinal in range(4):
                rows.append(
                    _write_cache(
                        root / f"{dataset}_{source_ordinal}.npz",
                        plan,
                        dataset=dataset,
                        family=family,
                        source_ordinal=source_ordinal,
                        source_index=source_index,
                        poison_labels=True,
                    )
                )
                source_index += 1
        original = runner.fit_raw_pca

        def fake_fit(factory):
            first = list(factory())
            second = list(factory())
            assert sum(len(block) for block in first) == 64
            assert sum(len(block) for block in second) == 64
            return _pca(sample_count=64)

        runner.fit_raw_pca = fake_fit
        try:
            fitted = runner.fit_pca_from_cache_rows(
                plan,
                rows,
                outer_family="half_cylinder",
                held_out="boeing_747",
                expected_sample_count=64,
            )
        finally:
            runner.fit_raw_pca = original
        assert fitted.model.sample_count == 64
        assert fitted.audit["labels_opened_for_pca"] is False
        assert fitted.audit["fmt_features_opened"] is False
        assert fitted.audit["cache_count"] == 16
        assert all(
            cache["opened_members"] == ("raw_features", "metadata_json")
            for cache in fitted.audit["ordered_fit_caches"]
        )


def _final_fit(model: RawPCARepresentation) -> runner.RawPCAFitResult:
    identities = (
        ("delta_wing", "deltaWing_resampled"),
        ("delta_wing", "deltaWing_LBM"),
        ("f22_raptor", "f22raptor"),
        ("channel", "channel"),
        ("boeing_747", "boeing747"),
    )
    caches = [
        {
            "dataset": dataset,
            "family": family,
            "source_ordinal": ordinal,
            "source_index": cache_index,
            "path": f"/synthetic/primitive_cache/train/{dataset}_{ordinal}.npz",
            "size_bytes": 1,
            "file_sha256": "a" * 64,
            "row_count": 1,
            "raw_features_sha256": "b" * 64,
            "opened_members": ["raw_features", "metadata_json"],
            "valid_labels_opened": False,
            "fmt_features_opened": False,
        }
        for cache_index, (family, dataset, ordinal) in enumerate(
            (family, dataset, ordinal)
            for family, dataset in identities
            for ordinal in range(4)
        )
    ]
    assert model.sample_count == len(caches) == 20
    return runner.RawPCAFitResult(
        model=model,
        audit={
            "held_out": "final",
            "outer_raw_features_opened": False,
            "ordered_fit_family_set": (
                "delta_wing",
                "f22_raptor",
                "channel",
                "boeing_747",
            ),
            "ordered_fit_caches": caches,
            "fit_population": "every_valid_raw_row_irrespective_of_label",
        },
    )


def _tiny_final_plan(plan: runner.Plan, outer: str, count: int) -> runner.Plan:
    counts = dict(plan.expected_sample_counts)
    counts[(outer, "final")] = count
    return replace(plan, expected_sample_counts=counts)


def test_final_pca_artifact_round_trip_authentication_and_no_overwrite():
    plan = _tiny_final_plan(runner.load_plan(CONFIG), "half_cylinder", 20)
    fit_families = ("delta_wing", "f22_raptor", "channel", "boeing_747")
    git_commit = "1" * 40
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        pca_path, manifest_path, _, manifest_sha = runner.write_final_pca_artifact(
            output,
            _final_fit(_pca(20)),
            plan=plan,
            outer_family="half_cylinder",
            fit_families=fit_families,
            git_commit=git_commit,
        )
        verified = runner.authenticate_and_rebuild_final_pca(
            pca_path,
            manifest_path,
            plan=plan,
            outer_family="half_cylinder",
            fit_families=fit_families,
            git_commit=git_commit,
            expected_manifest_file_sha256=manifest_sha,
        )
        assert verified.model.sample_count == 20
        assert verified.manifest["labels_opened_for_pca"] is False
        assert verified.manifest["outer_raw_features_opened"] is False
        _expect_error(
            FileExistsError,
            runner.write_final_pca_artifact,
            output,
            _final_fit(_pca(20)),
            plan=plan,
            outer_family="half_cylinder",
            fit_families=fit_families,
            git_commit=git_commit,
        )


def test_scaler_and_calibration_manifests_bind_authenticated_final_pca():
    plan = _tiny_final_plan(runner.load_plan(CONFIG), "half_cylinder", 20)
    fit_families = ("delta_wing", "f22_raptor", "channel", "boeing_747")
    git_commit = "2" * 40
    selected = runner.TailCandidateSpec(
        "raw_pca161", 1, 0.0, "fixed_top_fraction", 0.05
    )
    features = np.zeros((4, RAW_OUTPUT_WIDTH), dtype=np.float32)
    features[:, 0] = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    scales = np.asarray([0, 0, 1000, 1000], dtype=np.int64)
    model = PerScaleNegativeTailModel(
        features,
        scales,
        ks=(1,),
        device="cpu",
        query_chunk_size=2,
        library_chunk_size=2,
    )
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        pca_path, pca_manifest_path, _, pca_manifest_sha = (
            runner.write_final_pca_artifact(
                output,
                _final_fit(_pca(20)),
                plan=plan,
                outer_family="half_cylinder",
                fit_families=fit_families,
                git_commit=git_commit,
            )
        )
        pca = runner.authenticate_and_rebuild_final_pca(
            pca_path,
            pca_manifest_path,
            plan=plan,
            outer_family="half_cylinder",
            fit_families=fit_families,
            git_commit=git_commit,
            expected_manifest_file_sha256=pca_manifest_sha,
        )
        scaler_path, scaler_manifest_path, _, scaler_manifest_sha = (
            runner.write_final_scaler_artifact(
                output,
                model,
                plan=plan,
                selected=selected,
                pca=pca,
                outer_family="half_cylinder",
                fit_families=fit_families,
                git_commit=git_commit,
            )
        )
        scaler = runner.authenticate_and_rebuild_final_scaler(
            scaler_path,
            scaler_manifest_path,
            plan=plan,
            selected=selected,
            pca=pca,
            outer_family="half_cylinder",
            fit_families=fit_families,
            git_commit=git_commit,
            expected_manifest_file_sha256=scaler_manifest_sha,
        )
        calibration_path, calibration_manifest_path, _, calibration_manifest_sha = (
            runner.write_final_calibration_artifact(
                output,
                model,
                plan=plan,
                selected=selected,
                pca=pca,
                scaler=scaler,
                outer_family="half_cylinder",
                fit_families=fit_families,
                git_commit=git_commit,
            )
        )
        calibration = runner.authenticate_and_rebuild_final_calibration(
            calibration_path,
            calibration_manifest_path,
            plan=plan,
            selected=selected,
            pca=pca,
            scaler=scaler,
            outer_family="half_cylinder",
            fit_families=fit_families,
            git_commit=git_commit,
            expected_manifest_file_sha256=calibration_manifest_sha,
        )
        assert scaler.manifest["final_pca"]["file"]["sha256"] == pca.pca_file_sha256
        assert (
            calibration.manifest["final_pca"]["manifest"]["file_sha256"]
            == pca.manifest_file_sha256
        )
        assert calibration.model.ks == (1,)


def test_fresh_prediction_replay_failure_prevents_reference_open():
    persisted = {
        "prediction": np.asarray([False, True], dtype=bool),
        "spatial_score": np.asarray([0.1, 0.9], dtype=np.float64),
        "spatial_denominator": np.asarray([1.0, 1.0], dtype=np.float64),
    }
    reference_opened = []

    def damaged_replay():
        output = {name: np.array(values, copy=True) for name, values in persisted.items()}
        output["prediction"][0] = True
        return output

    def references():
        reference_opened.append(True)
        return "labels"

    _expect_error(
        ValueError,
        runner.replay_then_open_references,
        persisted,
        damaged_replay,
        references,
    )
    assert reference_opened == []
    replayed, labels = runner.replay_then_open_references(
        persisted,
        lambda: {name: np.array(values, copy=True) for name, values in persisted.items()},
        references,
    )
    assert labels == "labels"
    assert reference_opened == [True]
    assert np.array_equal(replayed["prediction"], persisted["prediction"])


def test_outer_reference_function_rejects_any_unsealed_prediction():
    plan = runner.load_plan(CONFIG)
    forged = runner.VerifiedOuterPrediction(
        manifest_path=Path("forged.json"),
        manifest_file_sha256="0" * 64,
        prediction_file_sha256="0" * 64,
        manifest={},
        arrays={},
        replayed_outer=(),
        _authentication_seal=object(),
    )
    _expect_error(
        ValueError,
        runner.load_outer_references_after_prediction,
        plan,
        forged,
    )


def test_raw_runner_atomic_publish_never_replaces_a_concurrent_winner():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        destination = root / "artifact.bin"
        original_link = runner.os.link
        race_triggered = []

        def racing_link(source, target, *args, **kwargs):
            Path(target).write_bytes(b"concurrent-winner")
            race_triggered.append(True)
            return original_link(source, target, *args, **kwargs)

        with patch.object(runner.os, "link", new=racing_link):
            _expect_error(
                FileExistsError,
                runner._atomic_bytes,
                destination,
                b"losing-payload",
            )
        assert race_triggered == [True]
        assert destination.read_bytes() == b"concurrent-winner"
        assert list(root.glob("*.partial")) == []
        assert list(root.glob(".*.partial")) == []

        npz_destination = root / "winner.npz"
        npz_destination.write_bytes(b"existing-winner")
        _expect_error(
            FileExistsError,
            runner._atomic_npz,
            npz_destination,
            {"value": np.arange(4, dtype=np.int32)},
        )
        assert npz_destination.read_bytes() == b"existing-winner"


def test_raw_runner_authenticated_reader_rejects_inode_replacement_identity():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "stable.bin"
        path.write_bytes(b"same-size-content")
        expected_sha = sha256_file(path)
        real_from_stat = runner._OpenFileIdentity.from_stat
        calls = []

        def changed_final_path_identity(value):
            identity = real_from_stat(value)
            calls.append(identity)
            if len(calls) == 5:
                return replace(identity, inode=identity.inode + 1)
            return identity

        with patch.object(
            runner._OpenFileIdentity,
            "from_stat",
            side_effect=changed_final_path_identity,
        ):
            _expect_error(
                ValueError,
                runner._read_authenticated_bytes,
                path,
                expected_size=path.stat().st_size,
                expected_sha256=expected_sha,
            )
        assert len(calls) == 5

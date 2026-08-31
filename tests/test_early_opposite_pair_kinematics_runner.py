from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import io
import inspect
import json
from pathlib import Path
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.early_kinematic_preparation import (  # noqa: E402
    CleanSourceIdentity,
    REQUIRED_SOURCE_PATHS,
)
from pathline_template_matching import (  # noqa: E402
    early_kinematic_preparation as preparation_module,
)
from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_json_sha256,
    sha256_file,
)
from scripts import (  # noqa: E402
    aggregate_verify_early_opposite_pair_kinematics_1_1 as aggregate_module,
)
from scripts import run_verify_early_opposite_pair_kinematics_1_1 as runner  # noqa: E402
from scripts.run_verify_scale_conditioned_retrieval_1_1 import CacheRow  # noqa: E402


CONFIG = ROOT / "config" / "Verify_EarlyOppositePairKinematics_1.1.yaml"


def _expect_error(error_types, function, *args, contains: str | None = None, **kwargs):
    try:
        function(*args, **kwargs)
    except error_types as error:
        if contains is not None:
            assert contains in str(error), str(error)
        return
    raise AssertionError("expected an exception")


def _self_hashed(value: dict[str, object]) -> dict[str, object]:
    output = dict(value)
    output["content_sha256"] = canonical_json_sha256(output)
    return output


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def _fake_source_identity() -> CleanSourceIdentity:
    return CleanSourceIdentity(
        git_commit="1" * 40,
        worktree_clean=True,
        source_file_sha256_items=tuple(
            (path, hashlib.sha256(path.encode("utf-8")).hexdigest())
            for path in REQUIRED_SOURCE_PATHS
        ),
    )


def _population_fixture(root: Path):
    identity = _fake_source_identity()
    input_rows: list[dict[str, object]] = []
    population_rows: list[dict[str, object]] = []
    total = 0
    for dataset, family in runner.PRODUCTION_CONTRACT.dataset_family_pairs:
        for ordinal in range(4):
            source_index = ordinal * 17
            input_rows.append(
                {
                    "dataset": dataset,
                    "physical_family": family,
                    "source_ordinal": ordinal,
                    "source_index": source_index,
                }
            )
            row_root = root / dataset / f"source_{ordinal:02d}_index_{source_index:06d}"
            sidecar = row_root / "seed_time_kinematic4.npz"
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_bytes(f"sidecar:{dataset}:{ordinal}".encode("utf-8"))
            sidecar_sha = sha256_file(sidecar)
            combined = hashlib.sha256(f"combined:{dataset}:{ordinal}".encode()).hexdigest()
            completion = _self_hashed(
                {
                    "dataset": dataset,
                    "source_ordinal": ordinal,
                    "sidecar_file_sha256": sidecar_sha,
                    "sidecar_combined_array_sha256": combined,
                }
            )
            completion_path = row_root / "SIDECAR_COMPLETE.json"
            completion_sha = _write_json(completion_path, completion)
            count = ordinal + 1
            total += count
            population_rows.append(
                {
                    "dataset": dataset,
                    "physical_family": family,
                    "source_ordinal": ordinal,
                    "source_index": source_index,
                    "completion_relative_path": completion_path.relative_to(root).as_posix(),
                    "completion_size_bytes": completion_path.stat().st_size,
                    "completion_file_sha256": completion_sha,
                    "sidecar_relative_path": sidecar.relative_to(root).as_posix(),
                    "sidecar_size_bytes": sidecar.stat().st_size,
                    "sidecar_file_sha256": sidecar_sha,
                    "sidecar_combined_array_sha256": combined,
                    "sidecar_row_count": count,
                }
            )
    input_manifest = {
        "rows": input_rows,
        "content_sha256": "2" * 64,
    }
    input_path = root.parent / "KINEMATIC_INPUT.json"
    synthetic_path = root.parent / "SYNTHETIC_PASS.json"
    population = _self_hashed(
        {
            "schema": runner.POPULATION_MANIFEST_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": "passed",
            "git_commit": identity.git_commit,
            "worktree_clean": True,
            "verify_config_sha256": runner.EXPECTED_CONFIG_SHA256,
            "source_file_sha256": dict(identity.source_file_sha256_items),
            "source_file_sha256_content_sha256": identity.source_content_sha256,
            "input_manifest_path": str(input_path.resolve()),
            "input_manifest_file_sha256": "3" * 64,
            "input_manifest_content_sha256": input_manifest["content_sha256"],
            "synthetic_pass_path": str(synthetic_path.resolve()),
            "synthetic_pass_file_sha256": "4" * 64,
            "composite_descriptor_ids": {
                name: f"{name}_sha256_{'5' * 64}"
                for name in runner.REPRESENTATIONS
            },
            "sidecar_count": 32,
            "sidecar_row_count_total": total,
            "rows": population_rows,
            "rows_content_sha256": canonical_json_sha256(population_rows),
            "forbidden_dataset_access": False,
            "manifest_write_order": "last_after_all_32_completion_markers_and_sidecars_were_authenticated",
        }
    )
    population_path = root / "SIDECAR_POPULATION.json"
    population_sha = _write_json(population_path, population)
    return (
        identity,
        input_manifest,
        input_path.resolve(),
        synthetic_path.resolve(),
        population_path,
        population_sha,
    )


def test_early_plan_candidate_and_independent_schema_contracts_are_exact():
    plan = runner.load_plan(CONFIG)
    assert plan.sha256 == runner.EXPECTED_CONFIG_SHA256
    assert plan.representations == runner.REPRESENTATIONS
    assert len(runner.candidate_specs(plan)) == 3060
    assert plan.required_fold_files == runner.REQUIRED_FOLD_FILES
    assert len(plan.required_fold_files) == 15
    assert runner.RESULT_SCHEMA != "pathline_template_matching.per_scale_negative_metric_result.v1"
    aggregate_module._validate_plan_output_contract(plan)
    assert aggregate_module.EARLY_STOP_CERTIFICATE_SCHEMA == (
        plan.raw["success_stop_rule"]["early_stop_certificate_schema"]
    )


def test_canonical_json_descriptor_order_is_rebuilt_and_values_are_authenticated():
    expected = {
        name: f"{name}_sha256_{index:064x}"
        for index, name in enumerate(runner.REPRESENTATIONS, start=1)
    }
    observed = dict(sorted(expected.items()))
    assert tuple(observed) != runner.REPRESENTATIONS

    authenticated = runner._ordered_composite_descriptor_ids(observed, expected)
    assert tuple(authenticated) == runner.REPRESENTATIONS
    assert dict(authenticated) == expected

    missing = dict(observed)
    missing.pop(runner.REPRESENTATIONS[0])
    _expect_error(
        ValueError,
        runner._ordered_composite_descriptor_ids,
        missing,
        expected,
        contains="population changed",
    )
    tampered = dict(observed)
    tampered[runner.REPRESENTATIONS[0]] = "tampered"
    _expect_error(
        ValueError,
        runner._ordered_composite_descriptor_ids,
        tampered,
        expected,
        contains="values drifted",
    )


def test_preparation_evidence_commit_is_pinned_separately_from_fold_commit():
    current = _fake_source_identity()
    assert current.git_commit != runner.PREPARATION_ARTIFACT_GIT_COMMIT
    preparation = runner._preparation_artifact_identity(current)
    assert preparation.git_commit == runner.PREPARATION_ARTIFACT_GIT_COMMIT
    assert preparation.worktree_clean is True
    assert preparation.source_file_sha256_items == current.source_file_sha256_items


def test_bind_early_evidence_accepts_sorted_json_keys_but_keeps_current_fold_identity():
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory).resolve()
        plan = replace(runner.load_plan(CONFIG), output_root=base)
        current = _fake_source_identity()
        input_path = base / "KINEMATIC_INPUT.json"
        synthetic_path = base / "SYNTHETIC_PASS.json"
        sidecar_root = base / "kinematic_cache" / "train"
        population_path = sidecar_root / "SIDECAR_POPULATION.json"
        input_sha = "3" * 64
        synthetic_sha = "4" * 64
        population_sha = "5" * 64
        expected_ids = {
            name: f"{name}_sha256_{index:064x}"
            for index, name in enumerate(runner.REPRESENTATIONS, start=1)
        }
        input_manifest = {
            "synthetic_pass": {
                "path": str(synthetic_path),
                "file_sha256": synthetic_sha,
            },
            "content_sha256": "6" * 64,
        }
        population = {
            "git_commit": runner.PREPARATION_ARTIFACT_GIT_COMMIT,
            "composite_descriptor_ids": dict(sorted(expected_ids.items())),
            "sidecar_count": 32,
            "rows": [{} for _index in range(32)],
            "content_sha256": "7" * 64,
        }
        contracts = {
            name: {"descriptor_id": descriptor_id}
            for name, descriptor_id in expected_ids.items()
        }

        with (
            patch.object(runner, "capture_clean_source_identity", return_value=current),
            patch.object(runner, "authenticate_synthetic_pass_marker") as synthetic_auth,
            patch.object(
                runner,
                "authenticate_kinematic_input_manifest",
                return_value=input_manifest,
            ) as input_auth,
            patch.object(
                runner,
                "_authenticate_population_envelope_without_sidecar_member_open",
                return_value=population,
            ) as population_auth,
            patch.object(
                runner,
                "composite_descriptor_contracts",
                return_value=contracts,
            ),
        ):
            bound = runner.bind_early_evidence(
                plan,
                kinematic_input_manifest_path=input_path,
                kinematic_input_manifest_file_sha256=input_sha,
                synthetic_pass_path=synthetic_path,
                synthetic_pass_file_sha256=synthetic_sha,
                sidecar_root=sidecar_root,
                sidecar_population_manifest_path=population_path,
                sidecar_population_manifest_file_sha256=population_sha,
            )

        assert bound.source_identity == current
        assert tuple(bound.composite_descriptor_ids) == runner.REPRESENTATIONS
        assert dict(bound.composite_descriptor_ids) == expected_ids
        for mocked in (synthetic_auth, input_auth, population_auth):
            assert (
                mocked.call_args.kwargs["identity"].git_commit
                == runner.PREPARATION_ARTIFACT_GIT_COMMIT
            )


def test_composite_projection_appends_the_same_seed4_without_hidden_transform():
    row = CacheRow("cylinder3d", "half_cylinder", 0, 0, Path("cache.npz"), 1, "0" * 64)
    fmt = np.arange(3 * 161, dtype=np.float32).reshape(3, 161)
    seed4 = np.asarray(
        [[1.0, 2.0, -3.0, 4.0], [5.0, 6.0, 7.0, -8.0], [9.0, 10.0, 11.0, 12.0]],
        dtype=np.float32,
    )
    cache = runner.EarlyCacheProjection(
        row=row,
        fmt_features=fmt,
        seed_kinematic4=seed4,
        scale_ids=np.asarray([0, 1, 1000], dtype=np.int32),
        center_indices=np.asarray([0, 1, 2], dtype=np.int64),
        block_indices=np.asarray([0, 0, 1], dtype=np.int8),
        assigned_row_indices=np.asarray([0, 1, 64002], dtype=np.int64),
        labels=None,
        metadata=MappingProxyType({}),
        sidecar_file_sha256="1" * 64,
        sidecar_combined_array_sha256="2" * 64,
    )
    for representation in runner.REPRESENTATIONS:
        observed = runner.composite_representation_features(cache, representation)
        parent = runner.representation_features(
            fmt, runner.PARENT_REPRESENTATION[representation]
        )
        assert observed.dtype == np.dtype(np.float32)
        assert observed.shape == (3, runner.COMPOSITE_WIDTH[representation])
        assert np.array_equal(observed[:, : parent.shape[1]], parent)
        assert np.array_equal(observed[:, -4:], seed4)


def test_composite_features_drive_the_parent_per_scale_tail_model_end_to_end():
    plan = runner.load_plan(CONFIG)
    row = CacheRow("cylinder3d", "half_cylinder", 0, 0, Path("cache.npz"), 1, "0" * 64)
    fmt = np.zeros((8, 161), dtype=np.float32)
    fmt[:, 0] = np.linspace(0.0, 0.7, 8, dtype=np.float32)
    seed4 = np.zeros((8, 4), dtype=np.float32)
    seed4[:, 0] = np.asarray([0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0], dtype=np.float32)
    cache = runner.EarlyCacheProjection(
        row=row,
        fmt_features=fmt,
        seed_kinematic4=seed4,
        scale_ids=np.zeros(8, dtype=np.int32),
        center_indices=np.arange(8, dtype=np.int64),
        block_indices=np.zeros(8, dtype=np.int8),
        assigned_row_indices=np.arange(8, dtype=np.int64),
        labels=np.asarray([False, False, False, False, True, True, True, True]),
        metadata=MappingProxyType({}),
        sidecar_file_sha256="1" * 64,
        sidecar_combined_array_sha256="2" * 64,
    )
    model = runner._fit_tail_model(
        [cache], "fmt161_plus_seed4", plan, device="cpu", ks=(1,)
    )
    queried = runner._query_cache_batch(
        model,
        [cache],
        "fmt161_plus_seed4",
        plan,
        device="cpu",
        ks=(1,),
    )[1][0]
    assert queried["retrieval_supported"].all()
    assert queried["calibration_supported"].all()
    assert np.isfinite(queried["raw_distance"]).all()
    assert np.isfinite(queried["tail_anomaly"]).all()


def test_population_envelope_authenticates_exact_32_files_without_npz_deserialization():
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        root = base / "kinematic_cache" / "train"
        (
            identity,
            input_manifest,
            input_path,
            synthetic_path,
            population_path,
            population_sha,
        ) = _population_fixture(root)
        original_np_load = np.load

        def forbidden_np_load(*_args, **_kwargs):
            raise AssertionError("population envelope must not open a sidecar member")

        np.load = forbidden_np_load
        try:
            authenticated = runner._authenticate_population_envelope_without_sidecar_member_open(
                population_path,
                expected_file_sha256=population_sha,
                sidecar_root=root.resolve(),
                input_manifest=input_manifest,
                input_manifest_path=input_path,
                input_manifest_file_sha256="3" * 64,
                synthetic_pass_path=synthetic_path,
                synthetic_pass_file_sha256="4" * 64,
                identity=identity,
            )
        finally:
            np.load = original_np_load
        assert authenticated["sidecar_count"] == 32
        assert len(authenticated["rows"]) == 32

        value = json.loads(population_path.read_text(encoding="utf-8"))
        value["rows"][0], value["rows"][1] = value["rows"][1], value["rows"][0]
        value["rows_content_sha256"] = canonical_json_sha256(value["rows"])
        without = dict(value)
        without.pop("content_sha256")
        value["content_sha256"] = canonical_json_sha256(without)
        tampered_sha = _write_json(population_path, value)
        _expect_error(
            ValueError,
            runner._authenticate_population_envelope_without_sidecar_member_open,
            population_path,
            expected_file_sha256=tampered_sha,
            sidecar_root=root.resolve(),
            input_manifest=input_manifest,
            input_manifest_path=input_path,
            input_manifest_file_sha256="3" * 64,
            synthetic_pass_path=synthetic_path,
            synthetic_pass_file_sha256="4" * 64,
            identity=identity,
            contains="reordered",
        )


def test_every_early_artifact_writer_uses_hard_link_no_replace():
    source = inspect.getsource(runner)
    assert "os.replace" not in source
    assert "os.link(temporary, path, follow_symlinks=False)" in source
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "artifact.json"
        first = {"writer": 0}
        first_sha = runner._atomic_json(path, first)
        before = path.read_bytes()
        _expect_error(FileExistsError, runner._atomic_json, path, {"writer": 1})
        assert path.read_bytes() == before
        assert sha256_file(path) == first_sha

        raced = Path(directory) / "race.json"

        def publish(index: int):
            try:
                runner._atomic_json(raced, {"writer": index})
                return True
            except FileExistsError:
                return False

        with ThreadPoolExecutor(max_workers=8) as pool:
            outcomes = list(pool.map(publish, range(8)))
        assert sum(outcomes) == 1
        assert json.loads(raced.read_text(encoding="utf-8"))["writer"] in range(8)


def test_outer_sidecar_and_label_gate_order_is_explicit_and_irreversible():
    source = inspect.getsource(runner.run)
    selection_auth = source.index("authenticate_selected_candidate")
    outer_sidecar_open = source.index("load_early_cache_projection", selection_auth)
    prediction_write = source.index("write_outer_prediction", outer_sidecar_open)
    label_gate = source.index("evaluate_outer_prediction", prediction_write)
    assert selection_auth < outer_sidecar_open < prediction_write < label_gate
    replay = inspect.getsource(runner.load_outer_references_after_prediction)
    assert replay.index("authenticate_outer_prediction") < replay.index("include_labels=True")
    assert "metadata_json_opened" in inspect.getsource(runner.write_outer_prediction)
    access = inspect.getsource(runner.evaluate_outer_prediction)
    assert "sidecar_file_sha256" in access
    assert "label_member_opened_after_prediction_authentication" in access


def test_selection_evidence_swap_restore_cannot_parse_path_bytes_or_open_outer():
    plan = runner.load_plan(CONFIG)
    candidates = runner.candidate_specs(plan)

    def summary_bytes(candidate):
        row = runner._candidate_payload(candidate)
        row.update(
            {
                "accuracy": 0.5,
                "average_precision": 0.5,
                "f1": 0.5,
                "balanced_accuracy": 0.5,
                "auroc": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "retrieval_support_fraction": 1.0,
                "calibration_support_fraction": 1.0,
                "spatial_imputed_fraction": 0.0,
                "spatial_unimputable_fraction": 0.0,
                "inner_family_count": 4,
                "group_count": 1,
            }
        )
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(
            stream,
            fieldnames=runner.SUMMARY_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(row)
        return stream.getvalue().encode("utf-8")

    authenticated_summary = summary_bytes(candidates[0])
    malicious_path_summary = summary_bytes(candidates[1])
    authenticated_sha = hashlib.sha256(authenticated_summary).hexdigest()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        summary_path = root / "inner_candidate_summary.csv"
        group_path = root / "inner_group_metrics.csv"
        audit_path = root / "inner_fit_audits.json"
        summary_path.write_bytes(malicious_path_summary)
        group_path.write_bytes(b"malicious group bytes")
        audit_path.write_bytes(b"malicious audit bytes")
        observed_candidate_ids = []
        outer_opened = []
        original_validate = runner._validate_candidate_csv_identity

        def authenticated_bytes(path, *, expected_sha256):
            if path == summary_path:
                assert expected_sha256 == authenticated_sha
                return authenticated_summary
            return b""

        def record_candidate(raw, candidate):
            observed_candidate_ids.append(str(raw["candidate_id"]))
            return original_validate(raw, candidate)

        def forbidden_outer(*_args, **_kwargs):
            outer_opened.append(True)
            raise AssertionError("outer references opened after failed selection evidence")

        with (
            patch.object(
                runner,
                "_read_authenticated_bytes",
                side_effect=authenticated_bytes,
            ),
            patch.object(
                runner,
                "_validate_candidate_csv_identity",
                side_effect=record_candidate,
            ),
            patch.object(
                runner,
                "load_outer_references_after_prediction",
                side_effect=forbidden_outer,
            ),
        ):
            _expect_error(
                ValueError,
                runner._authenticate_inner_selection_evidence,
                plan=plan,
                selected=None,
                outer_family="half_cylinder",
                inner_group_metrics_path=group_path,
                inner_group_metrics_sha256="1" * 64,
                inner_candidate_summary_path=summary_path,
                inner_candidate_summary_sha256=authenticated_sha,
                inner_fit_audits_path=audit_path,
                inner_fit_audits_sha256="2" * 64,
                contains="complete 3060-candidate set",
            )
        assert observed_candidate_ids == [candidates[0].candidate_id]
        assert candidates[1].candidate_id not in observed_candidate_ids
        assert outer_opened == []

    source = inspect.getsource(runner._authenticate_inner_selection_evidence)
    assert source.count("_read_authenticated_bytes(") == 3
    assert "io.StringIO(summary_bytes.decode" in source
    assert "io.StringIO(group_bytes.decode" in source
    assert "json.loads(audit_bytes.decode" in source
    assert "_stable_file_identity(" not in source
    assert ".read_text(" not in source
    assert ".open(" not in source
    run_source = inspect.getsource(runner.run)
    candidate_match = run_source.index("selected == _in_memory_selected")
    summary_match = run_source.index(
        "selected_summary[name] == _in_memory_selected_summary[name]"
    )
    outer_feature_open = run_source.index("load_early_cache_projection", summary_match)
    assert candidate_match < summary_match < outer_feature_open


def test_verified_array_freeze_is_immutable_bytes_backed():
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    frozen = runner._deep_freeze({"values": source})["values"]
    assert np.array_equal(frozen, source)
    assert frozen.flags.writeable is False
    assert frozen.flags.owndata is False
    _expect_error(ValueError, frozen.setflags, write=True)


def test_parent_cache_single_fd_rejects_final_path_inode_replacement():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "parent_cache.npz"
        path.write_bytes(b"same-size-authenticated-cache-bytes")
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
            def consume_authenticated_descriptor():
                with runner._authenticated_open_file(
                    path,
                    expected_size=path.stat().st_size,
                    expected_sha256=expected_sha,
                ) as opened:
                    assert opened.stream.read() == path.read_bytes()

            _expect_error(
                ValueError,
                consume_authenticated_descriptor,
                contains="path or descriptor changed",
            )
        assert len(calls) == 5
    loader_source = inspect.getsource(runner.load_parent_cache_projection)
    assert "_authenticated_open_file" in loader_source
    assert "np.load(opened.stream" in loader_source
    assert "np.load(row.path" not in loader_source
    for consumer, forbidden_path_load in (
        (runner.authenticate_and_rebuild_final_scaler, "np.load(scaler_path"),
        (runner.authenticate_and_rebuild_final_calibration, "np.load(calibration_path"),
        (runner.authenticate_outer_prediction, "np.load(prediction_path"),
    ):
        source = inspect.getsource(consumer)
        assert "_authenticated_open_file" in source
        assert "np.load(opened.stream" in source
        assert forbidden_path_load not in source


def test_preparation_json_consumers_use_authenticated_single_fd_snapshots():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "authenticated.json"
        path.write_bytes(b'{"status":"passed"}\n')
        expected_sha = sha256_file(path)
        real_from_stat = preparation_module._SnapshotFileIdentity.from_stat
        calls = []

        def changed_final_path_identity(value):
            identity = real_from_stat(value)
            calls.append(identity)
            if len(calls) == 4:
                return replace(identity, inode=identity.inode + 1)
            return identity

        with patch.object(
            preparation_module._SnapshotFileIdentity,
            "from_stat",
            side_effect=changed_final_path_identity,
        ):
            _expect_error(
                RuntimeError,
                preparation_module._read_authenticated_bytes,
                path,
                expected_size=path.stat().st_size,
                expected_sha256=expected_sha,
                contains="path or descriptor changed",
            )
        assert len(calls) == 4

    consumers = (
        preparation_module._authenticate_synthetic_evidence,
        preparation_module.authenticate_synthetic_pass_marker,
        preparation_module._authenticate_portable_marker,
        preparation_module.authenticate_kinematic_input_manifest,
        preparation_module.authenticate_row_completion,
        preparation_module.authenticate_sidecar_population_manifest,
    )
    for consumer in consumers:
        source = inspect.getsource(consumer)
        assert "_json_from_authenticated_snapshot" in source
        assert ".read_text(" not in source

    parent_projection = inspect.getsource(preparation_module._narrow_parent_evidence)
    assert "_read_authenticated_bytes" in parent_projection
    assert "np.load(io.BytesIO(snapshot.content)" in parent_projection
    assert "np.load(path" not in parent_projection
    assert "_stable_file_identity(path)" not in parent_projection


def test_aggregator_replays_label_free_snapshots_before_performance_artifacts():
    label_free = set(aggregate_module.LABEL_FREE_PRE_RESULT_FILES)
    assert "result_manifest.json" not in label_free
    assert "outer_group_metrics.csv" not in label_free
    assert "outer_summary.json" not in label_free
    assert "outer_reference_access_audit.json" not in label_free
    source = inspect.getsource(aggregate_module._authenticate_fold)
    replay = source.index("runner.evaluate_outer_prediction")
    result_open = source.index('_read_file_snapshot(fold_path / "result_manifest.json")')
    metrics_open = source.index('snapshots[name] = _read_file_snapshot', result_open)
    assert replay < result_open < metrics_open


def test_ibex_stages_require_profile_capped_population_and_clean_source_gates():
    common = (ROOT / "ibex" / "verify_early_opposite_pair_kinematics_1.1_common.sh").read_text(
        encoding="utf-8"
    )
    assert "git status --porcelain=v1 --untracked-files=all" in common
    assert 'git show "${expected_commit}:${source}"' in common
    assert "EARLY_CONFIG_SHA256=e6bac456" in common
    assert f"EARLY_RUNNER_SHA256={sha256_file(ROOT / 'scripts' / 'run_verify_early_opposite_pair_kinematics_1_1.py')}" in common
    assert f"EARLY_AGGREGATOR_SHA256={sha256_file(ROOT / 'scripts' / 'aggregate_verify_early_opposite_pair_kinematics_1_1.py')}" in common
    assert f"EARLY_PREPARER_SHA256={sha256_file(ROOT / 'scripts' / 'prepare_verify_early_opposite_pair_kinematics_1_1.py')}" in common
    assert "tangaroa" in common and "smokebuoyancy" in common
    assert "ptm_stage_unchanged" in common
    assert "python tests/test_all.py" in common
    identity_block = common.split("readonly -a EARLY_IDENTITY_SOURCES=(", 1)[1].split("\n)", 1)[0]
    identity_paths = tuple(line.strip() for line in identity_block.splitlines() if line.strip())
    assert identity_paths == REQUIRED_SOURCE_PATHS

    sidecars = (ROOT / "ibex" / "verify_early_opposite_pair_kinematics_1.1_sidecars_limited.sh").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --array=0-31%2" in sidecars
    assert sidecars.index("authenticate-profile") < sidecars.index("build-sidecar")
    assert "--row-index 0" in sidecars

    preparation = (ROOT / "ibex" / "verify_early_opposite_pair_kinematics_1.1_synthetic_input.sh").read_text(
        encoding="utf-8"
    )
    assert "ptm_full_preflight" in preparation
    assert "tests/test_early_kinematic_preparation.py" in preparation
    assert "tests/test_seed_time_kinematic_sidecar.py" in preparation

    folds = (ROOT / "ibex" / "verify_early_opposite_pair_kinematics_1.1_all_folds.sh").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --array=1-4%2" in folds
    assert "authenticate-population" not in folds
    assert "--sidecar-population-manifest-sha256" in folds
    assert "early_stop_certificate.json" in folds
    assert "certificate[\"stop_version\"] is False" in folds

    first = (ROOT / "ibex" / "verify_early_opposite_pair_kinematics_1.1_first_fold.sh").read_text(
        encoding="utf-8"
    )
    assert "--outer-family half_cylinder" in first
    assert "#SBATCH --array" not in first
    first_auth = (ROOT / "ibex" / "verify_early_opposite_pair_kinematics_1.1_first_fold_auth.sh").read_text(
        encoding="utf-8"
    )
    assert "--mode single-fold" in first_auth
    assert "early_stop_certificate.json" in first_auth

    aggregate = (ROOT / "ibex" / "verify_early_opposite_pair_kinematics_1.1_aggregate_five.sh").read_text(
        encoding="utf-8"
    )
    assert "--mode complete-five-fold" in aggregate
    assert aggregate.count("--run-dir") == 5
    assert "FIRST_FOLD_JOB_ID" in aggregate
    assert "REMAINING_FOLD_ARRAY_JOB_ID" in aggregate


def test_remaining_fold_release_recomputes_certificate_and_binds_source_fold():
    source = (
        ROOT / "ibex" / "verify_early_opposite_pair_kinematics_1.1_all_folds.sh"
    ).read_text(encoding="utf-8")
    gate = source.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    compile(gate, "early_remaining_fold_release_gate", "exec")
    assert "expected_names = {" in gate
    assert 'assert {path.name for path in root.iterdir()} == expected_names' in gate
    assert 'completion = self_hashed("AGGREGATE_COMPLETE.json")' in gate
    assert 'manifest = self_hashed(completion["aggregate_manifest_file"])' in gate
    assert 'certificate = self_hashed(record["path"])' in gate
    assert 'table_snapshot.sha256 == report["outer_family_summary_file_sha256"]' in gate
    assert 'Path(source_fold["run_directory"]).resolve() == expected_fold' in gate
    assert 'fold_completion_snapshot.sha256 == source_fold["completion_file_sha256"]' in gate
    assert 'result["artifacts"] == artifacts' in gate
    assert 'fold[field] == outer_summary[field]' in gate
    assert "aggregate._early_stop_certificate(" in gate
    assert "certificate == expected_certificate" in gate
    assert 'certificate["stop_version"] is False' in gate
    assert 'result_early["kinematic_input_manifest"] == early["kinematic_input_manifest"]' in gate
    assert 'result_early["sidecar_population_manifest"] == early["sidecar_population_manifest"]' in gate
    assert "aggregate._require_preparation_release_binding(" in gate
    assert 'input_manifest["composite_descriptor_ids"]' not in gate
    assert 'input_manifest["git_commit"] == commit' not in gate

    plan = runner.load_plan(CONFIG)
    fold = {field: 0.8 for field in aggregate_module.FAMILY_METRIC_FIELDS}
    fold.update({field: 1 for field in aggregate_module.FAMILY_COUNT_FIELDS})
    fold.update(
        {
            "outer_family": "half_cylinder",
            "run_directory": "/expected/first/fold",
            "numerical_git_commit": "1" * 40,
            "config_sha256": plan.sha256,
            "input_manifest_sha256": plan.manifest_sha256,
            "input_manifest_rows_sha256": plan.manifest_rows_sha256,
            "requested_device": "cpu",
            "selected_candidate_id": "candidate",
            "completion_file_sha256": "2" * 64,
            "completion_content_sha256": "3" * 64,
            "result_manifest_file_sha256": "4" * 64,
            "result_manifest_content_sha256": "5" * 64,
            "outer_group_metrics_file_sha256": "6" * 64,
        }
    )
    expected = runner._manifest_with_self_hash(
        aggregate_module._early_stop_certificate(
            plan,
            [fold],
            numerical_git_commit="1" * 40,
        )
    )
    forged = dict(expected)
    forged["observed_outer_families"] = ["delta_wing"]
    forged.pop("content_sha256")
    forged = runner._manifest_with_self_hash(forged)
    assert forged != expected


def test_remaining_release_binding_uses_real_input_schema_and_pinned_producer():
    producer = runner.PREPARATION_ARTIFACT_GIT_COMMIT
    current = "1" * 40
    descriptor_ids = {
        name: f"{name}_sha256_{index:064x}"
        for index, name in enumerate(runner.REPRESENTATIONS, start=1)
    }
    early = {
        "clean_git_commit": current,
        "composite_descriptor_ids": dict(descriptor_ids),
        "kinematic_input_manifest": {"producer_git_commit": producer},
        "synthetic_pass": {"producer_git_commit": producer},
        "sidecar_population_manifest": {"producer_git_commit": producer},
    }
    input_manifest = {
        "git_commit": producer,
        "composite_descriptors": {
            name: {
                "composite_representation": name,
                "descriptor_id": descriptor_id,
            }
            for name, descriptor_id in descriptor_ids.items()
        },
    }
    synthetic = {
        "git_commit": producer,
        "composite_descriptor_ids": dict(descriptor_ids),
    }
    population = {
        "git_commit": producer,
        "composite_descriptor_ids": dict(descriptor_ids),
    }

    def authenticate(
        early_value=early,
        input_value=input_manifest,
        synthetic_value=synthetic,
        population_value=population,
    ):
        aggregate_module._require_preparation_release_binding(
            early=early_value,
            input_manifest=input_value,
            synthetic_marker=synthetic_value,
            population_manifest=population_value,
            current_fold_commit=current,
        )

    authenticate()

    tampered_input = json.loads(json.dumps(input_manifest))
    tampered_input["composite_descriptors"][runner.REPRESENTATIONS[0]][
        "descriptor_id"
    ] = "tampered"
    _expect_error(ValueError, authenticate, input_value=tampered_input, contains="drifted")

    tampered_synthetic = json.loads(json.dumps(synthetic))
    tampered_synthetic["composite_descriptor_ids"][runner.REPRESENTATIONS[0]] = "tampered"
    _expect_error(
        ValueError,
        authenticate,
        synthetic_value=tampered_synthetic,
        contains="drifted",
    )

    tampered_population = json.loads(json.dumps(population))
    tampered_population["composite_descriptor_ids"][runner.REPRESENTATIONS[0]] = "tampered"
    _expect_error(
        ValueError,
        authenticate,
        population_value=tampered_population,
        contains="drifted",
    )

    tampered_nested = json.loads(json.dumps(early))
    tampered_nested["synthetic_pass"]["producer_git_commit"] = current
    _expect_error(
        ValueError,
        authenticate,
        early_value=tampered_nested,
        contains="producer commit drifted",
    )

    for artifact_name, artifact in (
        ("input", input_manifest),
        ("synthetic", synthetic),
        ("population", population),
    ):
        tampered = json.loads(json.dumps(artifact))
        tampered["git_commit"] = current
        kwargs = {f"{artifact_name}_value": tampered}
        if artifact_name == "population":
            kwargs = {"population_value": tampered}
        _expect_error(
            ValueError,
            authenticate,
            contains="producer commit drifted",
            **kwargs,
        )


def test_aggregate_stop_rule_requires_complete_five_families_and_early_schemas():
    plan = runner.load_plan(CONFIG)
    rows = []
    for family, f1 in zip(plan.family_order, (0.80, 0.75, 0.70, 0.65, 0.60)):
        row = {field: 0.8 for field in aggregate_module.FAMILY_METRIC_FIELDS}
        row.update({field: 1 for field in aggregate_module.FAMILY_COUNT_FIELDS})
        row["outer_family"] = family
        row["f1"] = f1
        rows.append(row)
    inputs, outcomes = aggregate_module._stop_rule(plan, rows)
    np.testing.assert_allclose(inputs["family_macro"]["f1"], 0.70)
    assert all(outcomes.values())
    assert "runner.REFERENCE_AUDIT_SCHEMA" in inspect.getsource(
        aggregate_module._authenticate_reference_audit
    )
    aggregate_source = inspect.getsource(aggregate_module.aggregate)
    assert "bind_early_evidence" in aggregate_source
    assert "aggregator_commit == expected_fold_commit" in aggregate_source

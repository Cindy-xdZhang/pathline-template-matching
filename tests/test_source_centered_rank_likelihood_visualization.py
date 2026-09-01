from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT, ROOT / "tests"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_json_sha256,
    sha256_file,
)
from pathline_template_matching.source_centered_rank_likelihood_visualization import (  # noqa: E402
    CENTER_COUNT,
    PANEL_TITLES,
    SCENE_ARRAY_NAMES,
    adapt_unique_primary_prediction,
    adapt_valid_primary_prediction,
    bind_rank_likelihood_table_only_projection,
    bind_rank_likelihood_valid_projection,
    combine_rank_likelihood_parent_scenes,
    publish_bytes_without_overwrite,
    scene_arrays,
)
from scripts import (  # noqa: E402
    aggregate_verify_source_centered_rank_likelihood_template_1_1 as aggregate,
)
from scripts import (  # noqa: E402
    audit_source_centered_rank_likelihood_template_visualizations as auditor,
)
from scripts import (  # noqa: E402
    render_source_centered_rank_likelihood_template_visualizations as reporter,
)
from scripts import (  # noqa: E402
    run_verify_source_centered_rank_likelihood_template_1_1 as runner,
)
from test_source_centered_visualization import (  # noqa: E402
    _synthetic_scene_inputs,
    _unique_prediction as _old_unique_prediction,
    _valid_prediction as _old_valid_prediction,
)


CONFIG = ROOT / "config" / "Other_SourceCenteredRankLikelihoodTemplateVisualization_1.1.yaml"


def _raises(function, *args, contains: str, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - assertion helper checks fail-closed text
        assert contains.casefold() in str(exc).casefold(), str(exc)
    else:
        raise AssertionError(f"expected exception containing {contains!r}")


def _write_self_hashed(path: Path, payload: dict) -> str:
    value = dict(payload)
    value["content_sha256"] = canonical_json_sha256(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return sha256_file(path)


def _rank_unique() -> dict[str, np.ndarray]:
    old = _old_unique_prediction()
    combined = np.asarray(old["legacy_valid"]) | np.asarray(old["expanded_valid"])
    result = {
        name: np.zeros(CENTER_COUNT, dtype=dtype)
        for name, dtype in runner.UNIQUE_PREDICTION_DTYPES.items()
    }
    result.update(
        {
            "unique_dataset": old["unique_dataset"],
            "unique_source_ordinal": old["unique_source_ordinal"],
            "unique_source_index": old["unique_source_index"],
            "unique_center_seed_index": old["unique_center_seed_index"],
            "unique_legacy_valid": old["legacy_valid"],
            "unique_expanded_valid": old["expanded_valid"],
            "unique_combined_valid": combined,
            "unique_primary_spatial_score": old["paired_score"],
            "unique_primary_prediction": old["paired_prediction"],
        }
    )
    return {name: np.asarray(result[name], dtype=dtype) for name, dtype in runner.UNIQUE_PREDICTION_DTYPES.items()}


def _rank_valid() -> dict[str, np.ndarray]:
    old = _old_valid_prediction()
    count = len(old["valid_center_seed_index"])
    result = {
        name: np.zeros(count, dtype=dtype)
        for name, dtype in runner.VALID_PREDICTION_DTYPES.items()
    }
    for name in (
        "valid_dataset",
        "valid_source_ordinal",
        "valid_source_index",
        "valid_scale_id",
        "valid_center_seed_index",
        "valid_scale_block_index",
        "valid_assigned_row_index",
    ):
        result[name] = old[name]
    result["valid_primary_score"] = old["valid_paired_score"]
    result["valid_primary_prediction"] = old["valid_paired_prediction"]
    return {name: np.asarray(result[name], dtype=dtype) for name, dtype in runner.VALID_PREDICTION_DTYPES.items()}


def test_frozen_config_is_primary_only_four_flow_and_exact_18_file_contract() -> None:
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == reporter.CONFIG_SHA256
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    method = value["method_parent"]
    assert value["experiment"] == reporter.EXPERIMENT
    assert value["status"].startswith("frozen_before_reading_any_rank_likelihood")
    assert method["config_sha256"] == runner.EXPECTED_CONFIG_SHA256
    assert method["release_mode_required"] == "complete_five_fold_aggregate"
    assert tuple(method["required_fold_files"]) == runner.REQUIRED_FOLD_FILES
    assert len(method["required_fold_files"]) == 18
    assert tuple(method["prediction_arrays"]["unique"]) == tuple(runner.UNIQUE_PREDICTION_DTYPES)
    assert tuple(method["prediction_arrays"]["valid_projection"]) == tuple(runner.VALID_PREDICTION_DTYPES)
    assert method["plotted_arm"] == "dual_histogram_llr"
    assert method["negative_ecdf_control_is_not_plotted"] is True
    assert method["direct_rank_mean_diagnostic_is_not_plotted"] is True
    assert value["figure_contract"]["evidence_hierarchy"]["aggregate_controls"] == "table_only_not_plotted"
    datasets = value["query"]["datasets"]
    assert value["query"]["source_ordinal"] == 2
    assert tuple(row["id"] for row in datasets) == reporter.DATASETS
    assert {row["id"]: row["source_index"] for row in datasets} == reporter.EXPECTED_SOURCE_INDEX


def test_primary_adapter_reuses_exact_parent_join_without_control_fields() -> None:
    (legacy_metadata, legacy_arrays), (expanded_metadata, expanded_arrays) = _synthetic_scene_inputs()
    unique = _rank_unique()
    valid = _rank_valid()
    adapted_unique = adapt_unique_primary_prediction(unique)
    adapted_valid = adapt_valid_primary_prediction(valid)
    assert set(adapted_unique) == {
        "unique_dataset",
        "unique_source_ordinal",
        "unique_source_index",
        "unique_center_seed_index",
        "paired_score",
        "legacy_valid",
        "expanded_valid",
        "paired_prediction",
    }
    assert set(adapted_valid).isdisjoint({"valid_control_score", "valid_direct_rank_mean_score"})
    scene = combine_rank_likelihood_parent_scenes(
        legacy_metadata=legacy_metadata,
        legacy_arrays=legacy_arrays,
        expanded_metadata=expanded_metadata,
        expanded_arrays=expanded_arrays,
        unique_prediction=unique,
        title="Synthetic",
    )
    projection = bind_rank_likelihood_valid_projection(
        legacy_metadata=legacy_metadata,
        legacy_arrays=legacy_arrays,
        expanded_metadata=expanded_metadata,
        expanded_arrays=expanded_arrays,
        unique_prediction=unique,
        valid_prediction=valid,
    )
    controls = {
        arm: bind_rank_likelihood_table_only_projection(
            arm=arm,
            legacy_metadata=legacy_metadata,
            legacy_arrays=legacy_arrays,
            expanded_metadata=expanded_metadata,
            expanded_arrays=expanded_arrays,
            unique_prediction=unique,
            valid_prediction=valid,
        )
        for arm in reporter.TABLE_ONLY_ARMS
    }
    assert scene.center_seed_index.tolist() == [0, 1, 2]
    assert scene.prediction.tolist() == [False, True, True]
    assert projection.center_seed_index.tolist() == [0, 1, 1, 2]
    assert all(item.center_seed_index.tolist() == [0, 1, 1, 2] for item in controls.values())
    payload = scene_arrays(scene, "{}")
    assert tuple(payload) == SCENE_ARRAY_NAMES
    assert "primary_score" in payload and "paired_score" not in payload


def test_primary_adapter_fails_on_combined_mask_or_valid_projection_drift() -> None:
    unique = _rank_unique()
    unique["unique_combined_valid"] = unique["unique_combined_valid"].copy()
    unique["unique_combined_valid"][0] = False
    _raises(adapt_unique_primary_prediction, unique, contains="combined-valid")
    valid = _rank_valid()
    del valid["valid_primary_prediction"]
    _raises(adapt_valid_primary_prediction, valid, contains="incomplete")


def test_atomic_byte_publication_is_complete_and_never_replaces() -> None:
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "frozen_config.yaml"
        publish_bytes_without_overwrite(target, b"first")
        assert target.read_bytes() == b"first"
        _raises(
            publish_bytes_without_overwrite,
            target,
            b"second",
            contains="overwrite",
        )
        assert target.read_bytes() == b"first"
        assert not tuple(target.parent.glob(f".{target.name}.*.partial"))


def test_output_root_must_be_disjoint_from_every_immutable_input_root() -> None:
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory).resolve()
        immutable = base / "release"
        immutable.mkdir()
        _raises(
            reporter._require_disjoint_output_root,
            immutable / "new_report",
            (immutable,),
            contains="overlaps",
        )
        _raises(
            reporter._require_disjoint_output_root,
            base / "future_output",
            (base / "future_output" / "fold",),
            contains="overlaps",
        )
        reporter._require_disjoint_output_root(base / "report", (immutable,))


def test_producer_metric_replay_preserves_single_class_and_blank_nan_semantics() -> None:
    observed = reporter._metric_values(
        np.zeros(4, dtype=np.bool_),
        np.zeros(4, dtype=np.bool_),
        np.linspace(0.0, 1.0, 4, dtype=np.float64),
    )
    assert observed["balanced_accuracy"] == 0.5
    assert np.isnan(observed["average_precision"])
    expected = {
        field: str(int(observed[field])) for field in reporter.METRIC_INTEGER_FIELDS
    }
    expected.update(
        {
            field: "" if np.isnan(float(observed[field])) else f"{float(observed[field]):.12g}"
            for field in reporter.METRIC_FLOAT_FIELDS
        }
    )
    reporter._compare_metrics(observed, expected, label="single-class")
    expected["average_precision"] = "not-a-number"
    _raises(
        reporter._compare_metrics,
        observed,
        expected,
        label="single-class",
        contains="not numeric",
    )


def _synthetic_fold(root: Path, family: str, commit: str, plan: runner.Plan) -> tuple[str, str]:
    root.mkdir()
    primary = runner.candidate_specs(plan)[0]
    control = runner.control_specs(plan)[0]
    primary_payload = runner._json_safe(runner._candidate_payload(primary))
    control_payload = runner._json_safe(runner._control_payload(control))
    selected_payload = {
        "schema": runner.SELECTED_SCHEMA,
        "experiment": runner.EXPERIMENT,
        "git_commit": commit,
        "config_sha256": runner.EXPECTED_CONFIG_SHA256,
        "outer_family": family,
        "primary_candidate": primary_payload,
        "selected_control": control_payload,
        "outer_results_visible_to_selection": False,
        "outer_sidecar_members_opened": False,
        "outer_labels_opened": False,
    }
    _write_self_hashed(root / "selected_candidate.json", selected_payload)
    (root / "outer_predictions.npz").write_bytes(b"opaque-not-an-npz")
    prediction_manifest = {
        "schema": runner.PREDICTION_MANIFEST_SCHEMA,
        "prediction_schema": runner.PREDICTION_SCHEMA,
        "experiment": runner.EXPERIMENT,
        "git_commit": commit,
        "config_sha256": runner.EXPECTED_CONFIG_SHA256,
        "outer_family": family,
        "primary_candidate": primary_payload,
        "selected_control": control_payload,
        "prediction_file": {
            "path": "outer_predictions.npz",
            "size_bytes": (root / "outer_predictions.npz").stat().st_size,
            "sha256": sha256_file(root / "outer_predictions.npz"),
        },
        "arrays": {name: {} for name in runner.PREDICTION_DTYPES},
        "outer_labels_opened": False,
        "parent_control_prediction_opened": False,
        "fmt_features_opened": False,
        "raw_features_opened": False,
        "reference_labels_all_opened": False,
    }
    _write_self_hashed(root / "outer_prediction_manifest.json", prediction_manifest)
    for name in runner.REQUIRED_FOLD_FILES:
        if name in {
            "selected_candidate.json",
            "outer_predictions.npz",
            "outer_prediction_manifest.json",
            "result_manifest.json",
            "RUN_COMPLETE.json",
        }:
            continue
        (root / name).write_bytes(f"opaque:{family}:{name}".encode("utf-8"))
    release_evidence = {
        "config_sha256": runner.EXPECTED_CONFIG_SHA256,
        "parent_binding_file_sha256": "a" * 64,
        "binding_completion_file_sha256": "b" * 64,
        "historical_source_centered_evidence": runner._json_safe(plan.source_evidence),
    }
    fit_families = [item for item in runner.FAMILY_ORDER if item != family]
    fold_source = {
        "parent_binding": {"path": "/opaque/binding.json", "file_sha256": "a" * 64, "content_sha256": "d" * 64},
        "binding_completion": {"path": "/opaque/BINDING_COMPLETE.json", "file_sha256": "b" * 64},
        "historical_source_centered_evidence": release_evidence["historical_source_centered_evidence"],
        "fit_families": fit_families,
        "config_sha256": runner.EXPECTED_CONFIG_SHA256,
        "fmt_features_opened": False,
        "raw_features_opened": False,
        "reference_labels_all_opened": False,
    }
    artifact_names = [name for name in runner.REQUIRED_FOLD_FILES if name not in {"result_manifest.json", "RUN_COMPLETE.json"}]
    result = {
        "schema": runner.RESULT_SCHEMA,
        "experiment": runner.EXPERIMENT,
        "status": "completed",
        "git_commit": commit,
        "config_sha256": runner.EXPECTED_CONFIG_SHA256,
        "outer_family": family,
        "source_centered_evidence": fold_source,
        "primary_candidate": primary_payload,
        "selected_control": control_payload,
        "selected_candidate_file_sha256": sha256_file(root / "selected_candidate.json"),
        "artifacts": {
            name: {
                "size_bytes": (root / name).stat().st_size,
                "sha256": sha256_file(root / name),
            }
            for name in artifact_names
        },
    }
    result_sha = _write_self_hashed(root / "result_manifest.json", result)
    completion = {
        "schema": runner.COMPLETE_SCHEMA,
        "experiment": runner.EXPERIMENT,
        "outer_family": family,
        "git_commit": commit,
        "config_sha256": runner.EXPECTED_CONFIG_SHA256,
        "result_manifest_file": "result_manifest.json",
        "result_manifest_file_sha256": result_sha,
        "result_manifest_content_sha256": json.loads((root / "result_manifest.json").read_text(encoding="utf-8"))["content_sha256"],
    }
    completion_sha = _write_self_hashed(root / "RUN_COMPLETE.json", completion)
    return completion_sha, result_sha


def _synthetic_release(root: Path, folds: dict[str, tuple[Path, str, str]], commit: str, plan: runner.Plan) -> None:
    root.mkdir()
    (root / "outer_family_summary.csv").write_text("outer_family\n" + "\n".join(runner.FAMILY_ORDER) + "\n", encoding="utf-8")
    evidence = {
        "config_sha256": runner.EXPECTED_CONFIG_SHA256,
        "parent_binding_file_sha256": "a" * 64,
        "binding_completion_file_sha256": "b" * 64,
        "historical_source_centered_evidence": runner._json_safe(plan.source_evidence),
    }
    rows = [
        {
            "outer_family": family,
            "run_directory": str(folds[family][0]),
            "completion_file_sha256": folds[family][1],
            "result_manifest_file_sha256": folds[family][2],
        }
        for family in runner.FAMILY_ORDER
    ]
    report = {
        "schema": aggregate.AGGREGATE_SUMMARY_SCHEMA,
        "experiment": runner.EXPERIMENT,
        "status": "completed",
        "mode": "complete_five_fold_aggregate",
        "config_sha256": runner.EXPECTED_CONFIG_SHA256,
        "aggregator_git_commit": commit,
        "fold_git_commit": commit,
        "source_centered_evidence": evidence,
        "outer_families": list(runner.FAMILY_ORDER),
    }
    report_sha = _write_self_hashed(root / "aggregate_summary.json", report)
    manifest = {
        "schema": aggregate.AGGREGATE_MANIFEST_SCHEMA,
        "experiment": runner.EXPERIMENT,
        "status": "completed",
        "mode": "complete_five_fold_aggregate",
        "config_sha256": runner.EXPECTED_CONFIG_SHA256,
        "aggregator_git_commit": commit,
        "fold_git_commit": commit,
        "source_centered_evidence": evidence,
        "outer_family_summary_file": "outer_family_summary.csv",
        "outer_family_summary_file_sha256": sha256_file(root / "outer_family_summary.csv"),
        "report_file": "aggregate_summary.json",
        "report_file_sha256": report_sha,
        "source_folds": rows,
    }
    manifest_sha = _write_self_hashed(root / "aggregate_manifest.json", manifest)
    completion = {
        "schema": aggregate.AGGREGATE_COMPLETE_SCHEMA,
        "experiment": runner.EXPERIMENT,
        "status": "completed",
        "mode": "complete_five_fold_aggregate",
        "config_sha256": runner.EXPECTED_CONFIG_SHA256,
        "aggregator_git_commit": commit,
        "fold_git_commit": commit,
        "source_centered_evidence": evidence,
        "aggregate_manifest_file": "aggregate_manifest.json",
        "aggregate_manifest_file_sha256": manifest_sha,
        "report_file": "aggregate_summary.json",
        "report_file_sha256": report_sha,
    }
    _write_self_hashed(root / "AGGREGATE_COMPLETE.json", completion)


def test_complete_five_and_two_required_folds_authenticate_opaque_all_18_files() -> None:
    plan = runner.load_plan(ROOT / "config" / "Verify_SourceCenteredRankLikelihoodTemplate_1.1.yaml")
    commit = "c" * 40
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        folds: dict[str, tuple[Path, str, str]] = {}
        for family in runner.FAMILY_ORDER:
            fold = base / f"fold_{family}"
            completion_sha, result_sha = _synthetic_fold(fold, family, commit, plan)
            folds[family] = (fold, completion_sha, result_sha)
        release_root = base / "release"
        _synthetic_release(release_root, folds, commit, plan)
        with patch("numpy.load", side_effect=AssertionError("NPZ member opened before manifest")):
            release = reporter.authenticate_release_root(
                release_root, plan=plan, expected_method_commit=commit
            )
            half = reporter.authenticate_fold_root(
                folds["half_cylinder"][0],
                expected_family="half_cylinder",
                release_record=release.source_folds["half_cylinder"],
                expected_method_commit=commit,
                plan=plan,
                release_evidence=release.source_centered_evidence,
            )
            boeing = reporter.authenticate_fold_root(
                folds["boeing_747"][0],
                expected_family="boeing_747",
                release_record=release.source_folds["boeing_747"],
                expected_method_commit=commit,
                plan=plan,
                release_evidence=release.source_centered_evidence,
            )
        assert len(half.evidence) == len(boeing.evidence) == 18
        assert half.primary_candidate["arm"] == "dual_histogram_llr"
        assert boeing.selected_control["arm"] == "negative_ecdf"
        (folds["half_cylinder"][0] / "extra.txt").write_text("extra", encoding="utf-8")
        _raises(
            reporter.authenticate_fold_root,
            folds["half_cylinder"][0],
            expected_family="half_cylinder",
            release_record=release.source_folds["half_cylinder"],
            expected_method_commit=commit,
            plan=plan,
            release_evidence=release.source_centered_evidence,
            contains="exactly 18",
        )


def test_reporter_orders_opaque_input_manifest_before_any_npz_or_metric_read() -> None:
    source = inspect.getsource(reporter.render_bundle)
    manifest = source.index('_atomic_json(output_root / "input_manifest.json"')
    prediction = source.index("groups = load_prediction_groups")
    metrics = source.index("producer_metrics = read_producer_metric_rows")
    parent = source.index("parent_by_block[block] = _load_parent_scene")
    assert manifest < prediction < metrics < parent
    assert '"fold_sidecar_or_label_member_access": False' in source
    assert '"all_18_files_authenticated_per_required_fold": True' in source


def test_panel_b_and_machine_qa_never_mislabel_primary_as_fmt() -> None:
    assert PANEL_TITLES[1] == "Source-rank likelihood template classification"
    assert "FMT" not in PANEL_TITLES[1]
    render_source = inspect.getsource(reporter)
    render_function_source = inspect.getsource(
        reporter.render_source_rank_likelihood_triptych
    )
    audit_source = inspect.getsource(auditor)
    assert '"plotted_arm": "dual_histogram_llr"' in render_source
    assert '"controls_not_plotted": True' in render_source
    assert '"table_only_control_metrics_reported": True' in render_source
    assert "bind_rank_likelihood_table_only_projection" in render_source
    assert render_function_source.count("_savefig_without_overwrite") == 3
    assert "figure.savefig(requested" not in render_function_source
    assert "shutil.copyfile" not in render_source
    assert "np.load(" not in audit_source
    assert '"--json-out"' not in audit_source
    assert "publish_file_without_overwrite" in audit_source
    for token in (
        "complete_pending_local_rendered_qa",
        "audit_panel_alignment.py",
        "audit_pdf_text.py",
        "audit_figure_collisions.py",
        "delivery_qa_summary.json",
        "refusing to overwrite",
        "Nature-figure QA tool changed during execution",
        "final machine-plus-QA file set changed",
    ):
        assert token in audit_source


def _run_standalone() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
        and callable(value)
        and not inspect.signature(value).parameters
    ]
    assert len(tests) == 9
    for function in tests:
        function()


if __name__ == "__main__":
    _run_standalone()

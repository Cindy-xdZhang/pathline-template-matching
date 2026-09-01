from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "src", ROOT, ROOT / "tests"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from pathline_template_matching.negative_tail_visualization import PredictionGroup, exact_bind_prediction_group  # noqa: E402
from scripts import render_class_conditional_template_score_visualizations as report  # noqa: E402
from pathline_template_matching.portable_flow import canonical_array_sha256, canonical_json_sha256, sha256_file  # noqa: E402


def _raises(function, *args, contains: str, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except ValueError as error:
        assert contains in str(error), str(error)
        return
    raise AssertionError("expected ValueError")


def _source(family: str, experiment: str, root: Path) -> report.SourceReleaseSpec:
    return report.SourceReleaseSpec(
        family, report.EVIDENCE_SOURCES[family], experiment, "1" * 40,
        root / f"{family}.yaml", "2" * 64, root / f"{family}_release",
        "AGGREGATE_COMPLETE.json" if family == "half_cylinder" else "DIAGNOSTIC_COMPLETE.json",
        "3" * 64, root / f"{family}_fold",
    )


def _plan(root: Path) -> report.ReportPlan:
    return report.ReportPlan(
        root / "report.yaml", "4" * 64, {},
        (_source("half_cylinder", report.VERIFY_EXPERIMENT, root),
         _source("boeing_747", report.BOEING_DIAGNOSTIC_EXPERIMENT, root)),
        root / "parent", "Other_Parent_1.1", "5" * 40, "6" * 64, "7" * 64, "8" * 64,
        tuple(report.DatasetSpec(n, report.DISPLAY_NAMES[n], report.DATASET_TO_FAMILY[n]) for n in report.TARGET_DATASETS),
        tuple(report.BlockSpec(n, report.BLOCK_INDEX[n], *report.BLOCK_SCALE_RANGE[n]) for n in report.BLOCKS),
        2, 2500, 360,
    )


def test_contract_is_two_single_fold_releases_without_production_config() -> None:
    source = (ROOT / "scripts" / "render_class_conditional_template_score_visualizations.py").read_text(encoding="utf-8")
    assert report.REPORT_CONFIG_SCHEMA.endswith("visualization_config.v2")
    assert report.SOURCE_RELEASE_MODE == "two_authenticated_single_fold_releases"
    assert "authenticate_aggregate_chain" not in source
    assert "authenticate_single_fold_release" not in source
    assert "authenticate_diagnostic_release" not in source
    assert "aggregate_fresh_replay_authenticated_families" not in source
    manifest_write = source.index('_atomic_json(output_root / "input_manifest.json"')
    prediction_open = source.index("groups = load_prediction_groups(plan, releases)")
    metric_open = source.index("parent_metrics = read_outer_group_metrics(plan, releases)")
    assert manifest_write < prediction_open < metric_open
    assert '"report_time_fresh_replay": False' in source
    assert not (ROOT / "config" / "Other_ClassConditionalTemplateScoreVisualization_1.1.yaml").exists()


def test_non_confirmatory_claims_and_per_release_evidence(tmp_path: Path) -> None:
    evidence = report.SourceReleasesEvidence(
        report.SOURCE_RELEASE_MODE, {},
        {"half_cylinder": {"formal_confirmation": False}, "boeing_747": {"formal_confirmation": False}},
        {}, (),
    )
    assert report._source_release_claims(evidence) == {
        "source_release_mode": report.SOURCE_RELEASE_MODE,
        "source_release_count": 2,
        "authenticated_outer_families": ["half_cylinder", "boeing_747"],
        "complete_five_fold": False,
        "five_fold_success_evaluated": False,
        "formal_confirmation": False,
    }
    assert [row["evidence_source"] for row in report._source_release_manifest_rows(_plan(tmp_path), evidence)] == list(report.EVIDENCE_SOURCES.values())


def test_boeing_forbids_stop_success_and_macro_claims() -> None:
    for value in ({"stop_version": False}, {"five_fold_success": None}, {"five_family_macro_f1": 0.1}):
        _raises(report._require_no_forbidden_release_claims, value, path="synthetic", contains="forbidden Boeing")
    report._require_no_forbidden_release_claims({"formal_confirmation": False}, path="synthetic")


def test_method_projection_allows_release_identity_only() -> None:
    common = {"score": {"combine": "mean"}, "threshold": {"comparison": "strict_greater_than"},
              "prediction_array_contract": "unchanged_parent_19_arrays", "fold_transaction": "unchanged_parent_15_files"}
    half = {"experiment": report.VERIFY_EXPERIMENT, "config": {"sha256": "1" * 64}, **common}
    other = {"experiment": report.BOEING_DIAGNOSTIC_EXPERIMENT, "config": {"sha256": "2" * 64}, "adapter": "boeing", **common}
    assert report._method_binding_projection(half) == report._method_binding_projection(other)
    changed = {**other, "threshold": {"comparison": "greater_than_or_equal"}}
    assert report._method_binding_projection(half) != report._method_binding_projection(changed)


def test_dataset_scope_is_three_half_and_one_boeing() -> None:
    assert [report.DATASET_TO_FAMILY[n] for n in report.TARGET_DATASETS] == ["half_cylinder"] * 3 + ["boeing_747"]
    assert len(report.TARGET_DATASETS) * len(report.BLOCKS) == 8


def test_identity_join_rejects_reordered_rows() -> None:
    metadata = {"dataset": "cylinder3d", "source_ordinal": 2, "source_index": 4, "scale_block_id": "legacy_0000_0999"}
    parent = {"valid_center_seed_index": np.array([0, 1]), "valid_assigned_row_index": np.array([0, 1]), "valid_scale_id": np.array([2, 3]), "valid_scale_block_index": np.array([0, 0])}
    group = PredictionGroup(
        dataset="cylinder3d", source_ordinal=2, source_index=4,
        block="legacy_0000_0999", outer_family="half_cylinder", candidate={},
        center_seed_index=np.array([1, 0]), assigned_row_index=np.array([1, 0]),
        scale_id=np.array([3, 2]), scale_block_index=np.array([0, 0]),
        spatial_score=np.ones(2), prediction=np.array([True, False]),
    )
    _raises(exact_bind_prediction_group, metadata, parent, group, contains="row order")


def test_existing_output_rejected_before_config_access(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    try:
        report.run_from_config(config_path=tmp_path / "missing.yaml", config_sha256="0" * 64,
                               output_root=output, expected_reporting_commit="1" * 40)
    except FileExistsError:
        return
    raise AssertionError("expected immutable output rejection")


def test_environment_records_cpu_and_scheduler_identity() -> None:
    value = report._environment_record("cpu")
    assert value["requested_device"] == "cpu"
    assert isinstance(value["hostname"], str) and value["hostname"]
    assert "slurm_job_id" in value and "slurm_array_task_id" in value


def test_prediction_contract_is_exactly_nineteen_typed_arrays() -> None:
    assert len(report.PREDICTION_ARRAY_NAMES) == 19
    assert tuple(report.PREDICTION_DTYPES) == report.PREDICTION_ARRAY_NAMES
    assert report.PREDICTION_DTYPES["prediction"] == "|b1"
    assert report.PREDICTION_DTYPES["spatial_score"] == "<f8"


def test_opaque_release_and_fold_authentication_precedes_array_open() -> None:
    source = (ROOT / "scripts" / "render_class_conditional_template_score_visualizations.py").read_text(encoding="utf-8")
    release_auth = source.index("releases = authenticate_source_release_chains(plan)")
    input_write = source.index('_atomic_json(output_root / "input_manifest.json"')
    array_open = source.index("groups = load_prediction_groups(plan, releases)")
    assert release_auth < input_write < array_open
    assert "set(FOLD_FILE_NAMES)" in source
    assert 'source.get("artifact_count") == 13' in source


def test_parent_provenance_and_scene_invariance_are_fail_closed() -> None:
    source = (ROOT / "scripts" / "render_class_conditional_template_score_visualizations.py").read_text(encoding="utf-8")
    for required in (
        "parent result manifest SHA changed",
        "parent completion SHA changed",
        "parent result config changed",
        "child scene changed immutable parent array",
        "only_prediction_and_metadata_changed",
    ):
        assert required in source


def test_real_synthetic_boeing_release_envelope_and_mutation(tmp_path: Path) -> None:
    import json
    from dataclasses import replace
    import test_other_class_conditional_template_score_boeing_diagnostic_aggregate as fixture

    plan, _bound, fold, output, _payload = fixture._publish(tmp_path)
    spec = report.SourceReleaseSpec(
        "boeing_747", report.EVIDENCE_SOURCES["boeing_747"],
        report.BOEING_DIAGNOSTIC_EXPERIMENT, fixture.COMMIT,
        fixture.CONFIG.resolve(), plan.sha256, output.resolve(),
        "DIAGNOSTIC_COMPLETE.json", sha256_file(output / "DIAGNOSTIC_COMPLETE.json"),
        fold.path.resolve(),
    )
    release_report = json.loads((output / "boeing_diagnostic_report.json").read_text(encoding="utf-8"))
    synthetic_summary = fixture.aggregate._self_hashed(
        {"class_conditional_support": release_report["class_conditional_support"]}
    )
    (fold.path / "outer_summary.json").write_bytes(
        fixture.aggregate._canonical_json_bytes(synthetic_summary)
    )
    fake_fold = report.FoldEvidence(
        "boeing_747", fold.path.resolve(),
        {"input_manifest": {"sha256": release_report["input_manifest_sha256"],
                            "rows_content_sha256": release_report["input_manifest_rows_sha256"]}},
        {}, {}, {}, {},
        fixture.aggregate._method_binding(plan, fixture.COMMIT),
    )
    with patch.object(report, "_authenticate_fold_chain", return_value=(fake_fold, [])):
        evidence, release, files = report._authenticate_boeing_release_opaque(spec)
    assert evidence.outer_family == "boeing_747"
    assert release["formal_confirmation"] is False
    assert len(files) == 4

    completion_path = output / "DIAGNOSTIC_COMPLETE.json"
    changed = json.loads(completion_path.read_text(encoding="utf-8"))
    changed.pop("content_sha256")
    changed["unexpected"] = True
    changed = fixture.aggregate._self_hashed(changed)
    completion_path.write_bytes(fixture.aggregate._canonical_json_bytes(changed))
    changed_spec = replace(spec, completion_sha256=sha256_file(completion_path))
    with patch.object(report, "_authenticate_fold_chain", return_value=(fake_fold, [])):
        _raises(report._authenticate_boeing_release_opaque, changed_spec, contains="provenance changed")


def _synthetic_candidate() -> dict:
    return {
        "candidate_id": "representation=chirality_all35|k=1|sigma=0.0|fixed_top_fraction=0.05",
        "representation": "chirality_all35", "k": 1, "sigma": 0.0,
        "decision_rule": "fixed_top_fraction", "decision_value": 0.05,
    }


def _synthetic_binding(experiment: str, config_sha: str, commit: str) -> dict:
    return {
        "experiment": experiment, "config": {"sha256": config_sha},
        "score": {"combine": "equal_mean_over_jointly_supported_families",
                  "inner_support": "2_of_3_joint_families", "outer_support": "3_of_4_joint_families"},
        "threshold": {"comparison": "strict_greater_than", "equality_prediction": "negative"},
        "prediction_array_contract": "unchanged_parent_19_arrays",
        "fold_transaction": "unchanged_parent_15_files", "numerical_git_commit": commit,
    }


def _write_self_hash(path: Path, payload: dict) -> tuple[dict, str]:
    value = dict(payload)
    value["content_sha256"] = canonical_json_sha256(value)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return value, sha256_file(path)


def _write_synthetic_fold(spec: report.SourceReleaseSpec, plan: report.ReportPlan) -> dict:
    root = spec.fold_root
    root.mkdir(parents=True)
    candidate = _synthetic_candidate()
    binding = _synthetic_binding(spec.experiment, spec.config_sha256, spec.numerical_commit)
    datasets = [d.dataset for d in plan.datasets if d.outer_family == spec.outer_family]
    values = {name: [] for name in report.PREDICTION_ARRAY_NAMES}
    audits = []
    fit = [family for family in report.FAMILY_ORDER if family != spec.outer_family]
    for dataset_index, dataset in enumerate(datasets):
        for ordinal in range(4):
            source_index = 100 * dataset_index + ordinal
            for block, block_index in report.BLOCK_INDEX.items():
                for center, score in enumerate((0.25, 0.75)):
                    row = {
                        "dataset": dataset, "source_ordinal": ordinal, "source_index": source_index,
                        "scale_id": block_index * 1000 + center, "center_seed_index": center,
                        "scale_block_index": block_index,
                        "assigned_row_index": block_index * plan.assigned_per_block + center,
                        "raw_negative_distance": 1.0 - score, "tail_probability": 1.0 - score,
                        "tail_anomaly": score, "spatial_score": score, "spatial_denominator": 4.0,
                        "retrieval_supported": True, "calibration_supported": True,
                        "spatial_imputed": False, "spatial_unimputable": False,
                        "calibration_mode": 1, "scaler_mode": 1, "prediction": score > 0.5,
                    }
                    for name in values:
                        values[name].append(row[name])
                audits.append({
                    "dataset": dataset, "source_ordinal": ordinal, "source_index": source_index,
                    "block": block, "sample_count": 2, "retrieval_supported_count": 2,
                    "calibration_supported_count": 2, "imputed_count": 0,
                    "unimputable_count": 0, "prediction_count": 1,
                    "class_conditional_support": {"sample_count": 2, "family_order": fit,
                        "required_joint_family_count": 3,
                        "joint_supported_family_count_histogram": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 2},
                        "families": {family: {} for family in fit}},
                })
    arrays = {name: np.asarray(values[name], dtype=np.dtype(report.PREDICTION_DTYPES[name]))
              for name in report.PREDICTION_ARRAY_NAMES}
    np.savez(root / "outer_predictions.npz", **arrays)
    _write_self_hash(root / "selected_candidate.json", {
        "schema": report.PREDICTION_SELECTED_SCHEMA, "experiment": spec.experiment,
        "git_commit": spec.numerical_commit, "config_sha256": spec.config_sha256,
        "outer_family": spec.outer_family, "candidate": candidate, report.METHOD_BINDING_KEY: binding})
    _write_self_hash(root / "outer_prediction_manifest.json", {
        "schema": report.PREDICTION_MANIFEST_SCHEMA, "prediction_schema": report.PREDICTION_SCHEMA,
        "experiment": spec.experiment, "git_commit": spec.numerical_commit,
        "config_sha256": spec.config_sha256, "outer_family": spec.outer_family,
        "selected_candidate": candidate, "valid_labels_opened": False,
        "metadata_json_opened": False, "array_count": 19, "row_count": len(arrays["dataset"]),
        "arrays": {name: {"dtype": array.dtype.str, "shape": list(array.shape),
                           "sha256": canonical_array_sha256(array)} for name, array in arrays.items()},
        "prediction_file": {"path": "outer_predictions.npz", "size_bytes": (root / "outer_predictions.npz").stat().st_size,
                            "sha256": sha256_file(root / "outer_predictions.npz")},
        "group_audits": audits, report.METHOD_BINDING_KEY: binding})
    for name in report.FOLD_ARTIFACT_NAMES:
        path = root / name
        if not path.exists():
            path.write_text(f"synthetic {name}\n", encoding="utf-8")
    identities = {name: {"size_bytes": (root / name).stat().st_size, "sha256": sha256_file(root / name)}
                  for name in report.FOLD_ARTIFACT_NAMES}
    result, result_sha = _write_self_hash(root / "result_manifest.json", {
        "schema": report.PREDICTION_RESULT_SCHEMA, "experiment": spec.experiment, "status": "completed",
        "git_commit": spec.numerical_commit, "config_sha256": spec.config_sha256,
        "outer_family": spec.outer_family, "selected_candidate": candidate,
        "artifacts": identities, report.METHOD_BINDING_KEY: binding})
    _, completion_sha = _write_self_hash(root / "RUN_COMPLETE.json", {
        "schema": report.PREDICTION_COMPLETE_SCHEMA, "experiment": spec.experiment,
        "git_commit": spec.numerical_commit, "config_sha256": spec.config_sha256,
        "outer_family": spec.outer_family, "result_manifest_file": "result_manifest.json",
        "result_manifest_file_sha256": result_sha,
        "result_manifest_content_sha256": result["content_sha256"], report.METHOD_BINDING_KEY: binding})
    return {"outer_family": spec.outer_family, "run_directory": str(root.resolve()),
            "completion_file_sha256": completion_sha, "result_manifest_file_sha256": result_sha,
            "artifact_count": 13, "artifacts": identities}


def test_real_fold_chain_prediction_members_metrics_and_join(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    folds = {}
    for spec in plan.source_releases:
        source = _write_synthetic_fold(spec, plan)
        fold, _files = report._authenticate_fold_chain(spec, source)
        folds[spec.outer_family] = fold
    releases = report.SourceReleasesEvidence(report.SOURCE_RELEASE_MODE, folds, {}, {}, ())
    groups = report.load_prediction_groups(plan, releases)
    assert len(groups) == 8
    loaded = groups[("cylinder3d", report.BLOCKS[0])]
    metadata = {"dataset": "cylinder3d", "source_ordinal": 2,
                "source_index": loaded.group.source_index, "scale_block_id": report.BLOCKS[0]}
    parent = {"valid_center_seed_index": loaded.group.center_seed_index.copy(),
              "valid_assigned_row_index": loaded.group.assigned_row_index.copy(),
              "valid_scale_id": loaded.group.scale_id.copy(),
              "valid_scale_block_index": loaded.group.scale_block_index.copy()}
    prediction, _score = exact_bind_prediction_group(metadata, parent, loaded.group)
    metric = report.recompute_complete_metric_row(reference=np.asarray([False, True]), loaded=loaded)
    report.compare_metric_rows(metric, dict(metric))
    assert prediction.tolist() == [False, True] and metric["accuracy"] == 1.0

    prediction_path = folds["half_cylinder"].root / "outer_predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as archive:
        mutated = {name: np.asarray(archive[name]) for name in archive.files}
    mutated["spatial_score"] = mutated["spatial_score"].copy()
    mutated["spatial_score"][0] += 0.01
    np.savez(prediction_path, **mutated)
    _raises(report.load_prediction_groups, plan, releases, contains="array SHA changed")

    drifted = dict(metric)
    drifted["accuracy"] += 1.0e-6
    _raises(report.compare_metric_rows, metric, drifted, contains="metric mismatch: accuracy")


def _run_standalone() -> None:
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    assert len(tests) == 13
    for function in tests:
        if inspect.signature(function).parameters:
            with tempfile.TemporaryDirectory() as directory:
                function(Path(directory))
        else:
            function()


if __name__ == "__main__":
    _run_standalone()

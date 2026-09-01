from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.class_conditional_template_score import (  # noqa: E402
    ClassConditionalTemplateScoreModel,
)
from pathline_template_matching.early_kinematic_preparation import (  # noqa: E402
    CleanSourceIdentity,
    REQUIRED_SOURCE_PATHS,
)
from scripts import run_verify_class_conditional_template_score_1_1 as runner  # noqa: E402
from scripts import run_verify_early_opposite_pair_kinematics_1_1 as parent  # noqa: E402
from scripts.run_verify_scale_conditioned_retrieval_1_1 import CacheRow  # noqa: E402


CONFIG = ROOT / "config" / "Verify_ClassConditionalTemplateScore_1.1.yaml"


def _expect_error(error_types, function, *args, contains: str | None = None, **kwargs):
    try:
        function(*args, **kwargs)
    except error_types as error:
        if contains is not None:
            assert contains in str(error), str(error)
        return error
    raise AssertionError("expected an exception")


def _cache(
    family: str,
    family_index: int,
    *,
    labels: bool,
    query: bool = False,
) -> runner.EarlyCacheProjection:
    if query:
        values = np.asarray([0.1, 5.1, 0.1, 5.1], dtype=np.float32)
        scales = np.asarray([0, 0, 1000, 1000], dtype=np.int32)
        block_indices = np.asarray([0, 0, 1, 1], dtype=np.int8)
        centers = np.asarray([0, 1, 0, 1], dtype=np.int64)
        targets = None
    else:
        values = np.asarray(
            [
                0.0,
                0.2,
                5.0,
                5.2,
                0.0,
                0.2,
                5.0,
                5.2,
            ],
            dtype=np.float32,
        ) + np.float32(0.03 * family_index)
        scales = np.asarray([0, 0, 0, 0, 1000, 1000, 1000, 1000], dtype=np.int32)
        block_indices = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
        centers = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        targets = np.asarray(
            [False, False, True, True, False, False, True, True],
            dtype=np.bool_,
        )
    fmt = np.zeros((len(values), 161), dtype=np.float32)
    fmt[:, 0] = values
    seed4 = np.zeros((len(values), 4), dtype=np.float32)
    seed4[:, 0] = values * np.float32(0.1)
    assigned = block_indices.astype(np.int64) * 64000 + centers
    row = CacheRow(
        dataset=f"synthetic_{family}",
        family=family,
        source_ordinal=0,
        source_index=family_index,
        path=Path(f"synthetic_{family}.npz"),
        size_bytes=1,
        sha256=f"{family_index + 1:064x}",
    )
    return runner.EarlyCacheProjection(
        row=row,
        fmt_features=fmt,
        seed_kinematic4=seed4,
        scale_ids=scales,
        center_indices=centers,
        block_indices=block_indices,
        assigned_row_indices=assigned,
        labels=targets if labels else None,
        metadata=MappingProxyType({}),
        sidecar_file_sha256="a" * 64,
        sidecar_combined_array_sha256="b" * 64,
    )


def _synthetic_model():
    plan = replace(
        runner.load_plan(CONFIG),
        ks=(1,),
        sigmas=(0.0,),
        thresholds=(0.5,),
        query_chunk_size=16,
        library_chunk_size=16,
    )
    families = plan.family_order[:3]
    caches = [_cache(family, index, labels=True) for index, family in enumerate(families)]
    model = runner._fit_tail_model(
        caches,
        "fmt161_plus_seed4",
        plan,
        device="cpu",
        ks=(1,),
    )
    return plan, caches, model


def test_plan_candidate_and_parent_transaction_contracts_are_exact():
    plan = runner.load_plan(CONFIG)
    assert plan.sha256 == runner.EXPECTED_CONFIG_SHA256
    assert plan.family_order == runner.FAMILY_ORDER
    assert plan.representations == runner.REPRESENTATIONS
    assert plan.required_fold_files == runner.REQUIRED_FOLD_FILES
    assert len(runner.REQUIRED_FOLD_FILES) == 15
    assert tuple(runner.PREDICTION_ARRAY_DTYPES) == tuple(parent.PREDICTION_ARRAY_DTYPES)
    assert len(runner.PREDICTION_ARRAY_DTYPES) == 19
    candidates = runner.candidate_specs(plan)
    assert len(candidates) == 3060
    assert len({candidate.candidate_id for candidate in candidates}) == 3060
    threshold = next(
        candidate
        for candidate in candidates
        if candidate.decision_rule == "calibrated_tail_anomaly_threshold"
    )
    fixed = next(
        candidate
        for candidate in candidates
        if candidate.decision_rule == "fixed_top_fraction"
    )
    assert "comparator=strict_greater_than" in threshold.candidate_id
    assert "comparator=" not in fixed.candidate_id
    assert plan.output_root == runner.DEFAULT_OUTPUT_ROOT


def test_plan_fails_closed_before_parsing_a_changed_child_config():
    with tempfile.TemporaryDirectory() as directory:
        changed = Path(directory) / CONFIG.name
        changed.write_bytes(CONFIG.read_bytes() + b"\n")
        _expect_error(
            ValueError,
            runner.load_plan,
            changed,
            contains="frozen child config SHA-256 drifted",
        )


def test_strict_threshold_ties_are_negative_in_scalar_and_series_paths():
    scores = np.asarray([0.49, 0.50, 0.51, 0.50], dtype=np.float64)
    labels = np.asarray([False, True, True, False], dtype=np.bool_)
    eligible = np.asarray([True, True, True, False], dtype=np.bool_)
    centers = np.arange(4, dtype=np.int64)
    candidate = runner.TailCandidateSpec(
        "fmt161_plus_seed4",
        1,
        0.0,
        "calibrated_tail_anomaly_threshold",
        0.50,
    )
    prediction = runner.candidate_predictions(candidate, scores, centers, eligible)
    assert np.array_equal(prediction, np.asarray([False, False, True, False]))
    metric = runner._threshold_confusion_series(
        labels, scores, eligible, (0.50,)
    )[0]
    assert metric["true_positive"] == 1
    assert metric["false_positive"] == 0
    assert metric["true_negative"] == 2
    assert metric["false_negative"] == 1


def test_family_aware_fit_and_query_map_every_compatibility_field():
    plan, caches, model = _synthetic_model()
    assert isinstance(model, ClassConditionalTemplateScoreModel)
    assert model.family_order == plan.family_order[:3]
    assert model.required_family_count == 2
    audit = model.fit_audit
    assert audit["fit_family_count"] == 3
    assert tuple(audit["family_order"]) == plan.family_order[:3]
    queried = runner._query_cache_batch(
        model,
        caches,
        "fmt161_plus_seed4",
        plan,
        device="cpu",
        ks=(1,),
    )[1]
    assert len(queried) == 3
    for values, cache in zip(queried, caches, strict=True):
        assert values["family_order"] == model.family_order
        assert values["required_family_count"] == 2
        assert values["raw_distance"].dtype == np.dtype(np.float32)
        assert values["retrieval_supported"].all()
        assert values["calibration_supported"].all()
        assert np.array_equal(values["joint_family_count"], np.full(cache.count, 3))
        assert np.allclose(
            values["tail_probability"] + values["tail_anomaly"], 1.0,
            rtol=0.0,
            atol=1.0e-15,
        )
        assert np.array_equal(
            values["calibration_mode"], np.ones(cache.count, dtype=np.int8)
        )
        for field in runner._PER_FAMILY_SUPPORT_FIELDS:
            support = values[field]
            assert support.dtype == np.dtype(np.bool_)
            assert support.shape == (cache.count, 3)
            assert support.all()


def test_fit_family_with_an_absent_positive_class_uses_joint_support_gate():
    plan, caches, _model = _synthetic_model()
    missing_family = plan.family_order[2]
    absent_positive = replace(
        caches[2], labels=np.zeros(caches[2].count, dtype=np.bool_)
    )
    model = runner._fit_tail_model(
        [caches[0], caches[1], absent_positive],
        "fmt161_plus_seed4",
        plan,
        device="cpu",
        ks=(1,),
    )
    assert model.calibrator_for(missing_family, positive=True) is None
    query = _cache("held_out", 9, labels=False, query=True)
    values = runner._query_cache_batch(
        model,
        [query],
        "fmt161_plus_seed4",
        plan,
        device="cpu",
        ks=(1,),
    )[1][0]
    assert values["retrieval_supported"].all()
    assert values["calibration_supported"].all()
    assert np.array_equal(values["joint_family_count"], np.full(query.count, 2))
    missing_index = values["family_order"].index(missing_family)
    assert not values["per_family_positive_retrieval_supported"][:, missing_index].any()
    assert not values["per_family_positive_calibration_supported"][:, missing_index].any()
    assert values["per_family_negative_retrieval_supported"][:, missing_index].all()


def test_outer_builder_keeps_19_arrays_and_replays_manifest_support_audits():
    plan, _caches, model = _synthetic_model()
    query = _cache("held_out", 9, labels=False, query=True)
    selected = runner.TailCandidateSpec(
        "fmt161_plus_seed4",
        1,
        0.0,
        "calibrated_tail_anomaly_threshold",
        0.50,
    )
    arrays, groups = runner.build_outer_prediction_arrays(
        [query], model, selected, plan, device="cpu"
    )
    replay_arrays, replay_groups = runner.build_outer_prediction_arrays(
        [query], model, selected, plan, device="cpu"
    )
    assert tuple(arrays) == tuple(runner.PREDICTION_ARRAY_DTYPES)
    assert len(arrays) == 19
    assert len(groups) == 2
    assert groups == replay_groups
    for name, dtype in runner.PREDICTION_ARRAY_DTYPES.items():
        assert arrays[name].dtype == dtype
        if np.issubdtype(dtype, np.floating):
            assert np.array_equal(
                arrays[name], replay_arrays[name], equal_nan=True
            )
        else:
            assert np.array_equal(arrays[name], replay_arrays[name])
    for group in groups:
        support = group["class_conditional_support"]
        assert support["sample_count"] == 2
        assert support["required_joint_family_count"] == 2
        assert support["joint_supported_family_count_histogram"] == {
            "0": 0,
            "1": 0,
            "2": 0,
            "3": 2,
        }
        for family in model.family_order:
            family_audit = support["families"][family]
            for class_name in ("positive", "negative"):
                for support_name in ("retrieval", "calibration"):
                    assert family_audit[f"{class_name}_{support_name}_count"] == 2
                    assert family_audit[f"{class_name}_{support_name}_fraction"] == 1.0
    persisted_groups = json.loads(json.dumps(groups, sort_keys=True))
    summary = runner._aggregate_group_support_audits(persisted_groups)
    assert summary["sample_count"] == 4
    assert summary["joint_supported_family_count_histogram"]["3"] == 4
    for family in model.family_order:
        assert summary["families"][family]["positive_retrieval_count"] == 4
        assert summary["families"][family]["negative_calibration_fraction"] == 1.0


def test_runtime_rebinds_only_inside_transaction_and_restores_after_error():
    _mini_plan, _caches, model = _synthetic_model()
    plan = runner.load_plan(CONFIG)
    commit = "1" * 40
    names = (
        "EXPERIMENT",
        "TailCandidateSpec",
        "PerScaleNegativeTailModel",
        "_fit_tail_model",
        "_query_cache_batch",
        "candidate_predictions",
        "_threshold_confusion_series",
        "build_outer_prediction_arrays",
        "evaluate_outer_prediction",
        "_manifest_with_self_hash",
        "_authenticate_self_hash",
    )
    before = {name: getattr(parent, name) for name in names}
    try:
        with runner.class_conditional_parent_runtime(plan, commit):
            assert parent.EXPERIMENT == runner.EXPERIMENT
            assert parent.TailCandidateSpec is runner.TailCandidateSpec
            assert parent.PerScaleNegativeTailModel is ClassConditionalTemplateScoreModel
            assert tuple(
                candidate.candidate_id for candidate in parent.candidate_specs(plan)
            ) == tuple(
                candidate.candidate_id for candidate in runner.candidate_specs(plan)
            )
            rebuilt = parent.PerScaleNegativeTailModel.from_artifacts(
                model.scaler.export_arrays(), model.tail_calibrator.export_arrays()
            )
            assert rebuilt.family_order == model.family_order
            manifest = parent._manifest_with_self_hash({"schema": "synthetic.v1"})
            assert manifest[runner.METHOD_BINDING_KEY] == runner._method_binding(
                plan, commit
            )
            parent._authenticate_self_hash(manifest)
            _expect_error(
                ValueError,
                lambda: runner.class_conditional_parent_runtime(plan, commit).__enter__(),
                contains="nested or concurrent use is forbidden",
            )
            raise RuntimeError("synthetic transaction exit")
    except RuntimeError as error:
        assert str(error) == "synthetic transaction exit"
    for name, value in before.items():
        assert getattr(parent, name) is value


def test_fresh_final_artifact_chain_precedes_any_parent_outer_feature_access():
    plan = runner.load_plan(CONFIG)
    selected = runner.candidate_specs(plan)[0]
    events = []
    scaler = object()
    calibration = object()

    def authenticate_scaler(*_args, **_kwargs):
        events.append("final_scaler_fresh_auth")
        return scaler

    def authenticate_calibration(*_args, **kwargs):
        assert kwargs["scaler"] is scaler
        events.append("final_calibration_fresh_auth")
        return calibration

    def authenticate_selected(*_args, **kwargs):
        assert kwargs["scaler"] is scaler
        assert kwargs["calibration"] is calibration
        events.append("selected_candidate_fresh_auth")
        return object()

    def parent_evaluator(*_args, **_kwargs):
        # This is the first boundary capable of calling
        # load_early_cache_projection(..., include_labels=False).
        events.append("parent_outer_feature_access_boundary")
        return [], []

    kwargs = {
        "outer_family": "half_cylinder",
        "git_commit": "1" * 40,
        "device": "cpu",
        "expected_scaler_manifest_sha256": "2" * 64,
        "expected_calibration_manifest_sha256": "3" * 64,
        "expected_selected_candidate_sha256": "4" * 64,
        "expected_prediction_manifest_sha256": "5" * 64,
        "inner_group_metrics_path": Path("inner_group_metrics.csv"),
        "inner_group_metrics_sha256": "6" * 64,
        "inner_candidate_summary_path": Path("inner_candidate_summary.csv"),
        "inner_candidate_summary_sha256": "7" * 64,
        "inner_fit_audits_path": Path("inner_fit_audits.json"),
        "inner_fit_audits_sha256": "8" * 64,
    }
    with (
        patch.object(
            parent,
            "authenticate_and_rebuild_final_scaler",
            side_effect=authenticate_scaler,
        ),
        patch.object(
            parent,
            "authenticate_and_rebuild_final_calibration",
            side_effect=authenticate_calibration,
        ),
        patch.object(
            parent,
            "authenticate_selected_candidate",
            side_effect=authenticate_selected,
        ),
        patch.object(
            runner,
            "_INHERITED_EVALUATE_OUTER_PREDICTION",
            side_effect=parent_evaluator,
        ),
        patch.object(
            runner,
            "_support_audit_from_prediction_manifest",
            return_value={},
        ),
    ):
        rows, access = runner._evaluate_outer_prediction_bound(
            plan, selected, Path("synthetic_output"), **kwargs
        )
    assert isinstance(rows, runner.OuterMetricRows)
    assert access == []
    assert events == [
        "final_scaler_fresh_auth",
        "final_calibration_fresh_auth",
        "selected_candidate_fresh_auth",
        "parent_outer_feature_access_boundary",
    ]


def _parent_artifact_authentication_rebuilds_class_conditional_model(
    tmp_path: Path,
):
    plan, _caches, model = _synthetic_model()
    commit = "3" * 40
    identity = CleanSourceIdentity(
        git_commit=commit,
        worktree_clean=True,
        source_file_sha256_items=tuple(
            (path, hashlib.sha256(path.encode("utf-8")).hexdigest())
            for path in REQUIRED_SOURCE_PATHS
        ),
    )
    bound = replace(
        plan,
        source_identity=identity,
        kinematic_input_manifest_path=tmp_path / "KINEMATIC_INPUT.json",
        kinematic_input_manifest_file_sha256="4" * 64,
        kinematic_input_manifest_content_sha256="5" * 64,
        synthetic_pass_path=tmp_path / "SYNTHETIC_PASS.json",
        synthetic_pass_file_sha256="6" * 64,
        sidecar_root=tmp_path / "kinematic_cache" / "train",
        sidecar_population_manifest_path=tmp_path / "SIDECAR_POPULATION.json",
        sidecar_population_manifest_file_sha256="7" * 64,
        sidecar_population_manifest_content_sha256="8" * 64,
        sidecar_population=MappingProxyType(
            {"git_commit": parent.PREPARATION_ARTIFACT_GIT_COMMIT, "rows": ()}
        ),
        composite_descriptor_ids=MappingProxyType(
            {name: f"{name}_synthetic" for name in runner.REPRESENTATIONS}
        ),
    )
    selected = runner.TailCandidateSpec(
        "fmt161_plus_seed4",
        1,
        0.0,
        "calibrated_tail_anomaly_threshold",
        0.50,
    )
    fit_families = list(model.family_order)
    with runner.class_conditional_parent_runtime(bound, commit):
        scaler_path, scaler_manifest_path, _scaler_sha, scaler_manifest_sha = (
            parent.write_final_scaler_artifact(
                tmp_path,
                model,
                plan=bound,
                selected=selected,
                outer_family="channel",
                fit_families=fit_families,
                git_commit=commit,
            )
        )
        scaler = parent.authenticate_and_rebuild_final_scaler(
            scaler_path,
            scaler_manifest_path,
            plan=bound,
            selected=selected,
            outer_family="channel",
            fit_families=fit_families,
            git_commit=commit,
            expected_manifest_file_sha256=scaler_manifest_sha,
        )
        calibration_path, calibration_manifest_path, _calibration_sha, calibration_manifest_sha = (
            parent.write_final_calibration_artifact(
                tmp_path,
                model,
                plan=bound,
                selected=selected,
                scaler=scaler,
                outer_family="channel",
                fit_families=fit_families,
                git_commit=commit,
            )
        )
        calibration = parent.authenticate_and_rebuild_final_calibration(
            calibration_path,
            calibration_manifest_path,
            plan=bound,
            selected=selected,
            scaler=scaler,
            outer_family="channel",
            fit_families=fit_families,
            git_commit=commit,
            expected_manifest_file_sha256=calibration_manifest_sha,
        )
        assert isinstance(calibration.model, ClassConditionalTemplateScoreModel)
        assert calibration.model.family_order == model.family_order
        assert calibration.model.ks == model.ks
        assert (
            parent._json_safe(calibration.manifest[runner.METHOD_BINDING_KEY])
            == parent._json_safe(runner._method_binding(bound, commit))
        )


def test_parent_artifact_authentication_rebuilds_class_conditional_model():
    with tempfile.TemporaryDirectory() as directory:
        _parent_artifact_authentication_rebuilds_class_conditional_model(
            Path(directory)
        )


def test_outer_summary_requires_and_preserves_authenticated_support_payload():
    support = {
        "schema": "synthetic_support.v1",
        "sample_count": 4,
        "joint_supported_family_count_histogram": {"0": 0, "3": 4},
    }
    rows = runner.OuterMetricRows([{"x": 1}], support)
    with patch.object(
        runner,
        "_INHERITED_OUTER_SUMMARY",
        return_value={"schema": "parent", "experiment": "parent"},
    ):
        summary = runner._outer_summary(rows, "half_cylinder")
        assert summary["schema"] == runner.OUTER_SUMMARY_SCHEMA
        assert summary["experiment"] == runner.EXPERIMENT
        assert summary["class_conditional_support"] == support
        _expect_error(
            ValueError,
            runner._outer_summary,
            [{"x": 1}],
            "half_cylinder",
            contains="authenticated class-conditional support audits",
        )


def test_method_binding_records_strict_semantics_support_audit_and_core_hash():
    plan = runner.load_plan(CONFIG)
    binding = runner._method_binding(plan, "2" * 40)
    assert binding["threshold"]["comparison"] == "strict_greater_than"
    assert binding["threshold"]["equality_prediction"] == "negative"
    assert binding["score"]["probability_claim"] is False
    assert binding["support_audit"]["fresh_replay"].startswith("recomputed")
    assert binding["prediction_array_contract"] == "unchanged_parent_19_arrays"
    assert binding["fold_transaction"] == "unchanged_parent_15_files"
    assert binding["core"]["sha256"] == hashlib.sha256(
        runner.CORE_PATH.read_bytes()
    ).hexdigest()
    fields = binding["compatibility_prediction_fields"]
    assert "not_a_probability_claim" in fields["tail_probability"]
    assert "compatibility_sentinel" in fields["calibration_mode_1"]


def _run_without_pytest() -> None:
    """Allow the synthetic suite to run in minimal Ibex/local environments."""

    for name, function in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        function()
        print(f"PASS {name}")


if __name__ == "__main__":
    _run_without_pytest()

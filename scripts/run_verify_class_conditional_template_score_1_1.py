#!/usr/bin/env python3
"""Production fold runner for Verify_ClassConditionalTemplateScore_1.1.

This adapter deliberately reuses the authenticated Early runner's complete
nested split, sidecar join, spatial transform, label gate, metrics, and
15-file transaction.  The only numerical substitutions are the frozen
family/class template fit, its query/calibration replay, and strict ``>``
threshold decisions.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
import threading
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.class_conditional_template_score import (  # noqa: E402
    ClassConditionalTemplateScoreModel,
    FamilyFitBatch,
    strict_threshold_predictions,
)
from pathline_template_matching.negative_tail_calibration import (  # noqa: E402
    CALIBRATION_NONE,
)
from pathline_template_matching.portable_flow import sha256_file  # noqa: E402
from scripts import run_verify_early_opposite_pair_kinematics_1_1 as inherited  # noqa: E402
from scripts.run_verify_scale_conditioned_retrieval_1_1 import (  # noqa: E402
    _classification_from_confusion,
    _git_identity,
    _require,
)


EXPERIMENT = "Verify_ClassConditionalTemplateScore_1.1"
PARENT_EXPERIMENT = inherited.EXPERIMENT
EXPECTED_CONFIG_SHA256 = (
    "814f95d2ec58f751a91082d588f790b3592a891963810013ad92ab704febbdea"
)
EXPECTED_PARENT_CONFIG_SHA256 = (
    "e6bac4568025f42cf0a9effd78620e5ab4ba5653429a7023bd91816f29512767"
)
EXPECTED_PARENT_RUNNER_SHA256 = (
    "e999960ac06d3fedd355e1d6135d9e69316bfe1e798318a22dadf5a8e2063796"
)
EXPECTED_PARENT_AGGREGATOR_SHA256 = (
    "631909159387cba854f471b3179ff0f0cd97404905e29b74589b2b8cf71f089e"
)
EXPECTED_PARENT_NUMERICAL_GIT_COMMIT = (
    "2c3774dca0d81db8edd5645e63576526b9e276f7"
)
EXPECTED_CORE_SHA256 = (
    "9c009376f7cea1481f6f47a49362d54d0e78530717f480fda3e8a109f841ef99"
)

CONFIG_PATH = ROOT / "config" / "Verify_ClassConditionalTemplateScore_1.1.yaml"
PARENT_CONFIG_PATH = ROOT / "config" / "Verify_EarlyOppositePairKinematics_1.1.yaml"
PARENT_RUNNER_PATH = ROOT / "scripts" / "run_verify_early_opposite_pair_kinematics_1_1.py"
PARENT_AGGREGATOR_PATH = ROOT / "scripts" / "aggregate_verify_early_opposite_pair_kinematics_1_1.py"
CORE_PATH = ROOT / "src" / "pathline_template_matching" / "class_conditional_template_score.py"

FAMILY_ORDER = inherited.FAMILY_ORDER
REPRESENTATIONS = inherited.REPRESENTATIONS
K_VALUES = inherited.K_VALUES
SIGMAS = inherited.SIGMAS
SCORE_THRESHOLDS = inherited.TAIL_THRESHOLDS
GRID_SHAPE = inherited.GRID_SHAPE
BLOCK_NAMES = inherited.BLOCK_NAMES
FROZEN_CANDIDATE_COUNT = 3060
DEFAULT_OUTPUT_ROOT = Path(
    "/ibex/user/zhanx0o/pathline-template-matching/"
    "Verify_ClassConditionalTemplateScore_1.1"
)

SCALER_ARTIFACT_SCHEMA = (
    "pathline_template_matching.class_conditional_template_score_scaler.v1"
)
SCALER_MANIFEST_SCHEMA = (
    "pathline_template_matching.class_conditional_template_score_scaler_manifest.v1"
)
CALIBRATION_ARTIFACT_SCHEMA = (
    "pathline_template_matching.class_conditional_template_bundle.v1"
)
CALIBRATION_MANIFEST_SCHEMA = (
    "pathline_template_matching.class_conditional_template_bundle_manifest.v1"
)
SELECTED_SCHEMA = (
    "pathline_template_matching.class_conditional_template_score_selected_candidate.v1"
)
PREDICTION_SCHEMA = (
    "pathline_template_matching.class_conditional_template_score_outer_prediction.v1"
)
PREDICTION_MANIFEST_SCHEMA = (
    "pathline_template_matching.class_conditional_template_score_outer_prediction_manifest.v1"
)
INNER_AUDIT_SCHEMA = (
    "pathline_template_matching.class_conditional_template_score_inner_fit_audits.v1"
)
OUTER_SUMMARY_SCHEMA = (
    "pathline_template_matching.class_conditional_template_score_outer_summary.v1"
)
REFERENCE_AUDIT_SCHEMA = (
    "pathline_template_matching.class_conditional_template_score_outer_reference_access.v1"
)
RESULT_SCHEMA = "pathline_template_matching.class_conditional_template_score_result.v1"
COMPLETE_SCHEMA = (
    "pathline_template_matching.class_conditional_template_score_run_complete.v1"
)
METHOD_BINDING_KEY = "class_conditional_template_score_method"

# The inherited files are intentionally retained.  Their contents acquire the
# child schemas and method binding while the transaction is active.
REQUIRED_FOLD_FILES = inherited.REQUIRED_FOLD_FILES
PREDICTION_ARRAY_DTYPES = inherited.PREDICTION_ARRAY_DTYPES
Plan = inherited.Plan
EarlyCacheProjection = inherited.EarlyCacheProjection
_atomic_csv = inherited._atomic_csv
_atomic_json = inherited._atomic_json
_atomic_npz = inherited._atomic_npz

_INHERITED_LOAD_PLAN = inherited.load_plan
_INHERITED_BIND_EARLY_EVIDENCE = inherited.bind_early_evidence
_INHERITED_OUTER_SUMMARY = inherited._outer_summary
_INHERITED_EVALUATE_OUTER_PREDICTION = inherited.evaluate_outer_prediction
_INHERITED_GIT_IDENTITY = _git_identity
_PARENT_RUNTIME_LOCK = threading.Lock()


@dataclass(frozen=True)
class TailCandidateSpec:
    """Parent-compatible candidate with an explicit strict comparator ID."""

    representation: str
    k: int
    sigma: float
    decision_rule: str
    decision_value: float

    @property
    def candidate_id(self) -> str:
        sigma = format(self.sigma, ".1f")
        if self.decision_rule == "fixed_top_fraction":
            decision = f"fixed_top_fraction={self.decision_value:.2f}"
        elif self.decision_rule == "calibrated_tail_anomaly_threshold":
            decision = (
                f"calibrated_tail_anomaly_threshold={self.decision_value:.2f}"
                "|comparator=strict_greater_than"
            )
        else:
            raise ValueError(f"unsupported decision rule: {self.decision_rule}")
        return (
            f"representation={self.representation}|k={self.k}|sigma={sigma}|"
            f"{decision}"
        )


class OuterMetricRows(list[dict[str, Any]]):
    """Metric rows carrying their authenticated manifest-only support audit."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        support_audit: Mapping[str, Any],
    ) -> None:
        super().__init__(dict(row) for row in rows)
        self.support_audit = dict(support_audit)


def _lower_hex(value: object, *, length: int = 64) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is missing")
    assert isinstance(value, Mapping)
    return value


def _candidate_count(plan: Plan) -> int:
    return len(plan.representations) * len(plan.ks) * len(plan.sigmas) * (
        1 + len(plan.thresholds)
    )


def candidate_specs(plan: Plan) -> tuple[TailCandidateSpec, ...]:
    """Enumerate the exact 3060 candidates without consulting labels."""

    candidates = tuple(
        TailCandidateSpec(representation, k, sigma, decision_rule, decision_value)
        for representation in plan.representations
        for k in plan.ks
        for sigma in plan.sigmas
        for decision_rule, decision_value in (
            ("fixed_top_fraction", plan.fixed_top_fraction),
            *(
                ("calibrated_tail_anomaly_threshold", threshold)
                for threshold in plan.thresholds
            ),
        )
    )
    _require(len(candidates) == FROZEN_CANDIDATE_COUNT, "candidate count drifted")
    _require(
        len({candidate.candidate_id for candidate in candidates}) == len(candidates),
        "candidate ID collision",
    )
    return candidates


def load_plan(config_path: str | Path = CONFIG_PATH) -> Plan:
    """Authenticate the child contract and derive only unchanged parent state."""

    path = Path(config_path).resolve()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    _require(digest == EXPECTED_CONFIG_SHA256, "frozen child config SHA-256 drifted")
    raw = yaml.safe_load(payload.decode("utf-8"))
    _require(isinstance(raw, Mapping), "child config root is invalid")
    assert isinstance(raw, Mapping)
    _require(
        raw.get("schema")
        == "pathline_template_matching.verify_class_conditional_template_score_config.v1"
        and raw.get("experiment") == EXPERIMENT
        and raw.get("phase") == "exposed_train_only_nested_family_validation"
        and raw.get("status") == "frozen_pre_run_not_implemented",
        "child experiment identity or immutable freeze status drifted",
    )
    freeze = _mapping(raw.get("freeze"), "freeze history")
    _require(
        freeze.get("frozen_before_first_read_of_any_new_version_real_array_or_result")
        is True
        and freeze.get("new_version_runner_exists") is False
        and freeze.get("new_version_real_feature_label_valid_rate_prediction_or_metric_read")
        is False,
        "pre-read freeze history drifted",
    )

    direct = _mapping(raw.get("direct_parent"), "direct parent")
    _require(
        direct.get("experiment") == PARENT_EXPERIMENT
        and direct.get("inheritance") == "direct_only"
        and direct.get("numerical_git_commit")
        == EXPECTED_PARENT_NUMERICAL_GIT_COMMIT
        and direct.get("config_sha256") == EXPECTED_PARENT_CONFIG_SHA256
        and direct.get("runner_sha256") == EXPECTED_PARENT_RUNNER_SHA256
        and direct.get("aggregator_sha256") == EXPECTED_PARENT_AGGREGATOR_SHA256,
        "direct-parent identity drifted",
    )
    _require(
        Path(str(direct.get("config_path")))
        == Path("config/Verify_EarlyOppositePairKinematics_1.1.yaml"),
        "direct-parent config path drifted",
    )
    _require(
        sha256_file(PARENT_CONFIG_PATH) == EXPECTED_PARENT_CONFIG_SHA256
        and sha256_file(PARENT_RUNNER_PATH) == EXPECTED_PARENT_RUNNER_SHA256
        and sha256_file(PARENT_AGGREGATOR_PATH)
        == EXPECTED_PARENT_AGGREGATOR_SHA256
        and sha256_file(CORE_PATH) == EXPECTED_CORE_SHA256,
        "authenticated direct-parent source changed",
    )
    parent = _INHERITED_LOAD_PLAN(PARENT_CONFIG_PATH)
    _require(
        parent.sha256 == EXPECTED_PARENT_CONFIG_SHA256
        and parent.required_fold_files == REQUIRED_FOLD_FILES,
        "authenticated parent plan drifted",
    )

    evidence = _mapping(raw.get("evidence_scope"), "evidence scope")
    _require(
        evidence.get("level") == "exposed_development_only"
        and evidence.get("formal_confirmation") is False
        and evidence.get("forbidden_datasets") == ["tangaroa", "smokeBuoyancy"],
        "evidence scope drifted",
    )
    identity = _mapping(raw.get("input_identity"), "input identity")
    manifest = _mapping(
        identity.get("train_cache_input_manifest"), "train cache manifest"
    )
    _require(
        Path(str(manifest.get("path"))) == parent.manifest_path
        and int(manifest.get("size_bytes", -1)) == parent.manifest_size
        and manifest.get("sha256") == parent.manifest_sha256
        and manifest.get("rows_content_sha256") == parent.manifest_rows_sha256
        and int(manifest.get("row_count", -1)) == 32,
        "child and parent cache populations differ",
    )
    _require(
        identity.get("cache_builder_git_commit") == parent.cache_commit
        and identity.get("main_config_sha256") == parent.parent_config_sha256
        and identity.get("descriptor_id") == parent.descriptor_id
        and identity.get("cache_schema") == parent.cache_schema,
        "child cache identity differs from the direct parent",
    )

    families_raw = _mapping(raw.get("families"), "family mapping")
    split = _mapping(raw.get("nested_split"), "nested split")
    families = {
        str(family): tuple(str(dataset) for dataset in datasets)
        for family, datasets in families_raw.items()
    }
    _require(
        tuple(families) == FAMILY_ORDER
        and families == dict(parent.families)
        and tuple(split.get("outer_order", ())) == FAMILY_ORDER
        and tuple(split.get("inner_order", ())) == FAMILY_ORDER
        and int(split.get("outer_fit_family_count", -1)) == 4
        and int(split.get("inner_fit_family_count", -1)) == 3
        and split.get("outer_labels_available_to_fit_or_selection") is False,
        "complete-family nested split drifted",
    )

    representations = _mapping(raw.get("representations"), "representations")
    _require(
        tuple(representations.get("order", ())) == REPRESENTATIONS,
        "representation order drifted",
    )
    for name, width in inherited.COMPOSITE_WIDTH.items():
        spec = _mapping(representations.get(name), name)
        _require(int(spec.get("width", -1)) == width, f"{name} width drifted")

    scaler = _mapping(raw.get("shared_negative_scaler"), "shared scaler")
    conformity = _mapping(
        raw.get("family_class_exact_scale_conformity"), "conformity method"
    )
    retrieval = _mapping(conformity.get("retrieval"), "class retrieval")
    calibration = _mapping(conformity.get("calibration"), "class calibration")
    support = _mapping(
        raw.get("joint_family_support_and_score"), "joint-family support"
    )
    inner_gate = _mapping(support.get("inner_gate"), "inner support gate")
    outer_gate = _mapping(support.get("final_outer_gate"), "outer support gate")
    _require(
        scaler.get("conditioning") == "exact_numeric_scale_id"
        and float(scaler.get("shrinkage_lambda", -1)) == 64.0
        and scaler.get("population_is_shared_across_fit_families") is True
        and scaler.get("population_is_shared_across_both_class_libraries_and_queries")
        is True,
        "shared negative scaler drifted",
    )
    _require(
        tuple(int(value) for value in conformity.get("ks", ())) == K_VALUES
        and conformity.get("conditioning_fields")
        == ["fit_physical_family", "class", "exact_numeric_scale_id"]
        and conformity.get("cross_family_pooling") == "forbidden"
        and conformity.get("cross_class_pooling") == "forbidden"
        and conformity.get("retrieval_fallback") == "forbidden"
        and retrieval.get("query_distance")
        == "kth_nearest_distance_to_matching_family_class_exact_scale_library"
        and int(retrieval.get("query_chunk_size", -1)) == parent.query_chunk_size
        and int(retrieval.get("library_chunk_size", -1))
        == parent.library_chunk_size
        and calibration.get("self_exclusion")
        == "set_only_current_row_distance_to_positive_infinity"
        and calibration.get("scaler_refit_per_leave_one_out_row") is False
        and float(calibration.get("shrinkage_lambda", -1)) == 64.0,
        "family/class retrieval or calibration drifted",
    )
    _require(
        int(inner_gate.get("available_fit_families", -1)) == 3
        and int(inner_gate.get("minimum_jointly_supported_families", -1)) == 2
        and int(outer_gate.get("available_fit_families", -1)) == 4
        and int(outer_gate.get("minimum_jointly_supported_families", -1)) == 3
        and support.get("per_family_score") == "S_f=0.5*(1+q_positive-q_negative)"
        and support.get("family_weights") == "exactly_equal",
        "strict-majority support or score formula drifted",
    )

    transform = _mapping(raw.get("group_transform"), "group transform")
    decisions = _mapping(raw.get("decision_candidates"), "decision grid")
    top = _mapping(decisions.get("fixed_top_fraction"), "fixed-top decision")
    threshold = _mapping(
        decisions.get("class_conditional_score_threshold"), "score threshold"
    )
    thresholds = tuple(float(value) for value in threshold.get("values", ()))
    _require(
        tuple(float(value) for value in transform.get("gaussian_sigmas_grid_indices", ()))
        == SIGMAS
        and tuple(int(value) for value in transform.get("grid_shape_zyx", ()))
        == GRID_SHAPE
        and transform.get("input_score") == "raw_class_conditional_score_S"
        and transform.get("positive_sigma_policy")
        == "joint_support_mask_normalized_spatial_imputation"
        and thresholds == SCORE_THRESHOLDS
        and threshold.get("comparison") == "score_strictly_greater_than_threshold"
        and threshold.get("equality_prediction") == "negative"
        and float(top.get("fraction", -1)) == 0.05
        and int(decisions.get("frozen_candidate_count", -1))
        == FROZEN_CANDIDATE_COUNT,
        "spatial transform or decision grid drifted",
    )

    final_gate = _mapping(
        raw.get("final_refit_and_outer_label_gate"), "outer label gate"
    )
    _require(
        final_gate.get(
            "outer_prediction_must_be_freshly_recomputed_written_closed_and_authenticated_before_parent_valid_labels_open"
        )
        is True
        and final_gate.get(
            "fit_selection_support_orientation_prevalence_or_threshold_use_of_outer_labels"
        )
        == "forbidden",
        "outer reference gate drifted",
    )
    output = _mapping(raw.get("output"), "output contract")
    output_root = Path(str(output.get("root")))
    _require(
        output_root == DEFAULT_OUTPUT_ROOT
        and output.get("overwrite") == "forbidden"
        and output.get("atomic_publish") == "hard_link_without_replace",
        "output contract drifted",
    )

    merged_raw = dict(raw)
    # The inherited artifact writer reads only this historical parent field.
    # Preserve it byte-for-byte while retaining the complete child config.
    merged_raw["parent_identity"] = parent.raw["parent_identity"]
    plan = replace(
        parent,
        path=path,
        sha256=digest,
        raw=merged_raw,
        family_order=FAMILY_ORDER,
        families=families,
        dataset_to_family={
            dataset: family
            for family in FAMILY_ORDER
            for dataset in families[family]
        },
        representations=REPRESENTATIONS,
        ks=K_VALUES,
        sigmas=SIGMAS,
        thresholds=thresholds,
        fixed_top_fraction=0.05,
        grid_shape=GRID_SHAPE,
        gaussian_truncate=float(transform["gaussian_truncate"]),
        query_chunk_size=int(retrieval["query_chunk_size"]),
        library_chunk_size=int(retrieval["library_chunk_size"]),
        shrinkage_lambda=float(scaler["shrinkage_lambda"]),
        output_root=output_root,
        required_fold_files=REQUIRED_FOLD_FILES,
    )
    _require(_candidate_count(plan) == FROZEN_CANDIDATE_COUNT, "grid size drifted")
    return plan


def bind_early_evidence(
    plan: Plan,
    *,
    kinematic_input_manifest_path: str | Path,
    kinematic_input_manifest_file_sha256: str,
    synthetic_pass_path: str | Path,
    synthetic_pass_file_sha256: str,
    sidecar_root: str | Path,
    sidecar_population_manifest_path: str | Path,
    sidecar_population_manifest_file_sha256: str,
) -> Plan:
    """Authenticate the unchanged Early evidence before rebinding its runner."""

    identity = _mapping(plan.raw.get("input_identity"), "input identity")
    kinematic = _mapping(
        identity.get("kinematic_input_manifest"), "kinematic input manifest"
    )
    synthetic = _mapping(identity.get("parent_synthetic_pass"), "synthetic pass")
    population = _mapping(identity.get("sidecar_population"), "sidecar population")
    frozen_root = Path(str(identity.get("sidecar_root"))).resolve()
    _require(
        Path(kinematic_input_manifest_path).resolve()
        == Path(str(kinematic.get("path"))).resolve()
        and kinematic_input_manifest_file_sha256 == kinematic.get("sha256")
        and Path(synthetic_pass_path).resolve()
        == Path(str(synthetic.get("path"))).resolve()
        and synthetic_pass_file_sha256 == synthetic.get("sha256")
        and Path(sidecar_root).resolve() == frozen_root
        and Path(sidecar_population_manifest_path).resolve()
        == Path(str(population.get("path"))).resolve()
        and sidecar_population_manifest_file_sha256 == population.get("sha256"),
        "runtime Early evidence differs from the frozen child config",
    )
    # The parent's envelope authenticator derives the sidecar path from its
    # output root.  Give it that unchanged root only for evidence binding, then
    # restore the child's independent immutable result root.
    parent_root = frozen_root.parent.parent
    bound = _INHERITED_BIND_EARLY_EVIDENCE(
        replace(plan, output_root=parent_root),
        kinematic_input_manifest_path=kinematic_input_manifest_path,
        kinematic_input_manifest_file_sha256=kinematic_input_manifest_file_sha256,
        synthetic_pass_path=synthetic_pass_path,
        synthetic_pass_file_sha256=synthetic_pass_file_sha256,
        sidecar_root=sidecar_root,
        sidecar_population_manifest_path=sidecar_population_manifest_path,
        sidecar_population_manifest_file_sha256=(
            sidecar_population_manifest_file_sha256
        ),
    )
    return replace(bound, output_root=plan.output_root)


def _fit_tail_model(
    caches: Sequence[EarlyCacheProjection],
    representation: str,
    plan: Plan,
    *,
    device: str,
    ks: Sequence[int] | None = None,
) -> ClassConditionalTemplateScoreModel:
    """Fit one natural, unbalanced family/class library per fit family."""

    observed = {cache.row.family for cache in caches}
    _require(observed.issubset(set(plan.family_order)), "unknown fit family")
    family_order = tuple(
        family for family in plan.family_order if family in observed
    )
    _require(
        len(family_order) in (3, 4) and observed == set(family_order),
        "class-conditional fit requires exactly three inner or four final families",
    )
    batches: dict[str, FamilyFitBatch] = {}
    for family in family_order:
        family_caches = [cache for cache in caches if cache.row.family == family]
        _require(family_caches, f"{family}: fit family is empty")
        _require(
            all(cache.labels is not None for cache in family_caches),
            f"{family}: fit labels are unavailable",
        )
        features = np.ascontiguousarray(
            np.concatenate(
                [
                    inherited.composite_representation_features(cache, representation)
                    for cache in family_caches
                ],
                axis=0,
            ),
            dtype=np.float32,
        )
        scales = np.ascontiguousarray(
            np.concatenate([cache.scale_ids for cache in family_caches]),
            dtype=np.int64,
        )
        labels = np.ascontiguousarray(
            np.concatenate(
                [np.asarray(cache.labels, dtype=np.bool_) for cache in family_caches]
            ),
            dtype=np.bool_,
        )
        batches[family] = FamilyFitBatch(features, scales, labels)
    return ClassConditionalTemplateScoreModel(
        batches,
        family_order=family_order,
        ks=plan.ks if ks is None else tuple(int(value) for value in ks),
        shrinkage_lambda=plan.shrinkage_lambda,
        device=device,
        query_chunk_size=plan.query_chunk_size,
        library_chunk_size=plan.library_chunk_size,
    )


_PER_FAMILY_SUPPORT_FIELDS = (
    "per_family_positive_retrieval_supported",
    "per_family_positive_calibration_supported",
    "per_family_negative_retrieval_supported",
    "per_family_negative_calibration_supported",
)


def _query_cache_batch(
    model: ClassConditionalTemplateScoreModel,
    caches: Sequence[EarlyCacheProjection],
    representation: str,
    plan: Plan,
    *,
    device: str,
    ks: Sequence[int] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Query once and expose parent fields plus manifest-only support audits."""

    requested = model.ks if ks is None else tuple(int(value) for value in ks)
    offsets = np.cumsum([0, *(cache.count for cache in caches)], dtype=np.int64)
    if caches:
        features = np.ascontiguousarray(
            np.concatenate(
                [
                    inherited.composite_representation_features(cache, representation)
                    for cache in caches
                ],
                axis=0,
            ),
            dtype=np.float32,
        )
        scales = np.ascontiguousarray(
            np.concatenate([cache.scale_ids for cache in caches]), dtype=np.int64
        )
    else:
        features = np.empty(
            (0, inherited.COMPOSITE_WIDTH[representation]), dtype=np.float32
        )
        scales = np.empty(0, dtype=np.int64)
    result = model.query(
        features,
        scales,
        ks=requested,
        device=device,
        query_chunk_size=plan.query_chunk_size,
        library_chunk_size=plan.library_chunk_size,
    )
    scaler_modes = model.scaler.mode_for_scales(scales)
    output: dict[int, list[dict[str, Any]]] = {}
    for k in requested:
        score = np.asarray(result.scores[k], dtype=np.float64)
        retrieval = np.asarray(result.retrieval_supported[k], dtype=np.bool_)
        joint = np.asarray(result.joint_supported[k], dtype=np.bool_)
        joint_count = np.asarray(result.joint_family_count[k], dtype=np.int16)
        support_matrices = {
            name: np.asarray(getattr(result, name)[k], dtype=np.bool_)
            for name in _PER_FAMILY_SUPPORT_FIELDS
        }
        expected_shape = (len(scales), len(model.family_order))
        _require(
            all(values.shape == expected_shape for values in support_matrices.values()),
            "per-family support audit shape drifted",
        )
        positive_retrieval = support_matrices[
            "per_family_positive_retrieval_supported"
        ]
        positive_calibration = support_matrices[
            "per_family_positive_calibration_supported"
        ]
        negative_retrieval = support_matrices[
            "per_family_negative_retrieval_supported"
        ]
        negative_calibration = support_matrices[
            "per_family_negative_calibration_supported"
        ]
        _require(
            np.array_equal(
                retrieval,
                np.count_nonzero(
                    positive_retrieval & negative_retrieval, axis=1
                )
                >= model.required_family_count,
            )
            and np.array_equal(
                joint_count,
                np.count_nonzero(
                    positive_retrieval
                    & positive_calibration
                    & negative_retrieval
                    & negative_calibration,
                    axis=1,
                ).astype(np.int16),
            )
            and np.array_equal(
                joint, joint_count >= model.required_family_count
            ),
            "aggregate and per-family support audits disagree",
        )
        tail = np.ascontiguousarray(1.0 - score, dtype=np.float64)
        modes = np.where(joint, 1, CALIBRATION_NONE).astype(np.int8)
        parts: list[dict[str, Any]] = []
        for index in range(len(caches)):
            selected = slice(int(offsets[index]), int(offsets[index + 1]))
            part: dict[str, Any] = {
                "raw_distance": result.mean_negative_distances[k][selected],
                "tail_probability": tail[selected],
                "tail_anomaly": score[selected],
                "retrieval_supported": retrieval[selected],
                "calibration_supported": joint[selected],
                "calibration_mode": modes[selected],
                "scaler_mode": scaler_modes[selected],
                "joint_family_count": joint_count[selected],
                "family_order": model.family_order,
                "required_family_count": model.required_family_count,
            }
            part.update(
                {
                    name: values[selected]
                    for name, values in support_matrices.items()
                }
            )
            parts.append(part)
        output[int(k)] = parts
    return output


def candidate_predictions(
    candidate: TailCandidateSpec,
    scores: object,
    center_indices: object,
    eligible: object,
) -> np.ndarray:
    """Apply the unchanged fixed-top rule or the frozen strict threshold."""

    values = np.asarray(scores, dtype=np.float64)
    centers = np.asarray(center_indices)
    allowed = np.asarray(eligible)
    _require(
        values.ndim == 1
        and centers.shape == values.shape
        and allowed.shape == values.shape,
        "invalid candidate population",
    )
    _require(
        centers.dtype.kind in "iu" and allowed.dtype == np.dtype(np.bool_),
        "invalid candidate identity/support dtype",
    )
    _require(np.isfinite(values).all(), "candidate scores must be finite")
    if candidate.decision_rule == "fixed_top_fraction":
        return inherited.fixed_top_fraction_predictions(
            values,
            centers,
            allowed & (values > 0.0),
            fraction=candidate.decision_value,
        )
    if candidate.decision_rule == "calibrated_tail_anomaly_threshold":
        return strict_threshold_predictions(
            values, allowed, threshold=candidate.decision_value
        )
    raise ValueError(f"unsupported decision rule: {candidate.decision_rule}")


def _threshold_confusion_series(
    labels: np.ndarray,
    scores: np.ndarray,
    eligible: np.ndarray,
    thresholds: Sequence[float],
) -> list[dict[str, int | float]]:
    """Vectorized strict-``>`` confusion series; exact ties remain negative."""

    targets = np.asarray(labels, dtype=np.bool_)
    values = np.asarray(scores, dtype=np.float64)
    allowed = np.asarray(eligible, dtype=np.bool_)
    _require(
        targets.shape == values.shape == allowed.shape and len(targets) > 0,
        "invalid threshold population",
    )
    _require(np.isfinite(values).all(), "threshold scores must be finite")
    eligible_scores = values[allowed]
    eligible_labels = targets[allowed]
    order = np.argsort(eligible_scores, kind="mergesort")
    sorted_scores = eligible_scores[order]
    sorted_labels = eligible_labels[order].astype(np.int64)
    suffix_positive = np.concatenate(
        (
            np.cumsum(sorted_labels[::-1], dtype=np.int64)[::-1],
            np.zeros(1, dtype=np.int64),
        )
    )
    total_positive = int(targets.sum())
    total_negative = len(targets) - total_positive
    output: list[dict[str, int | float]] = []
    for threshold in thresholds:
        first = int(
            np.searchsorted(sorted_scores, float(threshold), side="right")
        )
        predicted_count = len(sorted_scores) - first
        true_positive = int(suffix_positive[first])
        false_positive = predicted_count - true_positive
        output.append(
            _classification_from_confusion(
                tp=true_positive,
                fp=false_positive,
                tn=total_negative - false_positive,
                fn=total_positive - true_positive,
            )
        )
    return output


def _group_support_audit(values: Mapping[str, Any], selected: np.ndarray) -> dict[str, Any]:
    family_order = tuple(str(value) for value in values["family_order"])
    count = int(np.count_nonzero(selected))
    joint_counts = np.asarray(values["joint_family_count"])[selected]
    histogram = np.bincount(
        joint_counts.astype(np.int64), minlength=len(family_order) + 1
    )
    families: dict[str, Any] = {}
    for family_index, family in enumerate(family_order):
        audit: dict[str, int | float] = {}
        for class_name in ("positive", "negative"):
            for support_name in ("retrieval", "calibration"):
                field = f"per_family_{class_name}_{support_name}_supported"
                supported_count = int(
                    np.count_nonzero(np.asarray(values[field])[selected, family_index])
                )
                audit[f"{class_name}_{support_name}_count"] = supported_count
                audit[f"{class_name}_{support_name}_fraction"] = (
                    supported_count / count if count else 0.0
                )
        families[family] = audit
    return {
        "schema": "pathline_template_matching.class_conditional_group_support_audit.v1",
        "sample_count": count,
        "family_order": list(family_order),
        "required_joint_family_count": int(values["required_family_count"]),
        "joint_supported_family_count_histogram": {
            str(index): int(value) for index, value in enumerate(histogram)
        },
        "families": families,
    }


def build_outer_prediction_arrays(
    caches: Sequence[EarlyCacheProjection],
    model: ClassConditionalTemplateScoreModel,
    selected: TailCandidateSpec,
    plan: Plan,
    *,
    device: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Build the unchanged 19 arrays plus replayable group support audits."""

    _require(
        caches
        and all(cache.labels is None and not cache.metadata for cache in caches),
        "outer projection must be label-free",
    )
    query = _query_cache_batch(
        model,
        caches,
        selected.representation,
        plan,
        device=device,
        ks=(selected.k,),
    )[selected.k]
    parts: dict[str, list[np.ndarray]] = {
        name: [] for name in PREDICTION_ARRAY_DTYPES
    }
    group_audits: list[dict[str, Any]] = []
    for cache_index, cache in enumerate(caches):
        values = query[cache_index]
        for block_index, block_name in enumerate(BLOCK_NAMES):
            block = np.asarray(cache.block_indices == block_index)
            _require(
                block.any(),
                f"{cache.row.dataset}/{cache.row.source_ordinal}/{block_name}: empty group",
            )
            centers = np.asarray(cache.center_indices[block], dtype=np.int64)
            spatial = inherited.spatial_calibrated_tail_scores(
                values["tail_anomaly"][block],
                values["calibration_supported"][block],
                centers,
                sigma=selected.sigma,
                grid_shape=plan.grid_shape,
                truncate=plan.gaussian_truncate,
            )
            eligible = spatial.calibration_supported | spatial.imputed
            prediction = candidate_predictions(
                selected, spatial.scores, centers, eligible
            )
            count = int(block.sum())
            parts["dataset"].append(
                np.full(
                    count,
                    cache.row.dataset,
                    dtype=PREDICTION_ARRAY_DTYPES["dataset"],
                )
            )
            parts["source_ordinal"].append(
                np.full(count, cache.row.source_ordinal, dtype=np.int16)
            )
            parts["source_index"].append(
                np.full(count, cache.row.source_index, dtype=np.int64)
            )
            parts["scale_id"].append(cache.scale_ids[block])
            parts["center_seed_index"].append(cache.center_indices[block])
            parts["scale_block_index"].append(cache.block_indices[block])
            parts["assigned_row_index"].append(cache.assigned_row_indices[block])
            parts["raw_negative_distance"].append(values["raw_distance"][block])
            parts["tail_probability"].append(values["tail_probability"][block])
            parts["tail_anomaly"].append(values["tail_anomaly"][block])
            parts["spatial_score"].append(spatial.scores)
            parts["spatial_denominator"].append(spatial.denominator)
            parts["retrieval_supported"].append(
                values["retrieval_supported"][block]
            )
            parts["calibration_supported"].append(spatial.calibration_supported)
            parts["spatial_imputed"].append(spatial.imputed)
            parts["spatial_unimputable"].append(spatial.unimputable)
            parts["calibration_mode"].append(values["calibration_mode"][block])
            parts["scaler_mode"].append(values["scaler_mode"][block])
            parts["prediction"].append(prediction)
            group_audits.append(
                {
                    "dataset": cache.row.dataset,
                    "source_ordinal": cache.row.source_ordinal,
                    "source_index": cache.row.source_index,
                    "block": block_name,
                    "sample_count": count,
                    "retrieval_supported_count": int(
                        values["retrieval_supported"][block].sum()
                    ),
                    "calibration_supported_count": int(
                        spatial.calibration_supported.sum()
                    ),
                    "imputed_count": int(spatial.imputed.sum()),
                    "unimputable_count": int(spatial.unimputable.sum()),
                    "calibration_mode_counts": {
                        str(mode): int(
                            np.count_nonzero(values["calibration_mode"][block] == mode)
                        )
                        for mode in range(6)
                    },
                    "scaler_mode_counts": {
                        str(mode): int(
                            np.count_nonzero(values["scaler_mode"][block] == mode)
                        )
                        for mode in range(4)
                    },
                    "prediction_count": int(prediction.sum()),
                    "class_conditional_support": _group_support_audit(values, block),
                }
            )
    arrays = {
        name: np.ascontiguousarray(np.concatenate(parts[name]), dtype=dtype)
        for name, dtype in PREDICTION_ARRAY_DTYPES.items()
    }
    return arrays, group_audits


def _aggregate_group_support_audits(
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _require(groups, "prediction support audit is empty")
    first = _mapping(
        groups[0].get("class_conditional_support"), "group support audit"
    )
    family_order = tuple(str(value) for value in first.get("family_order", ()))
    required = int(first.get("required_joint_family_count", -1))
    histogram = np.zeros(len(family_order) + 1, dtype=np.int64)
    family_counts = {
        family: {
            f"{class_name}_{support_name}_count": 0
            for class_name in ("positive", "negative")
            for support_name in ("retrieval", "calibration")
        }
        for family in family_order
    }
    total = 0
    for group in groups:
        audit = _mapping(
            group.get("class_conditional_support"), "group support audit"
        )
        _require(
            tuple(audit.get("family_order", ())) == family_order
            and int(audit.get("required_joint_family_count", -1)) == required,
            "group support family binding drifted",
        )
        count = int(audit.get("sample_count", -1))
        _require(count == int(group.get("sample_count", -2)), "support count drifted")
        total += count
        observed_histogram = _mapping(
            audit.get("joint_supported_family_count_histogram"),
            "joint support histogram",
        )
        expected_histogram_keys = {
            str(index) for index in range(len(histogram))
        }
        _require(
            set(observed_histogram) == expected_histogram_keys,
            "joint support histogram bins drifted",
        )
        group_histogram_total = 0
        for index in range(len(histogram)):
            value = int(observed_histogram[str(index)])
            _require(value >= 0, "joint support histogram is negative")
            histogram[index] += value
            group_histogram_total += value
        _require(group_histogram_total == count, "group support histogram drifted")
        family_audits = _mapping(audit.get("families"), "family support audits")
        _require(set(family_audits) == set(family_order), "family support set drifted")
        for family in family_order:
            values = _mapping(family_audits[family], f"{family} support audit")
            for field in family_counts[family]:
                family_counts[family][field] += int(values.get(field, -1))
    _require(int(histogram.sum()) == total, "joint histogram population drifted")
    families: dict[str, Any] = {}
    for family in family_order:
        values: dict[str, int | float] = dict(family_counts[family])
        for field, count in tuple(family_counts[family].items()):
            values[field.replace("_count", "_fraction")] = (
                count / total if total else 0.0
            )
        families[family] = values
    return {
        "schema": "pathline_template_matching.class_conditional_outer_support_summary.v1",
        "sample_count": total,
        "family_order": list(family_order),
        "required_joint_family_count": required,
        "joint_supported_family_count_histogram": {
            str(index): int(value) for index, value in enumerate(histogram)
        },
        "families": families,
    }


def _support_audit_from_prediction_manifest(
    path: Path, expected_file_sha256: str
) -> dict[str, Any]:
    payload = path.read_bytes()
    _require(
        _lower_hex(expected_file_sha256)
        and hashlib.sha256(payload).hexdigest() == expected_file_sha256,
        "prediction manifest changed before support-summary construction",
    )
    manifest = json.loads(payload.decode("utf-8"))
    _require(isinstance(manifest, Mapping), "prediction manifest root is invalid")
    inherited._authenticate_self_hash(manifest)
    groups = manifest.get("group_audits")
    _require(isinstance(groups, list), "prediction group audits are missing")
    return _aggregate_group_support_audits(groups)


def _evaluate_outer_prediction_bound(
    plan: Plan,
    selected: TailCandidateSpec,
    output_directory: Path,
    **kwargs: Any,
) -> tuple[OuterMetricRows, list[dict[str, Any]]]:
    # The inherited fresh-replay evaluator opens label-free outer projections
    # before it repeats these three artifact authentications.  The child
    # contract is stricter: a complete final scaler -> calibration -> selected
    # chain must fresh-authenticate before the first outer feature member is
    # opened.  Perform that closed-chain gate here, before entering any parent
    # evaluator code that can access an outer cache.  The parent's duplicate
    # authentication remains useful because it binds the prediction replay.
    fit_families = [
        family
        for family in plan.family_order
        if family != str(kwargs["outer_family"])
    ]
    fresh_scaler = inherited.authenticate_and_rebuild_final_scaler(
        Path(output_directory) / "final_per_scale_scaler.npz",
        Path(output_directory) / "final_per_scale_scaler_manifest.json",
        plan=plan,
        selected=selected,
        outer_family=str(kwargs["outer_family"]),
        fit_families=fit_families,
        git_commit=str(kwargs["git_commit"]),
        expected_manifest_file_sha256=str(
            kwargs["expected_scaler_manifest_sha256"]
        ),
    )
    fresh_calibration = inherited.authenticate_and_rebuild_final_calibration(
        Path(output_directory) / "final_tail_calibration.npz",
        Path(output_directory) / "final_tail_calibration_manifest.json",
        plan=plan,
        selected=selected,
        scaler=fresh_scaler,
        outer_family=str(kwargs["outer_family"]),
        fit_families=fit_families,
        git_commit=str(kwargs["git_commit"]),
        expected_manifest_file_sha256=str(
            kwargs["expected_calibration_manifest_sha256"]
        ),
    )
    inherited.authenticate_selected_candidate(
        Path(output_directory) / "selected_candidate.json",
        plan=plan,
        selected=selected,
        scaler=fresh_scaler,
        calibration=fresh_calibration,
        inner_group_metrics_path=Path(kwargs["inner_group_metrics_path"]),
        inner_group_metrics_sha256=str(kwargs["inner_group_metrics_sha256"]),
        inner_candidate_summary_path=Path(
            kwargs["inner_candidate_summary_path"]
        ),
        inner_candidate_summary_sha256=str(
            kwargs["inner_candidate_summary_sha256"]
        ),
        inner_fit_audits_path=Path(kwargs["inner_fit_audits_path"]),
        inner_fit_audits_sha256=str(kwargs["inner_fit_audits_sha256"]),
        outer_family=str(kwargs["outer_family"]),
        git_commit=str(kwargs["git_commit"]),
        expected_file_sha256=str(kwargs["expected_selected_candidate_sha256"]),
    )
    # Do not retain a second full template/calibration population while the
    # inherited prediction replay rebuilds the same immutable artifacts.
    del fresh_calibration, fresh_scaler
    metrics, access = _INHERITED_EVALUATE_OUTER_PREDICTION(
        plan, selected, output_directory, **kwargs
    )
    support = _support_audit_from_prediction_manifest(
        Path(output_directory) / "outer_prediction_manifest.json",
        str(kwargs["expected_prediction_manifest_sha256"]),
    )
    return OuterMetricRows(metrics, support), access


def _outer_summary(
    rows: Sequence[Mapping[str, Any]], outer_family: str
) -> dict[str, Any]:
    summary = _INHERITED_OUTER_SUMMARY(rows, outer_family)
    summary["schema"] = OUTER_SUMMARY_SCHEMA
    summary["experiment"] = EXPERIMENT
    _require(
        isinstance(rows, OuterMetricRows),
        "outer summary requires authenticated class-conditional support audits",
    )
    summary["class_conditional_support"] = rows.support_audit
    return summary


def _method_binding(plan: Plan, git_commit: str) -> dict[str, Any]:
    """Return the complete child method identity embedded in every JSON file."""

    _require(
        _lower_hex(git_commit, length=40),
        "numerical Git commit must be a full lowercase SHA-1",
    )
    direct = _mapping(plan.raw.get("direct_parent"), "direct parent")
    return {
        "schema": "pathline_template_matching.class_conditional_template_score_method_binding.v1",
        "experiment": EXPERIMENT,
        "config": {"path": str(plan.path), "sha256": plan.sha256},
        "core": {"path": str(CORE_PATH), "sha256": EXPECTED_CORE_SHA256},
        "direct_parent": {
            "experiment": PARENT_EXPERIMENT,
            "numerical_git_commit": direct["numerical_git_commit"],
            "config_sha256": EXPECTED_PARENT_CONFIG_SHA256,
            "runner_sha256": EXPECTED_PARENT_RUNNER_SHA256,
            "aggregator_sha256": EXPECTED_PARENT_AGGREGATOR_SHA256,
        },
        "fit": {
            "shared_scaler": "all_fit_family_natural_negatives_exact_scale",
            "templates": "full_natural_family_class_exact_scale",
            "calibration": "same_family_same_class_LOO_with_frozen_cross_scale_prior",
            "family_order": list(plan.family_order),
            "k": list(plan.ks),
            "shrinkage_lambda": plan.shrinkage_lambda,
        },
        "score": {
            "per_family": "0.5*(1+q_positive-q_negative)",
            "combine": "equal_mean_over_jointly_supported_families",
            "inner_support": "2_of_3_joint_families",
            "outer_support": "3_of_4_joint_families",
            "probability_claim": False,
        },
        "threshold": {
            "serialized_compatibility_name": "calibrated_tail_anomaly_threshold",
            "scientific_input": "class_conditional_template_score",
            "comparison": "strict_greater_than",
            "equality_prediction": "negative",
            "candidate_id_encodes_comparator": True,
        },
        "compatibility_prediction_fields": {
            "raw_negative_distance": "mean_negative_distance_over_strict_majority_joint_class_retrieval_families",
            "retrieval_supported": "strict_majority_positive_and_negative_retrieval",
            "calibration_supported": "strict_majority_joint_positive_and_negative_retrieval_and_calibration",
            "tail_anomaly": "class_conditional_template_score",
            "tail_probability": "one_minus_class_conditional_template_score_not_a_probability_claim",
            "calibration_mode_0": "joint_calibration_unsupported",
            "calibration_mode_1": "joint_calibration_supported_compatibility_sentinel_not_a_tail_mode",
        },
        "support_audit": {
            "storage": "prediction_manifest_group_audits_not_prediction_npz",
            "joint_family_count_histogram": True,
            "per_family_positive_negative_retrieval_calibration_counts_and_fractions": True,
            "fresh_replay": "recomputed_then_exact_JSON_equality_authenticated",
        },
        "prediction_array_contract": "unchanged_parent_19_arrays",
        "fold_transaction": "unchanged_parent_15_files",
        "numerical_git_commit": git_commit,
    }


def _set_inherited_global(name: str, value: Any) -> None:
    """One injectable mutation seam used by restoration-failure tests."""

    setattr(inherited, name, value)


@contextmanager
def class_conditional_parent_runtime(
    plan: Plan, git_commit: str
) -> Iterator[None]:
    """Exclusively and transactionally bind the inherited Early machinery."""

    acquired = _PARENT_RUNTIME_LOCK.acquire(blocking=False)
    _require(
        acquired,
        "class-conditional parent runtime is already active; nested or concurrent use is forbidden",
    )
    attempted: list[str] = []
    old: dict[str, Any] = {}
    restoration_failures: list[tuple[str, BaseException]] = []
    try:
        binding = _method_binding(plan, git_commit)
        original_manifest = inherited._manifest_with_self_hash
        original_authenticate_self_hash = inherited._authenticate_self_hash

        def bound_load_plan(config_path: str | Path = CONFIG_PATH) -> Plan:
            active_path = Path(config_path).resolve()
            _require(active_path == plan.path, "inherited runner requested another config")
            _require(
                sha256_file(active_path) == plan.sha256,
                "child config changed after authentication",
            )
            _require(
                sha256_file(PARENT_CONFIG_PATH) == EXPECTED_PARENT_CONFIG_SHA256
                and sha256_file(PARENT_RUNNER_PATH) == EXPECTED_PARENT_RUNNER_SHA256
                and sha256_file(PARENT_AGGREGATOR_PATH)
                == EXPECTED_PARENT_AGGREGATOR_SHA256
                and sha256_file(CORE_PATH) == binding["core"]["sha256"],
                "bound method source changed during the fold",
            )
            return plan

        def bound_bind_early_evidence(active_plan: Plan, **kwargs: Any) -> Plan:
            _require(
                active_plan.path == plan.path and active_plan.sha256 == plan.sha256,
                "inherited evidence bind received another plan",
            )
            expected = {
                "kinematic_input_manifest_path": plan.kinematic_input_manifest_path,
                "kinematic_input_manifest_file_sha256": (
                    plan.kinematic_input_manifest_file_sha256
                ),
                "synthetic_pass_path": plan.synthetic_pass_path,
                "synthetic_pass_file_sha256": plan.synthetic_pass_file_sha256,
                "sidecar_root": plan.sidecar_root,
                "sidecar_population_manifest_path": (
                    plan.sidecar_population_manifest_path
                ),
                "sidecar_population_manifest_file_sha256": (
                    plan.sidecar_population_manifest_file_sha256
                ),
            }
            _require(set(kwargs) == set(expected), "Early evidence arguments drifted")
            for name, expected_value in expected.items():
                observed = kwargs[name]
                if name.endswith("_path") or name == "sidecar_root":
                    _require(
                        Path(observed).resolve() == Path(expected_value).resolve(),
                        f"{name} drifted after authentication",
                    )
                else:
                    _require(
                        observed == expected_value,
                        f"{name} drifted after authentication",
                    )
            return plan

        def bound_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
            values = dict(payload)
            if values.get("schema") == (
                "pathline_template_matching.early_opposite_pair_kinematics_outer_reference_access.v1"
            ):
                values["schema"] = REFERENCE_AUDIT_SCHEMA
            existing = values.get(METHOD_BINDING_KEY)
            _require(
                existing is None or existing == binding,
                "artifact method binding conflicts with active method",
            )
            values[METHOD_BINDING_KEY] = binding
            return original_manifest(values)

        def bound_authenticate_self_hash(manifest: Mapping[str, Any]) -> None:
            original_authenticate_self_hash(manifest)
            _require(
                manifest.get(METHOD_BINDING_KEY) == binding,
                "artifact class-conditional method binding drifted",
            )

        def clean_exact_git_identity() -> tuple[str, bool]:
            observed, dirty = _INHERITED_GIT_IDENTITY()
            _require(not dirty, "numerical run requires a clean Git worktree")
            _require(observed == git_commit, "numerical Git commit changed")
            return observed, dirty

        replacements: dict[str, Any] = {
            "EXPERIMENT": EXPERIMENT,
            "EXPECTED_CONFIG_SHA256": EXPECTED_CONFIG_SHA256,
            "FROZEN_CANDIDATE_COUNT": FROZEN_CANDIDATE_COUNT,
            "TailCandidateSpec": TailCandidateSpec,
            "PerScaleNegativeTailModel": ClassConditionalTemplateScoreModel,
            "SCALER_ARTIFACT_SCHEMA": SCALER_ARTIFACT_SCHEMA,
            "SCALER_MANIFEST_SCHEMA": SCALER_MANIFEST_SCHEMA,
            "CALIBRATION_ARTIFACT_SCHEMA": CALIBRATION_ARTIFACT_SCHEMA,
            "CALIBRATION_MANIFEST_SCHEMA": CALIBRATION_MANIFEST_SCHEMA,
            "SELECTED_SCHEMA": SELECTED_SCHEMA,
            "PREDICTION_SCHEMA": PREDICTION_SCHEMA,
            "PREDICTION_MANIFEST_SCHEMA": PREDICTION_MANIFEST_SCHEMA,
            "INNER_AUDIT_SCHEMA": INNER_AUDIT_SCHEMA,
            "OUTER_SUMMARY_SCHEMA": OUTER_SUMMARY_SCHEMA,
            "REFERENCE_AUDIT_SCHEMA": REFERENCE_AUDIT_SCHEMA,
            "RESULT_SCHEMA": RESULT_SCHEMA,
            "COMPLETE_SCHEMA": COMPLETE_SCHEMA,
            "load_plan": bound_load_plan,
            "bind_early_evidence": bound_bind_early_evidence,
            "_fit_tail_model": _fit_tail_model,
            "_query_cache_batch": _query_cache_batch,
            "candidate_predictions": candidate_predictions,
            "_threshold_confusion_series": _threshold_confusion_series,
            "build_outer_prediction_arrays": build_outer_prediction_arrays,
            "evaluate_outer_prediction": _evaluate_outer_prediction_bound,
            "_outer_summary": _outer_summary,
            "_manifest_with_self_hash": bound_manifest,
            "_authenticate_self_hash": bound_authenticate_self_hash,
            "_git_identity": clean_exact_git_identity,
        }
        old = {name: getattr(inherited, name) for name in replacements}
        for name, value in replacements.items():
            attempted.append(name)
            _set_inherited_global(name, value)
        yield
    finally:
        for name in reversed(attempted):
            try:
                _set_inherited_global(name, old[name])
            except BaseException as error:  # pragma: no cover - catastrophic corruption
                restoration_failures.append((name, error))
        _PARENT_RUNTIME_LOCK.release()
        if restoration_failures:
            names = ", ".join(name for name, _ in restoration_failures)
            raise RuntimeError(
                f"failed to restore inherited parent globals: {names}"
            ) from restoration_failures[0][1]


def evaluate_outer_prediction(
    plan: Plan,
    selected: TailCandidateSpec,
    output_directory: Path,
    *,
    outer_family: str,
    git_commit: str,
    device: str,
    expected_scaler_manifest_sha256: str,
    expected_calibration_manifest_sha256: str,
    expected_selected_candidate_sha256: str,
    expected_prediction_manifest_sha256: str,
    inner_group_metrics_path: Path,
    inner_group_metrics_sha256: str,
    inner_candidate_summary_path: Path,
    inner_candidate_summary_sha256: str,
    inner_fit_audits_path: Path,
    inner_fit_audits_sha256: str,
) -> tuple[OuterMetricRows, list[dict[str, Any]]]:
    """Aggregator-facing fresh replay under the complete child binding."""

    kwargs = {
        "outer_family": outer_family,
        "git_commit": git_commit,
        "device": device,
        "expected_scaler_manifest_sha256": expected_scaler_manifest_sha256,
        "expected_calibration_manifest_sha256": (
            expected_calibration_manifest_sha256
        ),
        "expected_selected_candidate_sha256": expected_selected_candidate_sha256,
        "expected_prediction_manifest_sha256": expected_prediction_manifest_sha256,
        "inner_group_metrics_path": inner_group_metrics_path,
        "inner_group_metrics_sha256": inner_group_metrics_sha256,
        "inner_candidate_summary_path": inner_candidate_summary_path,
        "inner_candidate_summary_sha256": inner_candidate_summary_sha256,
        "inner_fit_audits_path": inner_fit_audits_path,
        "inner_fit_audits_sha256": inner_fit_audits_sha256,
    }
    with class_conditional_parent_runtime(plan, git_commit):
        return _evaluate_outer_prediction_bound(
            plan, selected, Path(output_directory), **kwargs
        )


def run(
    config_path: str | Path,
    outer_family: str,
    output_dir: str | Path,
    *,
    device: str,
    kinematic_input_manifest_path: str | Path,
    kinematic_input_manifest_file_sha256: str,
    synthetic_pass_path: str | Path,
    synthetic_pass_file_sha256: str,
    sidecar_root: str | Path,
    sidecar_population_manifest_path: str | Path,
    sidecar_population_manifest_file_sha256: str,
    expected_config_sha256: str | None = EXPECTED_CONFIG_SHA256,
) -> dict[str, Any]:
    """Run one immutable nested-family fold from a clean exact revision."""

    plan = load_plan(config_path)
    _require(outer_family in plan.family_order, f"unknown outer family: {outer_family}")
    if expected_config_sha256 is not None:
        _require(
            plan.sha256 == expected_config_sha256,
            "frozen config SHA-256 mismatch",
        )
    git_commit, dirty = _INHERITED_GIT_IDENTITY()
    _require(not dirty, "Ibex numerical run requires a clean committed Git worktree")
    plan = bind_early_evidence(
        plan,
        kinematic_input_manifest_path=kinematic_input_manifest_path,
        kinematic_input_manifest_file_sha256=kinematic_input_manifest_file_sha256,
        synthetic_pass_path=synthetic_pass_path,
        synthetic_pass_file_sha256=synthetic_pass_file_sha256,
        sidecar_root=sidecar_root,
        sidecar_population_manifest_path=sidecar_population_manifest_path,
        sidecar_population_manifest_file_sha256=(
            sidecar_population_manifest_file_sha256
        ),
    )
    _require(
        plan.source_identity is not None
        and plan.source_identity.git_commit == git_commit,
        "Early evidence and numerical runner commits differ",
    )
    with class_conditional_parent_runtime(plan, git_commit):
        result = inherited.run(
            config_path,
            outer_family,
            output_dir,
            device=device,
            kinematic_input_manifest_path=kinematic_input_manifest_path,
            kinematic_input_manifest_file_sha256=(
                kinematic_input_manifest_file_sha256
            ),
            synthetic_pass_path=synthetic_pass_path,
            synthetic_pass_file_sha256=synthetic_pass_file_sha256,
            sidecar_root=sidecar_root,
            sidecar_population_manifest_path=sidecar_population_manifest_path,
            sidecar_population_manifest_file_sha256=(
                sidecar_population_manifest_file_sha256
            ),
            expected_config_sha256=plan.sha256,
        )
    _require(
        result.get("schema") == RESULT_SCHEMA
        and result.get("experiment") == EXPERIMENT
        and result.get(METHOD_BINDING_KEY) == _method_binding(plan, git_commit),
        "completed result lost its class-conditional binding",
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--expected-config-sha256", default=EXPECTED_CONFIG_SHA256)
    parser.add_argument("--outer-family", required=True, choices=FAMILY_ORDER)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--kinematic-input-manifest", required=True)
    parser.add_argument("--kinematic-input-manifest-sha256", required=True)
    parser.add_argument("--synthetic-pass", required=True)
    parser.add_argument("--synthetic-pass-sha256", required=True)
    parser.add_argument("--sidecar-root", required=True)
    parser.add_argument("--sidecar-population-manifest", required=True)
    parser.add_argument("--sidecar-population-manifest-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    run(
        arguments.config,
        arguments.outer_family,
        arguments.output_dir,
        device=arguments.device,
        kinematic_input_manifest_path=arguments.kinematic_input_manifest,
        kinematic_input_manifest_file_sha256=(
            arguments.kinematic_input_manifest_sha256
        ),
        synthetic_pass_path=arguments.synthetic_pass,
        synthetic_pass_file_sha256=arguments.synthetic_pass_sha256,
        sidecar_root=arguments.sidecar_root,
        sidecar_population_manifest_path=arguments.sidecar_population_manifest,
        sidecar_population_manifest_file_sha256=(
            arguments.sidecar_population_manifest_sha256
        ),
        expected_config_sha256=arguments.expected_config_sha256,
    )


if __name__ == "__main__":
    main()

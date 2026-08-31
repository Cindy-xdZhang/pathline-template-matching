#!/usr/bin/env python3
"""Frozen nested-family runner for Verify_EarlyOppositePairKinematics_1.1.

The runner owns independent Early schemas, sidecar joins, artifacts, and the
outer-reference gate.  It reuses only the unchanged numerical primitives from
the authenticated PerScale parent; every feature projection is the frozen
parent FMT representation followed by the same four seed-time coordinates.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import gc
import hashlib
import io
import json
import math
import os
import platform
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.negative_tail_calibration import (  # noqa: E402
    CALIBRATION_NONE,
    SHRINKAGE_LAMBDA,
)
from pathline_template_matching.per_scale_negative_metric import (  # noqa: E402
    SCALER_ARRAY_NAMES,
    PerScaleNegativeScaler,
    PerScaleNegativeTailModel,
)
from pathline_template_matching.nested_scale_validation import (  # noqa: E402
    fixed_top_fraction_predictions,
    representation_features,
)
from pathline_template_matching.early_kinematic_preparation import (  # noqa: E402
    CleanSourceIdentity,
    POPULATION_MANIFEST_SCHEMA,
    PRODUCTION_CONTRACT,
    authenticate_kinematic_input_manifest,
    authenticate_synthetic_pass_marker,
    capture_clean_source_identity,
    composite_descriptor_contracts,
)
from pathline_template_matching.seed_time_kinematic_sidecar import (  # noqa: E402
    load_seed_time_kinematic_sidecar,
)
from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_array_sha256,
    canonical_json_sha256,
    sha256_file,
)
from scripts.run_verify_scale_conditioned_retrieval_1_1 import (  # noqa: E402
    CacheProjection,
    CacheRow,
    _classification_counts,
    _configure_execution,
    _git_identity,
    _json_safe,
    _json_safe_content_sha256,
    _ranking_metrics_one_sort,
    _require,
    _stable_file_identity,
    _threshold_confusion_series,
    _utc_now,
    _validate_cache_arrays as _validate_parent_cache_arrays,
    _validate_outer_prediction_projection as _validate_parent_label_free_projection,
    load_cache_rows,
)


EXPERIMENT = "Verify_EarlyOppositePairKinematics_1.1"
EXPECTED_CONFIG_SHA256 = (
    "e6bac4568025f42cf0a9effd78620e5ab4ba5653429a7023bd91816f29512767"
)
PREPARATION_ARTIFACT_GIT_COMMIT = (
    "fd0412dc134da9dba88d71d665fc2ad160e78e06"
)
FAMILY_ORDER = (
    "half_cylinder",
    "delta_wing",
    "f22_raptor",
    "channel",
    "boeing_747",
)
REPRESENTATIONS = (
    "fmt161_plus_seed4",
    "real_neighbor36_plus_seed4",
    "chirality_all35_plus_seed4",
)
PARENT_REPRESENTATION = MappingProxyType(
    {
        "fmt161_plus_seed4": "fmt161",
        "real_neighbor36_plus_seed4": "real_neighbor36",
        "chirality_all35_plus_seed4": "chirality_all35",
    }
)
COMPOSITE_WIDTH = MappingProxyType(
    {
        "fmt161_plus_seed4": 165,
        "real_neighbor36_plus_seed4": 40,
        "chirality_all35_plus_seed4": 39,
    }
)


def _preparation_artifact_identity(
    current_identity: CleanSourceIdentity,
) -> CleanSourceIdentity:
    """Bind immutable preparation evidence to its exact producer commit.

    The scientific preparation sources are authenticated again from the current
    clean checkout.  Only the Git commit field is replaced with the exact commit
    that produced the already sealed input, synthetic, and sidecar artifacts.
    Their embedded source-hash mapping must still match byte-for-byte during
    artifact authentication.
    """

    return replace(
        current_identity,
        git_commit=PREPARATION_ARTIFACT_GIT_COMMIT,
    )


def _ordered_composite_descriptor_ids(
    observed: object,
    expected: Mapping[str, str],
) -> Mapping[str, str]:
    """Authenticate descriptor IDs without trusting JSON object key order."""

    _require(isinstance(observed, Mapping), "composite descriptor population is not a mapping")
    _require(
        set(observed) == set(REPRESENTATIONS)
        and set(expected) == set(REPRESENTATIONS),
        "composite descriptor population changed",
    )
    ordered = {
        name: str(observed[name])
        for name in REPRESENTATIONS
    }
    expected_ordered = {
        name: str(expected[name])
        for name in REPRESENTATIONS
    }
    _require(
        ordered == expected_ordered,
        "composite descriptor population values drifted",
    )
    return MappingProxyType(ordered)
K_VALUES = (1, 5, 15, 31)
SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.0)
TAIL_THRESHOLDS = tuple(round(0.50 + 0.01 * index, 2) for index in range(50))
GRID_SHAPE = (40, 40, 40)
BLOCK_NAMES = ("legacy_2_1", "expanded_3_1")
SPATIAL_REPLAY_ULP_BOUNDS: Mapping[str, int] = MappingProxyType(
    {
        "spatial_score": 8,
        "spatial_denominator": 8,
    }
)
FROZEN_CANDIDATE_COUNT = 3060
CALIBRATION_ARTIFACT_SCHEMA = "pathline_template_matching.early_opposite_pair_kinematics_tail_calibration_artifact.v1"
CALIBRATION_MANIFEST_SCHEMA = (
    "pathline_template_matching.early_opposite_pair_kinematics_tail_calibration_manifest.v1"
)
SCALER_ARTIFACT_SCHEMA = "pathline_template_matching.early_opposite_pair_kinematics_per_scale_metric.v1"
SCALER_MANIFEST_SCHEMA = (
    "pathline_template_matching.early_opposite_pair_kinematics_per_scale_metric_manifest.v1"
)
PREDICTION_SCHEMA = "pathline_template_matching.early_opposite_pair_kinematics_outer_prediction.v1"
PREDICTION_MANIFEST_SCHEMA = (
    "pathline_template_matching.early_opposite_pair_kinematics_outer_prediction_manifest.v1"
)
SELECTED_SCHEMA = "pathline_template_matching.early_opposite_pair_kinematics_selected_candidate.v1"
RESULT_SCHEMA = "pathline_template_matching.early_opposite_pair_kinematics_result.v1"
COMPLETE_SCHEMA = "pathline_template_matching.early_opposite_pair_kinematics_run_complete.v1"
INNER_AUDIT_SCHEMA = "pathline_template_matching.early_opposite_pair_kinematics_inner_fit_audits.v1"
OUTER_SUMMARY_SCHEMA = "pathline_template_matching.early_opposite_pair_kinematics_outer_summary.v1"
REFERENCE_AUDIT_SCHEMA = "pathline_template_matching.early_opposite_pair_kinematics_outer_reference_access.v1"

REQUIRED_FOLD_FILES = (
    "inner_group_metrics.csv",
    "inner_candidate_summary.csv",
    "inner_fit_audits.json",
    "final_per_scale_scaler.npz",
    "final_per_scale_scaler_manifest.json",
    "final_tail_calibration.npz",
    "final_tail_calibration_manifest.json",
    "selected_candidate.json",
    "outer_predictions.npz",
    "outer_prediction_manifest.json",
    "outer_group_metrics.csv",
    "outer_summary.json",
    "outer_reference_access_audit.json",
    "result_manifest.json",
    "RUN_COMPLETE.json",
)


def _environment_audit(device: str) -> dict[str, Any]:
    """Record the requested backend without querying a CPU as a CUDA device."""

    requested_device = torch.device(device)
    cuda_available = torch.cuda.is_available()
    audit: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "requested_device": device,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "cuda_available": cuda_available,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if cuda_available:
        audit.update(
            {
                "cuda_version": torch.version.cuda,
                "cuda_device_count": torch.cuda.device_count(),
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            }
        )
        if requested_device.type == "cuda":
            audit.update(
                {
                    "cuda_device_name": torch.cuda.get_device_name(requested_device),
                    "cuda_device_capability": list(
                        torch.cuda.get_device_capability(requested_device)
                    ),
                }
            )
        else:
            audit["cuda_device_query_skipped"] = "requested_device_is_not_cuda"
    return audit


@dataclass(frozen=True)
class Plan:
    path: Path
    sha256: str
    raw: Mapping[str, Any]
    family_order: tuple[str, ...]
    families: Mapping[str, tuple[str, ...]]
    dataset_to_family: Mapping[str, str]
    manifest_path: Path
    manifest_size: int
    manifest_sha256: str
    manifest_schema: str
    manifest_rows_sha256: str
    cache_commit: str
    parent_config_sha256: str
    descriptor_id: str
    cache_schema: str
    representations: tuple[str, ...]
    ks: tuple[int, ...]
    sigmas: tuple[float, ...]
    thresholds: tuple[float, ...]
    fixed_top_fraction: float
    grid_shape: tuple[int, int, int]
    gaussian_truncate: float
    query_chunk_size: int
    library_chunk_size: int
    shrinkage_lambda: float
    output_root: Path
    required_fold_files: tuple[str, ...]
    source_identity: CleanSourceIdentity | None = None
    kinematic_input_manifest_path: Path | None = None
    kinematic_input_manifest_file_sha256: str | None = None
    kinematic_input_manifest_content_sha256: str | None = None
    synthetic_pass_path: Path | None = None
    synthetic_pass_file_sha256: str | None = None
    sidecar_root: Path | None = None
    sidecar_population_manifest_path: Path | None = None
    sidecar_population_manifest_file_sha256: str | None = None
    sidecar_population_manifest_content_sha256: str | None = None
    sidecar_population: Mapping[str, Any] | None = None
    composite_descriptor_ids: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass
class EarlyCacheProjection:
    """Exact parent projection joined to one authenticated kinematic sidecar."""

    row: CacheRow
    fmt_features: np.ndarray
    seed_kinematic4: np.ndarray
    scale_ids: np.ndarray
    center_indices: np.ndarray
    block_indices: np.ndarray
    assigned_row_indices: np.ndarray
    labels: np.ndarray | None
    metadata: Mapping[str, Any]
    sidecar_file_sha256: str
    sidecar_combined_array_sha256: str

    @property
    def count(self) -> int:
        return len(self.scale_ids)


@dataclass(frozen=True)
class TailCandidateSpec:
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
            decision = f"calibrated_tail_anomaly_threshold={self.decision_value:.2f}"
        else:  # guarded again here so an invalid object cannot be serialized
            raise ValueError(f"unsupported decision rule: {self.decision_rule}")
        return f"representation={self.representation}|k={self.k}|sigma={sigma}|{decision}"


@dataclass(frozen=True)
class SpatialTailScores:
    scores: np.ndarray
    denominator: np.ndarray
    calibration_supported: np.ndarray
    imputed: np.ndarray
    unimputable: np.ndarray


@dataclass(frozen=True)
class VerifiedScalerArtifact:
    manifest_path: Path
    manifest_file_sha256: str
    scaler_file_sha256: str
    manifest: Mapping[str, Any]
    scaler: PerScaleNegativeScaler
    _authentication_seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedCalibrationArtifact:
    manifest_path: Path
    manifest_file_sha256: str
    calibration_file_sha256: str
    manifest: Mapping[str, Any]
    model: PerScaleNegativeTailModel
    _authentication_seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedSelectedCandidate:
    path: Path
    file_sha256: str
    manifest: Mapping[str, Any]
    _authentication_seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedNegativeTailPrediction:
    manifest_path: Path
    manifest_file_sha256: str
    prediction_file_sha256: str
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    _authentication_seal: object = field(repr=False, compare=False)


_AUTHENTICATION_SEAL = object()


def _deep_freeze(value: Any) -> Any:
    """Return an immutable copy of JSON-like state and NumPy arrays."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        output = np.frombuffer(
            contiguous.tobytes(order="C"), dtype=contiguous.dtype
        ).reshape(contiguous.shape)
        return output
    return value


def _fsync_parent_directory(parent: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(os.fspath(parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_replace(path: Path, writer: Any, *, text_mode: bool) -> str:
    """Publish via one atomic hard-link no-replace operation and fsync both entries."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        mode = "w" if text_mode else "wb"
        options = {"encoding": "utf-8", "newline": ""} if text_mode else {}
        with os.fdopen(descriptor, mode, **options) as destination:
            descriptor = -1
            writer(destination)
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, path, follow_symlinks=False)
        _fsync_parent_directory(path.parent)
        temporary.unlink()
        _fsync_parent_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
            _fsync_parent_directory(path.parent)
    return sha256_file(path)


def _atomic_json(path: Path, value: Any) -> str:
    payload = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    return _publish_no_replace(path, lambda stream: stream.write(payload), text_mode=True)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return int(bool(value))
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return "" if not np.isfinite(numeric) else format(numeric, ".12g")
    return "" if value is None else value


def _atomic_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> str:
    def write(destination: Any) -> None:
        writer = csv.DictWriter(
            destination, fieldnames=list(fieldnames), extrasaction="raise"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})

    return _publish_no_replace(path, write, text_mode=True)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    return _publish_no_replace(
        path,
        lambda destination: np.savez_compressed(destination, **arrays),
        text_mode=False,
    )


def _lower_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def load_plan(config_path: str | Path) -> Plan:
    """Load and fail-closed validate the immutable Early numerical contract."""

    path = Path(config_path).resolve()
    config_bytes = path.read_bytes()
    digest = hashlib.sha256(config_bytes).hexdigest()
    _require(digest == EXPECTED_CONFIG_SHA256, "frozen config SHA-256 drifted")
    raw = yaml.safe_load(config_bytes.decode("utf-8"))
    _require(isinstance(raw, Mapping), "config root must be a mapping")
    _require(raw.get("experiment") == EXPERIMENT, "experiment identity drifted")
    _require(
        raw.get("phase") == "exposed_train_only_nested_family_validation",
        "phase drifted",
    )
    _require(
        raw.get("status") == "frozen_pre_run_not_implemented",
        "immutable freeze-status history drifted",
    )
    _require(
        raw.get("frozen_before_first_read_of_any_per_scale_outer_result") is True,
        "pre-result freeze history drifted",
    )

    evidence = raw.get("evidence_scope")
    _require(isinstance(evidence, Mapping), "evidence_scope is missing")
    _require(
        evidence.get("allowed_inputs")
        == "exact_32_mainExp_TemplateMatching_3.1_train_cache_shards_and_their_matching_train_portable_windows",
        "input scope drifted",
    )
    _require(
        evidence.get("forbidden_datasets") == ["tangaroa", "smokeBuoyancy"],
        "forbidden datasets drifted",
    )

    families_raw = raw.get("families")
    split = raw.get("nested_split")
    _require(
        isinstance(families_raw, Mapping) and isinstance(split, Mapping),
        "family split is missing",
    )
    family_order = tuple(str(value) for value in split.get("outer_order", ()))
    _require(family_order == FAMILY_ORDER, "outer family order drifted")
    _require(
        tuple(split.get("inner_order", ())) == FAMILY_ORDER,
        "inner family order drifted",
    )
    _require(
        split.get("outer_labels_available_to_selection") is False,
        "outer-label gate drifted",
    )
    families = {
        str(key): tuple(str(value) for value in values)
        for key, values in families_raw.items()
    }
    _require(tuple(families) == FAMILY_ORDER, "family mapping order drifted")
    datasets = [
        dataset for family in FAMILY_ORDER for dataset in families[family]
    ]
    _require(
        len(datasets) == 8 and len(set(datasets)) == 8,
        "families must contain eight unique datasets",
    )
    dataset_to_family = {
        dataset: family
        for family in FAMILY_ORDER
        for dataset in families[family]
    }

    parent = raw.get("parent_identity")
    _require(isinstance(parent, Mapping), "parent_identity is missing")
    _require(
        parent.get("negative_tail_config_sha256")
        == "4b6f05dd852990364aa3465d1c990d79532e6c859ab27a219f3d95817868ce3b",
        "NegativeTail parent identity drifted",
    )
    _require(
        parent.get("per_scale_config_sha256")
        == "b469b909466dda941d122629ba43cf94e872faceed73c5f0970e3cf66697dd79",
        "PerScale parent identity drifted",
    )
    manifest = parent.get("train_cache_input_manifest")
    _require(isinstance(manifest, Mapping), "parent input manifest is missing")
    _require(
        int(manifest.get("row_count", -1)) == 32,
        "parent input manifest row count drifted",
    )
    _require(
        _lower_hex(manifest.get("sha256"))
        and _lower_hex(manifest.get("rows_content_sha256")),
        "parent input manifest hash is invalid",
    )

    representations_raw = raw.get("representations")
    _require(isinstance(representations_raw, Mapping), "representations are missing")
    representations = tuple(representations_raw.get("order", ()))
    _require(representations == REPRESENTATIONS, "representation grid drifted")
    for representation, width in COMPOSITE_WIDTH.items():
        spec = representations_raw.get(representation)
        _require(
            isinstance(spec, Mapping)
            and int(spec.get("width", -1)) == width
            and spec.get("append") == "seed_kinematic4_in_frozen_order",
            f"{representation}: composite contract drifted",
        )

    retrieval = raw.get("retrieval")
    calibration = raw.get("negative_tail_calibration")
    transform = raw.get("group_transform")
    decisions = raw.get("decision_candidates")
    library = raw.get("library_and_feature_metric")
    _require(
        all(
            isinstance(value, Mapping)
            for value in (retrieval, calibration, transform, decisions, library)
        ),
        "numerical contract is incomplete",
    )
    assert isinstance(retrieval, Mapping)
    assert isinstance(calibration, Mapping)
    assert isinstance(transform, Mapping)
    assert isinstance(decisions, Mapping)
    assert isinstance(library, Mapping)

    ks = tuple(int(value) for value in retrieval.get("ks", ()))
    _require(ks == K_VALUES, "k grid drifted")
    _require(
        retrieval.get("conditioning") == "exact_numeric_scale_id",
        "same-scale retrieval drifted",
    )
    _require(
        retrieval.get("distance")
        == "exact_euclidean_after_fit_negative_exact_scale_diagonal_standardization",
        "distance drifted",
    )
    _require(
        library.get("scaler")
        == "fit_negative_exact_per_scale_mean_and_shrunk_within_scale_population_std",
        "per-scale scaler identity drifted",
    )
    _require(library.get("local_variance_ddof") == 0, "local variance ddof drifted")
    _require(
        float(library.get("shrinkage_lambda", -1)) == SHRINKAGE_LAMBDA,
        "scaler shrinkage lambda drifted",
    )
    _require(
        library.get("shrinkage_domain") == "variance_before_square_root",
        "scaler shrinkage domain drifted",
    )
    _require(
        float(library.get("std_floor_exclusive", -1)) == 1.0e-12,
        "scaler std floor drifted",
    )
    _require(
        library.get("no_local_row_policy")
        == "no_scale_library_and_retrieval_unsupported",
        "no-local-row policy drifted",
    )
    _require(
        calibration.get("reference_population")
        == "fit_family_natural_negative_rows_only",
        "calibration reference drifted",
    )
    _require(
        calibration.get("leave_one_out_definition")
        == "explicit_self_row_exclusion_after_one_fit_negative_per_scale_scaler_fit",
        "leave-one-out definition drifted",
    )
    _require(
        calibration.get("scaler_refit_per_leave_one_out_row") is False,
        "leave-one-out scaler refit drifted",
    )
    _require(
        calibration.get("duplicate_feature_policy")
        == "exclude_only_current_row_and_retain_other_zero_distance_rows",
        "duplicate policy drifted",
    )
    _require(
        calibration.get("scale_blocks")
        == {"legacy_2_1": [0, 999], "expanded_3_1": [1000, 1999]},
        "scale-block names or bounds drifted",
    )
    _require(
        float(calibration.get("shrinkage_lambda", -1))
        == SHRINKAGE_LAMBDA
        == 64.0,
        "tail shrinkage lambda drifted",
    )
    _require(
        calibration.get("retrieval_support_rule")
        == "scale_negative_count_greater_than_or_equal_to_k",
        "retrieval support drifted",
    )
    _require(
        calibration.get("local_reference_rule")
        == "scale_negative_count_greater_than_or_equal_to_k_plus_1",
        "calibration support drifted",
    )
    _require(
        transform.get("query_group_rank") == "forbidden",
        "query-group rank must remain forbidden",
    )
    _require(
        transform.get("input_score") == "fit_negative_tail_anomaly",
        "spatial input drifted",
    )
    sigmas = tuple(
        float(value)
        for value in transform.get("gaussian_sigmas_grid_indices", ())
    )
    _require(sigmas == SIGMAS, "Gaussian grid drifted")
    _require(
        tuple(int(value) for value in transform.get("grid_shape_zyx", ()))
        == GRID_SHAPE,
        "grid shape drifted",
    )
    _require(
        transform.get("positive_sigma_policy")
        == "calibration_support_mask_normalized_spatial_imputation",
        "spatial policy drifted",
    )

    threshold_spec = decisions.get("calibrated_tail_anomaly_threshold")
    top_spec = decisions.get("fixed_top_fraction")
    _require(
        isinstance(threshold_spec, Mapping) and isinstance(top_spec, Mapping),
        "decision definitions are missing",
    )
    assert isinstance(threshold_spec, Mapping)
    assert isinstance(top_spec, Mapping)
    thresholds = tuple(float(value) for value in threshold_spec.get("values", ()))
    _require(thresholds == TAIL_THRESHOLDS, "tail anomaly threshold grid drifted")
    _require(
        threshold_spec.get("comparison")
        == "score_greater_than_or_equal_to_threshold",
        "threshold comparison drifted",
    )
    fixed_fraction = float(top_spec.get("fraction", -1.0))
    _require(fixed_fraction == 0.05, "fixed top fraction drifted")
    _require(
        int(decisions.get("frozen_candidate_count", -1))
        == FROZEN_CANDIDATE_COUNT,
        "candidate count drifted",
    )

    outer_gate = raw.get("outer_label_gate")
    _require(isinstance(outer_gate, Mapping), "outer label gate is missing")
    _require(
        tuple(outer_gate.get("final_artifacts_required_before_any_outer_sidecar_feature_member_opens", ()))
        == (
            "final_per_scale_scaler",
            "final_tail_calibration",
            "selected_candidate",
        ),
        "outer final-artifact gate drifted",
    )
    _require(
        outer_gate.get("outer_prediction_must_be_written_and_freshly_recomputed_and_authenticated_before_parent_valid_labels_open")
        is True,
        "outer prediction/reference gate drifted",
    )

    output = raw.get("output")
    _require(isinstance(output, Mapping), "output contract is missing")
    assert isinstance(output, Mapping)
    _require(output.get("overwrite") == "forbidden", "overwrite policy drifted")
    _require(
        Path(str(output.get("sidecar_root"))).as_posix()
        == "/ibex/user/zhanx0o/pathline-template-matching/Verify_EarlyOppositePairKinematics_1.1/kinematic_cache/train",
        "sidecar root drifted",
    )

    return Plan(
        path=path,
        sha256=digest,
        raw=raw,
        family_order=family_order,
        families=families,
        dataset_to_family=dataset_to_family,
        manifest_path=Path(str(manifest["path"])),
        manifest_size=int(manifest["size_bytes"]),
        manifest_sha256=str(manifest["sha256"]),
        manifest_schema=str(manifest["schema"]),
        manifest_rows_sha256=str(manifest["rows_content_sha256"]),
        cache_commit=str(parent["cache_builder_git_commit"]),
        parent_config_sha256=str(parent["main_config_sha256"]),
        descriptor_id=str(parent["descriptor_id"]),
        cache_schema=str(parent["cache_schema"]),
        representations=representations,
        ks=ks,
        sigmas=sigmas,
        thresholds=thresholds,
        fixed_top_fraction=fixed_fraction,
        grid_shape=GRID_SHAPE,
        gaussian_truncate=float(transform["gaussian_truncate"]),
        query_chunk_size=int(retrieval["query_chunk_size"]),
        library_chunk_size=int(retrieval["library_chunk_size"]),
        shrinkage_lambda=float(calibration["shrinkage_lambda"]),
        output_root=Path(str(output["root"])),
        required_fold_files=REQUIRED_FOLD_FILES,
    )


_POPULATION_TOP_LEVEL = frozenset(
    {
        "schema", "experiment", "status", "git_commit", "worktree_clean",
        "verify_config_sha256", "source_file_sha256",
        "source_file_sha256_content_sha256", "input_manifest_path",
        "input_manifest_file_sha256", "input_manifest_content_sha256",
        "synthetic_pass_path", "synthetic_pass_file_sha256",
        "composite_descriptor_ids", "sidecar_count",
        "sidecar_row_count_total", "rows", "rows_content_sha256",
        "forbidden_dataset_access", "manifest_write_order", "content_sha256",
    }
)
_POPULATION_ROW_FIELDS = frozenset(
    {
        "dataset", "physical_family", "source_ordinal", "source_index",
        "completion_relative_path", "completion_size_bytes",
        "completion_file_sha256", "sidecar_relative_path",
        "sidecar_size_bytes", "sidecar_file_sha256",
        "sidecar_combined_array_sha256", "sidecar_row_count",
    }
)


def _self_hashed_json(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    """Authenticate an immutable JSON file and its canonical content hash."""

    _require(_lower_hex(expected_file_sha256), f"invalid expected SHA-256: {path}")
    before = path.stat()
    _require(before.st_size > 0, f"empty evidence file: {path}")
    payload = path.read_bytes()
    after = path.stat()
    _require(
        (before.st_size, before.st_mtime_ns, getattr(before, "st_ino", 0))
        == (after.st_size, after.st_mtime_ns, getattr(after, "st_ino", 0)),
        f"evidence changed while reading: {path}",
    )
    _require(
        hashlib.sha256(payload).hexdigest() == expected_file_sha256,
        f"evidence SHA-256 mismatch: {path}",
    )
    value = json.loads(payload.decode("utf-8"))
    _require(isinstance(value, dict), f"JSON evidence root is not a mapping: {path}")
    claimed = value.get("content_sha256")
    without = dict(value)
    without.pop("content_sha256", None)
    _require(
        _lower_hex(claimed) and claimed == canonical_json_sha256(without),
        f"JSON content SHA-256 mismatch: {path}",
    )
    return value


def _authenticate_population_envelope_without_sidecar_member_open(
    path: Path,
    *,
    expected_file_sha256: str,
    sidecar_root: Path,
    input_manifest: Mapping[str, Any],
    input_manifest_path: Path,
    input_manifest_file_sha256: str,
    synthetic_pass_path: Path,
    synthetic_pass_file_sha256: str,
    identity: CleanSourceIdentity,
) -> Mapping[str, Any]:
    """Authenticate the 32-file envelope without opening any sidecar NPZ member.

    This is intentionally narrower than preparation's full population replay:
    the final-artifact gate forbids opening the held-out family's sidecar feature
    member at startup.  Every sidecar byte file and completion marker is still
    size/SHA authenticated here; the selected family members are deserialized
    only at the phase where their feature access becomes legal.
    """

    _require(path == sidecar_root / "SIDECAR_POPULATION.json", "population path drifted")
    value = _self_hashed_json(path, expected_file_sha256)
    _require(set(value) == _POPULATION_TOP_LEVEL, "population member set drifted")
    expected_top = {
        "schema": POPULATION_MANIFEST_SCHEMA,
        "experiment": EXPERIMENT,
        "status": "passed",
        "git_commit": identity.git_commit,
        "worktree_clean": True,
        "verify_config_sha256": EXPECTED_CONFIG_SHA256,
        "source_file_sha256": dict(identity.source_file_sha256_items),
        "source_file_sha256_content_sha256": identity.source_content_sha256,
        "input_manifest_path": str(input_manifest_path),
        "input_manifest_file_sha256": input_manifest_file_sha256,
        "input_manifest_content_sha256": str(input_manifest["content_sha256"]),
        "synthetic_pass_path": str(synthetic_pass_path),
        "synthetic_pass_file_sha256": synthetic_pass_file_sha256,
        "sidecar_count": 32,
        "forbidden_dataset_access": False,
        "manifest_write_order": "last_after_all_32_completion_markers_and_sidecars_were_authenticated",
    }
    drift = {
        key: (value.get(key), expected)
        for key, expected in expected_top.items()
        if value.get(key) != expected
    }
    _require(not drift, f"population provenance drifted: {drift}")
    rows = value.get("rows")
    _require(
        isinstance(rows, list)
        and len(rows) == 32
        and value.get("rows_content_sha256") == canonical_json_sha256(rows),
        "population row set/hash drifted",
    )
    expected_input_rows = list(input_manifest["rows"])
    expected_files = {path.resolve()}
    total_rows = 0
    for input_row, row in zip(expected_input_rows, rows, strict=True):
        _require(isinstance(row, Mapping), "population row is not a mapping")
        _require(set(row) == _POPULATION_ROW_FIELDS, "population row member set drifted")
        identity_fields = {
            "dataset": str(input_row["dataset"]),
            "physical_family": str(input_row["physical_family"]),
            "source_ordinal": int(input_row["source_ordinal"]),
            "source_index": int(input_row["source_index"]),
        }
        _require(
            all(row.get(name) == expected for name, expected in identity_fields.items()),
            "population rows are missing, duplicated, extra, or reordered",
        )
        dataset_lower = str(row["dataset"]).lower()
        _require(
            "tangaroa" not in dataset_lower and "smoke" not in dataset_lower,
            "forbidden dataset entered sidecar population",
        )
        completion = (sidecar_root / str(row["completion_relative_path"])).resolve()
        sidecar = (sidecar_root / str(row["sidecar_relative_path"])).resolve()
        _require(
            completion.is_relative_to(sidecar_root)
            and sidecar.is_relative_to(sidecar_root),
            "population child path escapes sidecar root",
        )
        expected_files.update((completion, sidecar))
        for child, size_name, hash_name in (
            (completion, "completion_size_bytes", "completion_file_sha256"),
            (sidecar, "sidecar_size_bytes", "sidecar_file_sha256"),
        ):
            before = child.stat()
            digest = sha256_file(child)
            after = child.stat()
            _require(
                before.st_size == int(row[size_name])
                and after.st_size == before.st_size
                and before.st_mtime_ns == after.st_mtime_ns
                and digest == str(row[hash_name]),
                f"population child identity changed: {child}",
            )
        completion_value = _self_hashed_json(
            completion, str(row["completion_file_sha256"])
        )
        _require(
            completion_value.get("dataset") == row["dataset"]
            and int(completion_value.get("source_ordinal", -1))
            == int(row["source_ordinal"])
            and completion_value.get("sidecar_file_sha256")
            == row["sidecar_file_sha256"]
            and completion_value.get("sidecar_combined_array_sha256")
            == row["sidecar_combined_array_sha256"],
            "population/completion identity drifted",
        )
        total_rows += int(row["sidecar_row_count"])
    actual_files = {
        child.resolve() for child in sidecar_root.rglob("*") if child.is_file()
    }
    _require(actual_files == expected_files, "sidecar population file set drifted")
    _require(
        int(value.get("sidecar_row_count_total", -1)) == total_rows,
        "sidecar population total row count drifted",
    )
    frozen = _deep_freeze(value)
    _require(isinstance(frozen, Mapping), "population envelope did not freeze")
    return frozen


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
    """Authenticate all 32 sidecars and bind their immutable identities to a plan."""

    identity = capture_clean_source_identity(ROOT)
    preparation_identity = _preparation_artifact_identity(identity)
    input_path = Path(kinematic_input_manifest_path).resolve()
    synthetic_path = Path(synthetic_pass_path).resolve()
    root = Path(sidecar_root).resolve()
    population_path = Path(sidecar_population_manifest_path).resolve()
    _require(
        root == plan.output_root / "kinematic_cache" / "train",
        "runtime sidecar root differs from the frozen config",
    )
    _require(
        population_path == root / "SIDECAR_POPULATION.json",
        "population manifest path differs from the frozen root",
    )
    authenticate_synthetic_pass_marker(
        synthetic_path,
        expected_file_sha256=synthetic_pass_file_sha256,
        identity=preparation_identity,
        contract=PRODUCTION_CONTRACT,
    )
    input_manifest = authenticate_kinematic_input_manifest(
        input_path,
        expected_file_sha256=kinematic_input_manifest_file_sha256,
        identity=preparation_identity,
        contract=PRODUCTION_CONTRACT,
        authenticate_all_referenced_rows=False,
    )
    _require(
        Path(str(input_manifest["synthetic_pass"]["path"])).resolve()
        == synthetic_path
        and str(input_manifest["synthetic_pass"]["file_sha256"])
        == synthetic_pass_file_sha256,
        "input manifest binds a different synthetic pass",
    )
    population = _authenticate_population_envelope_without_sidecar_member_open(
        population_path,
        expected_file_sha256=sidecar_population_manifest_file_sha256,
        sidecar_root=root,
        input_manifest=input_manifest,
        input_manifest_path=input_path,
        input_manifest_file_sha256=kinematic_input_manifest_file_sha256,
        synthetic_pass_path=synthetic_path,
        synthetic_pass_file_sha256=synthetic_pass_file_sha256,
        identity=preparation_identity,
    )
    descriptor_contracts = composite_descriptor_contracts(
        preparation_identity,
        contract=PRODUCTION_CONTRACT,
    )
    expected_descriptor_ids = {
        name: str(descriptor_contracts[name]["descriptor_id"])
        for name in REPRESENTATIONS
    }
    descriptor_ids = _ordered_composite_descriptor_ids(
        population["composite_descriptor_ids"],
        expected_descriptor_ids,
    )
    _require(
        int(population["sidecar_count"]) == 32
        and len(population["rows"]) == 32,
        "sidecar population is incomplete",
    )
    return replace(
        plan,
        source_identity=identity,
        kinematic_input_manifest_path=input_path,
        kinematic_input_manifest_file_sha256=kinematic_input_manifest_file_sha256,
        kinematic_input_manifest_content_sha256=str(input_manifest["content_sha256"]),
        synthetic_pass_path=synthetic_path,
        synthetic_pass_file_sha256=synthetic_pass_file_sha256,
        sidecar_root=root,
        sidecar_population_manifest_path=population_path,
        sidecar_population_manifest_file_sha256=sidecar_population_manifest_file_sha256,
        sidecar_population_manifest_content_sha256=str(population["content_sha256"]),
        sidecar_population=population,
        composite_descriptor_ids=descriptor_ids,
    )


def _require_early_evidence_bound(plan: Plan) -> None:
    _require(plan.source_identity is not None, "clean Early source identity is unbound")
    _require(plan.kinematic_input_manifest_path is not None, "kinematic input manifest is unbound")
    _require(plan.sidecar_root is not None, "sidecar root is unbound")
    _require(plan.sidecar_population_manifest_path is not None, "sidecar population is unbound")
    _require(plan.sidecar_population is not None, "sidecar population payload is unbound")
    _require(
        plan.sidecar_population.get("git_commit") == PREPARATION_ARTIFACT_GIT_COMMIT,
        "preparation artifact producer commit is unbound",
    )
    _require(tuple(plan.composite_descriptor_ids) == REPRESENTATIONS, "descriptor IDs are unbound")


def _population_row(plan: Plan, row: CacheRow) -> Mapping[str, Any]:
    _require_early_evidence_bound(plan)
    assert plan.sidecar_population is not None
    matches = [
        value
        for value in plan.sidecar_population["rows"]
        if value["dataset"] == row.dataset
        and int(value["source_ordinal"]) == row.source_ordinal
        and int(value["source_index"]) == row.source_index
        and value["physical_family"] == row.family
    ]
    _require(len(matches) == 1, "sidecar population row does not resolve uniquely")
    return matches[0]


@dataclass(frozen=True)
class _OpenFileIdentity:
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    device: int
    inode: int
    mode: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_OpenFileIdentity":
        return cls(
            size_bytes=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            ctime_ns=int(value.st_ctime_ns),
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=int(value.st_mode),
        )


@dataclass(frozen=True)
class _AuthenticatedOpenFile:
    stream: BinaryIO
    size_bytes: int
    sha256: str


@contextmanager
def _authenticated_open_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> Iterator[_AuthenticatedOpenFile]:
    """Hash and consume one no-follow descriptor bound to its path inode."""

    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    path_before = _OpenFileIdentity.from_stat(source.stat(follow_symlinks=False))
    descriptor = os.open(os.fspath(source), flags)
    stream = os.fdopen(descriptor, "rb")
    try:
        descriptor_before = _OpenFileIdentity.from_stat(os.fstat(stream.fileno()))
        _require(
            descriptor_before == path_before,
            f"parent cache path changed before descriptor open: {source}",
        )
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            byte_count += len(block)
        descriptor_after_hash = _OpenFileIdentity.from_stat(os.fstat(stream.fileno()))
        _require(
            descriptor_after_hash == descriptor_before,
            f"parent cache descriptor changed while hashing: {source}",
        )
        actual_sha256 = digest.hexdigest()
        _require(
            byte_count == descriptor_before.size_bytes
            and descriptor_before.size_bytes == int(expected_size)
            and _lower_hex(expected_sha256)
            and actual_sha256 == expected_sha256,
            f"parent cache size or SHA-256 mismatch: {source}",
        )
        stream.seek(0)
        yield _AuthenticatedOpenFile(
            stream=stream,
            size_bytes=descriptor_before.size_bytes,
            sha256=actual_sha256,
        )
        descriptor_after_read = _OpenFileIdentity.from_stat(os.fstat(stream.fileno()))
        path_after = _OpenFileIdentity.from_stat(source.stat(follow_symlinks=False))
        _require(
            descriptor_after_read == descriptor_before and path_after == path_before,
            f"parent cache path or descriptor changed while loading: {source}",
        )
    finally:
        stream.close()


def _read_authenticated_bytes(path: Path, *, expected_sha256: str) -> bytes:
    """Return bytes read from the same FD that authenticated the expected SHA."""

    source = Path(path)
    expected_size = int(source.stat(follow_symlinks=False).st_size)
    with _authenticated_open_file(
        source,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
    ) as opened:
        content = opened.stream.read()
        _require(
            len(content) == opened.size_bytes,
            f"authenticated evidence byte count changed: {source}",
        )
    return content


def load_parent_cache_projection(
    plan: Plan,
    row: CacheRow,
    *,
    include_labels: bool,
) -> CacheProjection:
    """Load only the authorized parent members from one authenticated FD."""

    names = [
        "fmt_features",
        "valid_scale_id",
        "valid_center_seed_index",
        "valid_scale_block_index",
        "valid_assigned_row_index",
    ]
    if include_labels:
        names.append("valid_labels")
    with _authenticated_open_file(
        row.path,
        expected_size=row.size_bytes,
        expected_sha256=row.sha256,
    ) as opened:
        with np.load(opened.stream, allow_pickle=False) as archive:
            _require(
                all(name in archive.files for name in names),
                f"{row.path}: parent projection is incomplete",
            )
            arrays = {
                name: np.array(archive[name], copy=True, order="C")
                for name in names
            }
            if include_labels:
                _require(
                    "metadata_json" in archive.files,
                    f"{row.path}: metadata_json is missing",
                )
                metadata_scalar = np.array(
                    archive["metadata_json"], copy=True, order="C"
                )
                _require(
                    metadata_scalar.shape == (),
                    f"{row.path}: metadata_json is not scalar",
                )
                metadata: Mapping[str, Any] = json.loads(
                    str(metadata_scalar.item())
                )
            else:
                metadata = {}
    if include_labels:
        _validate_parent_cache_arrays(plan, row, metadata, arrays)
    else:
        _validate_parent_label_free_projection(row, arrays)
    return CacheProjection(
        row=row,
        fmt_features=np.ascontiguousarray(arrays["fmt_features"]),
        scale_ids=np.ascontiguousarray(arrays["valid_scale_id"]),
        center_indices=np.ascontiguousarray(arrays["valid_center_seed_index"]),
        block_indices=np.ascontiguousarray(arrays["valid_scale_block_index"]),
        assigned_row_indices=np.ascontiguousarray(arrays["valid_assigned_row_index"]),
        labels=(
            None
            if not include_labels
            else np.ascontiguousarray(arrays["valid_labels"])
        ),
        metadata=metadata,
    )


def load_early_cache_projection(
    plan: Plan,
    row: CacheRow,
    *,
    include_labels: bool,
) -> EarlyCacheProjection:
    """Open the parent and sidecar only at the caller's authorized gate phase."""

    _require_early_evidence_bound(plan)
    parent = load_parent_cache_projection(plan, row, include_labels=include_labels)
    population_row = _population_row(plan, row)
    assert plan.sidecar_root is not None
    sidecar_path = (plan.sidecar_root / str(population_row["sidecar_relative_path"])).resolve()
    _require(sidecar_path.is_relative_to(plan.sidecar_root), "sidecar path escapes root")
    loaded = load_seed_time_kinematic_sidecar(
        sidecar_path,
        expected_file_sha256=str(population_row["sidecar_file_sha256"]),
    )
    payload = loaded.payload
    identities = (
        (payload.valid_assigned_row_index, parent.assigned_row_indices, "assigned row"),
        (payload.valid_center_seed_index, parent.center_indices, "center seed"),
        (payload.valid_scale_block_index, parent.block_indices, "scale block"),
        (payload.valid_scale_id, parent.scale_ids, "scale ID"),
    )
    for sidecar_values, parent_values, name in identities:
        _require(np.array_equal(sidecar_values, parent_values), f"sidecar/parent {name} join drifted")
    seed4 = np.asarray(payload.seed_kinematic4)
    _require(
        seed4.dtype == np.dtype(np.float32)
        and seed4.shape == (parent.count, 4)
        and np.isfinite(seed4).all(),
        "sidecar seed_kinematic4 contract drifted",
    )
    _require(
        loaded.metadata["combined_array_sha256"]
        == population_row["sidecar_combined_array_sha256"],
        "sidecar combined array identity drifted",
    )
    return EarlyCacheProjection(
        row=row,
        fmt_features=parent.fmt_features,
        seed_kinematic4=np.ascontiguousarray(seed4),
        scale_ids=parent.scale_ids,
        center_indices=parent.center_indices,
        block_indices=parent.block_indices,
        assigned_row_indices=parent.assigned_row_indices,
        labels=parent.labels,
        metadata=parent.metadata,
        sidecar_file_sha256=loaded.file_sha256,
        sidecar_combined_array_sha256=str(loaded.metadata["combined_array_sha256"]),
    )


def composite_representation_features(
    cache: EarlyCacheProjection,
    representation: str,
) -> np.ndarray:
    _require(representation in PARENT_REPRESENTATION, "unknown composite representation")
    parent_name = PARENT_REPRESENTATION[representation]
    parent_features = representation_features(cache.fmt_features, parent_name)
    result = np.ascontiguousarray(
        np.concatenate((parent_features, cache.seed_kinematic4), axis=1),
        dtype=np.float32,
    )
    _require(
        result.shape == (cache.count, COMPOSITE_WIDTH[representation])
        and np.isfinite(result).all(),
        "composite feature contract drifted",
    )
    return result


def _early_artifact_binding(
    plan: Plan,
    *,
    representation: str,
    fit_families: Sequence[str],
) -> dict[str, Any]:
    _require_early_evidence_bound(plan)
    _require(representation in plan.composite_descriptor_ids, "descriptor ID is missing")
    assert plan.kinematic_input_manifest_path is not None
    assert plan.sidecar_population_manifest_path is not None
    assert plan.sidecar_population is not None
    producer_commit = str(plan.sidecar_population["git_commit"])
    return {
        "kinematic_input_manifest": {
            "path": str(plan.kinematic_input_manifest_path),
            "file_sha256": plan.kinematic_input_manifest_file_sha256,
            "content_sha256": plan.kinematic_input_manifest_content_sha256,
            "producer_git_commit": producer_commit,
        },
        "sidecar_population_manifest": {
            "path": str(plan.sidecar_population_manifest_path),
            "file_sha256": plan.sidecar_population_manifest_file_sha256,
            "content_sha256": plan.sidecar_population_manifest_content_sha256,
            "sidecar_count": 32,
            "producer_git_commit": producer_commit,
        },
        "composite_descriptor_id": plan.composite_descriptor_ids[representation],
        "representation": representation,
        "fit_families": list(fit_families),
        "config_sha256": plan.sha256,
        "clean_git_commit": plan.source_identity.git_commit if plan.source_identity else None,
    }


def _require_early_artifact_binding(
    manifest: Mapping[str, Any],
    plan: Plan,
    *,
    representation: str,
    fit_families: Sequence[str],
    label: str,
) -> None:
    _require(
        manifest.get("early_evidence")
        == _early_artifact_binding(
            plan,
            representation=representation,
            fit_families=fit_families,
        ),
        f"{label} Early evidence binding drifted",
    )

def candidate_specs(plan: Plan) -> tuple[TailCandidateSpec, ...]:
    candidates: list[TailCandidateSpec] = []
    for representation in plan.representations:
        for k in plan.ks:
            for sigma in plan.sigmas:
                candidates.append(TailCandidateSpec(representation, k, sigma, "fixed_top_fraction", plan.fixed_top_fraction))
                candidates.extend(
                    TailCandidateSpec(representation, k, sigma, "calibrated_tail_anomaly_threshold", threshold)
                    for threshold in plan.thresholds
                )
    _require(len(candidates) == FROZEN_CANDIDATE_COUNT, "candidate enumeration drifted")
    _require(len({candidate.candidate_id for candidate in candidates}) == len(candidates), "duplicate candidate ID")
    _require(all("rank" not in candidate.candidate_id for candidate in candidates), "rank wording leaked into candidate IDs")
    return tuple(candidates)


def _gaussian_kernel1d(sigma: float, truncate: float) -> np.ndarray:
    radius = int(truncate * sigma + 0.5)
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (axis / sigma) ** 2)
    return kernel / kernel.sum(dtype=np.float64)


def _constant_zero_convolve_axis(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="constant", constant_values=0.0)
    result = np.zeros_like(values, dtype=np.float64)
    base = [slice(None)] * values.ndim
    for offset, weight in enumerate(kernel):
        selected = list(base)
        selected[axis] = slice(offset, offset + values.shape[axis])
        result += float(weight) * padded[tuple(selected)]
    return result


def _require_portable_spatial_replay(
    field: str,
    stored_values: object,
    replayed_values: object,
) -> int:
    """Authenticate a spatial float64 field within its frozen cross-CPU ULP bound.

    Jobs 51064502 and 51064646 were completed without opening outer labels or
    metrics.  Across Intel Xeon Gold 6248, AMD EPYC 7702, and AMD EPYC 9655,
    only these two Gaussian-derived fields differed: at most six ULP for score
    and five ULP for denominator.  The bound of eight ULP is frozen here; all
    identities, raw distances, tail values, states, and predictions remain
    bitwise authenticated elsewhere.
    """

    _require(field in SPATIAL_REPLAY_ULP_BOUNDS, f"no portability bound is frozen for {field}")
    stored = np.asarray(stored_values)
    replayed = np.asarray(replayed_values)
    _require(stored.dtype == np.dtype(np.float64), f"stored {field} dtype is not float64")
    _require(replayed.dtype == np.dtype(np.float64), f"replayed {field} dtype is not float64")
    _require(stored.shape == replayed.shape, f"{field} replay shape drifted")
    _require(np.isfinite(stored).all() and np.isfinite(replayed).all(), f"{field} replay is nonfinite")
    _require(np.all(stored >= 0.0) and np.all(replayed >= 0.0), f"{field} replay is negative")
    _require(
        np.array_equal(stored == 0.0, replayed == 0.0),
        f"{field} replay zero mask drifted",
    )
    stored_bits = np.ascontiguousarray(stored).view(np.uint64)
    replayed_bits = np.ascontiguousarray(replayed).view(np.uint64)
    stored_greater = stored_bits >= replayed_bits
    ulp_distance = np.empty_like(stored_bits)
    np.subtract(stored_bits, replayed_bits, out=ulp_distance, where=stored_greater)
    np.subtract(replayed_bits, stored_bits, out=ulp_distance, where=~stored_greater)
    maximum = int(ulp_distance.max(initial=np.uint64(0)))
    bound = SPATIAL_REPLAY_ULP_BOUNDS[field]
    _require(maximum <= bound, f"{field} replay exceeds frozen {bound}-ULP portability bound: {maximum}")
    return maximum


def spatial_calibrated_tail_scores(
    anomaly_scores: object,
    calibration_supported: object,
    center_indices: object,
    *,
    sigma: float,
    grid_shape: Sequence[int] = GRID_SHAPE,
    truncate: float = 3.0,
) -> SpatialTailScores:
    """Mask-normalized Gaussian of absolute calibrated anomaly scores.

    This helper intentionally contains no rank calculation.  Unsupported rows
    are imputed only when a positive-sigma support-mask denominator is positive.
    """

    anomaly = np.asarray(anomaly_scores, dtype=np.float64)
    support = np.asarray(calibration_supported)
    centers = np.asarray(center_indices)
    shape = tuple(int(value) for value in grid_shape)
    if anomaly.ndim != 1 or support.shape != anomaly.shape or centers.shape != anomaly.shape:
        raise ValueError("anomaly, support, and centers must be matching one-dimensional arrays")
    if support.dtype != np.dtype(np.bool_):
        raise ValueError("calibration_supported must be boolean")
    if centers.dtype.kind not in "iu":
        raise ValueError("center_indices must be integers")
    if len(shape) != 3 or any(value < 1 for value in shape):
        raise ValueError("grid_shape must contain three positive dimensions")
    if not np.isfinite(anomaly).all() or np.any((anomaly < 0.0) | (anomaly > 1.0)):
        raise ValueError("anomaly scores must be finite and within [0, 1]")
    if not np.isfinite(float(sigma)) or sigma < 0.0:
        raise ValueError("sigma must be finite and nonnegative")
    if not np.isfinite(float(truncate)) or truncate <= 0.0:
        raise ValueError("truncate must be finite and positive")
    cell_count = int(np.prod(shape, dtype=np.int64))
    if np.any((centers < 0) | (centers >= cell_count)) or len(np.unique(centers)) != len(centers):
        raise ValueError("center indices must be unique valid flat C-order grid indices")
    scores = np.zeros(len(anomaly), dtype=np.float64)
    denominator = support.astype(np.float64)
    if sigma == 0.0:
        scores[support] = anomaly[support]
        imputed = np.zeros(len(anomaly), dtype=bool)
        unimputable = ~support
    else:
        numerator_grid = np.zeros(cell_count, dtype=np.float64)
        mask_grid = np.zeros(cell_count, dtype=np.float64)
        numerator_grid[centers] = anomaly * support
        mask_grid[centers] = support.astype(np.float64)
        numerator_grid = numerator_grid.reshape(shape)
        mask_grid = mask_grid.reshape(shape)
        kernel = _gaussian_kernel1d(float(sigma), float(truncate))
        for axis in range(3):
            numerator_grid = _constant_zero_convolve_axis(numerator_grid, kernel, axis)
            mask_grid = _constant_zero_convolve_axis(mask_grid, kernel, axis)
        numerator = numerator_grid.reshape(-1)[centers]
        denominator = mask_grid.reshape(-1)[centers]
        positive = denominator > 0.0
        scores[positive] = numerator[positive] / denominator[positive]
        imputed = (~support) & positive
        unimputable = (~support) & ~positive
    if not np.isfinite(scores).all() or not np.isfinite(denominator).all():
        raise RuntimeError("spatial calibrated-tail transform produced nonfinite values")
    if np.any((scores < -1e-14) | (scores > 1.0 + 1e-14)):
        raise RuntimeError("spatial calibrated-tail score left [0, 1]")
    return SpatialTailScores(
        scores=np.clip(scores, 0.0, 1.0),
        denominator=denominator,
        calibration_supported=support.copy(),
        imputed=imputed,
        unimputable=unimputable,
    )


def _candidate_payload(candidate: TailCandidateSpec) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "representation": candidate.representation,
        "k": candidate.k,
        "sigma": candidate.sigma,
        "decision_rule": candidate.decision_rule,
        "decision_value": candidate.decision_value,
    }


def candidate_predictions(
    candidate: TailCandidateSpec,
    scores: object,
    center_indices: object,
    eligible: object,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    centers = np.asarray(center_indices)
    allowed = np.asarray(eligible)
    _require(values.ndim == 1 and centers.shape == values.shape and allowed.shape == values.shape, "invalid candidate population")
    _require(centers.dtype.kind in "iu" and allowed.dtype == np.dtype(np.bool_), "invalid candidate identity/support dtype")
    _require(np.isfinite(values).all(), "candidate scores must be finite")
    if candidate.decision_rule == "fixed_top_fraction":
        return fixed_top_fraction_predictions(
            values,
            centers,
            allowed & (values > 0.0),
            fraction=candidate.decision_value,
        )
    if candidate.decision_rule == "calibrated_tail_anomaly_threshold":
        return np.asarray(allowed & (values >= candidate.decision_value), dtype=bool)
    raise ValueError(f"unsupported decision rule: {candidate.decision_rule}")


def _fit_tail_model(
    caches: Sequence[EarlyCacheProjection],
    representation: str,
    plan: Plan,
    *,
    device: str,
    ks: Sequence[int] | None = None,
) -> PerScaleNegativeTailModel:
    feature_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    for cache in caches:
        _require(cache.labels is not None, "fit cache labels are unavailable")
        negative = ~np.asarray(cache.labels, dtype=bool)
        if negative.any():
            represented = composite_representation_features(cache, representation)
            feature_parts.append(np.ascontiguousarray(represented[negative], dtype=np.float32))
            scale_parts.append(np.ascontiguousarray(cache.scale_ids[negative], dtype=np.int64))
    _require(feature_parts, "fit families contain no natural negatives")
    features = np.ascontiguousarray(np.concatenate(feature_parts, axis=0), dtype=np.float32)
    scales = np.ascontiguousarray(np.concatenate(scale_parts), dtype=np.int64)
    model = PerScaleNegativeTailModel(
        features,
        scales,
        ks=plan.ks if ks is None else tuple(int(value) for value in ks),
        shrinkage_lambda=plan.shrinkage_lambda,
        device=device,
        query_chunk_size=plan.query_chunk_size,
        library_chunk_size=plan.library_chunk_size,
    )
    del features, scales, feature_parts, scale_parts
    return model


def _query_cache_batch(
    model: PerScaleNegativeTailModel,
    caches: Sequence[EarlyCacheProjection],
    representation: str,
    plan: Plan,
    *,
    device: str,
    ks: Sequence[int] | None = None,
) -> dict[int, list[dict[str, np.ndarray]]]:
    requested = model.ks if ks is None else tuple(int(value) for value in ks)
    offsets = np.cumsum([0, *(cache.count for cache in caches)], dtype=np.int64)
    if caches:
        features = np.ascontiguousarray(
            np.concatenate(
                [composite_representation_features(cache, representation) for cache in caches],
                axis=0,
            ),
            dtype=np.float32,
        )
        scales = np.ascontiguousarray(
            np.concatenate([cache.scale_ids for cache in caches]), dtype=np.int64
        )
    else:
        width = COMPOSITE_WIDTH[representation]
        features = np.empty((0, width), dtype=np.float32)
        scales = np.empty(0, dtype=np.int64)
    result = model.query(
        features,
        scales,
        ks=requested,
        device=device,
        query_chunk_size=plan.query_chunk_size,
        library_chunk_size=plan.library_chunk_size,
    )
    query_scaler_modes = model.scaler.mode_for_scales(scales)
    del features, scales
    output: dict[int, list[dict[str, np.ndarray]]] = {}
    for k in requested:
        parts: list[dict[str, np.ndarray]] = []
        for index in range(len(caches)):
            selected = slice(int(offsets[index]), int(offsets[index + 1]))
            parts.append(
                {
                    "raw_distance": result.raw_distances[k][selected],
                    "tail_probability": result.tail_probabilities[k][selected],
                    "tail_anomaly": result.anomaly_scores[k][selected],
                    "retrieval_supported": result.retrieval_supported[k][selected],
                    "calibration_supported": result.calibration_supported[k][selected],
                    "calibration_mode": result.calibration_modes[k][selected],
                    "scaler_mode": query_scaler_modes[selected],
                }
            )
        output[k] = parts
    return output


def _subset_f1(labels: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return float("nan")
    return float(_classification_counts(labels[selected], predictions[selected])["f1"])


def _metric_row(
    *,
    outer_family: str,
    inner_family: str,
    cache: EarlyCacheProjection | CacheProjection,
    block_name: str,
    candidate: TailCandidateSpec,
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    retrieval_supported: np.ndarray,
    calibration_supported: np.ndarray,
    imputed: np.ndarray,
    unimputable: np.ndarray,
    calibration_modes: np.ndarray,
    scaler_modes: np.ndarray,
    ranking_metrics: tuple[float, float] | None = None,
) -> dict[str, Any]:
    retrieval = np.asarray(retrieval_supported, dtype=bool)
    calibration = np.asarray(calibration_supported, dtype=bool)
    imputed_values = np.asarray(imputed, dtype=bool)
    unimputable_values = np.asarray(unimputable, dtype=bool)
    all_rows = np.ones(len(labels), dtype=bool)
    _require(np.array_equal(calibration | imputed_values | unimputable_values, all_rows), "calibration spatial states do not cover group")
    _require(not np.any((calibration & imputed_values) | (calibration & unimputable_values) | (imputed_values & unimputable_values)), "calibration spatial states overlap")
    _require(not np.any(calibration & ~retrieval), "calibration support exceeds retrieval support")
    counts = _classification_counts(labels, predictions)
    if ranking_metrics is not None:
        average_precision, auroc = ranking_metrics
    elif np.asarray(labels, dtype=bool).any() and (~np.asarray(labels, dtype=bool)).any():
        average_precision, auroc = _ranking_metrics_one_sort(labels, scores)
    else:
        average_precision = auroc = float("nan")
    row: dict[str, Any] = {
        "outer_family": outer_family,
        "inner_family": inner_family,
        "dataset": cache.row.dataset,
        "source_ordinal": cache.row.source_ordinal,
        "block": block_name,
        **_candidate_payload(candidate),
        **counts,
        "average_precision": average_precision,
        "auroc": auroc,
        "retrieval_supported_count": int(retrieval.sum()),
        "calibration_supported_count": int(calibration.sum()),
        "imputed_count": int(imputed_values.sum()),
        "unimputable_count": int(unimputable_values.sum()),
        "retrieval_support_fraction": float(retrieval.mean()),
        "calibration_support_fraction": float(calibration.mean()),
        "spatial_imputed_fraction": float(imputed_values.mean()),
        "spatial_unimputable_fraction": float(unimputable_values.mean()),
        "retrieval_supported_subset_f1": _subset_f1(labels, predictions, retrieval),
        "calibration_supported_subset_f1": _subset_f1(labels, predictions, calibration),
        "imputed_subset_f1": _subset_f1(labels, predictions, imputed_values),
        "unimputable_subset_f1": _subset_f1(labels, predictions, unimputable_values),
    }
    for mode in range(6):
        row[f"calibration_mode_{mode}_count"] = int(np.count_nonzero(calibration_modes == mode))
    for mode in range(4):
        row[f"scaler_mode_{mode}_count"] = int(np.count_nonzero(scaler_modes == mode))
    return row


def _threshold_metric_rows(
    *,
    outer_family: str,
    inner_family: str,
    cache: EarlyCacheProjection | CacheProjection,
    block_name: str,
    candidates: Sequence[TailCandidateSpec],
    labels: np.ndarray,
    scores: np.ndarray,
    eligible: np.ndarray,
    retrieval_supported: np.ndarray,
    calibration_supported: np.ndarray,
    imputed: np.ndarray,
    unimputable: np.ndarray,
    calibration_modes: np.ndarray,
    scaler_modes: np.ndarray,
    ranking_metrics: tuple[float, float],
) -> list[dict[str, Any]]:
    """Evaluate all 50 frozen thresholds from sorted confusion series."""

    thresholds = [candidate.decision_value for candidate in candidates]
    _require(
        all(candidate.decision_rule == "calibrated_tail_anomaly_threshold" for candidate in candidates),
        "threshold batch contains a non-tail-threshold candidate",
    )
    whole = _threshold_confusion_series(labels, scores, eligible, thresholds)
    subset_masks = {
        "retrieval_supported_subset_f1": np.asarray(retrieval_supported, dtype=bool),
        "calibration_supported_subset_f1": np.asarray(calibration_supported, dtype=bool),
        "imputed_subset_f1": np.asarray(imputed, dtype=bool),
        "unimputable_subset_f1": np.asarray(unimputable, dtype=bool),
    }
    subset_series: dict[str, list[dict[str, int | float]] | None] = {}
    for name, mask in subset_masks.items():
        subset_series[name] = (
            _threshold_confusion_series(
                np.asarray(labels)[mask],
                np.asarray(scores)[mask],
                np.asarray(eligible)[mask],
                thresholds,
            )
            if mask.any()
            else None
        )
    retrieval = np.asarray(retrieval_supported, dtype=bool)
    calibration = np.asarray(calibration_supported, dtype=bool)
    imputed_values = np.asarray(imputed, dtype=bool)
    unimputable_values = np.asarray(unimputable, dtype=bool)
    modes = np.asarray(calibration_modes, dtype=np.int8)
    metric_modes = np.asarray(scaler_modes, dtype=np.int8)
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        row: dict[str, Any] = {
            "outer_family": outer_family,
            "inner_family": inner_family,
            "dataset": cache.row.dataset,
            "source_ordinal": cache.row.source_ordinal,
            "block": block_name,
            **_candidate_payload(candidate),
            **whole[index],
            "average_precision": ranking_metrics[0],
            "auroc": ranking_metrics[1],
            "retrieval_supported_count": int(retrieval.sum()),
            "calibration_supported_count": int(calibration.sum()),
            "imputed_count": int(imputed_values.sum()),
            "unimputable_count": int(unimputable_values.sum()),
            "retrieval_support_fraction": float(retrieval.mean()),
            "calibration_support_fraction": float(calibration.mean()),
            "spatial_imputed_fraction": float(imputed_values.mean()),
            "spatial_unimputable_fraction": float(unimputable_values.mean()),
        }
        for name, series in subset_series.items():
            row[name] = float("nan") if series is None else float(series[index]["f1"])
        for mode in range(6):
            row[f"calibration_mode_{mode}_count"] = int(np.count_nonzero(modes == mode))
        for mode in range(4):
            row[f"scaler_mode_{mode}_count"] = int(np.count_nonzero(metric_modes == mode))
        rows.append(row)
    return rows


def _inner_metric_rows(
    plan: Plan,
    caches: Sequence[EarlyCacheProjection],
    outer_family: str,
    *,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, TailCandidateSpec], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    candidates: dict[str, TailCandidateSpec] = {}
    audits: list[dict[str, Any]] = []
    nonouter = [family for family in plan.family_order if family != outer_family]
    for inner_family in nonouter:
        fit_families = [family for family in nonouter if family != inner_family]
        fit_caches = [cache for cache in caches if cache.row.family in fit_families]
        query_caches = [cache for cache in caches if cache.row.family == inner_family]
        _require(fit_caches and query_caches, f"{inner_family}: empty nested fold")
        for representation in plan.representations:
            print(f"[{_utc_now()}] outer={outer_family} inner={inner_family} representation={representation} fit_start", flush=True)
            model = _fit_tail_model(fit_caches, representation, plan, device=device)
            audit = dict(model.fit_audit)
            audit.update(
                {
                    "outer_family": outer_family,
                    "inner_family": inner_family,
                    "fit_families": fit_families,
                    "representation": representation,
                    "composite_descriptor_id": plan.composite_descriptor_ids[
                        representation
                    ],
                    "kinematic_input_manifest_file_sha256": (
                        plan.kinematic_input_manifest_file_sha256
                    ),
                    "sidecar_population_manifest_file_sha256": (
                        plan.sidecar_population_manifest_file_sha256
                    ),
                    "device": device,
                    "query_chunk_size": plan.query_chunk_size,
                    "library_chunk_size": plan.library_chunk_size,
                }
            )
            audits.append(audit)
            query = _query_cache_batch(model, query_caches, representation, plan, device=device)
            for cache_index, cache in enumerate(query_caches):
                _require(cache.labels is not None, "inner query labels are unavailable")
                for block_index, block_name in enumerate(BLOCK_NAMES):
                    block = np.asarray(cache.block_indices == block_index)
                    _require(block.any(), f"{cache.row.dataset}/{cache.row.source_ordinal}/{block_name}: empty group")
                    labels = np.asarray(cache.labels[block], dtype=bool)
                    centers = np.asarray(cache.center_indices[block], dtype=np.int64)
                    _require(labels.any() and (~labels).any(), f"{cache.row.dataset}/{cache.row.source_ordinal}/{block_name}: single-class group")
                    for k in plan.ks:
                        values = query[k][cache_index]
                        for sigma in plan.sigmas:
                            spatial = spatial_calibrated_tail_scores(
                                values["tail_anomaly"][block],
                                values["calibration_supported"][block],
                                centers,
                                sigma=sigma,
                                grid_shape=plan.grid_shape,
                                truncate=plan.gaussian_truncate,
                            )
                            eligible = spatial.calibration_supported | spatial.imputed
                            base = (
                                TailCandidateSpec(representation, k, sigma, "fixed_top_fraction", plan.fixed_top_fraction),
                                *(TailCandidateSpec(representation, k, sigma, "calibrated_tail_anomaly_threshold", threshold) for threshold in plan.thresholds),
                            )
                            for candidate in base:
                                previous = candidates.setdefault(candidate.candidate_id, candidate)
                                _require(previous == candidate, "candidate ID collision")
                            ranking = _ranking_metrics_one_sort(labels, spatial.scores)
                            fixed_prediction = candidate_predictions(base[0], spatial.scores, centers, eligible)
                            rows.append(
                                _metric_row(
                                    outer_family=outer_family,
                                    inner_family=inner_family,
                                    cache=cache,
                                    block_name=block_name,
                                    candidate=base[0],
                                    labels=labels,
                                    scores=spatial.scores,
                                    predictions=fixed_prediction,
                                    retrieval_supported=values["retrieval_supported"][block],
                                    calibration_supported=spatial.calibration_supported,
                                    imputed=spatial.imputed,
                                    unimputable=spatial.unimputable,
                                    calibration_modes=values["calibration_mode"][block],
                                    scaler_modes=values["scaler_mode"][block],
                                    ranking_metrics=ranking,
                                )
                            )
                            rows.extend(
                                _threshold_metric_rows(
                                    outer_family=outer_family,
                                    inner_family=inner_family,
                                    cache=cache,
                                    block_name=block_name,
                                    candidates=base[1:],
                                    labels=labels,
                                    scores=spatial.scores,
                                    eligible=eligible,
                                    retrieval_supported=values["retrieval_supported"][block],
                                    calibration_supported=spatial.calibration_supported,
                                    imputed=spatial.imputed,
                                    unimputable=spatial.unimputable,
                                    calibration_modes=values["calibration_mode"][block],
                                    scaler_modes=values["scaler_mode"][block],
                                    ranking_metrics=ranking,
                                )
                            )
            del model, query
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    _require(len(candidates) == FROZEN_CANDIDATE_COUNT, "inner candidate count drifted")
    return rows, candidates, audits


def _hierarchical_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    families = sorted({str(row["inner_family"]) for row in rows})
    family_values = []
    for family in families:
        values = np.asarray([float(row[field]) for row in rows if row["inner_family"] == family], dtype=np.float64)
        finite = values[np.isfinite(values)]
        family_values.append(float(np.mean(finite)) if len(finite) else float("nan"))
    finite_families = np.asarray(family_values, dtype=np.float64)
    finite_families = finite_families[np.isfinite(finite_families)]
    return float(np.mean(finite_families)) if len(finite_families) else float("nan")


def _aggregate_and_select(
    rows: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, TailCandidateSpec],
) -> tuple[list[dict[str, Any]], TailCandidateSpec, dict[str, Any]]:
    _require(rows and len(candidates) == FROZEN_CANDIDATE_COUNT, "inner selection population is incomplete")
    grouped = {candidate_id: [] for candidate_id in candidates}
    for row in rows:
        candidate_id = str(row["candidate_id"])
        _require(candidate_id in grouped, "unknown candidate metric")
        grouped[candidate_id].append(row)
    expected_keys: set[tuple[str, str, int, str]] | None = None
    summaries: list[dict[str, Any]] = []
    for candidate_id in sorted(grouped):
        candidate_rows = grouped[candidate_id]
        keys = {(str(row["inner_family"]), str(row["dataset"]), int(row["source_ordinal"]), str(row["block"])) for row in candidate_rows}
        _require(len(keys) == len(candidate_rows), f"{candidate_id}: duplicate inner group")
        if expected_keys is None:
            expected_keys = keys
        _require(keys == expected_keys, f"{candidate_id}: incomplete inner group set")
        _require(len({key[0] for key in keys}) == 4, f"{candidate_id}: expected four inner families")
        summary = _candidate_payload(candidates[candidate_id])
        for field in (
            "accuracy", "average_precision", "f1", "balanced_accuracy", "auroc", "precision", "recall",
            "retrieval_support_fraction", "calibration_support_fraction", "spatial_imputed_fraction", "spatial_unimputable_fraction",
        ):
            summary[field] = _hierarchical_mean(candidate_rows, field)
        summary["inner_family_count"] = 4
        summary["group_count"] = len(candidate_rows)
        summaries.append(summary)
    def selection_key(summary: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            -float(summary["f1"]),
            -float(summary["average_precision"]),
            -float(summary["balanced_accuracy"]),
            -float(summary["precision"]),
            -float(summary["recall"]),
            str(summary["candidate_id"]),
        )
    selected_summary = min(summaries, key=selection_key)
    selected = candidates[str(selected_summary["candidate_id"])]
    return summaries, selected, dict(selected_summary)


def _array_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": np.asarray(values).dtype.str,
            "shape": list(np.asarray(values).shape),
            "sha256": canonical_array_sha256(np.asarray(values)),
        }
        for name, values in sorted(arrays.items())
    }


def _manifest_with_self_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(_json_safe(payload))
    _require("content_sha256" not in output, "manifest already contains a self hash")
    output["content_sha256"] = _json_safe_content_sha256(output)
    return output


def _authenticate_self_hash(manifest: Mapping[str, Any]) -> None:
    content = dict(manifest)
    stored = content.pop("content_sha256", None)
    _require(_lower_hex(stored), "manifest self hash is missing")
    _require(stored == _json_safe_content_sha256(content), "manifest self hash mismatch")


def write_final_scaler_artifact(
    output_directory: Path,
    model: PerScaleNegativeTailModel,
    *,
    plan: Plan,
    selected: TailCandidateSpec,
    outer_family: str,
    fit_families: Sequence[str],
    git_commit: str,
) -> tuple[Path, Path, str, str]:
    """Publish the complete 2000-scale state before any outer feature open."""

    arrays = model.scaler.export_arrays()
    _require(tuple(arrays) == SCALER_ARRAY_NAMES, "scaler export member order drifted")
    scaler_path = output_directory / "final_per_scale_scaler.npz"
    scaler_sha = _atomic_npz(scaler_path, arrays)
    parent = plan.raw.get("parent_identity")
    _require(isinstance(parent, Mapping), "parent identity is unavailable")
    manifest = _manifest_with_self_hash(
        {
            "schema": SCALER_MANIFEST_SCHEMA,
            "artifact_schema": SCALER_ARTIFACT_SCHEMA,
            "experiment": EXPERIMENT,
            "created_utc": _utc_now(),
            "config_path": str(plan.path),
            "config_sha256": plan.sha256,
            "input_manifest_sha256": plan.manifest_sha256,
            "parent_negative_tail_config_sha256": parent[
                "negative_tail_config_sha256"
            ],
            "git_commit": git_commit,
            "outer_family": outer_family,
            "fit_families": list(fit_families),
            "representation": selected.representation,
            "early_evidence": _early_artifact_binding(
                plan,
                representation=selected.representation,
                fit_families=fit_families,
            ),
            "selected_candidate": _candidate_payload(selected),
            "outer_feature_member_opened": False,
            "artifact_file": {
                "path": scaler_path.name,
                "size_bytes": scaler_path.stat().st_size,
                "sha256": scaler_sha,
            },
            "array_count": len(arrays),
            "arrays": _array_manifest(arrays),
            "fit_audit": model.scaler.fit_audit,
        }
    )
    manifest_path = output_directory / "final_per_scale_scaler_manifest.json"
    manifest_file_sha = _atomic_json(manifest_path, manifest)
    return scaler_path, manifest_path, scaler_sha, manifest_file_sha


def authenticate_and_rebuild_final_scaler(
    scaler_path: Path,
    manifest_path: Path,
    *,
    plan: Plan,
    selected: TailCandidateSpec,
    outer_family: str,
    fit_families: Sequence[str],
    git_commit: str,
    expected_manifest_file_sha256: str,
) -> VerifiedScalerArtifact:
    manifest_bytes = manifest_path.read_bytes()
    manifest_file_sha = hashlib.sha256(manifest_bytes).hexdigest()
    _require(
        _lower_hex(expected_manifest_file_sha256)
        and manifest_file_sha == expected_manifest_file_sha256,
        "scaler manifest file SHA-256 mismatch",
    )
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    _require(isinstance(manifest, Mapping), "scaler manifest root is invalid")
    _authenticate_self_hash(manifest)
    parent = plan.raw.get("parent_identity")
    _require(isinstance(parent, Mapping), "parent identity is unavailable")
    _require(manifest.get("schema") == SCALER_MANIFEST_SCHEMA, "scaler manifest schema drifted")
    _require(manifest.get("artifact_schema") == SCALER_ARTIFACT_SCHEMA, "scaler artifact schema drifted")
    _require(manifest.get("experiment") == EXPERIMENT, "scaler experiment drifted")
    _require(manifest.get("config_sha256") == plan.sha256, "scaler config drifted")
    _require(manifest.get("input_manifest_sha256") == plan.manifest_sha256, "scaler input manifest drifted")
    _require(manifest.get("parent_negative_tail_config_sha256") == parent.get("negative_tail_config_sha256"), "scaler parent calibration drifted")
    _require(manifest.get("git_commit") == git_commit, "scaler Git binding drifted")
    _require(manifest.get("outer_family") == outer_family, "scaler outer-family binding drifted")
    _require(manifest.get("fit_families") == list(fit_families), "scaler fit-family binding drifted")
    _require(manifest.get("representation") == selected.representation, "scaler representation drifted")
    _require_early_artifact_binding(
        manifest,
        plan,
        representation=selected.representation,
        fit_families=fit_families,
        label="scaler",
    )
    _require(manifest.get("selected_candidate") == _json_safe(_candidate_payload(selected)), "scaler candidate drifted")
    _require(manifest.get("outer_feature_member_opened") is False, "scaler was not closed before outer feature access")
    file_record = manifest.get("artifact_file")
    _require(isinstance(file_record, Mapping), "scaler file identity is missing")
    assert isinstance(file_record, Mapping)
    _require(file_record.get("path") == scaler_path.name, "scaler file path drifted")
    records = manifest.get("arrays")
    _require(isinstance(records, Mapping), "scaler array manifest is missing")
    assert isinstance(records, Mapping)
    _require(set(records) == set(SCALER_ARRAY_NAMES), "scaler manifest member set drifted")
    _require(int(manifest.get("array_count", -1)) == len(SCALER_ARRAY_NAMES), "scaler array count drifted")
    with _authenticated_open_file(
        scaler_path,
        expected_size=int(file_record["size_bytes"]),
        expected_sha256=str(file_record["sha256"]),
    ) as opened:
        with np.load(opened.stream, allow_pickle=False) as archive:
            _require(set(archive.files) == set(SCALER_ARRAY_NAMES), "scaler NPZ member set drifted")
            arrays = {
                name: np.array(archive[name], copy=True, order="C")
                for name in archive.files
            }
        scaler_file_sha256 = opened.sha256
    for name in SCALER_ARRAY_NAMES:
        values = arrays[name]
        record = records[name]
        _require(isinstance(record, Mapping), f"scaler array record is invalid: {name}")
        assert isinstance(record, Mapping)
        _require(values.dtype.str == record.get("dtype"), f"scaler dtype mismatch: {name}")
        _require(list(values.shape) == record.get("shape"), f"scaler shape mismatch: {name}")
        _require(canonical_array_sha256(values) == record.get("sha256"), f"scaler array hash mismatch: {name}")
    rebuilt = PerScaleNegativeScaler.from_arrays(arrays)
    _require(_json_safe(rebuilt.fit_audit) == manifest.get("fit_audit"), "rebuilt scaler audit drifted")
    return VerifiedScalerArtifact(
        manifest_path=manifest_path,
        manifest_file_sha256=manifest_file_sha,
        scaler_file_sha256=scaler_file_sha256,
        manifest=_deep_freeze(manifest),
        scaler=rebuilt,
        _authentication_seal=_AUTHENTICATION_SEAL,
    )


def write_final_calibration_artifact(
    output_directory: Path,
    model: PerScaleNegativeTailModel,
    *,
    plan: Plan,
    selected: TailCandidateSpec,
    scaler: VerifiedScalerArtifact,
    outer_family: str,
    fit_families: Sequence[str],
    git_commit: str,
) -> tuple[Path, Path, str, str]:
    """Write NPZ first and the authenticated manifest last."""

    _require(scaler._authentication_seal is _AUTHENTICATION_SEAL, "calibration write requires authenticated scaler")
    model_scaler_arrays = model.scaler.export_arrays()
    authenticated_scaler_arrays = scaler.scaler.export_arrays()
    _require(
        all(
            canonical_array_sha256(model_scaler_arrays[name])
            == canonical_array_sha256(authenticated_scaler_arrays[name])
            for name in SCALER_ARRAY_NAMES
        ),
        "calibration model does not use the authenticated scaler state",
    )
    arrays = model.tail_calibrator.export_arrays()
    calibration_path = output_directory / "final_tail_calibration.npz"
    calibration_sha = _atomic_npz(calibration_path, arrays)
    file_size = calibration_path.stat().st_size
    manifest_payload = {
        "schema": CALIBRATION_MANIFEST_SCHEMA,
        "artifact_schema": CALIBRATION_ARTIFACT_SCHEMA,
        "experiment": EXPERIMENT,
        "created_utc": _utc_now(),
        "config_path": str(plan.path),
        "config_sha256": plan.sha256,
        "git_commit": git_commit,
        "outer_family": outer_family,
        "fit_families": list(fit_families),
        "selected_candidate": _candidate_payload(selected),
        "early_evidence": _early_artifact_binding(
            plan,
            representation=selected.representation,
            fit_families=fit_families,
        ),
        "final_per_scale_scaler_manifest": {
            "path": scaler.manifest_path.name,
            "file_sha256": scaler.manifest_file_sha256,
            "content_sha256": scaler.manifest["content_sha256"],
        },
        "final_per_scale_scaler_file_sha256": scaler.scaler_file_sha256,
        "outer_feature_member_opened": False,
        "artifact_file": {
            "path": calibration_path.name,
            "size_bytes": int(file_size),
            "sha256": calibration_sha,
        },
        "array_count": len(arrays),
        "arrays": _array_manifest(arrays),
        "fit_audit": model.tail_calibrator.fit_audit,
    }
    manifest = _manifest_with_self_hash(manifest_payload)
    manifest_path = output_directory / "final_tail_calibration_manifest.json"
    manifest_file_sha = _atomic_json(manifest_path, manifest)
    return calibration_path, manifest_path, calibration_sha, manifest_file_sha


def authenticate_and_rebuild_final_calibration(
    calibration_path: Path,
    manifest_path: Path,
    *,
    plan: Plan,
    selected: TailCandidateSpec,
    scaler: VerifiedScalerArtifact,
    outer_family: str,
    fit_families: Sequence[str],
    git_commit: str,
    expected_manifest_file_sha256: str,
) -> VerifiedCalibrationArtifact:
    """Authenticate every byte/array before calling ``from_arrays``."""

    _require(scaler._authentication_seal is _AUTHENTICATION_SEAL, "calibration authentication requires authenticated scaler")
    manifest_bytes = manifest_path.read_bytes()
    manifest_file_sha = hashlib.sha256(manifest_bytes).hexdigest()
    _require(_lower_hex(expected_manifest_file_sha256), "expected calibration manifest SHA-256 is invalid")
    _require(manifest_file_sha == expected_manifest_file_sha256, "calibration manifest file SHA-256 mismatch")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    _require(isinstance(manifest, Mapping), "calibration manifest root is invalid")
    _authenticate_self_hash(manifest)
    _require(manifest.get("schema") == CALIBRATION_MANIFEST_SCHEMA, "calibration manifest schema drifted")
    _require(manifest.get("artifact_schema") == CALIBRATION_ARTIFACT_SCHEMA, "calibration artifact schema drifted")
    _require(manifest.get("experiment") == EXPERIMENT, "calibration experiment drifted")
    _require(manifest.get("config_sha256") == plan.sha256, "calibration config drifted")
    _require(manifest.get("outer_family") == outer_family, "calibration outer-family binding drifted")
    _require(manifest.get("fit_families") == list(fit_families), "calibration fit-family binding drifted")
    _require(manifest.get("git_commit") == git_commit, "calibration Git binding drifted")
    _require(manifest.get("selected_candidate") == _json_safe(_candidate_payload(selected)), "calibration candidate drifted")
    _require_early_artifact_binding(
        manifest,
        plan,
        representation=selected.representation,
        fit_families=fit_families,
        label="calibration",
    )
    scaler_record = manifest.get("final_per_scale_scaler_manifest")
    _require(isinstance(scaler_record, Mapping), "calibration scaler binding is missing")
    assert isinstance(scaler_record, Mapping)
    _require(scaler_record.get("path") == scaler.manifest_path.name, "calibration scaler path drifted")
    _require(scaler_record.get("file_sha256") == scaler.manifest_file_sha256, "calibration scaler manifest SHA drifted")
    _require(scaler_record.get("content_sha256") == scaler.manifest.get("content_sha256"), "calibration scaler content SHA drifted")
    _require(manifest.get("final_per_scale_scaler_file_sha256") == scaler.scaler_file_sha256, "calibration scaler file SHA drifted")
    _require(manifest.get("outer_feature_member_opened") is False, "calibration was not closed before outer feature access")
    file_record = manifest.get("artifact_file")
    _require(isinstance(file_record, Mapping), "calibration file identity is missing")
    assert isinstance(file_record, Mapping)
    _require(file_record.get("path") == calibration_path.name, "calibration file path drifted")
    arrays_record = manifest.get("arrays")
    _require(isinstance(arrays_record, Mapping), "calibration array manifest is missing")
    assert isinstance(arrays_record, Mapping)
    _require(int(manifest.get("array_count", -1)) == len(arrays_record), "calibration array count drifted")
    with _authenticated_open_file(
        calibration_path,
        expected_size=int(file_record["size_bytes"]),
        expected_sha256=str(file_record["sha256"]),
    ) as opened:
        with np.load(opened.stream, allow_pickle=False) as archive:
            _require(set(archive.files) == set(arrays_record), "calibration NPZ member set drifted")
            arrays = {
                name: np.array(archive[name], copy=True, order="C")
                for name in archive.files
            }
        calibration_file_sha256 = opened.sha256
    for name, values in arrays.items():
        record = arrays_record[name]
        _require(isinstance(record, Mapping), f"calibration array record is invalid: {name}")
        assert isinstance(record, Mapping)
        _require(values.dtype.str == record.get("dtype"), f"calibration dtype mismatch: {name}")
        _require(list(values.shape) == record.get("shape"), f"calibration shape mismatch: {name}")
        _require(canonical_array_sha256(values) == record.get("sha256"), f"calibration array hash mismatch: {name}")
    # Security boundary: reconstruction is deliberately after file, manifest,
    # exact-member, dtype, shape, and per-array hash authentication above.
    rebuilt = PerScaleNegativeTailModel.from_artifacts(
        scaler.scaler.export_arrays(), arrays
    )
    _require(rebuilt.ks == (selected.k,), "rebuilt final calibrator k drifted")
    _require(
        _json_safe(rebuilt.tail_calibrator.fit_audit) == manifest.get("fit_audit"),
        "rebuilt calibration fit audit drifted from authenticated manifest",
    )
    return VerifiedCalibrationArtifact(
        manifest_path=manifest_path,
        manifest_file_sha256=manifest_file_sha,
        calibration_file_sha256=calibration_file_sha256,
        manifest=_deep_freeze(manifest),
        model=rebuilt,
        _authentication_seal=_AUTHENTICATION_SEAL,
    )


def write_selected_candidate(
    output_directory: Path,
    *,
    plan: Plan,
    selected: TailCandidateSpec,
    selected_summary: Mapping[str, Any],
    scaler: VerifiedScalerArtifact,
    calibration: VerifiedCalibrationArtifact,
    inner_group_metrics_path: Path,
    inner_group_metrics_sha256: str,
    inner_candidate_summary_path: Path,
    inner_candidate_summary_sha256: str,
    inner_fit_audits_path: Path,
    inner_fit_audits_sha256: str,
    outer_family: str,
    git_commit: str,
) -> tuple[Path, str, Mapping[str, Any]]:
    _require(scaler._authentication_seal is _AUTHENTICATION_SEAL, "selected candidate requires authenticated scaler")
    _require(calibration._authentication_seal is _AUTHENTICATION_SEAL, "selected candidate requires authenticated calibration")
    evidence_inputs = (
        ("inner_group_metrics", inner_group_metrics_path, inner_group_metrics_sha256),
        ("inner_candidate_summary", inner_candidate_summary_path, inner_candidate_summary_sha256),
        ("inner_fit_audits", inner_fit_audits_path, inner_fit_audits_sha256),
    )
    evidence: dict[str, Any] = {}
    for name, evidence_path, expected_sha in evidence_inputs:
        identity = _stable_file_identity(evidence_path, evidence_path.stat().st_size, expected_sha)
        evidence[name] = {
            "path": evidence_path.name,
            "size_bytes": identity["size_bytes"],
            "sha256": identity["sha256"],
        }
    payload = _manifest_with_self_hash(
        {
            "schema": SELECTED_SCHEMA,
            "experiment": EXPERIMENT,
            "created_utc": _utc_now(),
            "config_sha256": plan.sha256,
            "git_commit": git_commit,
            "outer_family": outer_family,
            "candidate": _candidate_payload(selected),
            "candidate_count": FROZEN_CANDIDATE_COUNT,
            "early_evidence": _early_artifact_binding(
                plan,
                representation=selected.representation,
                fit_families=[
                    family for family in plan.family_order if family != outer_family
                ],
            ),
            "inner_selection_summary": selected_summary,
            "inner_evidence": evidence,
            "final_per_scale_scaler_manifest": {
                "path": scaler.manifest_path.name,
                "file_sha256": scaler.manifest_file_sha256,
                "content_sha256": scaler.manifest["content_sha256"],
            },
            "final_per_scale_scaler_file": {
                "path": "final_per_scale_scaler.npz",
                "sha256": scaler.scaler_file_sha256,
            },
            "final_calibration_manifest": {
                "path": calibration.manifest_path.name,
                "file_sha256": calibration.manifest_file_sha256,
                "content_sha256": calibration.manifest["content_sha256"],
            },
            "final_calibration_file": {
                "path": "final_tail_calibration.npz",
                "sha256": calibration.calibration_file_sha256,
            },
            "outer_feature_member_opened": False,
        }
    )
    path = output_directory / "selected_candidate.json"
    return path, _atomic_json(path, payload), payload


def _validate_candidate_csv_identity(
    row: Mapping[str, str], candidate: TailCandidateSpec
) -> None:
    _require(row.get("candidate_id") == candidate.candidate_id, "inner CSV candidate ID drifted")
    _require(row.get("representation") == candidate.representation, "inner CSV representation drifted")
    _require(int(row["k"]) == candidate.k, "inner CSV k drifted")
    _require(float(row["sigma"]) == candidate.sigma, "inner CSV sigma drifted")
    _require(row.get("decision_rule") == candidate.decision_rule, "inner CSV decision rule drifted")
    _require(float(row["decision_value"]) == candidate.decision_value, "inner CSV decision value drifted")


def _authenticate_inner_selection_evidence(
    *,
    plan: Plan,
    selected: TailCandidateSpec | None,
    outer_family: str,
    inner_group_metrics_path: Path,
    inner_group_metrics_sha256: str,
    inner_candidate_summary_path: Path,
    inner_candidate_summary_sha256: str,
    inner_fit_audits_path: Path,
    inner_fit_audits_sha256: str,
) -> dict[str, Any]:
    """Recompute candidate completeness and the frozen selection tie-break."""

    candidates = {candidate.candidate_id: candidate for candidate in candidate_specs(plan)}
    summary_bytes = _read_authenticated_bytes(
        inner_candidate_summary_path,
        expected_sha256=inner_candidate_summary_sha256,
    )
    group_bytes = _read_authenticated_bytes(
        inner_group_metrics_path,
        expected_sha256=inner_group_metrics_sha256,
    )
    audit_bytes = _read_authenticated_bytes(
        inner_fit_audits_path,
        expected_sha256=inner_fit_audits_sha256,
    )
    summary_rows: dict[str, dict[str, Any]] = {}
    with io.StringIO(summary_bytes.decode("utf-8"), newline="") as source:
        reader = csv.DictReader(source)
        _require(tuple(reader.fieldnames or ()) == SUMMARY_FIELDS, "inner candidate summary fields drifted")
        for raw in reader:
            candidate_id = str(raw["candidate_id"])
            _require(candidate_id in candidates and candidate_id not in summary_rows, "inner candidate summary candidate set drifted")
            _validate_candidate_csv_identity(raw, candidates[candidate_id])
            row = {
                field: float(raw[field])
                for field in (
                    "accuracy", "average_precision", "f1", "balanced_accuracy", "auroc",
                    "precision", "recall", "retrieval_support_fraction",
                    "calibration_support_fraction", "spatial_imputed_fraction",
                    "spatial_unimputable_fraction",
                )
            }
            _require(np.isfinite(list(row.values())).all(), "inner candidate summary contains nonfinite selection evidence")
            _require(int(raw["inner_family_count"]) == 4 and int(raw["group_count"]) > 0, "inner candidate summary coverage drifted")
            row["candidate_id"] = candidate_id
            row["group_count"] = int(raw["group_count"])
            summary_rows[candidate_id] = row
    _require(set(summary_rows) == set(candidates), "inner candidate summary is not the complete 3060-candidate set")

    aggregate_fields = (
        "accuracy", "average_precision", "f1", "balanced_accuracy", "auroc",
        "precision", "recall", "retrieval_support_fraction",
        "calibration_support_fraction", "spatial_imputed_fraction",
        "spatial_unimputable_fraction",
    )
    family_sums: dict[str, dict[str, dict[str, float]]] = {
        candidate_id: {} for candidate_id in candidates
    }
    family_counts: dict[str, dict[str, int]] = {
        candidate_id: {} for candidate_id in candidates
    }
    group_keys: dict[str, set[tuple[str, str, int, str]]] = {
        candidate_id: set() for candidate_id in candidates
    }
    with io.StringIO(group_bytes.decode("utf-8"), newline="") as source:
        reader = csv.DictReader(source)
        _require(tuple(reader.fieldnames or ()) == METRIC_FIELDS, "inner group metric fields drifted")
        for raw in reader:
            candidate_id = str(raw["candidate_id"])
            _require(candidate_id in candidates, "inner group metrics contain an unknown candidate")
            _validate_candidate_csv_identity(raw, candidates[candidate_id])
            _require(raw.get("outer_family") == outer_family, "inner group outer-family binding drifted")
            inner_family = str(raw["inner_family"])
            _require(inner_family in plan.family_order and inner_family != outer_family, "inner group family drifted")
            _require(str(raw["dataset"]) in plan.families[inner_family], "inner group dataset/family binding drifted")
            block = str(raw["block"])
            _require(block in BLOCK_NAMES, "inner group block name drifted")
            key = (inner_family, str(raw["dataset"]), int(raw["source_ordinal"]), block)
            _require(key not in group_keys[candidate_id], "duplicate inner candidate group")
            group_keys[candidate_id].add(key)
            values = np.asarray([float(raw[field]) for field in aggregate_fields], dtype=np.float64)
            _require(np.isfinite(values).all(), "inner group metrics contain nonfinite candidate evidence")
            sums = family_sums[candidate_id].setdefault(
                inner_family, {field: 0.0 for field in aggregate_fields}
            )
            for field, value in zip(aggregate_fields, values):
                sums[field] += float(value)
            family_counts[candidate_id][inner_family] = family_counts[candidate_id].get(inner_family, 0) + 1
    expected_groups: set[tuple[str, str, int, str]] | None = None
    for candidate_id in sorted(candidates):
        groups = group_keys[candidate_id]
        _require(groups and len({key[0] for key in groups}) == 4, "inner candidate lacks four-family coverage")
        if expected_groups is None:
            expected_groups = groups
        _require(groups == expected_groups, "inner candidates do not share one complete group set")
        _require(len(groups) == int(summary_rows[candidate_id]["group_count"]), "inner summary group count drifted")
        for field in aggregate_fields:
            recomputed = float(
                np.mean(
                    [
                        family_sums[candidate_id][family][field]
                        / family_counts[candidate_id][family]
                        for family in sorted(family_sums[candidate_id])
                    ]
                )
            )
            _require(
                math.isclose(
                    recomputed,
                    float(summary_rows[candidate_id][field]),
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-9,
                ),
                f"inner candidate summary does not reproduce group metrics: {candidate_id}/{field}",
            )

    frozen_expected_groups = {
        (family, dataset, source_ordinal, block)
        for family in plan.family_order
        if family != outer_family
        for dataset in plan.families[family]
        for source_ordinal in range(4)
        for block in BLOCK_NAMES
    }
    _require(
        expected_groups == frozen_expected_groups,
        "inner evidence does not cover the exact nonouter dataset/source/block set",
    )

    def selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            -float(row["f1"]),
            -float(row["average_precision"]),
            -float(row["balanced_accuracy"]),
            -float(row["precision"]),
            -float(row["recall"]),
            str(row["candidate_id"]),
        )

    recomputed_selected = min(summary_rows.values(), key=selection_key)
    if selected is not None:
        _require(recomputed_selected["candidate_id"] == selected.candidate_id, "selected candidate does not match the authenticated tie-break")

    audits = json.loads(audit_bytes.decode("utf-8"))
    _require(isinstance(audits, Mapping), "inner fit audit root is invalid")
    _authenticate_self_hash(audits)
    _require(audits.get("schema") == INNER_AUDIT_SCHEMA, "inner fit audit schema drifted")
    _require(audits.get("experiment") == EXPERIMENT and audits.get("outer_family") == outer_family, "inner fit audit provenance drifted")
    fits = audits.get("fits")
    _require(isinstance(fits, list) and int(audits.get("fit_count", -1)) == 12 == len(fits), "inner fit audit count drifted")
    expected_pairs = {
        (inner_family, representation)
        for inner_family in plan.family_order
        if inner_family != outer_family
        for representation in plan.representations
    }
    actual_pairs = {(str(item["inner_family"]), str(item["representation"])) for item in fits if isinstance(item, Mapping)}
    _require(actual_pairs == expected_pairs, "inner fit audit fold/representation coverage drifted")
    for item in fits:
        _require(isinstance(item, Mapping), "inner fit audit row is invalid")
        representation = str(item["representation"])
        _require(
            item.get("composite_descriptor_id")
            == plan.composite_descriptor_ids[representation]
            and item.get("kinematic_input_manifest_file_sha256")
            == plan.kinematic_input_manifest_file_sha256
            and item.get("sidecar_population_manifest_file_sha256")
            == plan.sidecar_population_manifest_file_sha256,
            "inner fit Early evidence binding drifted",
        )
    selected_candidate = candidates[str(recomputed_selected["candidate_id"])]
    persisted_selected_summary = _candidate_payload(selected_candidate)
    for field in aggregate_fields:
        persisted_selected_summary[field] = float(recomputed_selected[field])
    persisted_selected_summary["inner_family_count"] = 4
    persisted_selected_summary["group_count"] = int(recomputed_selected["group_count"])
    _require(
        tuple(persisted_selected_summary) == SUMMARY_FIELDS,
        "authenticated selected-summary field contract drifted",
    )
    return {
        "candidate_count": len(candidates),
        "group_count_per_candidate": len(expected_groups or ()),
        "selected_candidate_id": str(recomputed_selected["candidate_id"]),
        "selected_candidate": selected_candidate,
        "selected_summary": persisted_selected_summary,
        "fit_audit_count": len(fits),
    }


def authenticate_selected_candidate(
    path: Path,
    *,
    plan: Plan,
    selected: TailCandidateSpec,
    scaler: VerifiedScalerArtifact,
    calibration: VerifiedCalibrationArtifact,
    inner_group_metrics_path: Path,
    inner_group_metrics_sha256: str,
    inner_candidate_summary_path: Path,
    inner_candidate_summary_sha256: str,
    inner_fit_audits_path: Path,
    inner_fit_audits_sha256: str,
    outer_family: str,
    git_commit: str,
    expected_file_sha256: str,
) -> VerifiedSelectedCandidate:
    """Authenticate the closed selection artifact before outer feature access."""

    _require(scaler._authentication_seal is _AUTHENTICATION_SEAL, "selection authentication requires authenticated scaler")
    _require(calibration._authentication_seal is _AUTHENTICATION_SEAL, "selection authentication requires authenticated calibration")
    payload = path.read_bytes()
    file_sha = hashlib.sha256(payload).hexdigest()
    _require(_lower_hex(expected_file_sha256) and file_sha == expected_file_sha256, "selected-candidate file SHA-256 mismatch")
    manifest = json.loads(payload.decode("utf-8"))
    _require(isinstance(manifest, Mapping), "selected-candidate root is invalid")
    _authenticate_self_hash(manifest)
    _require(manifest.get("schema") == SELECTED_SCHEMA, "selected-candidate schema drifted")
    _require(manifest.get("experiment") == EXPERIMENT, "selected-candidate experiment drifted")
    _require(manifest.get("config_sha256") == plan.sha256, "selected-candidate config drifted")
    _require(manifest.get("git_commit") == git_commit, "selected-candidate Git binding drifted")
    _require(manifest.get("outer_family") == outer_family, "selected-candidate outer-family binding drifted")
    _require(manifest.get("candidate") == _json_safe(_candidate_payload(selected)), "selected-candidate numerical rule drifted")
    _require(int(manifest.get("candidate_count", -1)) == FROZEN_CANDIDATE_COUNT, "selected-candidate count drifted")
    _require_early_artifact_binding(
        manifest,
        plan,
        representation=selected.representation,
        fit_families=[
            family for family in plan.family_order if family != outer_family
        ],
        label="selected candidate",
    )
    selected_summary = manifest.get("inner_selection_summary")
    _require(isinstance(selected_summary, Mapping) and selected_summary.get("candidate_id") == selected.candidate_id, "selected-candidate summary binding drifted")
    _require(manifest.get("outer_feature_member_opened") is False, "selection was not closed before outer feature access")
    scaler_manifest = manifest.get("final_per_scale_scaler_manifest")
    scaler_file = manifest.get("final_per_scale_scaler_file")
    _require(isinstance(scaler_manifest, Mapping) and isinstance(scaler_file, Mapping), "selected-candidate scaler binding is missing")
    assert isinstance(scaler_manifest, Mapping) and isinstance(scaler_file, Mapping)
    _require(scaler_manifest.get("path") == scaler.manifest_path.name, "selected scaler manifest path drifted")
    _require(scaler_manifest.get("file_sha256") == scaler.manifest_file_sha256, "selected scaler manifest SHA drifted")
    _require(scaler_manifest.get("content_sha256") == scaler.manifest.get("content_sha256"), "selected scaler content hash drifted")
    _require(scaler_file.get("path") == "final_per_scale_scaler.npz", "selected scaler file path drifted")
    _require(scaler_file.get("sha256") == scaler.scaler_file_sha256, "selected scaler file SHA drifted")
    calibration_manifest = manifest.get("final_calibration_manifest")
    calibration_file = manifest.get("final_calibration_file")
    _require(isinstance(calibration_manifest, Mapping) and isinstance(calibration_file, Mapping), "selected-candidate calibration binding is missing")
    assert isinstance(calibration_manifest, Mapping) and isinstance(calibration_file, Mapping)
    _require(calibration_manifest.get("path") == calibration.manifest_path.name, "selected calibration manifest path drifted")
    _require(calibration_manifest.get("file_sha256") == calibration.manifest_file_sha256, "selected calibration manifest SHA drifted")
    _require(calibration_manifest.get("content_sha256") == calibration.manifest.get("content_sha256"), "selected calibration content hash drifted")
    _require(calibration_file.get("path") == "final_tail_calibration.npz", "selected calibration file path drifted")
    _require(calibration_file.get("sha256") == calibration.calibration_file_sha256, "selected calibration file SHA drifted")
    evidence = manifest.get("inner_evidence")
    _require(isinstance(evidence, Mapping), "selected-candidate inner evidence binding is missing")
    assert isinstance(evidence, Mapping)
    evidence_inputs = (
        ("inner_group_metrics", inner_group_metrics_path, inner_group_metrics_sha256),
        ("inner_candidate_summary", inner_candidate_summary_path, inner_candidate_summary_sha256),
        ("inner_fit_audits", inner_fit_audits_path, inner_fit_audits_sha256),
    )
    for name, evidence_path, expected_sha in evidence_inputs:
        record = evidence.get(name)
        _require(isinstance(record, Mapping), f"selected-candidate evidence record is missing: {name}")
        assert isinstance(record, Mapping)
        _require(record.get("path") == evidence_path.name, f"selected-candidate evidence path drifted: {name}")
        _require(record.get("sha256") == expected_sha, f"selected-candidate evidence SHA drifted: {name}")
        _stable_file_identity(evidence_path, int(record["size_bytes"]), expected_sha)
    evidence_audit = _authenticate_inner_selection_evidence(
        plan=plan,
        selected=selected,
        outer_family=outer_family,
        inner_group_metrics_path=inner_group_metrics_path,
        inner_group_metrics_sha256=inner_group_metrics_sha256,
        inner_candidate_summary_path=inner_candidate_summary_path,
        inner_candidate_summary_sha256=inner_candidate_summary_sha256,
        inner_fit_audits_path=inner_fit_audits_path,
        inner_fit_audits_sha256=inner_fit_audits_sha256,
    )
    _require(evidence_audit["candidate_count"] == FROZEN_CANDIDATE_COUNT, "authenticated candidate evidence is incomplete")
    authenticated_summary = evidence_audit["selected_summary"]
    _require(isinstance(authenticated_summary, Mapping), "authenticated selected summary is missing")
    assert isinstance(authenticated_summary, Mapping)
    candidate_identity = _candidate_payload(selected)
    for field in ("candidate_id", "representation", "k", "sigma", "decision_rule", "decision_value"):
        expected_value = candidate_identity[field]
        actual_value = selected_summary.get(field)
        if isinstance(expected_value, float):
            _require(math.isclose(float(actual_value), expected_value, rel_tol=0.0, abs_tol=1.0e-12), f"selected summary candidate identity drifted: {field}")
        else:
            _require(actual_value == expected_value, f"selected summary candidate identity drifted: {field}")
    for field in (
        "accuracy", "average_precision", "f1", "balanced_accuracy", "auroc",
        "precision", "recall", "retrieval_support_fraction",
        "calibration_support_fraction", "spatial_imputed_fraction",
        "spatial_unimputable_fraction",
    ):
        _require(
            math.isclose(
                float(selected_summary[field]),
                float(authenticated_summary[field]),
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            ),
            f"selected summary does not match authenticated CSV evidence: {field}",
        )
    _require(int(selected_summary.get("inner_family_count", -1)) == 4, "selected summary inner-family count drifted")
    _require(int(selected_summary.get("group_count", -1)) == int(authenticated_summary["group_count"]), "selected summary group count drifted")
    return VerifiedSelectedCandidate(
        path=path,
        file_sha256=file_sha,
        manifest=_deep_freeze(manifest),
        _authentication_seal=_AUTHENTICATION_SEAL,
    )


PREDICTION_ARRAY_DTYPES: Mapping[str, np.dtype[Any]] = {
    "dataset": np.dtype("<U64"),
    "source_ordinal": np.dtype(np.int16),
    "source_index": np.dtype(np.int64),
    "scale_id": np.dtype(np.int32),
    "center_seed_index": np.dtype(np.int64),
    "scale_block_index": np.dtype(np.int8),
    "assigned_row_index": np.dtype(np.int64),
    "raw_negative_distance": np.dtype(np.float32),
    "tail_probability": np.dtype(np.float64),
    "tail_anomaly": np.dtype(np.float64),
    "spatial_score": np.dtype(np.float64),
    "spatial_denominator": np.dtype(np.float64),
    "retrieval_supported": np.dtype(np.bool_),
    "calibration_supported": np.dtype(np.bool_),
    "spatial_imputed": np.dtype(np.bool_),
    "spatial_unimputable": np.dtype(np.bool_),
    "calibration_mode": np.dtype(np.int8),
    "scaler_mode": np.dtype(np.int8),
    "prediction": np.dtype(np.bool_),
}


def build_outer_prediction_arrays(
    caches: Sequence[EarlyCacheProjection],
    model: PerScaleNegativeTailModel,
    selected: TailCandidateSpec,
    plan: Plan,
    *,
    device: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    _require(caches and all(cache.labels is None and not cache.metadata for cache in caches), "outer projection must be label-free")
    query = _query_cache_batch(
        model,
        caches,
        selected.representation,
        plan,
        device=device,
        ks=(selected.k,),
    )[selected.k]
    parts: dict[str, list[np.ndarray]] = {name: [] for name in PREDICTION_ARRAY_DTYPES}
    group_audits: list[dict[str, Any]] = []
    for cache_index, cache in enumerate(caches):
        values = query[cache_index]
        for block_index, block_name in enumerate(BLOCK_NAMES):
            block = np.asarray(cache.block_indices == block_index)
            _require(block.any(), f"outer {cache.row.dataset}/{cache.row.source_ordinal}/{block_name}: empty group")
            centers = np.asarray(cache.center_indices[block], dtype=np.int64)
            spatial = spatial_calibrated_tail_scores(
                values["tail_anomaly"][block],
                values["calibration_supported"][block],
                centers,
                sigma=selected.sigma,
                grid_shape=plan.grid_shape,
                truncate=plan.gaussian_truncate,
            )
            eligible = spatial.calibration_supported | spatial.imputed
            prediction = candidate_predictions(selected, spatial.scores, centers, eligible)
            count = int(block.sum())
            parts["dataset"].append(np.full(count, cache.row.dataset, dtype=PREDICTION_ARRAY_DTYPES["dataset"]))
            parts["source_ordinal"].append(np.full(count, cache.row.source_ordinal, dtype=np.int16))
            parts["source_index"].append(np.full(count, cache.row.source_index, dtype=np.int64))
            parts["scale_id"].append(cache.scale_ids[block])
            parts["center_seed_index"].append(cache.center_indices[block])
            parts["scale_block_index"].append(cache.block_indices[block])
            parts["assigned_row_index"].append(cache.assigned_row_indices[block])
            parts["raw_negative_distance"].append(values["raw_distance"][block])
            parts["tail_probability"].append(values["tail_probability"][block])
            parts["tail_anomaly"].append(values["tail_anomaly"][block])
            parts["spatial_score"].append(spatial.scores)
            parts["spatial_denominator"].append(spatial.denominator)
            parts["retrieval_supported"].append(values["retrieval_supported"][block])
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
                    "retrieval_supported_count": int(values["retrieval_supported"][block].sum()),
                    "calibration_supported_count": int(spatial.calibration_supported.sum()),
                    "imputed_count": int(spatial.imputed.sum()),
                    "unimputable_count": int(spatial.unimputable.sum()),
                    "calibration_mode_counts": {str(mode): int(np.count_nonzero(values["calibration_mode"][block] == mode)) for mode in range(6)},
                    "scaler_mode_counts": {str(mode): int(np.count_nonzero(values["scaler_mode"][block] == mode)) for mode in range(4)},
                    "prediction_count": int(prediction.sum()),
                }
            )
    arrays = {
        name: np.ascontiguousarray(np.concatenate(parts[name]), dtype=dtype)
        for name, dtype in PREDICTION_ARRAY_DTYPES.items()
    }
    return arrays, group_audits


def write_outer_prediction(
    output_directory: Path,
    arrays: Mapping[str, np.ndarray],
    group_audits: Sequence[Mapping[str, Any]],
    *,
    plan: Plan,
    selected: TailCandidateSpec,
    selected_artifact: VerifiedSelectedCandidate,
    scaler: VerifiedScalerArtifact,
    calibration: VerifiedCalibrationArtifact,
    outer_family: str,
    git_commit: str,
) -> tuple[Path, Path, str, str]:
    _require(selected_artifact._authentication_seal is _AUTHENTICATION_SEAL, "prediction write requires authenticated selected candidate")
    _require(scaler._authentication_seal is _AUTHENTICATION_SEAL, "prediction write requires authenticated scaler")
    _require(calibration._authentication_seal is _AUTHENTICATION_SEAL, "prediction write requires authenticated calibration")
    _require(set(arrays) == set(PREDICTION_ARRAY_DTYPES), "prediction member set is incomplete")
    prediction_path = output_directory / "outer_predictions.npz"
    prediction_sha = _atomic_npz(prediction_path, arrays)
    manifest = _manifest_with_self_hash(
        {
            "schema": PREDICTION_MANIFEST_SCHEMA,
            "prediction_schema": PREDICTION_SCHEMA,
            "experiment": EXPERIMENT,
            "created_utc": _utc_now(),
            "config_sha256": plan.sha256,
            "git_commit": git_commit,
            "outer_family": outer_family,
            "selected_candidate": _candidate_payload(selected),
            "early_evidence": _early_artifact_binding(
                plan,
                representation=selected.representation,
                fit_families=[
                    family for family in plan.family_order if family != outer_family
                ],
            ),
            "selected_candidate_artifact": {
                "path": selected_artifact.path.name,
                "file_sha256": selected_artifact.file_sha256,
                "content_sha256": selected_artifact.manifest["content_sha256"],
            },
            "final_per_scale_scaler_manifest": {
                "path": scaler.manifest_path.name,
                "file_sha256": scaler.manifest_file_sha256,
                "content_sha256": scaler.manifest["content_sha256"],
            },
            "final_per_scale_scaler_file_sha256": scaler.scaler_file_sha256,
            "final_calibration_manifest": {
                "path": calibration.manifest_path.name,
                "file_sha256": calibration.manifest_file_sha256,
                "content_sha256": calibration.manifest["content_sha256"],
            },
            "final_calibration_file_sha256": calibration.calibration_file_sha256,
            "valid_labels_opened": False,
            "metadata_json_opened": False,
            "prediction_file": {
                "path": prediction_path.name,
                "size_bytes": prediction_path.stat().st_size,
                "sha256": prediction_sha,
            },
            "array_count": len(arrays),
            "row_count": len(np.asarray(arrays["prediction"])),
            "arrays": _array_manifest(arrays),
            "group_audits": list(group_audits),
        }
    )
    manifest_path = output_directory / "outer_prediction_manifest.json"
    manifest_file_sha = _atomic_json(manifest_path, manifest)
    return prediction_path, manifest_path, prediction_sha, manifest_file_sha


def _validate_label_free_outer_scope(
    plan: Plan,
    outer_family: str,
    outer_projections: Sequence[EarlyCacheProjection],
    expected_outer_rows: Sequence[CacheRow],
) -> None:
    expected_datasets = tuple(plan.families[outer_family])
    _require(
        len(expected_outer_rows) == 4 * len(expected_datasets),
        "expected outer cache set is incomplete",
    )
    expected_keys = {
        (dataset, source_ordinal)
        for dataset in expected_datasets
        for source_ordinal in range(4)
    }
    actual_keys = {
        (row.dataset, row.source_ordinal) for row in expected_outer_rows
    }
    _require(actual_keys == expected_keys and len(actual_keys) == len(expected_outer_rows), "expected outer rows do not cover every dataset/source exactly once")
    _require(all(row.family == outer_family for row in expected_outer_rows), "expected outer row family drifted")
    _require(len(outer_projections) == len(expected_outer_rows), "label-free outer projection count drifted")
    for projection, expected_row in zip(outer_projections, expected_outer_rows):
        _require(projection.row == expected_row, "label-free outer projection row identity drifted")
        _require(projection.labels is None and not projection.metadata, "outer projection exposed label-bearing members")
        _require(projection.count > 0, "label-free outer projection is empty")
        _require(set(np.unique(projection.block_indices).tolist()) == {0, 1}, "outer projection lacks a complete two-block query group")


def authenticate_outer_prediction(
    prediction_path: Path,
    manifest_path: Path,
    *,
    plan: Plan,
    selected: TailCandidateSpec,
    selected_artifact: VerifiedSelectedCandidate,
    scaler: VerifiedScalerArtifact,
    calibration: VerifiedCalibrationArtifact,
    outer_projections: Sequence[EarlyCacheProjection],
    expected_outer_rows: Sequence[CacheRow],
    outer_family: str,
    git_commit: str,
    device: str,
    expected_manifest_file_sha256: str,
) -> VerifiedNegativeTailPrediction:
    _require(selected_artifact._authentication_seal is _AUTHENTICATION_SEAL, "prediction authentication requires authenticated selected candidate")
    _require(scaler._authentication_seal is _AUTHENTICATION_SEAL, "prediction authentication requires authenticated scaler")
    _require(calibration._authentication_seal is _AUTHENTICATION_SEAL, "prediction authentication requires authenticated calibration")
    _validate_label_free_outer_scope(
        plan, outer_family, outer_projections, expected_outer_rows
    )
    manifest_bytes = manifest_path.read_bytes()
    manifest_file_sha = hashlib.sha256(manifest_bytes).hexdigest()
    _require(_lower_hex(expected_manifest_file_sha256), "expected prediction manifest SHA-256 is invalid")
    _require(manifest_file_sha == expected_manifest_file_sha256, "prediction manifest file SHA-256 mismatch")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    _require(isinstance(manifest, Mapping), "prediction manifest root is invalid")
    _authenticate_self_hash(manifest)
    _require(manifest.get("schema") == PREDICTION_MANIFEST_SCHEMA, "prediction manifest schema drifted")
    _require(manifest.get("prediction_schema") == PREDICTION_SCHEMA, "prediction schema drifted")
    _require(manifest.get("experiment") == EXPERIMENT and manifest.get("config_sha256") == plan.sha256, "prediction provenance drifted")
    _require(manifest.get("outer_family") == outer_family, "prediction outer-family binding drifted")
    _require(manifest.get("git_commit") == git_commit, "prediction Git binding drifted")
    _require(manifest.get("selected_candidate") == _json_safe(_candidate_payload(selected)), "prediction candidate drifted")
    _require_early_artifact_binding(
        manifest,
        plan,
        representation=selected.representation,
        fit_families=[
            family for family in plan.family_order if family != outer_family
        ],
        label="outer prediction",
    )
    selected_record = manifest.get("selected_candidate_artifact")
    _require(isinstance(selected_record, Mapping), "prediction selected-candidate binding is missing")
    assert isinstance(selected_record, Mapping)
    _require(selected_record.get("path") == selected_artifact.path.name, "prediction selected-candidate path drifted")
    _require(selected_record.get("file_sha256") == selected_artifact.file_sha256, "prediction selected-candidate SHA drifted")
    _require(selected_record.get("content_sha256") == selected_artifact.manifest.get("content_sha256"), "prediction selected-candidate content hash drifted")
    scaler_record = manifest.get("final_per_scale_scaler_manifest")
    _require(isinstance(scaler_record, Mapping), "prediction scaler manifest reference is missing")
    assert isinstance(scaler_record, Mapping)
    _require(scaler_record.get("path") == scaler.manifest_path.name, "prediction scaler manifest path drifted")
    _require(scaler_record.get("file_sha256") == scaler.manifest_file_sha256, "prediction scaler manifest SHA drifted")
    _require(scaler_record.get("content_sha256") == scaler.manifest.get("content_sha256"), "prediction scaler content hash drifted")
    _require(manifest.get("final_per_scale_scaler_file_sha256") == scaler.scaler_file_sha256, "prediction scaler file SHA drifted")
    _require(manifest.get("valid_labels_opened") is False and manifest.get("metadata_json_opened") is False, "prediction opened outer reference early")
    calibration_record = manifest.get("final_calibration_manifest")
    _require(isinstance(calibration_record, Mapping), "prediction calibration manifest reference is missing")
    assert isinstance(calibration_record, Mapping)
    _require(calibration_record.get("file_sha256") == calibration.manifest_file_sha256, "prediction calibration manifest SHA drifted")
    _require(calibration_record.get("content_sha256") == calibration.manifest.get("content_sha256"), "prediction calibration content hash drifted")
    _require(manifest.get("final_calibration_file_sha256") == calibration.calibration_file_sha256, "prediction calibration file SHA drifted")
    file_record = manifest.get("prediction_file")
    _require(isinstance(file_record, Mapping), "prediction file identity is missing")
    assert isinstance(file_record, Mapping)
    _require(file_record.get("path") == prediction_path.name, "prediction file path drifted")
    records = manifest.get("arrays")
    _require(isinstance(records, Mapping), "prediction array manifest is missing")
    assert isinstance(records, Mapping)
    _require(set(records) == set(PREDICTION_ARRAY_DTYPES), "prediction manifest member set drifted")
    _require(int(manifest.get("array_count", -1)) == len(records), "prediction array count drifted")
    with _authenticated_open_file(
        prediction_path,
        expected_size=int(file_record["size_bytes"]),
        expected_sha256=str(file_record["sha256"]),
    ) as opened:
        with np.load(opened.stream, allow_pickle=False) as archive:
            _require(set(archive.files) == set(records), "prediction NPZ member set drifted")
            arrays = {
                name: np.array(archive[name], copy=True, order="C")
                for name in archive.files
            }
        prediction_file_sha256 = opened.sha256
    row_count = int(manifest.get("row_count", -1))
    _require(row_count > 0, "prediction artifact is empty")
    for name, dtype in PREDICTION_ARRAY_DTYPES.items():
        values = arrays[name]
        record = records[name]
        _require(isinstance(record, Mapping), f"prediction array record is invalid: {name}")
        assert isinstance(record, Mapping)
        _require(values.dtype == dtype and values.shape == (row_count,), f"prediction array contract drifted: {name}")
        _require(values.dtype.str == record.get("dtype") and list(values.shape) == record.get("shape"), f"prediction array manifest drifted: {name}")
        _require(canonical_array_sha256(values) == record.get("sha256"), f"prediction array hash mismatch: {name}")

    raw = arrays["raw_negative_distance"]
    tail = arrays["tail_probability"]
    anomaly = arrays["tail_anomaly"]
    score = arrays["spatial_score"]
    denominator = arrays["spatial_denominator"]
    retrieval = arrays["retrieval_supported"]
    calibration_supported = arrays["calibration_supported"]
    imputed = arrays["spatial_imputed"]
    unimputable = arrays["spatial_unimputable"]
    modes = arrays["calibration_mode"]
    scaler_modes = arrays["scaler_mode"]
    prediction = arrays["prediction"]
    scales = arrays["scale_id"]
    centers = arrays["center_seed_index"]
    blocks = arrays["scale_block_index"]
    assigned = arrays["assigned_row_index"]
    _require(np.all((scales >= 0) & (scales < 2000)), "prediction scale outside 0..1999")
    _require(np.all((centers >= 0) & (centers < int(np.prod(plan.grid_shape)))), "prediction center outside frozen grid")
    _require(np.all((blocks >= 0) & (blocks < 2)), "prediction block outside 0..1")
    _require(np.array_equal(blocks, (scales >= 1000).astype(np.int8)), "prediction scale/block identity drifted")
    _require(np.array_equal(assigned, blocks.astype(np.int64) * 64000 + centers), "prediction assigned-row identity drifted")
    _require(np.isfinite(raw[retrieval]).all() and np.isnan(raw[~retrieval]).all(), "raw-distance support sentinel drifted")
    _require(not np.any(calibration_supported & ~retrieval), "calibration support exceeds retrieval support")
    _require(np.isfinite(tail).all() and np.isfinite(anomaly).all() and np.isfinite(score).all() and np.isfinite(denominator).all(), "prediction score field is nonfinite")
    _require(np.all((tail >= 0.0) & (tail <= 1.0)) and np.all((anomaly >= 0.0) & (anomaly <= 1.0)) and np.all((score >= 0.0) & (score <= 1.0)), "prediction probability left [0,1]")
    _require(np.all(denominator >= 0.0), "spatial denominator is negative")
    _require(np.allclose(tail + anomaly, 1.0, rtol=0.0, atol=1e-15), "tail probability/anomaly complement drifted")
    _require(np.array_equal(tail[~calibration_supported], np.ones(np.count_nonzero(~calibration_supported))), "unsupported calibration tail sentinel drifted")
    _require(np.array_equal(anomaly[~calibration_supported], np.zeros(np.count_nonzero(~calibration_supported))), "unsupported calibration anomaly sentinel drifted")
    _require(np.all((modes >= 0) & (modes <= 5)), "calibration mode outside 0..5")
    _require(np.all((scaler_modes >= 0) & (scaler_modes <= 3)), "scaler mode outside 0..3")
    _require(np.array_equal(scaler_modes, scaler.scaler.mode_for_scales(scales)), "prediction scaler modes drifted from authenticated scaler")
    _require(np.array_equal(modes == CALIBRATION_NONE, ~calibration_supported), "mode 0/support equivalence drifted")
    all_rows = np.ones(row_count, dtype=bool)
    _require(np.array_equal(calibration_supported | imputed | unimputable, all_rows), "calibration/imputed/unimputable do not partition rows")
    _require(not np.any((calibration_supported & imputed) | (calibration_supported & unimputable) | (imputed & unimputable)), "calibration spatial states overlap")
    _require(np.all(score[unimputable] == 0.0) and not np.any(prediction[unimputable]), "unimputable rows must fail closed")
    group_audits = manifest.get("group_audits")
    _require(isinstance(group_audits, list) and group_audits, "prediction group audit is missing")
    group_keys: set[tuple[str, int, int, str]] = set()
    coverage = np.zeros(row_count, dtype=np.int16)
    for group in group_audits:
        _require(isinstance(group, Mapping), "prediction group audit row is invalid")
        assert isinstance(group, Mapping)
        _require(group.get("block") in BLOCK_NAMES, "prediction group block name drifted")
        key = (str(group["dataset"]), int(group["source_ordinal"]), int(group["source_index"]), str(group["block"]))
        _require(key not in group_keys, "duplicate prediction group audit")
        group_keys.add(key)
        selected_rows = (arrays["dataset"] == key[0]) & (arrays["source_ordinal"] == key[1]) & (arrays["source_index"] == key[2]) & (blocks == BLOCK_NAMES.index(key[3]))
        coverage[selected_rows] += 1
        _require(int(selected_rows.sum()) == int(group["sample_count"]), "prediction group count drifted")
        _require(len(np.unique(centers[selected_rows])) == int(selected_rows.sum()), "duplicate prediction center within group")
        _require(int(retrieval[selected_rows].sum()) == int(group["retrieval_supported_count"]), "retrieval support audit drifted")
        _require(int(calibration_supported[selected_rows].sum()) == int(group["calibration_supported_count"]), "calibration support audit drifted")
        _require(int(imputed[selected_rows].sum()) == int(group["imputed_count"]), "imputation audit drifted")
        _require(int(unimputable[selected_rows].sum()) == int(group["unimputable_count"]), "unimputable audit drifted")
        mode_counts = group.get("calibration_mode_counts")
        _require(isinstance(mode_counts, Mapping), "mode-count audit is missing")
        assert isinstance(mode_counts, Mapping)
        for mode in range(6):
            _require(int(np.count_nonzero(modes[selected_rows] == mode)) == int(mode_counts[str(mode)]), "calibration mode audit drifted")
        scaler_mode_counts = group.get("scaler_mode_counts")
        _require(isinstance(scaler_mode_counts, Mapping), "scaler-mode-count audit is missing")
        assert isinstance(scaler_mode_counts, Mapping)
        for mode in range(4):
            _require(int(np.count_nonzero(scaler_modes[selected_rows] == mode)) == int(scaler_mode_counts[str(mode)]), "scaler mode audit drifted")
        expected_spatial = spatial_calibrated_tail_scores(
            anomaly[selected_rows],
            calibration_supported[selected_rows],
            centers[selected_rows],
            sigma=selected.sigma,
            grid_shape=plan.grid_shape,
            truncate=plan.gaussian_truncate,
        )
        _require_portable_spatial_replay(
            "spatial_score", score[selected_rows], expected_spatial.scores
        )
        _require_portable_spatial_replay(
            "spatial_denominator",
            denominator[selected_rows],
            expected_spatial.denominator,
        )
        _require(np.array_equal(imputed[selected_rows], expected_spatial.imputed), "authenticated spatial imputation state drifted")
        _require(np.array_equal(unimputable[selected_rows], expected_spatial.unimputable), "authenticated spatial unimputable state drifted")
        expected_prediction = candidate_predictions(
            selected,
            expected_spatial.scores,
            centers[selected_rows],
            expected_spatial.calibration_supported | expected_spatial.imputed,
        )
        _require(np.array_equal(prediction[selected_rows], expected_prediction), "authenticated prediction does not match selected candidate")
        _require(int(prediction[selected_rows].sum()) == int(group["prediction_count"]), "prediction-count audit drifted")
    _require(np.array_equal(coverage, np.ones(row_count, dtype=np.int16)), "prediction groups do not uniquely cover every row")
    expected_arrays, expected_group_audits = build_outer_prediction_arrays(
        outer_projections,
        calibration.model,
        selected,
        plan,
        device=device,
    )
    _require(set(expected_arrays) == set(arrays), "recomputed outer prediction member set drifted")
    for name in sorted(arrays):
        if name in SPATIAL_REPLAY_ULP_BOUNDS:
            _require_portable_spatial_replay(name, arrays[name], expected_arrays[name])
        else:
            _require(
                canonical_array_sha256(expected_arrays[name])
                == canonical_array_sha256(arrays[name]),
                f"persisted outer prediction does not match calibrator query: {name}",
            )
    _require(
        _json_safe(expected_group_audits) == _json_safe(group_audits),
        "persisted outer group audit does not match calibrator query",
    )
    return VerifiedNegativeTailPrediction(
        manifest_path=manifest_path,
        manifest_file_sha256=manifest_file_sha,
        prediction_file_sha256=prediction_file_sha256,
        manifest=_deep_freeze(manifest),
        arrays=_deep_freeze(arrays),
        _authentication_seal=_AUTHENTICATION_SEAL,
    )


def load_outer_references_after_prediction(
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
) -> tuple[
    VerifiedNegativeTailPrediction,
    dict[tuple[str, int, int], tuple[np.ndarray, Mapping[str, Any]]],
    tuple[CacheRow, ...],
]:
    """Reauthenticate the full disk chain, then open every outer reference.

    No label-bearing cache member is opened until the complete expected outer
    cache scope, rebuilt calibrator query, selection evidence, prediction
    arrays, spatial transform, and candidate decisions all authenticate.
    """

    authenticated_manifest_rows, _ = load_cache_rows(plan)
    fresh_outer_rows = tuple(
        row for row in authenticated_manifest_rows if row.family == outer_family
    )
    fresh_outer_projections = tuple(
        load_early_cache_projection(plan, row, include_labels=False)
        for row in fresh_outer_rows
    )
    _validate_label_free_outer_scope(
        plan, outer_family, fresh_outer_projections, fresh_outer_rows
    )
    fit_families = [family for family in plan.family_order if family != outer_family]
    fresh_scaler = authenticate_and_rebuild_final_scaler(
        output_directory / "final_per_scale_scaler.npz",
        output_directory / "final_per_scale_scaler_manifest.json",
        plan=plan,
        selected=selected,
        outer_family=outer_family,
        fit_families=fit_families,
        git_commit=git_commit,
        expected_manifest_file_sha256=expected_scaler_manifest_sha256,
    )
    fresh_calibration = authenticate_and_rebuild_final_calibration(
        output_directory / "final_tail_calibration.npz",
        output_directory / "final_tail_calibration_manifest.json",
        plan=plan,
        selected=selected,
        scaler=fresh_scaler,
        outer_family=outer_family,
        fit_families=fit_families,
        git_commit=git_commit,
        expected_manifest_file_sha256=expected_calibration_manifest_sha256,
    )
    fresh_selected = authenticate_selected_candidate(
        output_directory / "selected_candidate.json",
        plan=plan,
        selected=selected,
        scaler=fresh_scaler,
        calibration=fresh_calibration,
        inner_group_metrics_path=inner_group_metrics_path,
        inner_group_metrics_sha256=inner_group_metrics_sha256,
        inner_candidate_summary_path=inner_candidate_summary_path,
        inner_candidate_summary_sha256=inner_candidate_summary_sha256,
        inner_fit_audits_path=inner_fit_audits_path,
        inner_fit_audits_sha256=inner_fit_audits_sha256,
        outer_family=outer_family,
        git_commit=git_commit,
        expected_file_sha256=expected_selected_candidate_sha256,
    )
    fresh_prediction = authenticate_outer_prediction(
        output_directory / "outer_predictions.npz",
        output_directory / "outer_prediction_manifest.json",
        plan=plan,
        selected=selected,
        selected_artifact=fresh_selected,
        scaler=fresh_scaler,
        calibration=fresh_calibration,
        outer_projections=fresh_outer_projections,
        expected_outer_rows=fresh_outer_rows,
        outer_family=outer_family,
        git_commit=git_commit,
        device=device,
        expected_manifest_file_sha256=expected_prediction_manifest_sha256,
    )

    references: dict[tuple[str, int, int], tuple[np.ndarray, Mapping[str, Any]]] = {}
    arrays = fresh_prediction.arrays
    # The loop starts only after complete-scope authentication above, so a
    # missing cache/group cannot be discovered after an earlier label open.
    for row in fresh_outer_rows:
        projection = load_early_cache_projection(plan, row, include_labels=True)
        _require(projection.labels is not None, "outer reference labels are unavailable")
        selected_rows = (arrays["dataset"] == row.dataset) & (arrays["source_ordinal"] == row.source_ordinal) & (arrays["source_index"] == row.source_index)
        _require(int(selected_rows.sum()) == projection.count, "outer prediction/reference row count mismatch")
        expected_order = np.concatenate(
            [np.flatnonzero(projection.block_indices == block) for block in (0, 1)]
        )
        _require(np.array_equal(arrays["scale_id"][selected_rows], projection.scale_ids[expected_order]), "outer scale identity mismatch")
        _require(np.array_equal(arrays["center_seed_index"][selected_rows], projection.center_indices[expected_order]), "outer center identity mismatch")
        _require(np.array_equal(arrays["assigned_row_index"][selected_rows], projection.assigned_row_indices[expected_order]), "outer assigned-row identity mismatch")
        key = (row.dataset, row.source_ordinal, row.source_index)
        _require(key not in references, "duplicate outer reference identity")
        labels = np.ascontiguousarray(projection.labels[expected_order])
        labels.setflags(write=False)
        references[key] = (labels, _deep_freeze(projection.metadata))
    _require(len(references) == len(fresh_outer_rows), "outer reference scope is incomplete")
    return fresh_prediction, references, fresh_outer_rows


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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    access_audit: list[dict[str, Any]] = []
    verified, references, rows = load_outer_references_after_prediction(
        plan,
        selected,
        output_directory,
        outer_family=outer_family,
        git_commit=git_commit,
        device=device,
        expected_scaler_manifest_sha256=expected_scaler_manifest_sha256,
        expected_calibration_manifest_sha256=expected_calibration_manifest_sha256,
        expected_selected_candidate_sha256=expected_selected_candidate_sha256,
        expected_prediction_manifest_sha256=expected_prediction_manifest_sha256,
        inner_group_metrics_path=inner_group_metrics_path,
        inner_group_metrics_sha256=inner_group_metrics_sha256,
        inner_candidate_summary_path=inner_candidate_summary_path,
        inner_candidate_summary_sha256=inner_candidate_summary_sha256,
        inner_fit_audits_path=inner_fit_audits_path,
        inner_fit_audits_sha256=inner_fit_audits_sha256,
    )
    arrays = verified.arrays
    for row in rows:
        labels, metadata = references[(row.dataset, row.source_ordinal, row.source_index)]
        row_selected = (arrays["dataset"] == row.dataset) & (arrays["source_ordinal"] == row.source_ordinal) & (arrays["source_index"] == row.source_index)
        row_indices = np.flatnonzero(row_selected)
        _require(len(row_indices) == len(labels), "outer label length mismatch")
        for block_index, block_name in enumerate(BLOCK_NAMES):
            within = arrays["scale_block_index"][row_indices] == block_index
            indices = row_indices[within]
            block_labels = labels[within]
            proxy = CacheProjection(row=row, fmt_features=np.empty((len(indices), 161), dtype=np.float32), scale_ids=arrays["scale_id"][indices], center_indices=arrays["center_seed_index"][indices], block_indices=arrays["scale_block_index"][indices], assigned_row_indices=arrays["assigned_row_index"][indices], labels=block_labels, metadata=metadata)
            metrics.append(
                _metric_row(
                    outer_family=outer_family,
                    inner_family="outer_evaluation_only",
                    cache=proxy,
                    block_name=block_name,
                    candidate=selected,
                    labels=block_labels,
                    scores=arrays["spatial_score"][indices],
                    predictions=arrays["prediction"][indices],
                    retrieval_supported=arrays["retrieval_supported"][indices],
                    calibration_supported=arrays["calibration_supported"][indices],
                    imputed=arrays["spatial_imputed"][indices],
                    unimputable=arrays["spatial_unimputable"][indices],
                    calibration_modes=arrays["calibration_mode"][indices],
                    scaler_modes=arrays["scaler_mode"][indices],
                )
            )
        access_audit.append(
            {
                "dataset": row.dataset,
                "source_ordinal": row.source_ordinal,
                "cache_path": str(row.path),
                "cache_file_sha256": row.sha256,
                "sidecar_file_sha256": _population_row(plan, row)[
                    "sidecar_file_sha256"
                ],
                "sidecar_combined_array_sha256": _population_row(plan, row)[
                    "sidecar_combined_array_sha256"
                ],
                "metadata_schema": metadata.get("schema"),
                "label_member_opened_after_prediction_authentication": True,
                "prediction_manifest_file_sha256": verified.manifest_file_sha256,
                "prediction_file_sha256": verified.prediction_file_sha256,
            }
        )
    return metrics, access_audit


METRIC_FIELDS = (
    "outer_family", "inner_family", "dataset", "source_ordinal", "block",
    "candidate_id", "representation", "k", "sigma", "decision_rule", "decision_value",
    "sample_count", "positive_count", "negative_count", "true_positive", "false_positive",
    "true_negative", "false_negative", "accuracy", "precision", "recall", "f1",
    "balanced_accuracy", "average_precision", "auroc", "retrieval_supported_count",
    "calibration_supported_count", "imputed_count", "unimputable_count",
    "retrieval_support_fraction", "calibration_support_fraction", "spatial_imputed_fraction",
    "spatial_unimputable_fraction", "retrieval_supported_subset_f1",
    "calibration_supported_subset_f1", "imputed_subset_f1", "unimputable_subset_f1",
    "calibration_mode_0_count", "calibration_mode_1_count", "calibration_mode_2_count",
    "calibration_mode_3_count", "calibration_mode_4_count", "calibration_mode_5_count",
    "scaler_mode_0_count", "scaler_mode_1_count", "scaler_mode_2_count",
    "scaler_mode_3_count",
)
SUMMARY_FIELDS = (
    "candidate_id", "representation", "k", "sigma", "decision_rule", "decision_value",
    "accuracy", "average_precision", "f1", "balanced_accuracy", "auroc", "precision", "recall",
    "retrieval_support_fraction", "calibration_support_fraction", "spatial_imputed_fraction",
    "spatial_unimputable_fraction", "inner_family_count", "group_count",
)


def _outer_summary(rows: Sequence[Mapping[str, Any]], outer_family: str) -> dict[str, Any]:
    _require(rows, "outer evaluation produced no groups")
    summary: dict[str, Any] = {
        "schema": OUTER_SUMMARY_SCHEMA,
        "experiment": EXPERIMENT,
        "outer_family": outer_family,
        "group_count": len(rows),
    }
    for field in (
        "accuracy", "average_precision", "f1", "balanced_accuracy", "auroc", "precision", "recall",
        "retrieval_support_fraction", "calibration_support_fraction", "spatial_imputed_fraction", "spatial_unimputable_fraction",
    ):
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[field] = float(np.mean(finite)) if len(finite) else float("nan")
    for field in (
        "sample_count", "positive_count", "negative_count", "true_positive", "false_positive",
        "true_negative", "false_negative", "retrieval_supported_count", "calibration_supported_count",
        "imputed_count", "unimputable_count", "calibration_mode_0_count", "calibration_mode_1_count",
        "calibration_mode_2_count", "calibration_mode_3_count", "calibration_mode_4_count",
        "calibration_mode_5_count", "scaler_mode_0_count", "scaler_mode_1_count",
        "scaler_mode_2_count", "scaler_mode_3_count",
    ):
        summary[field] = int(sum(int(row[field]) for row in rows))
    return summary


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
    plan = load_plan(config_path)
    _require(outer_family in plan.family_order, f"unknown outer family: {outer_family}")
    if expected_config_sha256 is not None:
        _require(plan.sha256 == expected_config_sha256, "frozen config SHA-256 mismatch")
    git_commit, dirty = _git_identity()
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
    _configure_execution(device)
    destination = Path(output_dir).resolve()
    _require(not destination.exists(), f"immutable output directory exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    print(f"[{_utc_now()}] {EXPERIMENT} outer={outer_family} commit={git_commit}", flush=True)

    cache_rows, input_manifest_identity = load_cache_rows(plan)
    nonouter_rows = [row for row in cache_rows if row.family != outer_family]
    outer_rows = [row for row in cache_rows if row.family == outer_family]
    _require(nonouter_rows and outer_rows, "outer split produced an empty side")
    # No outer cache file has been opened at this point.
    nonouter_caches = [load_early_cache_projection(plan, row, include_labels=True) for row in nonouter_rows]
    inner_rows, candidates, inner_fit_audits = _inner_metric_rows(plan, nonouter_caches, outer_family, device=device)
    summaries, _in_memory_selected, _in_memory_selected_summary = _aggregate_and_select(inner_rows, candidates)
    _require(tuple(inner_rows[0]) == METRIC_FIELDS, "inner metric CSV field contract drifted")
    _require(tuple(summaries[0]) == SUMMARY_FIELDS, "inner summary CSV field contract drifted")
    inner_metrics_path = destination / "inner_group_metrics.csv"
    inner_summary_path = destination / "inner_candidate_summary.csv"
    inner_fit_path = destination / "inner_fit_audits.json"
    inner_metrics_sha = _atomic_csv(inner_metrics_path, METRIC_FIELDS, inner_rows)
    inner_summary_sha = _atomic_csv(inner_summary_path, SUMMARY_FIELDS, summaries)
    inner_fit_sha = _atomic_json(
        inner_fit_path,
        _manifest_with_self_hash(
            {
                "schema": INNER_AUDIT_SCHEMA,
                "experiment": EXPERIMENT,
                "outer_family": outer_family,
                "fit_count": len(inner_fit_audits),
                "fits": inner_fit_audits,
            }
        ),
    )
    persisted_selection = _authenticate_inner_selection_evidence(
        plan=plan,
        selected=None,
        outer_family=outer_family,
        inner_group_metrics_path=inner_metrics_path,
        inner_group_metrics_sha256=inner_metrics_sha,
        inner_candidate_summary_path=inner_summary_path,
        inner_candidate_summary_sha256=inner_summary_sha,
        inner_fit_audits_path=inner_fit_path,
        inner_fit_audits_sha256=inner_fit_sha,
    )
    selected = persisted_selection["selected_candidate"]
    _require(isinstance(selected, TailCandidateSpec), "persisted selection did not reconstruct a candidate")
    selected_summary = persisted_selection["selected_summary"]
    _require(
        isinstance(selected_summary, Mapping) and tuple(selected_summary) == SUMMARY_FIELDS,
        "persisted selection did not reconstruct the complete selected summary",
    )
    _require(
        selected == _in_memory_selected,
        "persisted 12g selection drifted from the full-precision in-memory selection",
    )
    _require(
        all(
            (
                math.isclose(
                    float(selected_summary[name]),
                    float(_in_memory_selected_summary[name]),
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-9,
                )
                if isinstance(selected_summary[name], float)
                else selected_summary[name] == _in_memory_selected_summary[name]
            )
            for name in SUMMARY_FIELDS
        ),
        "persisted selected summary drifted from the full-precision in-memory summary",
    )
    del _in_memory_selected, _in_memory_selected_summary

    fit_families = [family for family in plan.family_order if family != outer_family]
    final_model = _fit_tail_model(
        nonouter_caches,
        selected.representation,
        plan,
        device=device,
        ks=(selected.k,),
    )
    scaler_path, scaler_manifest_path, _, scaler_manifest_sha = (
        write_final_scaler_artifact(
            destination,
            final_model,
            plan=plan,
            selected=selected,
            outer_family=outer_family,
            fit_families=fit_families,
            git_commit=git_commit,
        )
    )
    scaler = authenticate_and_rebuild_final_scaler(
        scaler_path,
        scaler_manifest_path,
        plan=plan,
        selected=selected,
        outer_family=outer_family,
        fit_families=fit_families,
        git_commit=git_commit,
        expected_manifest_file_sha256=scaler_manifest_sha,
    )
    calibration_path, calibration_manifest_path, _, calibration_manifest_sha = write_final_calibration_artifact(
        destination,
        final_model,
        plan=plan,
        selected=selected,
        scaler=scaler,
        outer_family=outer_family,
        fit_families=fit_families,
        git_commit=git_commit,
    )
    # Re-read from disk and authenticate before reconstruction.  The model is
    # allowed to contain any frozen n<k, n=k, and n>=k+1 support mixture.
    calibration = authenticate_and_rebuild_final_calibration(
        calibration_path,
        calibration_manifest_path,
        plan=plan,
        selected=selected,
        scaler=scaler,
        outer_family=outer_family,
        fit_families=fit_families,
        git_commit=git_commit,
        expected_manifest_file_sha256=calibration_manifest_sha,
    )
    del final_model, nonouter_caches
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    selected_path, selected_sha, selected_payload = write_selected_candidate(
        destination,
        plan=plan,
        selected=selected,
        selected_summary=selected_summary,
        scaler=scaler,
        calibration=calibration,
        inner_group_metrics_path=inner_metrics_path,
        inner_group_metrics_sha256=inner_metrics_sha,
        inner_candidate_summary_path=inner_summary_path,
        inner_candidate_summary_sha256=inner_summary_sha,
        inner_fit_audits_path=inner_fit_path,
        inner_fit_audits_sha256=inner_fit_sha,
        outer_family=outer_family,
        git_commit=git_commit,
    )
    selected_artifact = authenticate_selected_candidate(
        selected_path,
        plan=plan,
        selected=selected,
        scaler=scaler,
        calibration=calibration,
        inner_group_metrics_path=inner_metrics_path,
        inner_group_metrics_sha256=inner_metrics_sha,
        inner_candidate_summary_path=inner_summary_path,
        inner_candidate_summary_sha256=inner_summary_sha,
        inner_fit_audits_path=inner_fit_path,
        inner_fit_audits_sha256=inner_fit_sha,
        outer_family=outer_family,
        git_commit=git_commit,
        expected_file_sha256=selected_sha,
    )
    # Only after the persisted/rebuilt model and selected-candidate artifact
    # close may any outer feature member be opened.
    outer_caches = [load_early_cache_projection(plan, row, include_labels=False) for row in outer_rows]
    arrays, group_audits = build_outer_prediction_arrays(
        outer_caches,
        calibration.model,
        selected,
        plan,
        device=device,
    )
    prediction_path, prediction_manifest_path, prediction_file_sha, prediction_manifest_sha = write_outer_prediction(
        destination,
        arrays,
        group_audits,
        plan=plan,
        selected=selected,
        selected_artifact=selected_artifact,
        scaler=scaler,
        calibration=calibration,
        outer_family=outer_family,
        git_commit=git_commit,
    )
    del arrays
    outer_metric_rows, reference_rows = evaluate_outer_prediction(
        plan,
        selected,
        destination,
        outer_family=outer_family,
        git_commit=git_commit,
        device=device,
        expected_scaler_manifest_sha256=scaler_manifest_sha,
        expected_calibration_manifest_sha256=calibration_manifest_sha,
        expected_selected_candidate_sha256=selected_sha,
        expected_prediction_manifest_sha256=prediction_manifest_sha,
        inner_group_metrics_path=inner_metrics_path,
        inner_group_metrics_sha256=inner_metrics_sha,
        inner_candidate_summary_path=inner_summary_path,
        inner_candidate_summary_sha256=inner_summary_sha,
        inner_fit_audits_path=inner_fit_path,
        inner_fit_audits_sha256=inner_fit_sha,
    )
    _require(tuple(outer_metric_rows[0]) == METRIC_FIELDS, "outer metric CSV field contract drifted")
    outer_metrics_sha = _atomic_csv(destination / "outer_group_metrics.csv", METRIC_FIELDS, outer_metric_rows)
    outer_summary = _outer_summary(outer_metric_rows, outer_family)
    outer_summary_sha = _atomic_json(destination / "outer_summary.json", _manifest_with_self_hash(outer_summary))
    reference_audit = _manifest_with_self_hash(
        {
            "schema": REFERENCE_AUDIT_SCHEMA,
            "experiment": EXPERIMENT,
            "outer_family": outer_family,
            "first_open_phase": "after_outer_prediction_file_and_manifest_authentication",
            "prediction_manifest_file_sha256": prediction_manifest_sha,
            "prediction_file_sha256": prediction_file_sha,
            "row_count": len(reference_rows),
            "rows": reference_rows,
        }
    )
    reference_audit_sha = _atomic_json(destination / "outer_reference_access_audit.json", reference_audit)

    artifact_names = tuple(name for name in plan.required_fold_files if name not in {"result_manifest.json", "RUN_COMPLETE.json"})
    _require(set(artifact_names) == {
        "inner_group_metrics.csv", "inner_candidate_summary.csv", "inner_fit_audits.json",
        "final_per_scale_scaler.npz", "final_per_scale_scaler_manifest.json",
        "final_tail_calibration.npz", "final_tail_calibration_manifest.json", "selected_candidate.json",
        "outer_predictions.npz", "outer_prediction_manifest.json", "outer_group_metrics.csv",
        "outer_summary.json", "outer_reference_access_audit.json",
    }, "required artifact list drifted")
    result_manifest = _manifest_with_self_hash(
        {
            "schema": RESULT_SCHEMA,
            "experiment": EXPERIMENT,
            "status": "completed",
            "completed_utc": _utc_now(),
            "git_commit": git_commit,
            "config_path": str(plan.path),
            "config_sha256": plan.sha256,
            "input_manifest": input_manifest_identity,
            "early_evidence": _early_artifact_binding(
                plan,
                representation=selected.representation,
                fit_families=fit_families,
            ),
            "outer_family": outer_family,
            "selected_candidate": _candidate_payload(selected),
            "selected_candidate_file": selected_path.name,
            "selected_candidate_file_sha256": selected_sha,
            "selected_candidate_content_sha256": selected_payload["content_sha256"],
            "final_scaler_manifest_file_sha256": scaler.manifest_file_sha256,
            "final_scaler_file_sha256": scaler.scaler_file_sha256,
            "final_calibration_manifest_file_sha256": calibration.manifest_file_sha256,
            "final_calibration_file_sha256": calibration.calibration_file_sha256,
            "prediction_manifest_file_sha256": prediction_manifest_sha,
            "prediction_file_sha256": prediction_file_sha,
            "inner_group_metrics_file_sha256": inner_metrics_sha,
            "inner_candidate_summary_file_sha256": inner_summary_sha,
            "inner_fit_audits_file_sha256": inner_fit_sha,
            "outer_group_metrics_file_sha256": outer_metrics_sha,
            "outer_summary_file_sha256": outer_summary_sha,
            "outer_reference_access_audit_file_sha256": reference_audit_sha,
            "environment": _environment_audit(device),
            "artifacts": {
                name: {"size_bytes": (destination / name).stat().st_size, "sha256": sha256_file(destination / name)}
                for name in artifact_names
            },
        }
    )
    result_path = destination / "result_manifest.json"
    result_sha = _atomic_json(result_path, result_manifest)
    completion = _manifest_with_self_hash(
        {
            "schema": COMPLETE_SCHEMA,
            "experiment": EXPERIMENT,
            "outer_family": outer_family,
            "git_commit": git_commit,
            "config_sha256": plan.sha256,
            "result_manifest_file": result_path.name,
            "result_manifest_file_sha256": result_sha,
            "result_manifest_content_sha256": result_manifest["content_sha256"],
            "completed_utc": _utc_now(),
        }
    )
    _atomic_json(destination / "RUN_COMPLETE.json", completion)
    _require(set(path.name for path in destination.iterdir()) == set(plan.required_fold_files), "completed fold file set drifted")
    print(f"[{_utc_now()}] completed outer={outer_family} F1={outer_summary['f1']:.6f}", flush=True)
    return result_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "Verify_EarlyOppositePairKinematics_1.1.yaml"))
    parser.add_argument("--outer-family", required=True, choices=FAMILY_ORDER)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kinematic-input-manifest", required=True)
    parser.add_argument("--kinematic-input-manifest-sha256", required=True)
    parser.add_argument("--synthetic-pass", required=True)
    parser.add_argument("--synthetic-pass-sha256", required=True)
    parser.add_argument("--sidecar-root", required=True)
    parser.add_argument("--sidecar-population-manifest", required=True)
    parser.add_argument("--sidecar-population-manifest-sha256", required=True)
    parser.add_argument("--expected-config-sha256", default=EXPECTED_CONFIG_SHA256)
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

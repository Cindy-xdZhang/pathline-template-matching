#!/usr/bin/env python3
"""Mandatory resource smoke for Verify_ClassConditionalTemplateScore_1.1.

This program is deliberately not an evaluator.  It authenticates the frozen
Early input population, opens members only for the three pre-registered fit
families, builds the widest class-conditional model at ``k=31``, and exercises
that model only with deterministic synthetic queries.  It never enumerates a
decision candidate, writes a class assignment, or computes a quality metric.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

try:  # ``resource`` is unavailable on Windows but mandatory on Ibex/Linux.
    import resource as _resource
except ImportError:  # pragma: no cover - exercised by Windows development hosts
    _resource = None

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.class_conditional_template_score import (  # noqa: E402
    CLASS_NAMES,
    ClassConditionalTemplateScoreModel,
    FamilyFitBatch,
    strict_threshold_predictions,
)
from pathline_template_matching.early_kinematic_preparation import (  # noqa: E402
    POPULATION_MANIFEST_SCHEMA,
)
from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_array_sha256,
    canonical_json_sha256,
    sha256_file,
)
from pathline_template_matching.seed_time_kinematic_sidecar import (  # noqa: E402
    SIDECAR_ARCHIVE_MEMBER_NAMES,
)
from scripts import run_verify_class_conditional_template_score_1_1 as runner  # noqa: E402


EXPERIMENT = "Verify_ClassConditionalTemplateScore_1.1"
STAGE = "mandatory_resource_smoke"
EXPECTED_CONFIG_SHA256 = (
    "814f95d2ec58f751a91082d588f790b3592a891963810013ad92ab704febbdea"
)
CONFIG_PATH = ROOT / "config" / "Verify_ClassConditionalTemplateScore_1.1.yaml"
SMOKE_SOURCE_PATH = (
    ROOT / "scripts" / "run_verify_class_conditional_template_score_resource_smoke_1_1.py"
)
SMOKE_TEST_PATH = ROOT / "tests" / "test_class_conditional_template_score_resource_smoke.py"
SMOKE_WRAPPER_PATH = ROOT / "ibex" / "verify_class_conditional_template_score_1.1_resource_smoke.sh"

FIT_FAMILIES = ("f22_raptor", "channel", "boeing_747")
RESERVED_OUTER_FAMILY = "half_cylinder"
RESERVED_INNER_FAMILY = "delta_wing"
RESERVED_FAMILIES = (RESERVED_OUTER_FAMILY, RESERVED_INNER_FAMILY)
REPRESENTATION = "fmt161_plus_seed4"
K = 31
EXPECTED_FIT_ROW_COUNT = 12
SCALE_COUNT = 2000
FEATURE_WIDTH = 165
QUERY_CHUNK_SIZE = 1024
LIBRARY_CHUNK_SIZE = 8192
MEMORY_LIMIT_BYTES = 128 * 1024**3
WALLTIME_LIMIT_SECONDS = 4 * 60 * 60
RESOURCE_SMOKE_TIME_LIMIT = "04:00:00"
SLURM_MEMORY_PER_NODE_MIB = 128 * 1024
FROZEN_CONFIG_SLURM_ACCOUNT = "deepvortex"
RUNTIME_SLURM_ACCOUNT = "pi-hadwigm"
FROZEN_CONFIG_SLURM_PARTITION = "cpu"
RUNTIME_SLURM_PARTITION = "batch"

PARENT_ARCHIVE_MEMBERS = (
    "fmt_features",
    "valid_scale_id",
    "valid_center_seed_index",
    "valid_scale_block_index",
    "valid_assigned_row_index",
    "valid_labels",
    "metadata_json",
)
SIDECAR_MEMBERS = tuple(SIDECAR_ARCHIVE_MEMBER_NAMES)

AUDIT_SCHEMA = (
    "pathline_template_matching.verify_class_conditional_template_score_"
    "resource_smoke_audit.v1"
)
PASS_SCHEMA = (
    "pathline_template_matching.verify_class_conditional_template_score_"
    "resource_smoke_pass.v1"
)
AUDIT_NAME = "resource_smoke_audit.json"
PASS_NAME = "RESOURCE_SMOKE_PASS.json"

CONSTRUCTED_MODEL_FIELDS = frozenset(
    {
        "family_order",
        "feature_width",
        "k",
        "strict_majority_family_count",
        "natural_raw_family_class_row_count",
        "effective_retained_family_class_library_row_count",
        "natural_raw_present_exact_scale_count",
        "effective_retained_exact_scale_count",
        "natural_raw_only_no_scaler_exact_scale_count",
        "natural_raw_present_exact_scale_ids_sha256",
        "effective_retained_exact_scale_ids_sha256",
        "natural_raw_only_no_scaler_exact_scale_ids_sha256",
        "natural_raw_class_scale_counts_sha256",
        "effective_retained_class_scale_counts_sha256",
        "shared_negative_row_count",
        "full_family_class_library_row_count",
        "loo_reference_row_count_k31",
        "zero_distance_loo_reference_count_k31",
        "class_library_and_reference_audits",
        "scaler_arrays",
        "family_class_library_and_calibration_arrays",
        "all_constructed_arrays_finite",
        "exact_self_exclusion_duplicate_and_support_count_audits_passed",
    }
)
SYNTHETIC_QUERY_FIELDS = frozenset(
    {
        "construction",
        "query_row_count",
        "query_scale_domain",
        "all_natural_raw_present_exact_scales_exercised",
        "natural_raw_present_exact_scale_count",
        "effective_retained_exact_scale_count",
        "natural_raw_only_no_scaler_exact_scale_count",
        "natural_raw_present_exact_scale_ids_sha256",
        "effective_retained_exact_scale_ids_sha256",
        "natural_raw_only_no_scaler_exact_scale_ids_sha256",
        "all_natural_raw_only_no_scaler_scales_have_all_class_family_retrieval_and_calibration_unsupported",
        "strict_majority_joint_supported_row_count",
        "strict_majority_retrieval_supported_row_count",
        "joint_family_count_histogram",
        "support_count_arithmetic_passed",
        "all_supported_numerical_values_finite",
        "reference_labels_consulted_by_query_path",
    }
)
OPENED_FIT_ROW_FIELDS = frozenset(
    {
        "dataset",
        "physical_family",
        "source_ordinal",
        "source_index",
        "joined_row_count",
        "parent_cache",
        "parent_archive_members",
        "sidecar",
        "sidecar_archive_members",
        "exact_identity_join_passed",
    }
)
FIT_INPUT_ROW_FIELDS = OPENED_FIT_ROW_FIELDS | {
    "valid_row_count",
    "negative_row_count",
    "positive_row_count",
}
ARRAY_RECORD_FIELDS = frozenset(
    {"dtype", "shape", "size_bytes", "sha256", "finite_or_nonfloating"}
)
CLASS_LIBRARY_AUDIT_FIELDS = frozenset(
    {
        "library_row_count",
        "library_supported_scale_count_k31",
        "loo_reference_row_count_k31",
        "loo_supported_scale_count_k31",
        "zero_distance_loo_reference_count_k31",
        "exact_self_exclusion_count_identity_passed",
        "duplicate_rows_retained_in_count_and_zero_distance_references",
        "class_scale_counts_sha256",
    }
)
SCALER_ARRAY_SPECS = {
    "block_other_row_count_int64": ("<i8", (SCALE_COUNT,)),
    "effective_std_float64": ("<f8", (SCALE_COUNT, FEATURE_WIDTH)),
    "global_other_row_count_int64": ("<i8", (SCALE_COUNT,)),
    "local_mean_float64": ("<f8", (SCALE_COUNT, FEATURE_WIDTH)),
    "local_row_count_int64": ("<i8", (SCALE_COUNT,)),
    "local_support_bool": ("|b1", (SCALE_COUNT,)),
    "local_variance_float64": ("<f8", (SCALE_COUNT, FEATURE_WIDTH)),
    "prior_variance_float64": ("<f8", (SCALE_COUNT, FEATURE_WIDTH)),
    "scale_id_int32": ("<i4", (SCALE_COUNT,)),
    "scaler_mode_int8": ("|i1", (SCALE_COUNT,)),
    "shrunk_variance_float64": ("<f8", (SCALE_COUNT, FEATURE_WIDTH)),
}
CALIBRATOR_BASE_ARRAY_SPECS = {
    "serialization_version_int16": ("<i2", ()),
    "family_order_unicode": (
        np.asarray(FIT_FAMILIES, dtype=np.str_).dtype.str,
        (len(FIT_FAMILIES),),
    ),
    "family_order_copy_unicode": (
        np.asarray(FIT_FAMILIES, dtype=np.str_).dtype.str,
        (len(FIT_FAMILIES),),
    ),
    "required_family_count_int64": ("<i8", ()),
    "ks_int64": ("<i8", (1,)),
    "shrinkage_lambda_float64": ("<f8", ()),
    "class_present_bool": ("|b1", (len(FIT_FAMILIES), len(CLASS_NAMES))),
    "class_scale_counts_int64": (
        "<i8",
        (len(FIT_FAMILIES), len(CLASS_NAMES), SCALE_COUNT),
    ),
}

# These names must never occur as persisted JSON fields.  Hash strings and
# ordinary prose values are not substring-scanned because random SHA-256 text
# may contain short strings such as ``f1``.
FORBIDDEN_OUTPUT_FIELD_NAMES = frozenset(
    {
        "accuracy",
        "f1",
        "average_precision",
        "balanced_accuracy",
        "auroc",
        "precision",
        "recall",
        "metric",
        "metrics",
        "prediction",
        "predictions",
        "selected_candidate",
        "performance",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _lower_hex(value: object, *, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _is_nonbool_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _stable_file_identity(
    path: str | Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    source = Path(path).resolve()
    before = source.stat(follow_symlinks=False)
    digest = sha256_file(source)
    after = source.stat(follow_symlinks=False)
    _require(
        (before.st_size, before.st_mtime_ns)
        == (after.st_size, after.st_mtime_ns),
        f"file changed while hashing: {source}",
    )
    if expected_size is not None:
        _require(after.st_size == int(expected_size), f"file size drifted: {source}")
    if expected_sha256 is not None:
        _require(
            _lower_hex(expected_sha256, length=64) and digest == expected_sha256,
            f"file SHA-256 drifted: {source}",
        )
    return {
        "path": str(source),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _git_identity(expected_commit: str) -> dict[str, Any]:
    _require(_lower_hex(expected_commit, length=40), "expected Git commit is invalid")
    commit = subprocess.run(
        ("git", "rev-parse", "--verify", "HEAD^{commit}"),
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    _require(commit == expected_commit, "checkout differs from expected Git commit")
    _require(not dirty, "resource smoke requires a clean committed worktree")
    return {"git_commit": commit, "worktree_clean": True}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        _require(np.isfinite(numeric), "JSON evidence contains a non-finite value")
        return numeric
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _with_self_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(_json_safe(payload))
    _require("content_sha256" not in output, "self hash is already present")
    output["content_sha256"] = canonical_json_sha256(output)
    return output


def _authenticate_self_hashed_file(
    path: str | Path, *, expected_file_sha256: str | None = None
) -> dict[str, Any]:
    source = Path(path)
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_file_sha256 is not None:
        _require(digest == expected_file_sha256, f"evidence SHA-256 drifted: {source}")
    value = json.loads(payload.decode("utf-8"))
    _require(isinstance(value, dict), f"evidence root is not a mapping: {source}")
    content = dict(value)
    claimed = content.pop("content_sha256", None)
    _require(
        _lower_hex(claimed, length=64)
        and claimed == canonical_json_sha256(content),
        f"evidence self hash drifted: {source}",
    )
    return value


def _fsync_parent_directory(parent: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(
        os.fspath(parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json_no_replace(path: Path, value: Mapping[str, Any]) -> str:
    """Publish one JSON artifact with a hard-link no-replace operation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = -1
            destination.write(payload)
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
    return hashlib.sha256(payload).hexdigest()


def _assert_no_forbidden_output_fields(value: Any, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            _require(
                normalized not in FORBIDDEN_OUTPUT_FIELD_NAMES,
                f"forbidden resource-smoke output field at {path}: {key}",
            )
            _assert_no_forbidden_output_fields(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_forbidden_output_fields(item, path=f"{path}[{index}]")


def _validate_resource_smoke_contract(plan: runner.Plan) -> None:
    raw = plan.raw
    smoke = raw.get("resource_smoke")
    gates = raw.get("pre_run_gates")
    _require(isinstance(smoke, Mapping), "resource smoke contract is missing")
    _require(isinstance(gates, Mapping), "pre-run gate contract is missing")
    split = smoke.get("fixed_split_identity")
    worst = smoke.get("worst_case_path")
    slurm = smoke.get("slurm")
    _require(isinstance(split, Mapping), "resource smoke split is missing")
    _require(isinstance(worst, Mapping), "resource smoke worst path is missing")
    _require(isinstance(slurm, Mapping), "resource smoke Slurm request is missing")
    _require(smoke.get("mandatory") is True, "resource smoke is not mandatory")
    _require(
        split.get("reserved_outer_family_not_opened") == RESERVED_OUTER_FAMILY
        and split.get("reserved_inner_family_not_opened") == RESERVED_INNER_FAMILY
        and tuple(split.get("fit_families_opened", ())) == FIT_FAMILIES,
        "resource smoke family split drifted",
    )
    _require(
        worst.get("representation") == REPRESENTATION
        and int(worst.get("k", -1)) == K
        and worst.get("scales") == "all_exact_scales_present_in_the_three_fit_families"
        and worst.get("query_exercise")
        == "deterministic_label_free_synthetic_queries_only",
        "resource smoke numerical path drifted",
    )
    _require(
        slurm.get("partition") == FROZEN_CONFIG_SLURM_PARTITION
        and slurm.get("constraint") == "rome"
        and slurm.get("account") == FROZEN_CONFIG_SLURM_ACCOUNT
        and int(slurm.get("nodes", -1)) == 1
        and int(slurm.get("cpus_per_task", -1)) == 32
        and int(slurm.get("memory_gb", -1)) == 128
        and str(slurm.get("walltime")) == "04:00:00"
        and slurm.get("gpu") == "none",
        "resource smoke resource request drifted",
    )
    required = tuple(gates.get("required_before_first_real_outer_fold", ()))
    _require(
        required
        == (
            "resource_smoke_PASS_marker_authenticated",
            "no_new_version_outer_feature_label_prediction_or_metric_read_during_smoke",
            "exact_parent_input_manifests_reauthenticated",
        ),
        "resource-smoke downstream gate drifted",
    )
    _require(
        plan.representations[0] == REPRESENTATION
        and runner.inherited.COMPOSITE_WIDTH[REPRESENTATION] == FEATURE_WIDTH
        and plan.ks[-1] == K
        and plan.query_chunk_size == QUERY_CHUNK_SIZE
        and plan.library_chunk_size == LIBRARY_CHUNK_SIZE,
        "production model constants drifted",
    )


@dataclass
class AccessLedger:
    """Member-open ledger; envelope hashing is recorded separately."""

    records: list[dict[str, Any]] = field(default_factory=list)
    member_open_counts: dict[str, int] = field(
        default_factory=lambda: {
            family: 0 for family in (*RESERVED_FAMILIES, *FIT_FAMILIES)
        }
    )

    def authorize(self, family: str) -> None:
        _require(
            family in FIT_FAMILIES,
            f"member access is forbidden for non-fit family: {family}",
        )

    def append(
        self,
        *,
        row: Any,
        parent_identity: Mapping[str, Any],
        sidecar_identity: Mapping[str, Any],
        joined_row_count: int,
    ) -> None:
        family = str(row.family)
        self.authorize(family)
        _require(
            tuple(PARENT_ARCHIVE_MEMBERS)
            == (
                "fmt_features",
                "valid_scale_id",
                "valid_center_seed_index",
                "valid_scale_block_index",
                "valid_assigned_row_index",
                "valid_labels",
                "metadata_json",
            ),
            "parent member contract drifted",
        )
        _require(
            tuple(SIDECAR_MEMBERS) == tuple(SIDECAR_ARCHIVE_MEMBER_NAMES),
            "sidecar member contract drifted",
        )
        self.member_open_counts[family] += 1
        self.records.append(
            {
                "dataset": str(row.dataset),
                "physical_family": family,
                "source_ordinal": int(row.source_ordinal),
                "source_index": int(row.source_index),
                "joined_row_count": int(joined_row_count),
                "parent_cache": dict(parent_identity),
                "parent_archive_members": list(PARENT_ARCHIVE_MEMBERS),
                "sidecar": dict(sidecar_identity),
                "sidecar_archive_members": list(SIDECAR_MEMBERS),
                "exact_identity_join_passed": True,
            }
        )

    def validate(self) -> None:
        _require(
            len(self.records) == EXPECTED_FIT_ROW_COUNT,
            "fit member-open row count drifted",
        )
        _require(
            all(self.member_open_counts[family] == 4 for family in FIT_FAMILIES),
            "each fit family must open exactly four source members",
        )
        _require(
            all(self.member_open_counts[family] == 0 for family in RESERVED_FAMILIES),
            "a reserved family member was opened",
        )

    def as_json(self, *, population_sidecar_count: int) -> dict[str, Any]:
        self.validate()
        return {
            "semantic_separation": {
                "population_envelope_whole_file_authentication": (
                    "all sealed sidecar byte files were hashed without archive-member deserialization"
                ),
                "archive_member_deserialization": "only the three fixed fit families below",
            },
            "population_envelope_sidecar_count": int(population_sidecar_count),
            "member_open_counts_by_family": dict(self.member_open_counts),
            "reserved_family_zero_member_open": {
                family: self.member_open_counts[family] == 0
                for family in RESERVED_FAMILIES
            },
            "reserved_parent_cache_whole_file_open_count": 0,
            "reserved_sidecar_archive_member_open_count": 0,
            "reserved_sidecar_envelope_hash_note": (
                "sealed byte files are whole-file hashed by the mandatory 32-file "
                "population envelope gate without NPZ member deserialization"
            ),
            "opened_fit_rows": list(self.records),
        }


def _population_row(plan: runner.Plan, row: Any) -> Mapping[str, Any]:
    _require(plan.sidecar_population is not None, "sidecar population is unbound")
    candidates = [
        value
        for value in plan.sidecar_population["rows"]
        if str(value["dataset"]) == str(row.dataset)
        and int(value["source_ordinal"]) == int(row.source_ordinal)
    ]
    _require(len(candidates) == 1, "sidecar population row identity is not unique")
    value = candidates[0]
    _require(
        str(value["physical_family"]) == str(row.family)
        and int(value["source_index"]) == int(row.source_index),
        "sidecar population row identity drifted",
    )
    return value


def _selected_fit_rows(
    rows: Sequence[Any], *, fit_families: Sequence[str] = FIT_FAMILIES
) -> list[Any]:
    selected_families = tuple(str(value) for value in fit_families)
    _require(selected_families == FIT_FAMILIES, "fit-family set/order drifted")
    _require(
        len(rows) == 32
        and {str(row.family) for row in rows}
        == set((*RESERVED_FAMILIES, *FIT_FAMILIES)),
        "authenticated 32-row family population drifted",
    )
    selected = [row for row in rows if str(row.family) in FIT_FAMILIES]
    _require(len(selected) == EXPECTED_FIT_ROW_COUNT, "fit row count drifted")
    _require(
        all(str(row.family) not in RESERVED_FAMILIES for row in selected),
        "reserved family entered selected fit rows",
    )
    return selected


def _load_fit_projections(
    plan: runner.Plan,
    rows: Sequence[Any],
    ledger: AccessLedger,
    *,
    loader: Callable[..., Any] | None = None,
) -> list[Any]:
    selected = _selected_fit_rows(rows)
    selected_loader = loader or runner.inherited.load_early_cache_projection
    projections: list[Any] = []
    assert plan.sidecar_root is not None
    for row in selected:
        ledger.authorize(str(row.family))
        population = _population_row(plan, row)
        sidecar_path = (
            plan.sidecar_root / str(population["sidecar_relative_path"])
        ).resolve()
        _require(
            sidecar_path.is_relative_to(plan.sidecar_root),
            "sidecar path escapes the sealed root",
        )
        parent_identity = _stable_file_identity(
            row.path,
            expected_size=int(row.size_bytes),
            expected_sha256=str(row.sha256),
        )
        sidecar_identity = _stable_file_identity(
            sidecar_path,
            expected_size=int(population["sidecar_size_bytes"]),
            expected_sha256=str(population["sidecar_file_sha256"]),
        )
        projection = selected_loader(plan, row, include_labels=True)
        _require(
            projection.labels is not None
            and projection.fmt_features.shape == (projection.count, 161)
            and projection.seed_kinematic4.shape == (projection.count, 4)
            and projection.sidecar_file_sha256 == sidecar_identity["sha256"],
            "fit projection contract drifted",
        )
        ledger.append(
            row=row,
            parent_identity=parent_identity,
            sidecar_identity=sidecar_identity,
            joined_row_count=projection.count,
        )
        projections.append(projection)
    ledger.validate()
    return projections


def _natural_class_scale_counts(projections: Sequence[Any]) -> np.ndarray:
    """Count every natural family/class/exact-scale row before scaler filtering."""

    counts = np.zeros((len(FIT_FAMILIES), len(CLASS_NAMES), SCALE_COUNT), dtype=np.int64)
    family_index = {family: index for index, family in enumerate(FIT_FAMILIES)}
    for projection in projections:
        labels = np.asarray(projection.labels)
        scales = np.asarray(projection.scale_ids, dtype=np.int64)
        _require(
            labels.dtype == np.dtype(np.bool_)
            and labels.shape == scales.shape
            and len(labels) == projection.count,
            "fit labels or scales drifted",
        )
        index = family_index[str(projection.row.family)]
        _require(
            np.all((scales >= 0) & (scales < SCALE_COUNT)),
            "fit scale IDs are outside the frozen exact-scale domain",
        )
        for class_index, positive in enumerate((False, True)):
            counts[index, class_index] += np.bincount(
                scales[labels == positive], minlength=SCALE_COUNT
            ).astype(np.int64, copy=False)
    _require(
        int(counts[:, 0].sum(dtype=np.int64)) > 0,
        "fit families contain no natural-negative template",
    )
    return counts


def _effective_class_scale_counts(natural_counts: np.ndarray) -> np.ndarray:
    """Apply the pooled-natural-negative scaler support mask to both classes."""

    raw = np.asarray(natural_counts)
    _require(
        raw.shape == (len(FIT_FAMILIES), len(CLASS_NAMES), SCALE_COUNT)
        and raw.dtype == np.dtype(np.int64)
        and np.all(raw >= 0),
        "natural family/class exact-scale counts drifted",
    )
    pooled_negative_counts = raw[:, 0].sum(axis=0, dtype=np.int64)
    _require(
        int(pooled_negative_counts.sum(dtype=np.int64)) > 0,
        "fit families contain no natural-negative template",
    )
    scaler_supported = pooled_negative_counts > 0
    counts = np.where(scaler_supported[None, None, :], raw, 0).astype(
        np.int64, copy=False
    )
    _require(
        np.array_equal(
            counts[:, 0].sum(axis=0, dtype=np.int64), pooled_negative_counts
        ),
        "supported negative counts do not reproduce the pooled scaler population",
    )
    return counts


def _expected_class_scale_counts(projections: Sequence[Any]) -> np.ndarray:
    """Compatibility helper reproducing the effective fitted population."""

    return _effective_class_scale_counts(_natural_class_scale_counts(projections))


def _array_record(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    finite = True
    if array.dtype.kind in "fc":
        finite = bool(np.isfinite(array).all())
    _require(finite, "constructed model artifact contains a non-finite array")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "size_bytes": int(array.nbytes),
        "sha256": canonical_array_sha256(array),
        "finite_or_nonfloating": True,
    }


def _validate_persisted_array_record(
    value: object,
    *,
    name: str,
    expected_dtype: str,
    expected_shape: tuple[int, ...],
) -> Mapping[str, Any]:
    _require(
        isinstance(value, Mapping) and set(value) == ARRAY_RECORD_FIELDS,
        f"constructed array record field set drifted: {name}",
    )
    dtype_text = value.get("dtype")
    shape = value.get("shape")
    size = value.get("size_bytes")
    try:
        dtype = np.dtype(dtype_text)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"constructed array dtype is invalid: {name}") from error
    _require(
        isinstance(shape, list)
        and all(
            isinstance(dimension, int)
            and not isinstance(dimension, bool)
            and dimension >= 0
            for dimension in shape
        )
        and tuple(shape) == expected_shape
        and isinstance(size, int)
        and not isinstance(size, bool)
        and size >= 0
        and size == int(np.prod(expected_shape, dtype=np.int64)) * dtype.itemsize
        and (
            dtype.kind == "U"
            if expected_dtype == "unicode"
            else dtype.str == expected_dtype
        )
        and _lower_hex(value.get("sha256"), length=64)
        and value.get("finite_or_nonfloating") is True,
        f"constructed array record dtype/shape/size/hash drifted: {name}",
    )
    return value


def _validate_constructed_model_artifact_audit(constructed: object) -> None:
    """Authenticate the complete persisted array/class audit structure.

    Array byte hashes cannot be recomputed without reopening the intentionally
    discarded model arrays.  This gate therefore authenticates every record's
    exact schema, dtype, shape, byte arithmetic, and hash syntax, then binds all
    counts/hashes that remain derivable from the persisted class summaries.
    """

    _require(
        isinstance(constructed, Mapping)
        and set(constructed) == CONSTRUCTED_MODEL_FIELDS,
        "constructed-model field set drifted",
    )
    class_audits = constructed.get("class_library_and_reference_audits")
    _require(
        isinstance(class_audits, Mapping) and set(class_audits) == set(FIT_FAMILIES),
        "constructed class-library family set drifted",
    )
    total_library = 0
    total_negative = 0
    total_loo = 0
    total_zero_loo = 0
    class_present = np.zeros(
        (len(FIT_FAMILIES), len(CLASS_NAMES)), dtype=np.bool_
    )
    class_counts: dict[tuple[int, int], tuple[int, int]] = {}
    for family_index, family in enumerate(FIT_FAMILIES):
        family_audit = class_audits[family]
        _require(
            isinstance(family_audit, Mapping)
            and set(family_audit) == set(CLASS_NAMES),
            f"constructed class set drifted: {family}",
        )
        for class_index, class_name in enumerate(CLASS_NAMES):
            record = family_audit[class_name]
            _require(
                isinstance(record, Mapping)
                and set(record) == CLASS_LIBRARY_AUDIT_FIELDS,
                f"constructed class audit field set drifted: {family}/{class_name}",
            )
            count_names = (
                "library_row_count",
                "library_supported_scale_count_k31",
                "loo_reference_row_count_k31",
                "loo_supported_scale_count_k31",
                "zero_distance_loo_reference_count_k31",
            )
            _require(
                all(
                    isinstance(record.get(name), int)
                    and not isinstance(record.get(name), bool)
                    and int(record[name]) >= 0
                    for name in count_names
                )
                and record["library_supported_scale_count_k31"]
                <= int(constructed["effective_retained_exact_scale_count"])
                and record["loo_supported_scale_count_k31"]
                <= record["library_supported_scale_count_k31"]
                and record["library_supported_scale_count_k31"] * K
                <= record["library_row_count"]
                and record["loo_supported_scale_count_k31"] * (K + 1)
                <= record["loo_reference_row_count_k31"]
                and (record["loo_reference_row_count_k31"] > 0)
                == (record["loo_supported_scale_count_k31"] > 0)
                and record["loo_reference_row_count_k31"]
                <= record["library_row_count"]
                and record["zero_distance_loo_reference_count_k31"]
                <= record["loo_reference_row_count_k31"]
                and record.get("exact_self_exclusion_count_identity_passed") is True
                and record.get(
                    "duplicate_rows_retained_in_count_and_zero_distance_references"
                )
                is True
                and _lower_hex(record.get("class_scale_counts_sha256"), length=64),
                f"constructed class audit counts/hash drifted: {family}/{class_name}",
            )
            library_rows = int(record["library_row_count"])
            loo_rows = int(record["loo_reference_row_count_k31"])
            if library_rows == 0:
                _require(
                    all(int(record[name]) == 0 for name in count_names),
                    f"absent constructed class has nonzero support: {family}/{class_name}",
                )
            class_present[family_index, class_index] = library_rows > 0
            class_counts[(family_index, class_index)] = (library_rows, loo_rows)
            total_library += library_rows
            total_loo += loo_rows
            total_zero_loo += int(record["zero_distance_loo_reference_count_k31"])
            if class_name == "negative":
                total_negative += library_rows

    _require(
        total_library == constructed.get("full_family_class_library_row_count")
        == constructed.get("effective_retained_family_class_library_row_count")
        and total_negative == constructed.get("shared_negative_row_count")
        and total_loo == constructed.get("loo_reference_row_count_k31")
        and total_zero_loo
        == constructed.get("zero_distance_loo_reference_count_k31"),
        "constructed class summaries do not reproduce top-level totals",
    )

    scaler = constructed.get("scaler_arrays")
    _require(
        isinstance(scaler, Mapping)
        and bool(scaler)
        and set(scaler) == set(SCALER_ARRAY_SPECS),
        "constructed scaler array member set drifted",
    )
    for name, (dtype, shape) in SCALER_ARRAY_SPECS.items():
        _validate_persisted_array_record(
            scaler[name], name=f"scaler/{name}", expected_dtype=dtype, expected_shape=shape
        )
    _require(
        scaler["scale_id_int32"]["sha256"]
        == canonical_array_sha256(np.arange(SCALE_COUNT, dtype=np.int32)),
        "constructed scaler exact scale-ID content hash drifted",
    )

    calibrator = constructed.get("family_class_library_and_calibration_arrays")
    _require(isinstance(calibrator, Mapping) and bool(calibrator), "empty calibrator audit")
    expected_specs = dict(CALIBRATOR_BASE_ARRAY_SPECS)
    member_specs = {
        "serialization_version": ("<i2", ()),
        "negative_features": ("<f4", None),
        "negative_scale_offsets": ("<i8", (SCALE_COUNT + 1,)),
        "mean": ("<f8", (FEATURE_WIDTH,)),
        "raw_std": ("<f8", (FEATURE_WIDTH,)),
        "effective_std": ("<f8", (FEATURE_WIDTH,)),
        "zero_variance_feature_mask": ("|b1", (FEATURE_WIDTH,)),
        "ks": ("<i8", (1,)),
        "shrinkage_lambda": ("<f8", ()),
        f"loo_distances_k_{K}": ("<f4", None),
        f"loo_scale_offsets_k_{K}": ("<i8", (SCALE_COUNT + 1,)),
    }
    for (family_index, class_index), (library_rows, loo_rows) in class_counts.items():
        if library_rows == 0:
            continue
        prefix = f"calibrator_f{family_index}_c{class_index}__"
        for suffix, (dtype, shape) in member_specs.items():
            if suffix == "negative_features":
                resolved_shape = (library_rows, FEATURE_WIDTH)
            elif suffix == f"loo_distances_k_{K}":
                resolved_shape = (loo_rows,)
            else:
                assert shape is not None
                resolved_shape = shape
            expected_specs[f"{prefix}{suffix}"] = (dtype, resolved_shape)
    _require(
        set(calibrator) == set(expected_specs),
        "constructed family/class calibrator array member set drifted",
    )
    for name, (dtype, shape) in expected_specs.items():
        _validate_persisted_array_record(
            calibrator[name],
            name=f"calibrator/{name}",
            expected_dtype=dtype,
            expected_shape=shape,
        )
    expected_k_sha = canonical_array_sha256(np.asarray([K], dtype=np.int64))
    expected_lambda_sha = canonical_array_sha256(
        np.asarray(64.0, dtype=np.float64)
    )
    expected_serialization_sha = canonical_array_sha256(
        np.asarray(1, dtype=np.int16)
    )
    for (family_index, class_index), (library_rows, _loo_rows) in class_counts.items():
        if library_rows == 0:
            continue
        prefix = f"calibrator_f{family_index}_c{class_index}__"
        _require(
            calibrator[f"{prefix}serialization_version"]["sha256"]
            == calibrator["serialization_version_int16"]["sha256"]
            and calibrator[f"{prefix}ks"]["sha256"] == expected_k_sha
            and calibrator[f"{prefix}shrinkage_lambda"]["sha256"]
            == expected_lambda_sha,
            f"constructed calibrator member metadata drifted: {prefix}",
        )
    _require(
        calibrator["class_scale_counts_int64"]["sha256"]
        == constructed.get("effective_retained_class_scale_counts_sha256")
        and calibrator["serialization_version_int16"]["sha256"]
        == expected_serialization_sha
        and calibrator["class_present_bool"]["sha256"]
        == canonical_array_sha256(class_present)
        and calibrator["family_order_unicode"]["dtype"]
        == calibrator["family_order_copy_unicode"]["dtype"]
        and calibrator["family_order_unicode"]["sha256"]
        == calibrator["family_order_copy_unicode"]["sha256"]
        == canonical_array_sha256(
            np.asarray(
                FIT_FAMILIES,
                dtype=np.dtype(calibrator["family_order_unicode"]["dtype"]),
            )
        )
        and calibrator["ks_int64"]["sha256"] == expected_k_sha
        and calibrator["required_family_count_int64"]["sha256"]
        == canonical_array_sha256(np.asarray(2, dtype=np.int64))
        and calibrator["shrinkage_lambda_float64"]["sha256"]
        == expected_lambda_sha,
        "constructed calibrator derivable content binding drifted",
    )


def _audit_model_arrays(
    model: ClassConditionalTemplateScoreModel,
    natural_counts: np.ndarray,
    expected_counts: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    _require(
        model.family_order == FIT_FAMILIES
        and model.ks == (K,)
        and model.required_family_count == 2,
        "fit model family/k/support contract drifted",
    )
    raw_counts = np.asarray(natural_counts)
    effective_counts = np.asarray(expected_counts)
    _require(
        np.array_equal(
            effective_counts, _effective_class_scale_counts(raw_counts)
        ),
        "effective counts do not equal natural counts masked by scaler support",
    )
    raw_present_scales = np.flatnonzero(
        raw_counts.sum(axis=(0, 1), dtype=np.int64) > 0
    ).astype(np.int64)
    effective_present_scales = np.flatnonzero(
        effective_counts.sum(axis=(0, 1), dtype=np.int64) > 0
    ).astype(np.int64)
    raw_only_no_scaler_scales = np.setdiff1d(
        raw_present_scales, effective_present_scales, assume_unique=True
    ).astype(np.int64, copy=False)
    scaler_arrays = model.scaler.export_arrays()
    calibrator_arrays = model.tail_calibrator.export_arrays()
    observed_counts = np.asarray(calibrator_arrays["class_scale_counts_int64"])
    _require(
        np.array_equal(observed_counts, expected_counts),
        "family/class exact-scale library counts drifted",
    )
    _require(
        np.array_equal(
            np.asarray(calibrator_arrays["class_present_bool"]),
            expected_counts.sum(axis=2, dtype=np.int64) > 0,
        ),
        "family/class presence metadata drifted",
    )
    _require(
        np.array_equal(
            np.asarray(scaler_arrays["local_row_count_int64"]),
            expected_counts[:, 0].sum(axis=0, dtype=np.int64),
        ),
        "shared-negative scaler counts drifted",
    )

    class_audits: dict[str, Any] = {}
    total_library_rows = 0
    total_loo_rows = 0
    total_zero_loo = 0
    for family_index, family in enumerate(FIT_FAMILIES):
        family_audit: dict[str, Any] = {}
        for class_index, class_name in enumerate(CLASS_NAMES):
            prefix = f"calibrator_f{family_index}_c{class_index}__"
            counts = expected_counts[family_index, class_index]
            expected_loo = np.where(counts >= K + 1, counts, 0)
            library_rows = int(counts.sum(dtype=np.int64))
            loo_rows = int(expected_loo.sum(dtype=np.int64))
            if library_rows == 0:
                _require(
                    not any(name.startswith(prefix) for name in calibrator_arrays),
                    f"{family}/{class_name}: absent class unexpectedly has artifact members",
                )
                zero_loo = 0
            else:
                offsets = np.asarray(
                    calibrator_arrays[f"{prefix}negative_scale_offsets"],
                    dtype=np.int64,
                )
                loo_offsets = np.asarray(
                    calibrator_arrays[f"{prefix}loo_scale_offsets_k_{K}"],
                    dtype=np.int64,
                )
                loo = np.asarray(
                    calibrator_arrays[f"{prefix}loo_distances_k_{K}"],
                    dtype=np.float32,
                )
                features = np.asarray(
                    calibrator_arrays[f"{prefix}negative_features"]
                )
                _require(
                    np.array_equal(np.diff(offsets), counts)
                    and np.array_equal(np.diff(loo_offsets), expected_loo)
                    and features.shape == (library_rows, FEATURE_WIDTH)
                    and len(loo) == loo_rows,
                    f"{family}/{class_name}: library or LOO support counts drifted",
                )
                _require(
                    np.isfinite(features).all()
                    and np.isfinite(loo).all()
                    and np.all(loo >= 0.0),
                    f"{family}/{class_name}: library or LOO values are invalid",
                )
                for scale_id in np.flatnonzero(expected_loo):
                    start = int(loo_offsets[scale_id])
                    stop = int(loo_offsets[scale_id + 1])
                    _require(
                        np.all(loo[start : stop - 1] <= loo[start + 1 : stop])
                        if stop - start > 1
                        else True,
                        f"{family}/{class_name}: LOO references are not stably sorted",
                    )
                zero_loo = int(np.count_nonzero(loo == 0.0))
            total_library_rows += library_rows
            total_loo_rows += loo_rows
            total_zero_loo += zero_loo
            family_audit[class_name] = {
                "library_row_count": library_rows,
                "library_supported_scale_count_k31": int(np.count_nonzero(counts >= K)),
                "loo_reference_row_count_k31": loo_rows,
                "loo_supported_scale_count_k31": int(np.count_nonzero(expected_loo)),
                "zero_distance_loo_reference_count_k31": zero_loo,
                "exact_self_exclusion_count_identity_passed": True,
                "duplicate_rows_retained_in_count_and_zero_distance_references": True,
                "class_scale_counts_sha256": canonical_array_sha256(counts),
            }
        class_audits[family] = family_audit

    scaler_records = {
        name: _array_record(values) for name, values in sorted(scaler_arrays.items())
    }
    calibrator_records = {
        name: _array_record(values)
        for name, values in sorted(calibrator_arrays.items())
    }
    return (
        {
            "family_order": list(FIT_FAMILIES),
            "feature_width": FEATURE_WIDTH,
            "k": K,
            "strict_majority_family_count": 2,
            "natural_raw_family_class_row_count": int(raw_counts.sum()),
            "effective_retained_family_class_library_row_count": int(
                effective_counts.sum()
            ),
            "natural_raw_present_exact_scale_count": int(len(raw_present_scales)),
            "effective_retained_exact_scale_count": int(
                len(effective_present_scales)
            ),
            "natural_raw_only_no_scaler_exact_scale_count": int(
                len(raw_only_no_scaler_scales)
            ),
            "natural_raw_present_exact_scale_ids_sha256": canonical_array_sha256(
                raw_present_scales
            ),
            "effective_retained_exact_scale_ids_sha256": canonical_array_sha256(
                effective_present_scales
            ),
            "natural_raw_only_no_scaler_exact_scale_ids_sha256": canonical_array_sha256(
                raw_only_no_scaler_scales
            ),
            "natural_raw_class_scale_counts_sha256": canonical_array_sha256(
                raw_counts
            ),
            "effective_retained_class_scale_counts_sha256": canonical_array_sha256(
                effective_counts
            ),
            "shared_negative_row_count": int(expected_counts[:, 0].sum()),
            "full_family_class_library_row_count": total_library_rows,
            "loo_reference_row_count_k31": total_loo_rows,
            "zero_distance_loo_reference_count_k31": total_zero_loo,
            "class_library_and_reference_audits": class_audits,
            "scaler_arrays": scaler_records,
            "family_class_library_and_calibration_arrays": calibrator_records,
            "all_constructed_arrays_finite": True,
            "exact_self_exclusion_duplicate_and_support_count_audits_passed": True,
        },
        scaler_arrays,
        calibrator_arrays,
    )


def _deterministic_synthetic_queries(scale_ids: np.ndarray) -> np.ndarray:
    scales = np.asarray(scale_ids, dtype=np.int64)
    _require(scales.ndim == 1 and len(scales) > 0, "synthetic scale set is empty")
    # Values depend only on frozen integer indices, never on fitted rows or labels.
    row_term = ((scales % 17) - 8).astype(np.float32)[:, None] * np.float32(1e-3)
    column_term = (
        ((np.arange(FEATURE_WIDTH, dtype=np.int64) % 13) - 6).astype(np.float32)[
            None, :
        ]
        * np.float32(1e-4)
    )
    result = np.ascontiguousarray(row_term + column_term, dtype=np.float32)
    _require(np.isfinite(result).all(), "synthetic query construction failed")
    return result


def _exercise_synthetic_query_path(
    model: ClassConditionalTemplateScoreModel,
    natural_counts: np.ndarray,
    expected_counts: np.ndarray,
) -> dict[str, Any]:
    raw_counts = np.asarray(natural_counts)
    effective_counts = np.asarray(expected_counts)
    _require(
        np.array_equal(
            effective_counts, _effective_class_scale_counts(raw_counts)
        ),
        "synthetic query effective counts drifted from natural count support",
    )
    scaler_counts = np.asarray(model.scaler.local_row_counts, dtype=np.int64)
    # The query domain is the raw union before scaler filtering.  In particular,
    # a positive-only scale must not disappear merely because it cannot enter a
    # scaler-supported family/class library.
    scales = np.flatnonzero(
        raw_counts.sum(axis=(0, 1), dtype=np.int64) > 0
    ).astype(np.int64)
    effective_scales = np.flatnonzero(
        effective_counts.sum(axis=(0, 1), dtype=np.int64) > 0
    ).astype(np.int64)
    raw_only_no_scaler_scales = np.setdiff1d(
        scales, effective_scales, assume_unique=True
    ).astype(np.int64, copy=False)
    features = _deterministic_synthetic_queries(scales)
    result = model.query(
        features,
        scales,
        ks=(K,),
        device="cpu",
        query_chunk_size=QUERY_CHUNK_SIZE,
        library_chunk_size=LIBRARY_CHUNK_SIZE,
    )

    positive_retrieval = np.asarray(
        result.per_family_positive_retrieval_supported[K], dtype=np.bool_
    )
    positive_calibration = np.asarray(
        result.per_family_positive_calibration_supported[K], dtype=np.bool_
    )
    negative_retrieval = np.asarray(
        result.per_family_negative_retrieval_supported[K], dtype=np.bool_
    )
    negative_calibration = np.asarray(
        result.per_family_negative_calibration_supported[K], dtype=np.bool_
    )
    scaler_supported = scaler_counts[scales] > 0
    expected_retrieval = (effective_counts[:, :, scales] >= K) & scaler_supported
    expected_loo = np.where(effective_counts >= K + 1, effective_counts, 0)
    any_reference = expected_loo.sum(axis=2) > 0
    expected_calibration = expected_retrieval & any_reference[:, :, None]
    _require(
        np.array_equal(negative_retrieval, expected_retrieval[:, 0].T)
        and np.array_equal(positive_retrieval, expected_retrieval[:, 1].T)
        and np.array_equal(negative_calibration, expected_calibration[:, 0].T)
        and np.array_equal(positive_calibration, expected_calibration[:, 1].T),
        "synthetic query per-family support differs from fitted count arithmetic",
    )
    expected_joint_matrix = (
        expected_retrieval[:, 0]
        & expected_calibration[:, 0]
        & expected_retrieval[:, 1]
        & expected_calibration[:, 1]
    ).T
    expected_joint_count = expected_joint_matrix.sum(axis=1).astype(np.int16)
    expected_joint = expected_joint_count >= model.required_family_count
    expected_retrieval_majority = (
        (expected_retrieval[:, 0] & expected_retrieval[:, 1]).T.sum(axis=1)
        >= model.required_family_count
    )
    _require(
        np.array_equal(result.joint_family_count[K], expected_joint_count)
        and np.array_equal(result.joint_supported[K], expected_joint)
        and np.array_equal(result.retrieval_supported[K], expected_retrieval_majority),
        "synthetic query aggregate support differs from fitted count arithmetic",
    )
    raw_only_mask = np.isin(scales, raw_only_no_scaler_scales, assume_unique=True)
    _require(
        not np.any(scaler_supported[raw_only_mask])
        and not np.any(negative_retrieval[raw_only_mask])
        and not np.any(positive_retrieval[raw_only_mask])
        and not np.any(negative_calibration[raw_only_mask])
        and not np.any(positive_calibration[raw_only_mask]),
        "raw-only no-scaler scale unexpectedly supports a class/family query path",
    )
    scores = np.asarray(result.scores[K], dtype=np.float64)
    distances = np.asarray(result.mean_negative_distances[K], dtype=np.float32)
    _require(
        np.isfinite(scores).all()
        and np.all((scores >= 0.0) & (scores <= 1.0))
        and np.isfinite(distances[expected_retrieval_majority]).all()
        and np.isnan(distances[~expected_retrieval_majority]).all(),
        "synthetic query numerical sentinels drifted",
    )
    histogram = np.bincount(
        expected_joint_count.astype(np.int64), minlength=len(FIT_FAMILIES) + 1
    )
    return {
        "construction": "deterministic_integer_index_formula_independent_of_fit_rows_and_labels",
        "query_row_count": int(len(scales)),
        "query_scale_domain": "union_of_all_natural_raw_exact_scales_before_scaler_filter",
        "all_natural_raw_present_exact_scales_exercised": True,
        "natural_raw_present_exact_scale_count": int(len(scales)),
        "effective_retained_exact_scale_count": int(len(effective_scales)),
        "natural_raw_only_no_scaler_exact_scale_count": int(
            len(raw_only_no_scaler_scales)
        ),
        "natural_raw_present_exact_scale_ids_sha256": canonical_array_sha256(
            scales
        ),
        "effective_retained_exact_scale_ids_sha256": canonical_array_sha256(
            effective_scales
        ),
        "natural_raw_only_no_scaler_exact_scale_ids_sha256": canonical_array_sha256(
            raw_only_no_scaler_scales
        ),
        "all_natural_raw_only_no_scaler_scales_have_all_class_family_retrieval_and_calibration_unsupported": True,
        "strict_majority_joint_supported_row_count": int(expected_joint.sum()),
        "strict_majority_retrieval_supported_row_count": int(
            expected_retrieval_majority.sum()
        ),
        "joint_family_count_histogram": {
            str(index): int(value) for index, value in enumerate(histogram)
        },
        "support_count_arithmetic_passed": True,
        "all_supported_numerical_values_finite": True,
        "reference_labels_consulted_by_query_path": False,
    }


def _synthetic_core_contract_gate(*, feature_width: int = FEATURE_WIDTH) -> dict[str, Any]:
    """Exercise self-exclusion, duplicate retention, round trip, and strict tie."""

    _require(feature_width >= 2, "synthetic core feature width is too small")
    family_batches: dict[str, FamilyFitBatch] = {}
    for family_index, family in enumerate(FIT_FAMILIES):
        one_class = np.zeros((K + 1, feature_width), dtype=np.float32)
        one_class[-1, 0] = np.float32(1.0 + family_index * 0.125)
        negative = one_class.copy()
        positive = one_class.copy()
        positive_only_no_scaler = one_class.copy()
        features = np.concatenate(
            (negative, positive, positive_only_no_scaler), axis=0
        )
        scales = np.concatenate(
            (
                np.zeros(2 * (K + 1), dtype=np.int64),
                np.ones(K + 1, dtype=np.int64),
            )
        )
        labels = np.concatenate(
            (
                np.zeros(K + 1, dtype=np.bool_),
                np.ones(2 * (K + 1), dtype=np.bool_),
            )
        )
        family_batches[family] = FamilyFitBatch(features, scales, labels)
    model = ClassConditionalTemplateScoreModel(
        family_batches,
        family_order=FIT_FAMILIES,
        ks=(K,),
        shrinkage_lambda=64.0,
        device="cpu",
        query_chunk_size=64,
        library_chunk_size=64,
    )
    calibrator_arrays = model.tail_calibrator.export_arrays()
    positive_loo_minima: list[float] = []
    for family_index in range(len(FIT_FAMILIES)):
        for class_index in range(len(CLASS_NAMES)):
            prefix = f"calibrator_f{family_index}_c{class_index}__"
            loo = np.asarray(
                calibrator_arrays[f"{prefix}loo_distances_k_{K}"],
                dtype=np.float32,
            )
            offsets = np.asarray(
                calibrator_arrays[f"{prefix}loo_scale_offsets_k_{K}"],
                dtype=np.int64,
            )
            _require(
                len(loo) == K + 1
                and int(offsets[1]) == K + 1
                and np.all(loo > 0.0),
                "synthetic self-exclusion/duplicate contract failed",
            )
            positive_loo_minima.append(float(loo.min()))
    query_features = np.zeros((1, feature_width), dtype=np.float32)
    query_scales = np.zeros(1, dtype=np.int64)
    before = model.query(query_features, query_scales, ks=(K,), device="cpu")
    scaler_arrays = model.scaler.export_arrays()
    restored = ClassConditionalTemplateScoreModel.from_artifacts(
        scaler_arrays, calibrator_arrays
    )
    after = restored.query(query_features, query_scales, ks=(K,), device="cpu")
    _require(
        np.array_equal(before.scores[K], after.scores[K])
        and np.array_equal(before.joint_supported[K], after.joint_supported[K])
        and bool(before.joint_supported[K][0])
        and float(before.scores[K][0]) == 0.5,
        "synthetic core artifact round trip drifted",
    )
    raw_domain_query = restored.query(
        np.zeros((2, feature_width), dtype=np.float32),
        np.asarray([0, 1], dtype=np.int64),
        ks=(K,),
        device="cpu",
    )
    unsupported_index = 1
    _require(
        not np.any(
            raw_domain_query.per_family_negative_retrieval_supported[K][
                unsupported_index
            ]
        )
        and not np.any(
            raw_domain_query.per_family_positive_retrieval_supported[K][
                unsupported_index
            ]
        )
        and not np.any(
            raw_domain_query.per_family_negative_calibration_supported[K][
                unsupported_index
            ]
        )
        and not np.any(
            raw_domain_query.per_family_positive_calibration_supported[K][
                unsupported_index
            ]
        ),
        "positive-only exact scale without a pooled negative scaler gained support",
    )
    tie_decision = strict_threshold_predictions(
        before.scores[K], before.joint_supported[K], threshold=0.5
    )
    _require(not bool(tie_decision[0]), "strict score tie must remain negative")
    return {
        "feature_width": int(feature_width),
        "family_count": len(FIT_FAMILIES),
        "rows_per_family_class": K + 1,
        "self_exclusion_forces_positive_k31_distance": all(
            value > 0.0 for value in positive_loo_minima
        ),
        "thirty_one_equal_duplicates_retained_per_family_class": True,
        "artifact_round_trip_exact": True,
        "strict_threshold_tie_contract_passed": True,
        "positive_only_no_scaler_exact_scale_all_class_family_support_false": True,
    }


def _peak_rss_bytes() -> int:
    _require(
        sys.platform.startswith("linux") and _resource is not None,
        "production resource smoke requires Linux getrusage",
    )
    # Linux ru_maxrss is KiB.  The smoke is never run as a child process.
    return int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss) * 1024


def _elapsed_seconds(process_started_monotonic: float) -> float:
    """Return the larger of Python elapsed time and wrapper-stage elapsed time."""

    process_elapsed = time.monotonic() - process_started_monotonic
    wrapper_start = os.environ.get("PTM_RESOURCE_SMOKE_WRAPPER_START_EPOCH")
    if wrapper_start is None:
        return process_elapsed
    try:
        wrapper_elapsed = time.time() - float(wrapper_start)
    except ValueError as error:
        raise RuntimeError("resource-smoke wrapper start time is invalid") from error
    _require(wrapper_elapsed >= 0.0, "resource-smoke wrapper clock moved backwards")
    return max(process_elapsed, wrapper_elapsed)


def _validated_scontrol_allocation(
    record: object,
    *,
    expected_job_id: str,
    expected_time_limit: str,
) -> dict[str, Any]:
    """Parse authoritative ``scontrol show job -o`` allocation fields."""

    _require(
        isinstance(record, str) and "\n" not in record and "\r" not in record,
        "scontrol job record must be one line",
    )
    record = record.strip()
    _require(bool(record), "scontrol job record must be nonempty after trimming")
    values: dict[str, str] = {}
    for name in (
        "JobId",
        "Partition",
        "Account",
        "NumNodes",
        "TimeLimit",
        "Features",
        "AllocTRES",
        "ReqTRES",
        "TresPerNode",
        "Gres",
    ):
        matches = re.findall(rf"(?:^|\s){re.escape(name)}=([^\s]*)", record)
        _require(len(matches) <= 1, f"scontrol field is duplicated: {name}")
        if matches:
            values[name] = matches[0]
    _require(
        set(
            (
                "JobId",
                "Partition",
                "Account",
                "NumNodes",
                "TimeLimit",
                "Features",
                "AllocTRES",
                "ReqTRES",
            )
        )
        <= set(values),
        "scontrol job record is missing an authoritative allocation field",
    )
    _require(
        values["JobId"] == expected_job_id,
        "scontrol JobId differs from SLURM_JOB_ID",
    )
    _require(
        values["Partition"] == RUNTIME_SLURM_PARTITION,
        f"authoritative scontrol Partition must be exactly {RUNTIME_SLURM_PARTITION}",
    )
    _require(
        values["Account"] == RUNTIME_SLURM_ACCOUNT,
        f"authoritative scontrol Account must be exactly {RUNTIME_SLURM_ACCOUNT}",
    )
    _require(
        values["NumNodes"] == "1", "Slurm allocation must contain exactly one node"
    )
    _require(
        values["TimeLimit"] == expected_time_limit,
        f"Slurm TimeLimit must be exactly {expected_time_limit}",
    )
    _require(
        values["Features"].casefold() == "rome",
        "authoritative scontrol Features must be exactly rome",
    )
    gpu_surfaces = {
        name: values.get(name, "(not_set)")
        for name in ("AllocTRES", "ReqTRES", "TresPerNode", "Gres")
    }
    _require(
        all("gpu" not in value.casefold() for value in gpu_surfaces.values()),
        "Slurm allocation/request contains a GPU resource",
    )
    _require(
        "node=1" in values["AllocTRES"].casefold(),
        "scontrol AllocTRES does not bind the one-node allocation",
    )
    return {
        "source": "scontrol_show_job_one_line",
        "job_id": values["JobId"],
        "partition": values["Partition"],
        "account": values["Account"],
        "num_nodes": 1,
        "time_limit": values["TimeLimit"],
        "features": values["Features"],
        "gpu_allocation": "none",
        "alloc_tres": gpu_surfaces["AllocTRES"],
        "req_tres": gpu_surfaces["ReqTRES"],
        "tres_per_node": gpu_surfaces["TresPerNode"],
        "gres": gpu_surfaces["Gres"],
        "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
    }


def _validate_resource_limits(*, peak_rss_bytes: int, elapsed_seconds: float) -> None:
    _require(
        isinstance(peak_rss_bytes, int)
        and not isinstance(peak_rss_bytes, bool)
        and 0 <= peak_rss_bytes < MEMORY_LIMIT_BYTES,
        "resource smoke peak RSS reached the strict 128 GiB limit",
    )
    _require(
        np.isfinite(elapsed_seconds)
        and 0.0 <= elapsed_seconds <= WALLTIME_LIMIT_SECONDS,
        "resource smoke exceeded the four-hour limit",
    )


def _runtime_environment_audit() -> dict[str, Any]:
    _require(sys.platform.startswith("linux"), "resource smoke must run on Linux")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", "")
    _require(job_id.isdigit(), "resource smoke must run inside a Slurm job")
    _require(cpus.isdigit(), "resource smoke CPU allocation is missing or invalid")
    allocation = _validated_scontrol_allocation(
        os.environ.get("PTM_SLURM_SCONTROL_JOB_RECORD"),
        expected_job_id=job_id,
        expected_time_limit=RESOURCE_SMOKE_TIME_LIMIT,
    )
    for name in ("SLURM_JOB_NUM_NODES", "SLURM_NNODES"):
        value = os.environ.get(name)
        _require(
            value is None or value == str(allocation["num_nodes"]),
            f"{name} differs from the authoritative scontrol node count",
        )
    _require(
        os.environ.get("SLURM_JOB_PARTITION", "") == allocation["partition"],
        "SLURM_JOB_PARTITION differs from authoritative scontrol Partition",
    )
    _require(
        os.environ.get("SLURM_JOB_ACCOUNT", "") == allocation["account"],
        "SLURM_JOB_ACCOUNT differs from authoritative scontrol Account",
    )
    for name in ("SLURM_JOB_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE"):
        value = os.environ.get(name)
        _require(
            value is None
            or value.strip().casefold() in {"", "0", "none", "(null)", "n/a"},
            f"{name} indicates an unexpected GPU allocation",
        )
    cpu_model = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[-1].strip()
                break
    result = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "slurm_job_id": job_id,
        "slurm_cpus_per_task": int(cpus),
        "slurm_job_partition": allocation["partition"],
        "slurm_job_account": allocation["account"],
        "slurm_scontrol_features": allocation["features"],
        "slurm_memory_per_node": os.environ.get("SLURM_MEM_PER_NODE", ""),
        "slurm_num_nodes": allocation["num_nodes"],
        "slurm_time_limit": allocation["time_limit"],
        "slurm_gpu_allocation": allocation["gpu_allocation"],
        "slurm_scontrol_source": allocation["source"],
        "slurm_scontrol_record_sha256": allocation["record_sha256"],
        "slurm_scontrol_alloc_tres": allocation["alloc_tres"],
        "slurm_scontrol_req_tres": allocation["req_tres"],
        "slurm_scontrol_tres_per_node": allocation["tres_per_node"],
        "slurm_scontrol_gres": allocation["gres"],
        "cpu_model": cpu_model,
        "requested_device": "cpu",
        "gpu_requested": False,
    }
    _validate_frozen_slurm_runtime_payload(result)
    return result


def _validate_frozen_slurm_runtime_payload(runtime: Mapping[str, Any]) -> None:
    """Authenticate the actual Slurm allocation against the frozen smoke request."""

    expected_fields = {
        "platform",
        "python",
        "slurm_job_id",
        "slurm_cpus_per_task",
        "slurm_job_partition",
        "slurm_job_account",
        "slurm_scontrol_features",
        "slurm_memory_per_node",
        "slurm_num_nodes",
        "slurm_time_limit",
        "slurm_gpu_allocation",
        "slurm_scontrol_source",
        "slurm_scontrol_record_sha256",
        "slurm_scontrol_alloc_tres",
        "slurm_scontrol_req_tres",
        "slurm_scontrol_tres_per_node",
        "slurm_scontrol_gres",
        "cpu_model",
        "requested_device",
        "gpu_requested",
    }
    constraint_value = runtime.get("slurm_scontrol_features")
    memory_value = runtime.get("slurm_memory_per_node")
    _require(
        isinstance(runtime, Mapping)
        and set(runtime) == expected_fields
        and isinstance(runtime.get("platform"), str)
        and bool(runtime["platform"])
        and isinstance(runtime.get("python"), str)
        and bool(runtime["python"])
        and isinstance(runtime.get("slurm_job_id"), str)
        and runtime["slurm_job_id"].isdigit()
        and _is_nonbool_int(runtime.get("slurm_cpus_per_task"))
        and runtime.get("slurm_cpus_per_task") == 32
        and runtime.get("slurm_job_partition") == RUNTIME_SLURM_PARTITION
        and runtime.get("slurm_job_account") == RUNTIME_SLURM_ACCOUNT
        and constraint_value == "rome"
        and isinstance(memory_value, str)
        and memory_value.isdigit()
        and int(memory_value) == SLURM_MEMORY_PER_NODE_MIB
        and _is_nonbool_int(runtime.get("slurm_num_nodes"))
        and runtime.get("slurm_num_nodes") == 1
        and runtime.get("slurm_time_limit") == RESOURCE_SMOKE_TIME_LIMIT
        and runtime.get("slurm_gpu_allocation") == "none"
        and runtime.get("slurm_scontrol_source")
        == "scontrol_show_job_one_line"
        and _lower_hex(
            runtime.get("slurm_scontrol_record_sha256"), length=64
        )
        and all(
            isinstance(runtime.get(name), str)
            and bool(runtime[name])
            and "gpu" not in runtime[name].casefold()
            for name in (
                "slurm_scontrol_alloc_tres",
                "slurm_scontrol_req_tres",
                "slurm_scontrol_tres_per_node",
                "slurm_scontrol_gres",
            )
        )
        and "node=1" in runtime["slurm_scontrol_alloc_tres"].casefold()
        and isinstance(runtime.get("cpu_model"), str)
        and "AMD EPYC" in runtime["cpu_model"]
        and runtime.get("requested_device") == "cpu"
        and runtime.get("gpu_requested") is False,
        "resource-smoke runtime differs from the frozen Slurm allocation",
    )


def _source_identity_records() -> dict[str, Any]:
    paths = (
        CONFIG_PATH,
        runner.CORE_PATH,
        Path(runner.__file__).resolve(),
        SMOKE_SOURCE_PATH,
        SMOKE_TEST_PATH,
        SMOKE_WRAPPER_PATH,
    )
    values = {
        str(path.relative_to(ROOT).as_posix()): sha256_file(path)
        for path in paths
    }
    return {
        "files": values,
        "files_content_sha256": canonical_json_sha256(values),
    }


def _frozen_release_contract() -> dict[str, Any]:
    """Read only the frozen configuration/Plan identities used by public auth."""

    plan = runner.load_plan(CONFIG_PATH)
    raw = plan.raw
    identity = raw.get("input_identity")
    output = raw.get("output")
    _require(
        isinstance(identity, Mapping) and isinstance(output, Mapping),
        "frozen release identity/output contract is missing",
    )
    source_names = {
        "train_cache_input_manifest": "train_cache_input_manifest",
        "kinematic_input_manifest": "kinematic_input_manifest",
        "parent_synthetic_pass": "parent_synthetic_pass",
        "sealed_sidecar_population": "sidecar_population",
    }
    evidence: dict[str, dict[str, Any]] = {}
    for audit_name, config_name in source_names.items():
        value = identity.get(config_name)
        _require(isinstance(value, Mapping), f"frozen evidence is missing: {config_name}")
        path = value.get("path")
        digest = value.get("sha256")
        _require(
            isinstance(path, str)
            and path.startswith("/")
            and _lower_hex(digest, length=64),
            f"frozen evidence path/SHA-256 drifted: {config_name}",
        )
        record: dict[str, Any] = {"path": path, "sha256": digest}
        if "size_bytes" in value:
            size = value.get("size_bytes")
            _require(
                isinstance(size, int) and not isinstance(size, bool) and size >= 0,
                f"frozen evidence size drifted: {config_name}",
            )
            record["size_bytes"] = size
        if config_name == "train_cache_input_manifest":
            schema = value.get("schema")
            rows_digest = value.get("rows_content_sha256")
            _require(
                schema
                == "pathline_template_matching.long_arc_train_cache_input.v1"
                and _lower_hex(rows_digest, length=64),
                "frozen train manifest semantic identity drifted",
            )
            record["schema"] = schema
            record["rows_content_sha256"] = rows_digest
        evidence[audit_name] = record
    output_root = output.get("root")
    _require(
        isinstance(output_root, str) and output_root.startswith("/"),
        "frozen output root drifted",
    )
    return {"output_root": output_root, "evidence": evidence}


def _validate_authenticated_evidence(
    evidence: object, *, frozen_contract: Mapping[str, Any]
) -> None:
    expected = frozen_contract.get("evidence")
    _require(
        isinstance(evidence, Mapping)
        and isinstance(expected, Mapping)
        and set(evidence) == set(expected),
        "authenticated evidence set drifted",
    )
    for name, frozen in expected.items():
        observed = evidence.get(name)
        expected_fields = set(frozen) | {"size_bytes"}
        _require(
            isinstance(frozen, Mapping)
            and isinstance(observed, Mapping)
            and set(observed) == expected_fields
            and observed.get("path") == frozen.get("path")
            and observed.get("sha256") == frozen.get("sha256")
            and isinstance(observed.get("size_bytes"), int)
            and not isinstance(observed.get("size_bytes"), bool)
            and int(observed["size_bytes"]) >= 0,
            f"authenticated evidence identity differs from frozen config: {name}",
        )
        for field, frozen_value in frozen.items():
            if field != "size_bytes":
                _require(
                    observed[field] == frozen_value,
                    f"authenticated evidence semantic identity differs from frozen config: {name}/{field}",
                )
        if "size_bytes" in frozen:
            _require(
                observed["size_bytes"] == frozen["size_bytes"],
                f"authenticated evidence size differs from frozen config: {name}",
            )


def _whole_file_authenticated_identity(
    identity: Mapping[str, Any], *, label: str
) -> tuple[bytes, Path]:
    """Return one stable byte snapshot after exact path/size/SHA authentication."""

    source = Path(str(identity["path"])).resolve()
    try:
        with source.open("rb") as stream:
            before = os.fstat(stream.fileno())
            payload = stream.read()
            after = os.fstat(stream.fileno())
        final = source.stat(follow_symlinks=False)
    except OSError as error:
        raise RuntimeError(f"{label} could not be read as a stable whole file") from error

    def signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(value.st_ctime_ns),
        )

    digest = hashlib.sha256(payload).hexdigest()
    _require(
        signature(before) == signature(after) == signature(final)
        and len(payload) == int(identity["size_bytes"])
        and digest == identity["sha256"],
        f"{label} changed or differs from its whole-file authenticated identity",
    )
    return payload, source


def _json_from_authenticated_snapshot(
    snapshot: tuple[bytes, Path], *, label: str
) -> tuple[Mapping[str, Any], Path]:
    """Parse JSON only after its supplied byte snapshot has been authenticated."""

    payload, source = snapshot
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not canonical UTF-8 JSON") from error
    _require(isinstance(value, Mapping), f"{label} root is not a mapping")
    return value, source


def _authenticated_evidence_whole_file_snapshots(
    evidence: Mapping[str, Any],
) -> dict[str, tuple[bytes, Path]]:
    """Authenticate all four persisted evidence files before semantic parsing."""

    labels = {
        "train_cache_input_manifest": "train cache input manifest",
        "kinematic_input_manifest": "kinematic input manifest",
        "parent_synthetic_pass": "parent synthetic PASS",
        "sealed_sidecar_population": "sealed sidecar population manifest",
    }
    _require(set(evidence) == set(labels), "authenticated evidence set drifted")
    snapshots: dict[str, tuple[bytes, Path]] = {}
    for name, label in labels.items():
        identity = evidence[name]
        _require(isinstance(identity, Mapping), f"{label} identity is invalid")
        snapshots[name] = _whole_file_authenticated_identity(identity, label=label)
    return snapshots


def _authenticated_input_metadata_rows(
    evidence: Mapping[str, Any],
    *,
    evidence_snapshots: Mapping[str, tuple[bytes, Path]],
) -> tuple[Mapping[tuple[str, int, int], Mapping[str, Any]], Mapping[tuple[str, int, int], Mapping[str, Any]], Path]:
    """Parse the two authenticated JSON snapshots, never NPZ members."""

    train_identity = evidence["train_cache_input_manifest"]
    population_identity = evidence["sealed_sidecar_population"]
    assert isinstance(train_identity, Mapping)
    assert isinstance(population_identity, Mapping)
    train, _train_path = _json_from_authenticated_snapshot(
        evidence_snapshots["train_cache_input_manifest"],
        label="train cache input manifest",
    )
    train_rows = train.get("rows")
    _require(
        train.get("schema") == train_identity.get("schema")
        and train.get("row_count") == 32
        and train.get("test_dataset_access") is False
        and isinstance(train_rows, list)
        and len(train_rows) == 32
        and train.get("rows_content_sha256")
        == train_identity.get("rows_content_sha256")
        == canonical_json_sha256(train_rows),
        "authenticated train manifest metadata population drifted",
    )
    train_by_key: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in train_rows:
        _require(isinstance(row, Mapping), "train metadata row is not a mapping")
        required = {
            "dataset",
            "source_ordinal",
            "source_index",
            "cache_path",
            "cache_size_bytes",
            "cache_file_sha256",
        }
        _require(required <= set(row), "train metadata row is missing identity fields")
        key = (
            str(row["dataset"]),
            int(row["source_ordinal"]),
            int(row["source_index"]),
        )
        _require(
            key not in train_by_key
            and int(row["source_ordinal"]) in range(4)
            and int(row["source_index"]) >= 0
            and isinstance(row["cache_path"], str)
            and int(row["cache_size_bytes"]) >= 0
            and _lower_hex(row["cache_file_sha256"], length=64),
            "train metadata source/file identity drifted",
        )
        train_by_key[key] = row

    population, population_path = _json_from_authenticated_snapshot(
        evidence_snapshots["sealed_sidecar_population"],
        label="sealed sidecar population manifest",
    )
    population_content = dict(population)
    stored_content_sha = population_content.pop("content_sha256", None)
    population_rows = population.get("rows")
    _require(
        _lower_hex(stored_content_sha, length=64)
        and stored_content_sha == canonical_json_sha256(population_content)
        and population.get("schema") == POPULATION_MANIFEST_SCHEMA
        and population.get("sidecar_count") == 32
        and isinstance(population_rows, list)
        and len(population_rows) == 32
        and population.get("rows_content_sha256")
        == canonical_json_sha256(population_rows),
        "authenticated sidecar population metadata drifted",
    )
    population_by_key: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in population_rows:
        _require(isinstance(row, Mapping), "sidecar metadata row is not a mapping")
        required = {
            "dataset",
            "physical_family",
            "source_ordinal",
            "source_index",
            "sidecar_relative_path",
            "sidecar_size_bytes",
            "sidecar_file_sha256",
            "sidecar_row_count",
        }
        _require(required <= set(row), "sidecar metadata row is missing identity fields")
        key = (
            str(row["dataset"]),
            int(row["source_ordinal"]),
            int(row["source_index"]),
        )
        relative = Path(str(row["sidecar_relative_path"]))
        _require(
            key not in population_by_key
            and not relative.is_absolute()
            and ".." not in relative.parts
            and int(row["sidecar_size_bytes"]) >= 0
            and int(row["sidecar_row_count"]) > 0
            and _lower_hex(row["sidecar_file_sha256"], length=64),
            "sidecar metadata source/file identity drifted",
        )
        population_by_key[key] = row
    _require(
        set(train_by_key) == set(population_by_key),
        "train and sidecar metadata source populations differ",
    )
    return train_by_key, population_by_key, population_path


def _validate_fit_access_against_authenticated_metadata(
    opened_rows: Sequence[Mapping[str, Any]],
    *,
    train_rows: Mapping[tuple[str, int, int], Mapping[str, Any]],
    population_rows: Mapping[tuple[str, int, int], Mapping[str, Any]],
    population_manifest_path: Path,
) -> None:
    _require(
        len(opened_rows) == EXPECTED_FIT_ROW_COUNT,
        "metadata-bound fit access row count drifted",
    )
    for opened in opened_rows:
        key = (
            str(opened["dataset"]),
            int(opened["source_ordinal"]),
            int(opened["source_index"]),
        )
        _require(
            key in train_rows and key in population_rows,
            "fit source is absent from authenticated input metadata",
        )
        train = train_rows[key]
        population = population_rows[key]
        parent = opened["parent_cache"]
        sidecar = opened["sidecar"]
        assert isinstance(parent, Mapping) and isinstance(sidecar, Mapping)
        sidecar_path = (
            population_manifest_path.parent / str(population["sidecar_relative_path"])
        ).resolve()
        _require(
            opened["physical_family"] == population["physical_family"]
            and parent["path"] == str(Path(str(train["cache_path"])).resolve())
            and parent["size_bytes"] == int(train["cache_size_bytes"])
            and parent["sha256"] == train["cache_file_sha256"]
            and sidecar["path"] == str(sidecar_path)
            and sidecar["size_bytes"] == int(population["sidecar_size_bytes"])
            and sidecar["sha256"] == population["sidecar_file_sha256"]
            and opened["joined_row_count"] == int(population["sidecar_row_count"]),
            "fit access file/join identity differs from authenticated metadata",
        )


def _validated_file_identity(value: object, *, label: str) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping)
        and set(value) == {"path", "size_bytes", "sha256"}
        and isinstance(value.get("path"), str)
        and bool(value["path"])
        and isinstance(value.get("size_bytes"), int)
        and not isinstance(value.get("size_bytes"), bool)
        and int(value["size_bytes"]) >= 0
        and _lower_hex(value.get("sha256"), length=64),
        f"{label} file identity drifted",
    )
    return dict(value)


def _canonical_opened_fit_row(value: object) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping) and set(value) == OPENED_FIT_ROW_FIELDS,
        "opened fit-row field set drifted",
    )
    family = value.get("physical_family")
    expected_dataset = {
        "f22_raptor": "f22raptor",
        "channel": "channel",
        "boeing_747": "boeing747",
    }
    _require(
        family in expected_dataset
        and value.get("dataset") == expected_dataset[family]
        and isinstance(value.get("source_ordinal"), int)
        and not isinstance(value.get("source_ordinal"), bool)
        and 0 <= int(value["source_ordinal"]) < 4
        and isinstance(value.get("source_index"), int)
        and not isinstance(value.get("source_index"), bool)
        and int(value["source_index"]) >= 0
        and isinstance(value.get("joined_row_count"), int)
        and not isinstance(value.get("joined_row_count"), bool)
        and int(value["joined_row_count"]) > 0
        and value.get("parent_archive_members") == list(PARENT_ARCHIVE_MEMBERS)
        and value.get("sidecar_archive_members") == list(SIDECAR_MEMBERS)
        and value.get("exact_identity_join_passed") is True,
        "opened fit-row identity/member contract drifted",
    )
    result = dict(value)
    result["parent_cache"] = _validated_file_identity(
        value.get("parent_cache"), label="parent cache"
    )
    result["sidecar"] = _validated_file_identity(
        value.get("sidecar"), label="sidecar"
    )
    return result


def _validate_fit_access_cross_binding(
    opened_rows: object, fit_rows: object
) -> None:
    _require(
        isinstance(opened_rows, list)
        and isinstance(fit_rows, list)
        and len(opened_rows) == EXPECTED_FIT_ROW_COUNT
        and len(fit_rows) == EXPECTED_FIT_ROW_COUNT,
        "fit access/input row count drifted",
    )
    opened = [_canonical_opened_fit_row(row) for row in opened_rows]
    projected_fit: list[dict[str, Any]] = []
    for value in fit_rows:
        _require(
            isinstance(value, Mapping) and set(value) == FIT_INPUT_ROW_FIELDS,
            "fit-input row field set drifted",
        )
        joined = value.get("joined_row_count")
        valid = value.get("valid_row_count")
        negative = value.get("negative_row_count")
        positive = value.get("positive_row_count")
        _require(
            isinstance(valid, int)
            and not isinstance(valid, bool)
            and isinstance(negative, int)
            and not isinstance(negative, bool)
            and isinstance(positive, int)
            and not isinstance(positive, bool)
            and valid == joined
            and negative >= 0
            and positive >= 0
            and negative + positive == valid,
            "fit-input joined/class row counts drifted",
        )
        projected_fit.append(
            _canonical_opened_fit_row(
                {name: value[name] for name in OPENED_FIT_ROW_FIELDS}
            )
        )
    expected_dataset_ordinals = {
        (dataset, family, ordinal)
        for family, dataset in (
            ("f22_raptor", "f22raptor"),
            ("channel", "channel"),
            ("boeing_747", "boeing747"),
        )
        for ordinal in range(4)
    }
    observed_dataset_ordinals = {
        (row["dataset"], row["physical_family"], row["source_ordinal"])
        for row in opened
    }
    complete_source_identities = {
        (
            row["dataset"],
            row["physical_family"],
            row["source_ordinal"],
            row["source_index"],
        )
        for row in opened
    }
    _require(
        opened == projected_fit
        and observed_dataset_ordinals == expected_dataset_ordinals
        and len(complete_source_identities) == EXPECTED_FIT_ROW_COUNT,
        "opened-fit and fit-input exact source/file identities differ",
    )


def _validate_output_directory(
    output_directory: Path, *, plan: runner.Plan, git_commit: str
) -> None:
    root = (plan.output_root / "resource_smoke").resolve()
    expected_job = os.environ.get("SLURM_JOB_ID", "")
    expected_name = f"slurm_{expected_job}_{git_commit[:12]}"
    _require(
        output_directory.resolve().parent == root
        and output_directory.name == expected_name,
        "resource smoke output directory identity drifted",
    )
    _require(not output_directory.exists(), "immutable resource-smoke output exists")


def _publish_smoke_evidence(
    output_directory: Path,
    audit_payload: Mapping[str, Any],
    marker_payload: Mapping[str, Any],
) -> tuple[Path, Path, str, str]:
    """Write the detailed audit first and the immutable PASS marker last."""

    _require(output_directory.is_dir(), "resource smoke output directory is missing")
    _require(not any(output_directory.iterdir()), "resource smoke output is not empty")
    audit = _with_self_hash(audit_payload)
    _assert_no_forbidden_output_fields(audit)
    audit_path = output_directory / AUDIT_NAME
    audit_sha = _atomic_json_no_replace(audit_path, audit)
    authenticated_audit = _authenticate_self_hashed_file(
        audit_path, expected_file_sha256=audit_sha
    )
    marker_base = dict(marker_payload)
    marker_base["audit"] = {
        "path": str(audit_path),
        "size_bytes": int(audit_path.stat().st_size),
        "sha256": audit_sha,
        "content_sha256": authenticated_audit["content_sha256"],
    }
    marker_base["write_order"] = (
        "last_after_detailed_audit_authentication_and_all_final_gates"
    )
    marker = _with_self_hash(marker_base)
    _assert_no_forbidden_output_fields(marker)
    marker_path = output_directory / PASS_NAME
    marker_sha = _atomic_json_no_replace(marker_path, marker)
    _authenticate_self_hashed_file(marker_path, expected_file_sha256=marker_sha)
    _require(
        {path.name for path in output_directory.iterdir()} == {AUDIT_NAME, PASS_NAME},
        "PASS marker was not the final and only post-audit artifact",
    )
    return audit_path, marker_path, audit_sha, marker_sha


def authenticate_resource_smoke_release(
    pass_path: str | Path,
    expected_file_sha256: str,
    expected_git_commit: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    """Read-only authentication gate required before the first real fold.

    The marker is only an entry point.  This function independently validates
    its self hash, the detailed audit's path/size/file/content hashes, both
    resource observations, the reserved-family access ledger, and the exact
    two-file immutable directory contract.  It never opens a parent cache or
    sidecar.
    """

    marker_path = Path(pass_path).resolve()
    _require(marker_path.name == PASS_NAME, "resource-smoke marker name drifted")
    _require(
        _lower_hex(expected_file_sha256, length=64),
        "expected resource-smoke marker SHA-256 is invalid",
    )
    _require(
        _lower_hex(expected_git_commit, length=40),
        "expected resource-smoke Git commit is invalid",
    )
    _require(
        expected_config_sha256 == EXPECTED_CONFIG_SHA256,
        "expected resource-smoke config SHA-256 drifted",
    )
    frozen_contract = _frozen_release_contract()
    marker_identity = _stable_file_identity(
        marker_path, expected_sha256=expected_file_sha256
    )
    marker = _authenticate_self_hashed_file(
        marker_path, expected_file_sha256=expected_file_sha256
    )
    expected_marker_fields = {
        "schema",
        "experiment",
        "stage",
        "status",
        "git_commit",
        "worktree_clean",
        "config_sha256",
        "elapsed_seconds",
        "linux_peak_rss_bytes",
        "memory_limit_bytes_exclusive",
        "walltime_limit_seconds_inclusive",
        "reserved_family_zero_member_open",
        "forbidden_dataset_member_open",
        "all_constructed_arrays_finite",
        "exact_path_and_resource_gates_passed",
        "audit",
        "write_order",
        "content_sha256",
    }
    _require(set(marker) == expected_marker_fields, "PASS marker field set drifted")
    _require(
        marker["schema"] == PASS_SCHEMA
        and marker["experiment"] == EXPERIMENT
        and marker["stage"] == STAGE
        and marker["status"] == "passed"
        and marker["git_commit"] == expected_git_commit
        and marker["worktree_clean"] is True
        and marker["config_sha256"] == expected_config_sha256
        and _is_nonbool_int(marker["linux_peak_rss_bytes"])
        and isinstance(marker["elapsed_seconds"], (int, float))
        and not isinstance(marker["elapsed_seconds"], bool)
        and np.isfinite(marker["elapsed_seconds"])
        and _is_nonbool_int(marker["memory_limit_bytes_exclusive"])
        and marker["memory_limit_bytes_exclusive"] == MEMORY_LIMIT_BYTES
        and _is_nonbool_int(marker["walltime_limit_seconds_inclusive"])
        and marker["walltime_limit_seconds_inclusive"] == WALLTIME_LIMIT_SECONDS
        and marker["reserved_family_zero_member_open"] is True
        and marker["forbidden_dataset_member_open"] is False
        and marker["all_constructed_arrays_finite"] is True
        and marker["exact_path_and_resource_gates_passed"] is True
        and marker["write_order"]
        == "last_after_detailed_audit_authentication_and_all_final_gates",
        "PASS marker identity or final gate drifted",
    )
    _validate_resource_limits(
        peak_rss_bytes=marker["linux_peak_rss_bytes"],
        elapsed_seconds=float(marker["elapsed_seconds"]),
    )

    children = list(marker_path.parent.iterdir())
    _require(
        len(children) == 2 and all(child.is_file() for child in children),
        "resource-smoke directory contains an extra or non-file child",
    )
    directory_files = {child.name: child.resolve() for child in children}
    _require(
        directory_files
        == {AUDIT_NAME: marker_path.parent / AUDIT_NAME, PASS_NAME: marker_path},
        "resource-smoke directory is not the exact two-file release",
    )
    audit_binding = marker.get("audit")
    _require(
        isinstance(audit_binding, Mapping)
        and set(audit_binding) == {"path", "size_bytes", "sha256", "content_sha256"}
        and _is_nonbool_int(audit_binding.get("size_bytes"))
        and int(audit_binding["size_bytes"]) >= 0
        and _lower_hex(audit_binding.get("sha256"), length=64)
        and _lower_hex(audit_binding.get("content_sha256"), length=64),
        "PASS marker audit binding drifted",
    )
    audit_path = Path(str(audit_binding["path"])).resolve()
    _require(
        audit_path == marker_path.parent / AUDIT_NAME,
        "PASS marker binds an audit outside its immutable directory",
    )
    audit_identity = _stable_file_identity(
        audit_path,
        expected_size=audit_binding["size_bytes"],
        expected_sha256=audit_binding["sha256"],
    )
    audit = _authenticate_self_hashed_file(
        audit_path, expected_file_sha256=audit_binding["sha256"]
    )
    _require(
        audit["content_sha256"] == audit_binding["content_sha256"],
        "PASS marker audit content hash drifted",
    )
    expected_audit_fields = {
        "schema",
        "experiment",
        "stage",
        "status",
        "evidence_scope",
        "git",
        "config",
        "production_source_identity",
        "runtime",
        "authenticated_evidence",
        "data_access",
        "fit_input_rows",
        "synthetic_arithmetic_gate",
        "constructed_model",
        "synthetic_query_path",
        "resource",
        "final_gates",
        "content_sha256",
    }
    _require(set(audit) == expected_audit_fields, "detailed audit field set drifted")
    _require(
        audit["schema"] == AUDIT_SCHEMA
        and audit["experiment"] == EXPERIMENT
        and audit["stage"] == STAGE
        and audit["status"] == "passed"
        and audit["evidence_scope"]
        == "resource_and_exact_path_only_no_method_quality_evidence",
        "detailed audit identity drifted",
    )
    git = audit.get("git")
    config = audit.get("config")
    _require(
        isinstance(git, Mapping)
        and set(git) == {"git_commit", "worktree_clean"}
        and git["git_commit"] == expected_git_commit
        and git["worktree_clean"] is True
        and isinstance(config, Mapping)
        and set(config) == {"path", "sha256"}
        and config["path"] == str(CONFIG_PATH.resolve())
        and config["sha256"] == expected_config_sha256,
        "detailed audit Git/config identity drifted",
    )
    source = audit.get("production_source_identity")
    current_source = _source_identity_records()
    expected_source_names = {
        path.relative_to(ROOT).as_posix()
        for path in (
            CONFIG_PATH,
            runner.CORE_PATH,
            Path(runner.__file__).resolve(),
            SMOKE_SOURCE_PATH,
            SMOKE_TEST_PATH,
            SMOKE_WRAPPER_PATH,
        )
    }
    _require(
        isinstance(source, Mapping)
        and set(source) == {"files", "files_content_sha256"}
        and isinstance(source["files"], Mapping)
        and set(source["files"]) == expected_source_names
        and source["files_content_sha256"]
        == canonical_json_sha256(source["files"])
        and all(_lower_hex(value, length=64) for value in source["files"].values()),
        "production source identity drifted",
    )
    _require(
        source == current_source,
        "production source identity differs from the current frozen checkout",
    )
    runtime = audit.get("runtime")
    _require(isinstance(runtime, Mapping), "resource-smoke runtime binding drifted")
    _validate_frozen_slurm_runtime_payload(runtime)
    frozen_output_root = frozen_contract.get("output_root")
    _require(
        isinstance(frozen_output_root, str), "frozen resource-smoke root is invalid"
    )
    expected_release_directory = (
        Path(frozen_output_root).resolve()
        / "resource_smoke"
        / f"slurm_{runtime['slurm_job_id']}_{expected_git_commit[:12]}"
    )
    _require(
        marker_path.parent == expected_release_directory,
        "resource-smoke release directory is outside the frozen output root or has a wrong basename",
    )
    evidence = audit.get("authenticated_evidence")
    _validate_authenticated_evidence(evidence, frozen_contract=frozen_contract)
    assert isinstance(evidence, Mapping)
    evidence_snapshots = _authenticated_evidence_whole_file_snapshots(evidence)
    train_metadata_rows, sidecar_metadata_rows, population_manifest_path = (
        _authenticated_input_metadata_rows(
            evidence,
            evidence_snapshots=evidence_snapshots,
        )
    )

    access = audit.get("data_access")
    _require(isinstance(access, Mapping), "data-access audit is missing")
    counts = access.get("member_open_counts_by_family")
    reserved = access.get("reserved_family_zero_member_open")
    opened = access.get("opened_fit_rows")
    _require(
        set(access)
        == {
            "semantic_separation",
            "population_envelope_sidecar_count",
            "member_open_counts_by_family",
            "reserved_family_zero_member_open",
            "reserved_parent_cache_whole_file_open_count",
            "reserved_sidecar_archive_member_open_count",
            "reserved_sidecar_envelope_hash_note",
            "opened_fit_rows",
        }
        and access.get("semantic_separation")
        == {
            "population_envelope_whole_file_authentication": (
                "all sealed sidecar byte files were hashed without archive-member deserialization"
            ),
            "archive_member_deserialization": "only the three fixed fit families below",
        }
        and _is_nonbool_int(access.get("population_envelope_sidecar_count"))
        and access.get("population_envelope_sidecar_count") == 32
        and isinstance(counts, Mapping)
        and set(counts) == set((*RESERVED_FAMILIES, *FIT_FAMILIES))
        and all(_is_nonbool_int(value) for value in counts.values())
        and counts
        == {
            RESERVED_OUTER_FAMILY: 0,
            RESERVED_INNER_FAMILY: 0,
            **{family: 4 for family in FIT_FAMILIES},
        }
        and isinstance(reserved, Mapping)
        and set(reserved) == set(RESERVED_FAMILIES)
        and all(reserved[family] is True for family in RESERVED_FAMILIES)
        and _is_nonbool_int(
            access.get("reserved_parent_cache_whole_file_open_count")
        )
        and access.get("reserved_parent_cache_whole_file_open_count") == 0
        and _is_nonbool_int(
            access.get("reserved_sidecar_archive_member_open_count")
        )
        and access.get("reserved_sidecar_archive_member_open_count") == 0
        and isinstance(opened, list)
        and len(opened) == EXPECTED_FIT_ROW_COUNT
        and all(row.get("physical_family") in FIT_FAMILIES for row in opened),
        "reserved-family or fit-family access audit drifted",
    )
    fit_input_rows = audit.get("fit_input_rows")
    _validate_fit_access_cross_binding(opened, fit_input_rows)
    assert isinstance(opened, list)
    assert isinstance(fit_input_rows, list)
    _validate_fit_access_against_authenticated_metadata(
        opened,
        train_rows=train_metadata_rows,
        population_rows=sidecar_metadata_rows,
        population_manifest_path=population_manifest_path,
    )
    fit_valid_row_count = sum(int(row["valid_row_count"]) for row in fit_input_rows)
    fit_negative_row_count = sum(
        int(row["negative_row_count"]) for row in fit_input_rows
    )
    arithmetic = audit.get("synthetic_arithmetic_gate")
    constructed = audit.get("constructed_model")
    query = audit.get("synthetic_query_path")
    _validate_constructed_model_artifact_audit(constructed)
    assert isinstance(constructed, Mapping)
    constructed_class_audits = constructed["class_library_and_reference_audits"]
    assert isinstance(constructed_class_audits, Mapping)
    for family in FIT_FAMILIES:
        family_fit_rows = [
            row for row in fit_input_rows if row["physical_family"] == family
        ]
        family_negative_row_count = sum(
            int(row["negative_row_count"]) for row in family_fit_rows
        )
        family_positive_row_count = sum(
            int(row["positive_row_count"]) for row in family_fit_rows
        )
        family_class_audit = constructed_class_audits[family]
        assert isinstance(family_class_audit, Mapping)
        negative_audit = family_class_audit["negative"]
        positive_audit = family_class_audit["positive"]
        assert isinstance(negative_audit, Mapping)
        assert isinstance(positive_audit, Mapping)
        _require(
            family_negative_row_count == negative_audit["library_row_count"]
            and family_positive_row_count >= positive_audit["library_row_count"],
            f"fit-input and constructed per-family class totals differ: {family}",
        )
    _require(
        isinstance(arithmetic, Mapping)
        and set(arithmetic)
        == {
            "feature_width",
            "family_count",
            "rows_per_family_class",
            "self_exclusion_forces_positive_k31_distance",
            "thirty_one_equal_duplicates_retained_per_family_class",
            "artifact_round_trip_exact",
            "strict_threshold_tie_contract_passed",
            "positive_only_no_scaler_exact_scale_all_class_family_support_false",
        }
        and _is_nonbool_int(arithmetic.get("feature_width"))
        and arithmetic.get("feature_width") == FEATURE_WIDTH
        and _is_nonbool_int(arithmetic.get("family_count"))
        and arithmetic.get("family_count") == len(FIT_FAMILIES)
        and _is_nonbool_int(arithmetic.get("rows_per_family_class"))
        and arithmetic.get("rows_per_family_class") == K + 1
        and arithmetic.get("self_exclusion_forces_positive_k31_distance") is True
        and arithmetic.get("thirty_one_equal_duplicates_retained_per_family_class")
        is True
        and arithmetic.get("artifact_round_trip_exact") is True
        and arithmetic.get("strict_threshold_tie_contract_passed") is True
        and arithmetic.get(
            "positive_only_no_scaler_exact_scale_all_class_family_support_false"
        )
        is True
        and isinstance(constructed, Mapping)
        and set(constructed) == CONSTRUCTED_MODEL_FIELDS
        and constructed.get("family_order") == list(FIT_FAMILIES)
        and _is_nonbool_int(constructed.get("feature_width"))
        and constructed.get("feature_width") == FEATURE_WIDTH
        and _is_nonbool_int(constructed.get("k"))
        and constructed.get("k") == K
        and _is_nonbool_int(constructed.get("strict_majority_family_count"))
        and constructed.get("strict_majority_family_count") == 2
        and constructed.get("all_constructed_arrays_finite") is True
        and constructed.get(
            "exact_self_exclusion_duplicate_and_support_count_audits_passed"
        )
        is True
        and all(
            isinstance(constructed.get(name), int)
            and not isinstance(constructed.get(name), bool)
            and int(constructed[name]) >= 0
            for name in (
                "natural_raw_family_class_row_count",
                "effective_retained_family_class_library_row_count",
                "natural_raw_present_exact_scale_count",
                "effective_retained_exact_scale_count",
                "natural_raw_only_no_scaler_exact_scale_count",
                "shared_negative_row_count",
                "full_family_class_library_row_count",
                "loo_reference_row_count_k31",
                "zero_distance_loo_reference_count_k31",
            )
        )
        and constructed["natural_raw_family_class_row_count"]
        >= constructed["effective_retained_family_class_library_row_count"]
        and constructed["natural_raw_family_class_row_count"]
        == fit_valid_row_count
        # Every natural negative makes its exact scale scaler-supported, so no
        # natural-negative row is removed by the pooled-negative support mask.
        and constructed["shared_negative_row_count"] == fit_negative_row_count
        and constructed["full_family_class_library_row_count"]
        == constructed["effective_retained_family_class_library_row_count"]
        and isinstance(
            constructed.get("natural_raw_present_exact_scale_count"), int
        )
        and isinstance(
            constructed.get("effective_retained_exact_scale_count"), int
        )
        and isinstance(
            constructed.get("natural_raw_only_no_scaler_exact_scale_count"), int
        )
        and constructed["natural_raw_present_exact_scale_count"]
        >= constructed["effective_retained_exact_scale_count"]
        and constructed["natural_raw_only_no_scaler_exact_scale_count"]
        == constructed["natural_raw_present_exact_scale_count"]
        - constructed["effective_retained_exact_scale_count"]
        and all(
            0 <= constructed[name] <= SCALE_COUNT
            for name in (
                "natural_raw_present_exact_scale_count",
                "effective_retained_exact_scale_count",
                "natural_raw_only_no_scaler_exact_scale_count",
            )
        )
        and constructed["natural_raw_present_exact_scale_count"]
        <= constructed["natural_raw_family_class_row_count"]
        and constructed["effective_retained_exact_scale_count"]
        <= constructed["effective_retained_family_class_library_row_count"]
        and constructed["effective_retained_exact_scale_count"]
        <= constructed["shared_negative_row_count"]
        and constructed["natural_raw_only_no_scaler_exact_scale_count"]
        <= constructed["natural_raw_family_class_row_count"]
        - constructed["effective_retained_family_class_library_row_count"]
        and _lower_hex(
            constructed.get("natural_raw_present_exact_scale_ids_sha256"),
            length=64,
        )
        and _lower_hex(
            constructed.get("effective_retained_exact_scale_ids_sha256"),
            length=64,
        )
        and _lower_hex(
            constructed.get(
                "natural_raw_only_no_scaler_exact_scale_ids_sha256"
            ),
            length=64,
        )
        and _lower_hex(
            constructed.get("natural_raw_class_scale_counts_sha256"), length=64
        )
        and _lower_hex(
            constructed.get("effective_retained_class_scale_counts_sha256"),
            length=64,
        )
        and isinstance(
            constructed.get("class_library_and_reference_audits"), Mapping
        )
        and set(constructed["class_library_and_reference_audits"])
        == set(FIT_FAMILIES)
        and isinstance(constructed.get("scaler_arrays"), Mapping)
        and isinstance(
            constructed.get("family_class_library_and_calibration_arrays"),
            Mapping,
        )
        and isinstance(query, Mapping)
        and set(query) == SYNTHETIC_QUERY_FIELDS
        and query.get("construction")
        == "deterministic_integer_index_formula_independent_of_fit_rows_and_labels"
        and query.get("query_scale_domain")
        == "union_of_all_natural_raw_exact_scales_before_scaler_filter"
        and query.get("all_natural_raw_present_exact_scales_exercised") is True
        and isinstance(query.get("query_row_count"), int)
        and not isinstance(query.get("query_row_count"), bool)
        and query["query_row_count"] >= 0
        and isinstance(query.get("natural_raw_present_exact_scale_count"), int)
        and isinstance(query.get("effective_retained_exact_scale_count"), int)
        and isinstance(
            query.get("natural_raw_only_no_scaler_exact_scale_count"), int
        )
        and query["natural_raw_present_exact_scale_count"]
        >= query["effective_retained_exact_scale_count"]
        and query["natural_raw_only_no_scaler_exact_scale_count"]
        == query["natural_raw_present_exact_scale_count"]
        - query["effective_retained_exact_scale_count"]
        and query["query_row_count"]
        == query["natural_raw_present_exact_scale_count"]
        and all(
            0 <= query[name] <= SCALE_COUNT
            for name in (
                "query_row_count",
                "natural_raw_present_exact_scale_count",
                "effective_retained_exact_scale_count",
                "natural_raw_only_no_scaler_exact_scale_count",
            )
        )
        and all(
            _lower_hex(query.get(name), length=64)
            for name in (
                "natural_raw_present_exact_scale_ids_sha256",
                "effective_retained_exact_scale_ids_sha256",
                "natural_raw_only_no_scaler_exact_scale_ids_sha256",
            )
        )
        and query.get(
            "all_natural_raw_only_no_scaler_scales_have_all_class_family_retrieval_and_calibration_unsupported"
        )
        is True
        and all(
            isinstance(query.get(name), int)
            and not isinstance(query.get(name), bool)
            and 0 <= int(query[name]) <= query["query_row_count"]
            for name in (
                "strict_majority_joint_supported_row_count",
                "strict_majority_retrieval_supported_row_count",
            )
        )
        and isinstance(query.get("joint_family_count_histogram"), Mapping)
        and set(query["joint_family_count_histogram"])
        == {str(index) for index in range(len(FIT_FAMILIES) + 1)}
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in query["joint_family_count_histogram"].values()
        )
        and sum(query["joint_family_count_histogram"].values())
        == query["query_row_count"]
        and query["strict_majority_joint_supported_row_count"]
        == sum(
            query["joint_family_count_histogram"][str(index)]
            for index in range(
                int(constructed["strict_majority_family_count"]),
                len(FIT_FAMILIES) + 1,
            )
        )
        and query["strict_majority_retrieval_supported_row_count"]
        >= query["strict_majority_joint_supported_row_count"]
        and query["strict_majority_joint_supported_row_count"]
        <= query["effective_retained_exact_scale_count"]
        and query["strict_majority_retrieval_supported_row_count"]
        <= query["effective_retained_exact_scale_count"]
        and query["joint_family_count_histogram"]["0"]
        >= query["natural_raw_only_no_scaler_exact_scale_count"]
        and sum(
            query["joint_family_count_histogram"][str(index)]
            for index in range(1, len(FIT_FAMILIES) + 1)
        )
        <= query["effective_retained_exact_scale_count"]
        and query.get("support_count_arithmetic_passed") is True
        and query.get("all_supported_numerical_values_finite") is True
        and query.get("reference_labels_consulted_by_query_path") is False,
        "synthetic arithmetic/model/query gate drifted",
    )
    scale_count_fields = (
        "natural_raw_present_exact_scale_count",
        "effective_retained_exact_scale_count",
        "natural_raw_only_no_scaler_exact_scale_count",
    )
    scale_sha_fields = (
        "natural_raw_present_exact_scale_ids_sha256",
        "effective_retained_exact_scale_ids_sha256",
        "natural_raw_only_no_scaler_exact_scale_ids_sha256",
    )
    _require(
        all(constructed[name] == query[name] for name in scale_count_fields)
        and all(constructed[name] == query[name] for name in scale_sha_fields),
        "constructed-model and synthetic-query exact scale sets are not identical",
    )
    resource_audit = audit.get("resource")
    _require(
        isinstance(resource_audit, Mapping)
        and set(resource_audit)
        == {
            "elapsed_seconds_before_audit_publish",
            "linux_peak_rss_bytes_before_audit_publish",
            "memory_limit_bytes_exclusive",
            "walltime_limit_seconds_inclusive",
            "peak_memory_strictly_below_limit",
            "elapsed_at_or_below_limit",
        }
        and _is_nonbool_int(resource_audit.get("memory_limit_bytes_exclusive"))
        and resource_audit.get("memory_limit_bytes_exclusive")
        == MEMORY_LIMIT_BYTES
        and _is_nonbool_int(
            resource_audit.get("walltime_limit_seconds_inclusive")
        )
        and resource_audit.get("walltime_limit_seconds_inclusive")
        == WALLTIME_LIMIT_SECONDS
        and _is_nonbool_int(
            resource_audit.get("linux_peak_rss_bytes_before_audit_publish")
        )
        and isinstance(
            resource_audit.get("elapsed_seconds_before_audit_publish"),
            (int, float),
        )
        and not isinstance(
            resource_audit.get("elapsed_seconds_before_audit_publish"), bool
        )
        and np.isfinite(resource_audit["elapsed_seconds_before_audit_publish"])
        and resource_audit.get("peak_memory_strictly_below_limit") is True
        and resource_audit.get("elapsed_at_or_below_limit") is True,
        "detailed resource gate drifted",
    )
    _validate_resource_limits(
        peak_rss_bytes=resource_audit["linux_peak_rss_bytes_before_audit_publish"],
        elapsed_seconds=float(resource_audit["elapsed_seconds_before_audit_publish"]),
    )
    _require(
        marker["linux_peak_rss_bytes"]
        >= resource_audit["linux_peak_rss_bytes_before_audit_publish"]
        and marker["elapsed_seconds"]
        >= resource_audit["elapsed_seconds_before_audit_publish"],
        "final resource observation predates or undercuts the detailed audit",
    )
    final_gates = audit.get("final_gates")
    _require(
        isinstance(final_gates, Mapping)
        and set(final_gates)
        == {
            "clean_git_identity_unchanged",
            "reserved_family_zero_member_open",
            "forbidden_dataset_member_open",
            "all_constructed_arrays_finite",
            "exact_self_exclusion_duplicate_and_support_counts",
            "raw_and_effective_scale_domains_distinguished",
            "synthetic_queries_only",
            "quality_outputs_absent",
        }
        and final_gates["clean_git_identity_unchanged"] is True
        and final_gates["reserved_family_zero_member_open"] is True
        and final_gates["forbidden_dataset_member_open"] is False
        and final_gates["all_constructed_arrays_finite"] is True
        and final_gates["exact_self_exclusion_duplicate_and_support_counts"]
        is True
        and final_gates["raw_and_effective_scale_domains_distinguished"] is True
        and final_gates["synthetic_queries_only"] is True
        and final_gates["quality_outputs_absent"] is True,
        "detailed final gate set drifted",
    )
    _assert_no_forbidden_output_fields(marker)
    _assert_no_forbidden_output_fields(audit)
    _require(
        marker_path.stat().st_mtime_ns >= audit_path.stat().st_mtime_ns,
        "PASS marker was not published after the detailed audit",
    )
    return {
        "schema": PASS_SCHEMA,
        "status": "authenticated",
        "git_commit": expected_git_commit,
        "config_sha256": expected_config_sha256,
        "marker": marker_identity,
        "audit": audit_identity,
        "resource_limits_passed": True,
        "reserved_family_zero_member_open": True,
        "directory_exact_two_files": True,
    }


def run(
    *,
    config_path: str | Path,
    expected_config_sha256: str,
    expected_git_commit: str,
    output_directory: str | Path,
    kinematic_input_manifest_path: str | Path,
    kinematic_input_manifest_sha256: str,
    synthetic_pass_path: str | Path,
    synthetic_pass_sha256: str,
    sidecar_root: str | Path,
    sidecar_population_manifest_path: str | Path,
    sidecar_population_manifest_sha256: str,
) -> dict[str, Any]:
    started = time.monotonic()
    _require(
        expected_config_sha256 == EXPECTED_CONFIG_SHA256,
        "caller expected config SHA-256 drifted",
    )
    initial_git = _git_identity(expected_git_commit)
    runtime = _runtime_environment_audit()
    plan = runner.load_plan(config_path)
    _require(plan.sha256 == EXPECTED_CONFIG_SHA256, "loaded config SHA-256 drifted")
    _validate_resource_smoke_contract(plan)
    output = Path(output_directory).resolve()
    _validate_output_directory(output, plan=plan, git_commit=expected_git_commit)
    output.mkdir(parents=True, exist_ok=False)

    arithmetic_gate = _synthetic_core_contract_gate()
    plan = runner.bind_early_evidence(
        plan,
        kinematic_input_manifest_path=kinematic_input_manifest_path,
        kinematic_input_manifest_file_sha256=kinematic_input_manifest_sha256,
        synthetic_pass_path=synthetic_pass_path,
        synthetic_pass_file_sha256=synthetic_pass_sha256,
        sidecar_root=sidecar_root,
        sidecar_population_manifest_path=sidecar_population_manifest_path,
        sidecar_population_manifest_file_sha256=sidecar_population_manifest_sha256,
    )
    rows, manifest_identity = runner.inherited.load_cache_rows(plan)
    ledger = AccessLedger()
    projections = _load_fit_projections(plan, rows, ledger)
    natural_counts = _natural_class_scale_counts(projections)
    expected_counts = _effective_class_scale_counts(natural_counts)
    fit_row_summary = []
    for projection, opened_row in zip(projections, ledger.records, strict=True):
        summary = dict(opened_row)
        summary.update(
            {
                "valid_row_count": int(projection.count),
                "negative_row_count": int(np.count_nonzero(~projection.labels)),
                "positive_row_count": int(np.count_nonzero(projection.labels)),
            }
        )
        fit_row_summary.append(summary)
    _validate_fit_access_cross_binding(ledger.records, fit_row_summary)

    model = runner._fit_tail_model(
        projections,
        REPRESENTATION,
        plan,
        device="cpu",
        ks=(K,),
    )
    model_audit, scaler_arrays, calibrator_arrays = _audit_model_arrays(
        model, natural_counts, expected_counts
    )
    # Free authenticated cache projections before artifact reconstruction; the
    # fitted model owns its complete natural family/class libraries.
    del projections
    gc.collect()
    restored = ClassConditionalTemplateScoreModel.from_artifacts(
        scaler_arrays, calibrator_arrays
    )
    synthetic_query_audit = _exercise_synthetic_query_path(
        restored, natural_counts, expected_counts
    )
    # A second independently reconstructed query must be byte-identical in all
    # structural support fields.  No class assignment is formed.
    second = ClassConditionalTemplateScoreModel.from_artifacts(
        scaler_arrays, calibrator_arrays
    )
    second_query_audit = _exercise_synthetic_query_path(
        second, natural_counts, expected_counts
    )
    _require(
        synthetic_query_audit == second_query_audit,
        "synthetic query reconstruction is not deterministic",
    )
    del model, restored, second, scaler_arrays, calibrator_arrays
    gc.collect()

    assert plan.sidecar_population is not None
    population_count = int(plan.sidecar_population["sidecar_count"])
    access_audit = ledger.as_json(population_sidecar_count=population_count)
    source_identity = _source_identity_records()
    evidence_files = {
        "train_cache_input_manifest": dict(manifest_identity),
        "kinematic_input_manifest": _stable_file_identity(
            kinematic_input_manifest_path,
            expected_sha256=kinematic_input_manifest_sha256,
        ),
        "parent_synthetic_pass": _stable_file_identity(
            synthetic_pass_path, expected_sha256=synthetic_pass_sha256
        ),
        "sealed_sidecar_population": _stable_file_identity(
            sidecar_population_manifest_path,
            expected_sha256=sidecar_population_manifest_sha256,
        ),
    }

    elapsed_before_audit = _elapsed_seconds(started)
    peak_before_audit = _peak_rss_bytes()
    _validate_resource_limits(
        peak_rss_bytes=peak_before_audit,
        elapsed_seconds=elapsed_before_audit,
    )
    final_git = _git_identity(expected_git_commit)
    _require(initial_git == final_git, "Git identity changed during resource smoke")
    ledger.validate()

    audit_payload = {
        "schema": AUDIT_SCHEMA,
        "experiment": EXPERIMENT,
        "stage": STAGE,
        "status": "passed",
        "evidence_scope": "resource_and_exact_path_only_no_method_quality_evidence",
        "git": final_git,
        "config": {"path": str(plan.path), "sha256": plan.sha256},
        "production_source_identity": source_identity,
        "runtime": runtime,
        "authenticated_evidence": evidence_files,
        "data_access": access_audit,
        "fit_input_rows": fit_row_summary,
        "synthetic_arithmetic_gate": arithmetic_gate,
        "constructed_model": model_audit,
        "synthetic_query_path": synthetic_query_audit,
        "resource": {
            "elapsed_seconds_before_audit_publish": elapsed_before_audit,
            "linux_peak_rss_bytes_before_audit_publish": peak_before_audit,
            "memory_limit_bytes_exclusive": MEMORY_LIMIT_BYTES,
            "walltime_limit_seconds_inclusive": WALLTIME_LIMIT_SECONDS,
            "peak_memory_strictly_below_limit": True,
            "elapsed_at_or_below_limit": True,
        },
        "final_gates": {
            "clean_git_identity_unchanged": True,
            "reserved_family_zero_member_open": True,
            "forbidden_dataset_member_open": False,
            "all_constructed_arrays_finite": True,
            "exact_self_exclusion_duplicate_and_support_counts": True,
            "raw_and_effective_scale_domains_distinguished": True,
            "synthetic_queries_only": True,
            "quality_outputs_absent": True,
        },
    }
    _assert_no_forbidden_output_fields(audit_payload)
    # Materialize the exact self-hashed audit JSON once before the final RSS
    # observation.  The subsequent immutable write cannot require a larger
    # serialization buffer than this preview.
    audit_preview = (
        json.dumps(
            _with_self_hash(audit_payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _require(len(audit_preview) > 0, "detailed audit serialization is empty")
    del audit_preview
    # Recheck the resource ceiling after building the complete audit payload and
    # immediately before the last marker transaction.
    final_elapsed = _elapsed_seconds(started)
    final_peak = _peak_rss_bytes()
    _validate_resource_limits(
        peak_rss_bytes=final_peak,
        elapsed_seconds=final_elapsed,
    )
    _require(_git_identity(expected_git_commit) == final_git, "final Git gate failed")
    marker_payload = {
        "schema": PASS_SCHEMA,
        "experiment": EXPERIMENT,
        "stage": STAGE,
        "status": "passed",
        "git_commit": expected_git_commit,
        "worktree_clean": True,
        "config_sha256": plan.sha256,
        "elapsed_seconds": final_elapsed,
        "linux_peak_rss_bytes": final_peak,
        "memory_limit_bytes_exclusive": MEMORY_LIMIT_BYTES,
        "walltime_limit_seconds_inclusive": WALLTIME_LIMIT_SECONDS,
        "reserved_family_zero_member_open": True,
        "forbidden_dataset_member_open": False,
        "all_constructed_arrays_finite": True,
        "exact_path_and_resource_gates_passed": True,
    }
    audit_path, marker_path, audit_sha, marker_sha = _publish_smoke_evidence(
        output, audit_payload, marker_payload
    )
    return {
        "output_directory": str(output),
        "audit_path": str(audit_path),
        "audit_sha256": audit_sha,
        "pass_marker_path": str(marker_path),
        "pass_marker_sha256": marker_sha,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kinematic-input-manifest", type=Path, required=True)
    parser.add_argument("--kinematic-input-manifest-sha256", required=True)
    parser.add_argument("--synthetic-pass", type=Path, required=True)
    parser.add_argument("--synthetic-pass-sha256", required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--sidecar-population-manifest", type=Path, required=True)
    parser.add_argument("--sidecar-population-manifest-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run(
        config_path=args.config,
        expected_config_sha256=args.expected_config_sha256,
        expected_git_commit=args.expected_git_commit,
        output_directory=args.output_dir,
        kinematic_input_manifest_path=args.kinematic_input_manifest,
        kinematic_input_manifest_sha256=args.kinematic_input_manifest_sha256,
        synthetic_pass_path=args.synthetic_pass,
        synthetic_pass_sha256=args.synthetic_pass_sha256,
        sidecar_root=args.sidecar_root,
        sidecar_population_manifest_path=args.sidecar_population_manifest,
        sidecar_population_manifest_sha256=args.sidecar_population_manifest_sha256,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

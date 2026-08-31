"""Fail-closed preparation artifacts for EarlyOppositePairKinematics.

This module intentionally stops before nested family validation.  It freezes
and authenticates the exact 32 train-cache/portable pairs, builds one additive
kinematic sidecar at a time, and closes the population with an immutable
manifest.  The input-freeze path never opens a portable NPZ; portable array
hashes are inherited from the already authenticated dataset JSON manifests.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .early_opposite_pair_kinematics import (
    FROZEN_KINEMATIC_FEATURE_ORDER,
    FROZEN_PRIMITIVE_ORDER,
    compute_seed_time_kinematic4,
    compute_seed_time_velocity_gradient,
)
from .arc_length_primitives import _interp4_quadrilinear_scalar
from .netcdf_io import FlowWindow3D
from .nested_scale_validation import representation_indices
from .portable_flow import (
    PORTABLE_FLOW_SCHEMA,
    canonical_array_sha256,
    canonical_json_sha256,
    load_portable_flow_window,
    sha256_file,
)
from .seed_time_kinematic_sidecar import (
    FORBIDDEN_PARENT_MEMBER_NAMES,
    FROZEN_DX_GRID_SCALE_BY_ID,
    PARENT_PROJECTION_MEMBER_NAMES,
    SIDECAR_PROVENANCE_BINDING_NAMES,
    ParentKinematicProjection,
    SeedTimeKinematicSidecarPayload,
    build_seed_time_kinematic_sidecar_payload,
    load_parent_kinematic_projection,
    load_seed_time_kinematic_sidecar,
    sample_seed_time_velocity_xyz,
    validate_sidecar_identity_join,
    write_seed_time_kinematic_sidecar,
)
from .vector_field import UnsteadyVectorField3D


EXPERIMENT = "Verify_EarlyOppositePairKinematics_1.1"
INPUT_MANIFEST_SCHEMA = (
    "pathline_template_matching.seed_time_opposite_pair_kinematics_input.v1"
)
SYNTHETIC_PASS_SCHEMA = (
    "pathline_template_matching.seed_time_opposite_pair_kinematics_synthetic_pass.v1"
)
SYNTHETIC_EVIDENCE_SCHEMA = (
    "pathline_template_matching.seed_time_opposite_pair_kinematics_synthetic_evidence.v1"
)
ROW_COMPLETION_SCHEMA = (
    "pathline_template_matching.seed_time_opposite_pair_kinematics_row_complete.v1"
)
POPULATION_MANIFEST_SCHEMA = (
    "pathline_template_matching.seed_time_opposite_pair_kinematics_population.v1"
)
COMPOSITE_DESCRIPTOR_SCHEMA = (
    "pathline_template_matching.seed_time_opposite_pair_composite_descriptor.v1"
)

EXPECTED_VERIFY_CONFIG_SHA256 = (
    "e6bac4568025f42cf0a9effd78620e5ab4ba5653429a7023bd91816f29512767"
)
EXPECTED_PARENT_MAIN_CONFIG_SHA256 = (
    "771980f14a6019a1f6e4bf03668d9f37dcf63495ae2dafa866312b12fc71855e"
)
EXPECTED_PARENT_CACHE_BUILDER_COMMIT = (
    "260a07ad380d64fc300cabe8926244e92d8ba04a"
)
EXPECTED_PARENT_INPUT_MANIFEST_SHA256 = (
    "e57d6b527acffb61da32a993f0c30a0e6435021679c7a3f1541dab8ba877b393"
)
EXPECTED_PARENT_INPUT_MANIFEST_SIZE = 24009
EXPECTED_PARENT_INPUT_ROWS_SHA256 = (
    "ceb6d0e3fb7a2c90fcaae98583f8d1def7ee75fa7968f38d2821ee3040ae156f"
)
EXPECTED_TRAIN_PORTABLE_MARKER_SHA256 = (
    "489d303b4430be7eded4fe39ab87107c778e1f7db2579cb9e3bb1fdfce209341"
)
PARENT_DESCRIPTOR_ID = "fmt_independent_3d_161d_sha256_25fce29499c9089e"
PARENT_CACHE_SCHEMA = "pathline_template_matching.phase31_cache.v1"
PARENT_INPUT_SCHEMA = "pathline_template_matching.long_arc_train_cache_input.v1"
PARENT_PORTABLE_MARKER_SCHEMA = (
    "pathline_template_matching.phase31_portable_population_pass.v1"
)
PARENT_DATASET_MANIFEST_SCHEMA = (
    "pathline_template_matching.portable_flow_dataset_manifest.v1"
)

DATASET_FAMILY_PAIRS = (
    ("cylinder3d", "half_cylinder"),
    ("halfcylinderRe640", "half_cylinder"),
    ("halfcylinderRe6400", "half_cylinder"),
    ("deltaWing_resampled", "delta_wing"),
    ("deltaWing_LBM", "delta_wing"),
    ("f22raptor", "f22_raptor"),
    ("channel", "channel"),
    ("boeing747", "boeing_747"),
)
FORBIDDEN_DATASET_TOKENS = ("tangaroa", "smokebuoyancy", "smoke_buoyancy")
SOURCE_COUNT_PER_DATASET = 4

REQUIRED_SOURCE_PATHS = (
    "config/Verify_EarlyOppositePairKinematics_1.1.yaml",
    "config/mainExp_TemplateMatching_3.1.yaml",
    "src/pathline_template_matching/early_opposite_pair_kinematics.py",
    "src/pathline_template_matching/seed_time_kinematic_sidecar.py",
    "src/pathline_template_matching/early_kinematic_preparation.py",
    "src/pathline_template_matching/portable_flow.py",
    "src/pathline_template_matching/nested_scale_validation.py",
    "src/pathline_template_matching/arc_length_primitives.py",
    "src/pathline_template_matching/vector_field.py",
    "src/pathline_template_matching/netcdf_io.py",
)
KINEMATIC_ALGORITHM_SOURCE_PATH = (
    "src/pathline_template_matching/early_opposite_pair_kinematics.py"
)
SAMPLER_SOURCE_PATH = "src/pathline_template_matching/seed_time_kinematic_sidecar.py"
NUMERICAL_DEPENDENCY_SOURCE_PATHS = (
    KINEMATIC_ALGORITHM_SOURCE_PATH,
    SAMPLER_SOURCE_PATH,
    "src/pathline_template_matching/arc_length_primitives.py",
    "src/pathline_template_matching/vector_field.py",
    "src/pathline_template_matching/portable_flow.py",
    "src/pathline_template_matching/netcdf_io.py",
)

SYNTHETIC_CHECK_NAMES = (
    "affine_gradient_every_frozen_dx",
    "rigid_translation_oracle",
    "rigid_rotation_oracle",
    "pure_strain_oracle",
    "isotropic_expansion_oracle",
    "proper_rotation_translation_invariance",
    "single_batch_chunk_permutation_invariance",
    "wrong_opposite_pair_order_rejected",
    "production_rk4_first_v1_match",
    "forbidden_parent_member_access_absent",
    "identity_join_missing_duplicate_extra_reorder_rejected",
)

_INPUT_TOP_LEVEL_NAMES = frozenset(
    {
        "schema",
        "experiment",
        "status",
        "git_commit",
        "worktree_clean",
        "verify_config_sha256",
        "source_file_sha256",
        "source_file_sha256_content_sha256",
        "synthetic_pass",
        "parent_input_manifest",
        "train_portable_population_marker",
        "parent_descriptor_id",
        "composite_descriptors",
        "input_scope",
        "portable_npz_opened_during_freeze",
        "forbidden_dataset_access",
        "row_count",
        "rows",
        "rows_content_sha256",
        "content_sha256",
    }
)
_INPUT_ROW_NAMES = frozenset(
    {
        "dataset",
        "physical_family",
        "split",
        "source_ordinal",
        "source_index",
        "parent_cache",
        "portable",
    }
)
_PARENT_ROW_NAMES = frozenset(
    {
        "path",
        "size_bytes",
        "file_sha256",
        "schema",
        "builder_git_commit",
        "config_sha256",
        "allowed_array_sha256",
        "opened_members",
    }
)
_PORTABLE_ROW_NAMES = frozenset(
    {
        "path",
        "size_bytes",
        "file_sha256",
        "schema",
        "builder_git_commit",
        "config_sha256",
        "dataset_manifest_path",
        "dataset_manifest_size_bytes",
        "dataset_manifest_file_sha256",
        "dataset_manifest_content_sha256",
        "portable_metadata_sha256",
        "array_sha256",
        "combined_array_sha256",
    }
)
_PORTABLE_ARRAY_NAMES = ("velocity", "x", "y", "z", "time")
_PARENT_MANIFEST_NAMES = frozenset(
    {
        "schema",
        "experiment",
        "parent_experiment",
        "git_commit",
        "main_config_sha256",
        "verify_config_sha256",
        "synthetic_pass_file_sha256",
        "train_portable_population_pass",
        "input_scope",
        "test_dataset_access",
        "row_count",
        "rows",
        "rows_content_sha256",
    }
)
_PARENT_MANIFEST_ROW_NAMES = frozenset(
    {
        "dataset",
        "source_ordinal",
        "source_index",
        "cache_path",
        "cache_size_bytes",
        "cache_file_sha256",
        "sidecar_path",
        "sidecar_size_bytes",
        "sidecar_file_sha256",
    }
)
_PORTABLE_MARKER_NAMES = frozenset(
    {
        "schema",
        "experiment",
        "status",
        "access_scope",
        "git_commit",
        "worktree_clean",
        "config_sha256",
        "dataset_registry_sha256",
        "portable_root",
        "dataset_count",
        "window_count",
        "synthetic_pass_file_sha256",
        "train_coverage_pass_file_sha256",
        "rows",
        "rows_content_sha256",
        "marker_write_order",
    }
)
_PORTABLE_MARKER_ROW_NAMES = frozenset(
    {
        "dataset",
        "split",
        "source_ordinal",
        "source_start_index",
        "relative_path",
        "file_size",
        "file_sha256",
        "manifest_relative_path",
        "manifest_file_sha256",
        "portable_metadata_sha256",
    }
)
_DATASET_MANIFEST_NAMES = frozenset(
    {
        "schema",
        "experiment",
        "config_path",
        "config_sha256",
        "dataset_registry_path",
        "dataset_registry_sha256",
        "builder_git_commit",
        "dataset",
        "physical_family",
        "split",
        "source_kind",
        "source_file",
        "source_file_size",
        "source_file_sha256",
        "source_total_frames",
        "selected_source_indices",
        "window_count",
        "windows",
        "manifest_content_sha256",
    }
)
_SYNTHETIC_MARKER_NAMES = frozenset(
    {
        "schema",
        "experiment",
        "status",
        "git_commit",
        "worktree_clean",
        "verify_config_sha256",
        "source_file_sha256",
        "source_file_sha256_content_sha256",
        "check_results",
        "check_results_content_sha256",
        "evidence",
        "evidence_content_sha256",
        "composite_descriptor_ids",
        "real_flow_or_cache_access",
        "forbidden_dataset_access",
        "marker_write_order",
        "content_sha256",
    }
)
_SYNTHETIC_EVIDENCE_NAMES = frozenset(
    {
        "schema",
        "experiment",
        "status",
        "git_commit",
        "worktree_clean",
        "verify_config_sha256",
        "source_file_sha256",
        "source_file_sha256_content_sha256",
        "check_count",
        "checks",
        "checks_content_sha256",
        "check_results",
        "check_results_content_sha256",
        "composite_descriptor_ids",
        "real_flow_or_cache_access",
        "forbidden_dataset_access",
        "content_sha256",
    }
)
_SYNTHETIC_CHECK_RECORD_NAMES = frozenset(
    {"name", "status", "detail", "detail_content_sha256"}
)
_ROW_COMPLETION_NAMES = frozenset(
    {
        "schema",
        "experiment",
        "status",
        "git_commit",
        "worktree_clean",
        "verify_config_sha256",
        "source_file_sha256",
        "source_file_sha256_content_sha256",
        "dataset",
        "physical_family",
        "split",
        "source_ordinal",
        "source_index",
        "input_manifest_path",
        "input_manifest_file_sha256",
        "input_manifest_content_sha256",
        "synthetic_pass_path",
        "synthetic_pass_file_sha256",
        "sidecar_relative_path",
        "sidecar_size_bytes",
        "sidecar_file_sha256",
        "sidecar_combined_array_sha256",
        "sidecar_row_count",
        "composite_descriptor_ids",
        "forbidden_parent_members_opened",
        "forbidden_dataset_access",
        "marker_write_order",
        "content_sha256",
    }
)
_POPULATION_MANIFEST_NAMES = frozenset(
    {
        "schema",
        "experiment",
        "status",
        "git_commit",
        "worktree_clean",
        "verify_config_sha256",
        "source_file_sha256",
        "source_file_sha256_content_sha256",
        "input_manifest_path",
        "input_manifest_file_sha256",
        "input_manifest_content_sha256",
        "synthetic_pass_path",
        "synthetic_pass_file_sha256",
        "composite_descriptor_ids",
        "sidecar_count",
        "sidecar_row_count_total",
        "rows",
        "rows_content_sha256",
        "forbidden_dataset_access",
        "manifest_write_order",
        "content_sha256",
    }
)
_POPULATION_ROW_NAMES = frozenset(
    {
        "dataset",
        "physical_family",
        "source_ordinal",
        "source_index",
        "completion_relative_path",
        "completion_size_bytes",
        "completion_file_sha256",
        "sidecar_relative_path",
        "sidecar_size_bytes",
        "sidecar_file_sha256",
        "sidecar_combined_array_sha256",
        "sidecar_row_count",
    }
)


def _lower_hex(value: object, length: int, *, name: str) -> str:
    selected = str(value)
    if len(selected) != length or any(
        character not in "0123456789abcdef" for character in selected
    ):
        raise ValueError(f"{name} must be a lowercase {length}-character digest")
    return selected


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    selected = int(value)
    if selected < 1:
        raise ValueError(f"{name} must be a positive integer")
    return selected


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    selected = int(value)
    if selected < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return selected


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(child) for child in value)
    return value


def _json_mutable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_mutable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_mutable(child) for child in value]
    return value


def _canonical_json_copy(value: object) -> Any:
    try:
        return json.loads(
            json.dumps(
                _json_mutable(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError("artifact payload must contain finite JSON values") from error


def _stable_file_identity(path: Path) -> tuple[int, str]:
    source = path.resolve()
    before = source.stat()
    digest = sha256_file(source)
    after = source.stat()
    before_identity = (
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
        int(getattr(before, "st_ino", 0)),
    )
    after_identity = (
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
        int(getattr(after, "st_ino", 0)),
    )
    if before_identity != after_identity:
        raise RuntimeError(f"file changed while it was authenticated: {source}")
    return int(after.st_size), digest


@dataclass(frozen=True)
class _AuthenticatedFileSnapshot:
    content: bytes
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class _SnapshotFileIdentity:
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    device: int
    inode: int
    mode: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_SnapshotFileIdentity":
        return cls(
            size_bytes=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            ctime_ns=int(value.st_ctime_ns),
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=int(value.st_mode),
        )


def _read_authenticated_bytes(
    path: str | Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> _AuthenticatedFileSnapshot:
    """Read and hash one no-follow FD bound to its path before and after."""

    source = Path(path).resolve()
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    path_before = _SnapshotFileIdentity.from_stat(
        source.stat(follow_symlinks=False)
    )
    descriptor = os.open(os.fspath(source), flags)
    try:
        descriptor_before = _SnapshotFileIdentity.from_stat(os.fstat(descriptor))
        if descriptor_before != path_before:
            raise RuntimeError(f"file path changed before descriptor open: {source}")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            byte_count += len(chunk)
        descriptor_after = _SnapshotFileIdentity.from_stat(os.fstat(descriptor))
        path_after = _SnapshotFileIdentity.from_stat(
            source.stat(follow_symlinks=False)
        )
        if descriptor_after != descriptor_before or path_after != path_before:
            raise RuntimeError(f"file path or descriptor changed while reading: {source}")
    finally:
        os.close(descriptor)
    actual_sha256 = digest.hexdigest()
    if byte_count != path_before.size_bytes:
        raise RuntimeError(f"file byte count changed while reading: {source}")
    if expected_size is not None and byte_count != int(expected_size):
        raise ValueError(f"file size mismatch: {source}")
    if expected_sha256 is not None:
        expected = _lower_hex(expected_sha256, 64, name=f"file SHA-256: {source}")
        if actual_sha256 != expected:
            raise ValueError(f"file SHA-256 mismatch: {source}")
    return _AuthenticatedFileSnapshot(
        content=b"".join(chunks),
        size_bytes=byte_count,
        sha256=actual_sha256,
    )


def _json_from_authenticated_snapshot(
    path: str | Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], _AuthenticatedFileSnapshot]:
    snapshot = _read_authenticated_bytes(
        path,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
    )
    value = json.loads(snapshot.content.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain a mapping: {path}")
    return value, snapshot


def _atomic_json_no_overwrite(path: Path, value: object) -> str:
    """Publish a complete JSON inode without an overwrite race."""

    output = path.resolve()
    if output.exists():
        raise FileExistsError(f"immutable artifact already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    try:
        with temporary.open("xb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            raise FileExistsError(f"immutable artifact already exists: {output}")
    finally:
        if temporary.exists():
            temporary.unlink()
    if os.name == "posix":
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(str(output.parent), flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return sha256_file(output)


def _with_content_sha256(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _canonical_json_copy(dict(value))
    if "content_sha256" in payload:
        raise ValueError("content_sha256 must be added only by the artifact writer")
    payload["content_sha256"] = canonical_json_sha256(payload)
    return payload


def _verify_content_sha256(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON mapping")
    payload = dict(value)
    claimed = payload.pop("content_sha256", None)
    if claimed != canonical_json_sha256(payload):
        raise ValueError(f"{name} content SHA-256 mismatch")
    payload["content_sha256"] = str(claimed)
    return payload


def _safe_relative(value: object, *, name: str) -> Path:
    selected = Path(str(value))
    if (
        selected.is_absolute()
        or not selected.parts
        or ".." in selected.parts
        or any(part in {"", "."} for part in selected.parts)
    ):
        raise ValueError(f"{name} must be a safe relative path")
    return selected


def _require_no_forbidden_path(path: str | Path, *, name: str) -> None:
    folded_parts = tuple(part.casefold() for part in Path(path).parts)
    for part in folded_parts:
        normalized = part.replace("-", "_").replace(" ", "_")
        if any(token in normalized for token in FORBIDDEN_DATASET_TOKENS):
            raise ValueError(f"{name} references a forbidden dataset path")


@dataclass(frozen=True)
class PreparationContract:
    """Immutable population/provenance contract for the preparation layer."""

    verify_config_sha256: str
    parent_main_config_sha256: str
    parent_cache_builder_commit: str
    parent_input_manifest_sha256: str
    parent_input_manifest_size: int
    parent_input_rows_sha256: str
    train_portable_marker_sha256: str
    dataset_family_pairs: tuple[tuple[str, str], ...] = DATASET_FAMILY_PAIRS
    source_count: int = SOURCE_COUNT_PER_DATASET

    def __post_init__(self) -> None:
        for name in (
            "verify_config_sha256",
            "parent_main_config_sha256",
            "parent_input_manifest_sha256",
            "parent_input_rows_sha256",
            "train_portable_marker_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _lower_hex(getattr(self, name), 64, name=name),
            )
        object.__setattr__(
            self,
            "parent_cache_builder_commit",
            _lower_hex(
                self.parent_cache_builder_commit,
                40,
                name="parent_cache_builder_commit",
            ),
        )
        object.__setattr__(
            self,
            "parent_input_manifest_size",
            _positive_integer(
                self.parent_input_manifest_size,
                name="parent_input_manifest_size",
            ),
        )
        object.__setattr__(
            self,
            "source_count",
            _positive_integer(self.source_count, name="source_count"),
        )
        pairs = tuple((str(dataset), str(family)) for dataset, family in self.dataset_family_pairs)
        datasets = tuple(dataset for dataset, _family in pairs)
        if (
            not pairs
            or len(set(datasets)) != len(datasets)
            or any(not dataset or not family for dataset, family in pairs)
        ):
            raise ValueError("dataset_family_pairs must be unique and nonempty")
        for dataset, _family in pairs:
            _require_no_forbidden_path(dataset, name="dataset")
        object.__setattr__(self, "dataset_family_pairs", pairs)

    @property
    def datasets(self) -> tuple[str, ...]:
        return tuple(dataset for dataset, _family in self.dataset_family_pairs)

    @property
    def family_by_dataset(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.dataset_family_pairs))

    @property
    def row_count(self) -> int:
        return len(self.dataset_family_pairs) * self.source_count


PRODUCTION_CONTRACT = PreparationContract(
    verify_config_sha256=EXPECTED_VERIFY_CONFIG_SHA256,
    parent_main_config_sha256=EXPECTED_PARENT_MAIN_CONFIG_SHA256,
    parent_cache_builder_commit=EXPECTED_PARENT_CACHE_BUILDER_COMMIT,
    parent_input_manifest_sha256=EXPECTED_PARENT_INPUT_MANIFEST_SHA256,
    parent_input_manifest_size=EXPECTED_PARENT_INPUT_MANIFEST_SIZE,
    parent_input_rows_sha256=EXPECTED_PARENT_INPUT_ROWS_SHA256,
    train_portable_marker_sha256=EXPECTED_TRAIN_PORTABLE_MARKER_SHA256,
)


@dataclass(frozen=True)
class CleanSourceIdentity:
    """A clean Git commit plus exact hashes of every preparation source."""

    git_commit: str
    worktree_clean: bool
    source_file_sha256_items: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "git_commit",
            _lower_hex(self.git_commit, 40, name="git_commit"),
        )
        if self.worktree_clean is not True:
            raise ValueError("production preparation requires a clean Git worktree")
        items = tuple((str(path), str(digest)) for path, digest in self.source_file_sha256_items)
        if tuple(path for path, _digest in items) != REQUIRED_SOURCE_PATHS:
            raise ValueError("source identity must contain the exact frozen source-path order")
        for path, digest in items:
            _safe_relative(path, name="source path")
            _lower_hex(digest, 64, name=f"source SHA-256 for {path}")
        object.__setattr__(self, "source_file_sha256_items", items)

    @property
    def source_file_sha256(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.source_file_sha256_items))

    @property
    def source_content_sha256(self) -> str:
        return canonical_json_sha256(dict(self.source_file_sha256_items))


def capture_clean_source_identity(project_root: str | Path) -> CleanSourceIdentity:
    """Capture exact source hashes only from a clean committed worktree."""

    root = Path(project_root).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("preparation artifacts require a clean committed worktree")
    hashes: list[tuple[str, str]] = []
    for relative in REQUIRED_SOURCE_PATHS:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise ValueError("required source resolves outside the project root") from error
        _size, digest = _stable_file_identity(source)
        hashes.append((relative, digest))
    commit_after = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_after = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit_after != commit or dirty_after:
        raise RuntimeError(
            "Git HEAD or exact clean status changed while source hashes were captured"
        )
    return CleanSourceIdentity(
        git_commit=commit,
        worktree_clean=True,
        source_file_sha256_items=tuple(hashes),
    )


def _validate_identity_against_contract(
    identity: CleanSourceIdentity, contract: PreparationContract
) -> None:
    source_hashes = identity.source_file_sha256
    if source_hashes[REQUIRED_SOURCE_PATHS[0]] != contract.verify_config_sha256:
        raise ValueError("clean source identity points to a different Verify config")
    if source_hashes[REQUIRED_SOURCE_PATHS[1]] != contract.parent_main_config_sha256:
        raise ValueError("clean source identity points to a different parent main config")


def composite_descriptor_contracts(
    identity: CleanSourceIdentity,
    *,
    contract: PreparationContract = PRODUCTION_CONTRACT,
) -> Mapping[str, Mapping[str, Any]]:
    """Return the three immutable, full-hash composite descriptor contracts."""

    _validate_identity_against_contract(identity, contract)
    definitions = (
        ("fmt161", "fmt161_plus_seed4", 161),
        ("real_neighbor36", "real_neighbor36_plus_seed4", 36),
        ("chirality_all35", "chirality_all35_plus_seed4", 35),
    )
    result: dict[str, Mapping[str, Any]] = {}
    source_hashes = identity.source_file_sha256
    for parent_name, composite_name, parent_width in definitions:
        indices = tuple(int(value) for value in representation_indices(parent_name))
        if len(indices) != parent_width:
            raise RuntimeError("parent representation width changed")
        payload = {
            "schema": COMPOSITE_DESCRIPTOR_SCHEMA,
            "parent_descriptor_id": PARENT_DESCRIPTOR_ID,
            "parent_representation": parent_name,
            "parent_representation_indices": list(indices),
            "parent_width": parent_width,
            "composite_representation": composite_name,
            "composite_width": parent_width + 4,
            "concatenation_order": "parent_FMT_coordinates_then_seed_kinematic4",
            "kinematic_algorithm_source_sha256": source_hashes[
                KINEMATIC_ALGORITHM_SOURCE_PATH
            ],
            "seed_time_sampler_source_sha256": source_hashes[SAMPLER_SOURCE_PATH],
            "numerical_dependency_source_sha256": {
                path: source_hashes[path]
                for path in NUMERICAL_DEPENDENCY_SOURCE_PATHS
            },
            "primitive_order": list(FROZEN_PRIMITIVE_ORDER),
            "sampling_contract": (
                "seven_synchronous_seed_velocities_at_portable_frame_zero_"
                "with_production_RK4_interpolator"
            ),
            "kinematic_feature_order": list(FROZEN_KINEMATIC_FEATURE_ORDER),
            "float_contract": {
                "velocity": "float32",
                "derivative": "float64",
                "kinematic_serialization": "float32",
            },
            "raw_kinematic_block_weight": 1.0,
        }
        digest = canonical_json_sha256(payload)
        payload["descriptor_id"] = f"{composite_name}_sha256_{digest}"
        result[composite_name] = _deep_freeze(payload)  # type: ignore[assignment]
    return MappingProxyType(result)


def _descriptor_json(
    descriptors: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return the immutable contracts in their canonical JSON representation."""

    return {
        name: _canonical_json_copy(dict(payload))
        for name, payload in descriptors.items()
    }


def _identity_payload(identity: CleanSourceIdentity) -> dict[str, Any]:
    return {
        "git_commit": identity.git_commit,
        "worktree_clean": True,
        "verify_config_sha256": identity.source_file_sha256[REQUIRED_SOURCE_PATHS[0]],
        "source_file_sha256": dict(identity.source_file_sha256_items),
        "source_file_sha256_content_sha256": identity.source_content_sha256,
    }


_SYNTHETIC_OFFSETS = np.asarray(
    (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ),
    dtype=np.float64,
)


def _synthetic_affine_velocity(
    matrix: np.ndarray,
    translation: np.ndarray,
    physical_dx: np.ndarray,
    *,
    centers: np.ndarray | None = None,
) -> np.ndarray:
    dx = np.asarray(physical_dx, dtype=np.float64)
    if centers is None:
        centers = np.zeros((len(dx), 3), dtype=np.float64)
    positions = centers[:, None, :] + dx[:, None, None] * _SYNTHETIC_OFFSETS[None]
    values = positions @ np.asarray(matrix, dtype=np.float64).T
    values += np.asarray(translation, dtype=np.float64)[None, None, :]
    return np.ascontiguousarray(values, dtype=np.float32)


def _synthetic_window(*, nonlinear: bool = False) -> FlowWindow3D:
    x = np.linspace(-1.0, 1.0, 17 if not nonlinear else 9, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, 17 if not nonlinear else 9, dtype=np.float64)
    z = np.linspace(-1.0, 1.0, 17 if not nonlinear else 9, dtype=np.float64)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    if nonlinear:
        first = np.stack(
            (
                0.25 + xx * yy + 0.1 * zz,
                -0.5 + yy * zz - 0.2 * xx,
                1.0 + zz * xx + 0.3 * yy,
            ),
            axis=-1,
        )
    else:
        matrix = np.asarray(
            ((0.5, -1.0, 0.25), (1.5, 0.75, -0.5), (-0.25, 1.0, 1.25)),
            dtype=np.float64,
        )
        first = np.stack((xx, yy, zz), axis=-1) @ matrix.T
        first += np.asarray((0.25, -0.5, 1.0), dtype=np.float64)
    second = first * np.float64(-7.0) + np.float64(123.0)
    return FlowWindow3D(
        velocity=np.ascontiguousarray(np.stack((first, second)), dtype=np.float32),
        coordinates_xyz=(x, y, z),
        time=np.asarray((8.0, 8.25), dtype=np.float64),
        source_path="synthetic-only.nc",
        source_start_index=0,
        spatial_strides={"x": 1, "y": 1, "z": 1},
        components=("u", "v", "w"),
        coordinate_sources={
            "x": "synthetic",
            "y": "synthetic",
            "z": "synthetic",
            "t": "synthetic",
        },
    )


def _synthetic_parent(scale_ids: Sequence[int]) -> ParentKinematicProjection:
    scale = np.asarray(tuple(scale_ids), dtype=np.int32)
    block = (scale >= 1000).astype(np.int8)
    legacy_count = int(np.sum(block == 0))
    expanded_count = int(np.sum(block == 1))
    center_count = max(legacy_count, expanded_count, 1)
    center = np.concatenate(
        (
            np.arange(legacy_count, dtype=np.int64),
            np.arange(expanded_count, dtype=np.int64),
        )
    )
    assigned = block.astype(np.int64) * center_count + center
    return ParentKinematicProjection(
        seeds_xyz=np.zeros((2 * center_count, 3), dtype=np.float64),
        valid_assigned_row_index=assigned,
        valid_center_seed_index=center,
        valid_scale_block_index=block,
        valid_scale_id=scale,
        center_sample_time=np.zeros((len(scale), 32), dtype=np.float32),
    )


def _expect_synthetic_value_error(function: Any, *args: Any, **kwargs: Any) -> None:
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise RuntimeError(f"synthetic rejection oracle did not reject: {function}")


def _synthetic_check_record(name: str, detail: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _canonical_json_copy(dict(detail))
    return {
        "name": name,
        "status": "passed",
        "detail": normalized,
        "detail_content_sha256": canonical_json_sha256(normalized),
    }


def _execute_synthetic_checks() -> tuple[dict[str, bool], list[dict[str, Any]]]:
    """Execute every frozen oracle against production sampler/core functions."""

    records: list[dict[str, Any]] = []
    matrix = np.asarray(
        ((1.5, -0.25, 0.75), (0.5, -2.0, 0.125), (-1.0, 0.375, 2.25)),
        dtype=np.float64,
    )
    all_dx = np.asarray(FROZEN_DX_GRID_SCALE_BY_ID * np.float64(0.2), dtype=np.float64)
    all_velocity = _synthetic_affine_velocity(matrix, np.zeros(3), all_dx)
    all_gradient = compute_seed_time_velocity_gradient(
        all_velocity, all_dx, primitive_order=FROZEN_PRIMITIVE_ORDER
    )
    if not np.allclose(all_gradient, matrix[None], rtol=2.0e-6, atol=2.0e-6):
        raise RuntimeError("affine gradient oracle failed across the frozen dx table")
    records.append(
        _synthetic_check_record(
            SYNTHETIC_CHECK_NAMES[0],
            {
                "row_count": 2000,
                "dx_sha256": canonical_array_sha256(all_dx),
                "gradient_sha256": canonical_array_sha256(all_gradient),
            },
        )
    )

    translation_velocity = np.asarray([[[3.0, -4.0, 7.0]] * 7], dtype=np.float32)
    translation_feature = compute_seed_time_kinematic4(
        translation_velocity,
        np.float64(0.25),
        primitive_order=FROZEN_PRIMITIVE_ORDER,
    )
    if not np.array_equal(translation_feature, np.zeros((1, 4), dtype=np.float32)):
        raise RuntimeError("rigid translation oracle failed")
    records.append(
        _synthetic_check_record(
            SYNTHETIC_CHECK_NAMES[1],
            {"feature_sha256": canonical_array_sha256(translation_feature)},
        )
    )

    angular = np.asarray((1.0, -2.0, 0.5), dtype=np.float64)
    rotation_matrix = np.asarray(
        (
            (0.0, -angular[2], angular[1]),
            (angular[2], 0.0, -angular[0]),
            (-angular[1], angular[0], 0.0),
        ),
        dtype=np.float64,
    )
    dx3 = np.asarray((0.125, 0.75, 2.5), dtype=np.float64)
    rotation_feature = compute_seed_time_kinematic4(
        _synthetic_affine_velocity(rotation_matrix, np.asarray((4.0, -1.0, 2.0)), dx3),
        dx3,
        primitive_order=FROZEN_PRIMITIVE_ORDER,
    )
    rotation_expected = np.broadcast_to(
        np.asarray(
            (2.0 * np.linalg.norm(angular), 0.0, 0.0, np.dot(angular, angular)),
            dtype=np.float32,
        ),
        rotation_feature.shape,
    )
    if not np.allclose(rotation_feature, rotation_expected, rtol=2.0e-6, atol=2.0e-6):
        raise RuntimeError("rigid rotation oracle failed")
    records.append(
        _synthetic_check_record(
            SYNTHETIC_CHECK_NAMES[2],
            {"feature_sha256": canonical_array_sha256(rotation_feature)},
        )
    )

    strain_rate = 2.0
    strain_matrix = np.diag((strain_rate, -strain_rate, 0.0))
    strain_feature = compute_seed_time_kinematic4(
        _synthetic_affine_velocity(strain_matrix, np.zeros(3), np.asarray((0.5,))),
        np.asarray((0.5,), dtype=np.float64),
        primitive_order=FROZEN_PRIMITIVE_ORDER,
    )
    if not np.allclose(
        strain_feature[0],
        (0.0, np.sqrt(2.0) * strain_rate, 0.0, -(strain_rate**2)),
        rtol=1.0e-6,
        atol=1.0e-6,
    ):
        raise RuntimeError("pure strain oracle failed")
    records.append(
        _synthetic_check_record(
            SYNTHETIC_CHECK_NAMES[3],
            {"feature_sha256": canonical_array_sha256(strain_feature)},
        )
    )

    expansion_rows: list[np.ndarray] = []
    for rate in (1.25, -0.75):
        feature = compute_seed_time_kinematic4(
            _synthetic_affine_velocity(
                rate * np.eye(3), np.zeros(3), np.asarray((0.5,), dtype=np.float64)
            ),
            np.asarray((0.5,), dtype=np.float64),
            primitive_order=FROZEN_PRIMITIVE_ORDER,
        )[0]
        expected = np.asarray(
            (0.0, np.sqrt(3.0) * abs(rate), 3.0 * rate, -1.5 * rate**2),
            dtype=np.float32,
        )
        if not np.allclose(feature, expected, rtol=1.0e-6, atol=1.0e-6):
            raise RuntimeError("signed isotropic expansion oracle failed")
        expansion_rows.append(feature)
    expansion = np.ascontiguousarray(expansion_rows, dtype=np.float32)
    records.append(
        _synthetic_check_record(
            SYNTHETIC_CHECK_NAMES[4],
            {"feature_sha256": canonical_array_sha256(expansion)},
        )
    )

    proper_rotation = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    centers = np.asarray(
        ((0.25, -0.5, 0.75), (-0.75, 1.0, 0.5), (1.0, 0.25, -0.5)),
        dtype=np.float64,
    )
    original = _synthetic_affine_velocity(
        matrix, np.asarray((1.0, -2.0, 0.5)), dx3, centers=centers
    )
    transformed_matrix = proper_rotation @ matrix @ proper_rotation.T
    transformed = _synthetic_affine_velocity(
        transformed_matrix,
        np.asarray((-4.0, 2.5, 1.0)),
        dx3,
        centers=centers @ proper_rotation.T + np.asarray((3.0, -1.0, 0.25)),
    )
    original_feature = compute_seed_time_kinematic4(
        original, dx3, primitive_order=FROZEN_PRIMITIVE_ORDER
    )
    transformed_feature = compute_seed_time_kinematic4(
        transformed, dx3, primitive_order=FROZEN_PRIMITIVE_ORDER
    )
    if not np.allclose(original_feature, transformed_feature, rtol=5.0e-6, atol=5.0e-6):
        raise RuntimeError("proper rotation/translation invariance oracle failed")
    records.append(
        _synthetic_check_record(
            SYNTHETIC_CHECK_NAMES[5],
            {
                "original_sha256": canonical_array_sha256(original_feature),
                "transformed_sha256": canonical_array_sha256(transformed_feature),
            },
        )
    )

    batch_dx = np.asarray((0.125, 0.25, 0.75, 1.5, 2.5), dtype=np.float64)
    batch_velocity = _synthetic_affine_velocity(
        matrix, np.asarray((1.0, 2.0, -3.0)), batch_dx
    )
    batch = compute_seed_time_kinematic4(
        batch_velocity, batch_dx, primitive_order=FROZEN_PRIMITIVE_ORDER
    )
    chunks = np.concatenate(
        (
            compute_seed_time_kinematic4(
                batch_velocity[:2], batch_dx[:2], primitive_order=FROZEN_PRIMITIVE_ORDER
            ),
            compute_seed_time_kinematic4(
                batch_velocity[2:], batch_dx[2:], primitive_order=FROZEN_PRIMITIVE_ORDER
            ),
        )
    )
    singles = np.concatenate(
        tuple(
            compute_seed_time_kinematic4(
                batch_velocity[index : index + 1],
                batch_dx[index],
                primitive_order=FROZEN_PRIMITIVE_ORDER,
            )
            for index in range(len(batch_dx))
        )
    )
    permutation = np.asarray((3, 0, 4, 1, 2), dtype=np.int64)
    permuted = compute_seed_time_kinematic4(
        batch_velocity[permutation],
        batch_dx[permutation],
        primitive_order=FROZEN_PRIMITIVE_ORDER,
    )[np.argsort(permutation)]
    if not (
        np.array_equal(batch, chunks)
        and np.array_equal(batch, singles)
        and np.array_equal(batch, permuted)
    ):
        raise RuntimeError("batch/chunk/permutation invariance oracle failed")
    records.append(
        _synthetic_check_record(
            SYNTHETIC_CHECK_NAMES[6],
            {"feature_sha256": canonical_array_sha256(batch)},
        )
    )

    wrong_velocity = np.array(all_velocity, copy=True)
    wrong_velocity[:, [1, 2], :] = wrong_velocity[:, [2, 1], :]
    wrong_gradient = compute_seed_time_velocity_gradient(
        wrong_velocity, all_dx, primitive_order=FROZEN_PRIMITIVE_ORDER
    )
    if np.allclose(wrong_gradient, matrix[None], rtol=2.0e-6, atol=2.0e-6):
        raise RuntimeError("wrong opposite-pair order was not detected by the affine oracle")
    _expect_synthetic_value_error(
        compute_seed_time_kinematic4,
        all_velocity[:1],
        all_dx[:1],
        primitive_order=(
            "center",
            "x_minus",
            "x_plus",
            "y_plus",
            "y_minus",
            "z_plus",
            "z_minus",
        ),
    )
    records.append(
        _synthetic_check_record(
            SYNTHETIC_CHECK_NAMES[7],
            {"wrong_gradient_sha256": canonical_array_sha256(wrong_gradient)},
        )
    )

    nonlinear_window = _synthetic_window(nonlinear=True)
    sample_centers = np.asarray(((0.0, 0.0, 0.0), (0.25, -0.125, 0.0)), dtype=np.float64)
    sample_dx = np.asarray((0.125, 0.0625), dtype=np.float64)
    sampled = sample_seed_time_velocity_xyz(nonlinear_window, sample_centers, sample_dx)
    field = UnsteadyVectorField3D.from_window(nonlinear_window)
    positions = sample_centers[:, None, :] + sample_dx[:, None, None] * _SYNTHETIC_OFFSETS[None]
    direct = np.empty_like(sampled)
    for row_index in range(len(sample_centers)):
        for line_index in range(7):
            direct[row_index, line_index] = np.asarray(
                _interp4_quadrilinear_scalar(
                    field.field,
                    field.domain_min,
                    field.grid_interval,
                    field.xdim,
                    field.ydim,
                    field.zdim,
                    field.tmin,
                    field.time_interval,
                    field.time_steps,
                    *positions[row_index, line_index],
                    field.tmin,
                ),
                dtype=np.float32,
            )
    if not np.array_equal(sampled, direct):
        raise RuntimeError("production RK4 first-v1 interpolation oracle failed")
    records.append(
        _synthetic_check_record(
            SYNTHETIC_CHECK_NAMES[8],
            {"sampled_velocity_sha256": canonical_array_sha256(sampled)},
        )
    )

    poison_parent = _synthetic_parent((0, 1000))
    allowed_arrays = {
        name: np.asarray(getattr(poison_parent, name))
        for name in PARENT_PROJECTION_MEMBER_NAMES
    }
    with tempfile.TemporaryDirectory() as directory:
        poison_path = Path(directory) / "synthetic_parent_with_poison.npz"
        poison = {
            "valid_labels": np.asarray((object(), object()), dtype=object),
            "reference_labels_all": np.asarray((object(),), dtype=object),
            "ivd_values_all": np.asarray((object(),), dtype=object),
            "ivd_volume": np.asarray((object(),), dtype=object),
            "metadata_json": np.asarray({"forbidden": True}, dtype=object),
        }
        with poison_path.open("xb") as stream:
            np.savez_compressed(stream, **allowed_arrays, **poison)
        loaded_parent = load_parent_kinematic_projection(
            poison_path,
            expected_size_bytes=poison_path.stat().st_size,
            expected_file_sha256=sha256_file(poison_path),
            expected_array_sha256={
                name: canonical_array_sha256(values)
                for name, values in allowed_arrays.items()
            },
        )
    if not FORBIDDEN_PARENT_MEMBER_NAMES.isdisjoint(loaded_parent.opened_member_names):
        raise RuntimeError("narrow parent loader opened a forbidden member")
    records.append(
        _synthetic_check_record(
            SYNTHETIC_CHECK_NAMES[9],
            {"opened_members": list(loaded_parent.opened_member_names)},
        )
    )

    base_parent = _synthetic_parent((0, 1000))
    base_payload = build_seed_time_kinematic_sidecar_payload(base_parent, _synthetic_window())
    missing_payload = SeedTimeKinematicSidecarPayload(
        valid_assigned_row_index=base_payload.valid_assigned_row_index[:1],
        valid_center_seed_index=base_payload.valid_center_seed_index[:1],
        valid_scale_block_index=base_payload.valid_scale_block_index[:1],
        valid_scale_id=base_payload.valid_scale_id[:1],
        seed_velocity_xyz=base_payload.seed_velocity_xyz[:1],
        seed_kinematic4=base_payload.seed_kinematic4[:1],
        physical_dx_by_scale=base_payload.physical_dx_by_scale,
    )
    _expect_synthetic_value_error(validate_sidecar_identity_join, base_parent, missing_payload)
    extra_payload = build_seed_time_kinematic_sidecar_payload(
        _synthetic_parent((0, 1, 1000)), _synthetic_window()
    )
    _expect_synthetic_value_error(validate_sidecar_identity_join, base_parent, extra_payload)
    duplicate = {
        name: np.array(getattr(base_parent, name), copy=True)
        for name in PARENT_PROJECTION_MEMBER_NAMES
    }
    duplicate["valid_assigned_row_index"][1] = duplicate["valid_assigned_row_index"][0]
    _expect_synthetic_value_error(ParentKinematicProjection, **duplicate)
    reordered = _synthetic_parent((0, 1))
    reordered_values = {
        name: np.array(getattr(reordered, name), copy=True)
        for name in PARENT_PROJECTION_MEMBER_NAMES
    }
    for name in (
        "valid_assigned_row_index",
        "valid_center_seed_index",
        "valid_scale_block_index",
        "valid_scale_id",
        "center_sample_time",
    ):
        reordered_values[name] = reordered_values[name][::-1]
    _expect_synthetic_value_error(ParentKinematicProjection, **reordered_values)
    records.append(
        _synthetic_check_record(
            SYNTHETIC_CHECK_NAMES[10],
            {"rejected_cases": ["missing", "duplicate", "extra", "reorder"]},
        )
    )

    if tuple(record["name"] for record in records) != SYNTHETIC_CHECK_NAMES:
        raise RuntimeError("synthetic check order changed")
    checks = {name: True for name in SYNTHETIC_CHECK_NAMES}
    return checks, records


def _synthetic_evidence_payload(
    identity: CleanSourceIdentity,
    *,
    contract: PreparationContract,
) -> dict[str, Any]:
    checks, records = _execute_synthetic_checks()
    descriptors = composite_descriptor_contracts(identity, contract=contract)
    return _with_content_sha256(
        {
            "schema": SYNTHETIC_EVIDENCE_SCHEMA,
            "experiment": EXPERIMENT,
            "status": "passed",
            **_identity_payload(identity),
            "check_count": len(records),
            "checks": records,
            "checks_content_sha256": canonical_json_sha256(records),
            "check_results": checks,
            "check_results_content_sha256": canonical_json_sha256(checks),
            "composite_descriptor_ids": {
                name: str(value["descriptor_id"])
                for name, value in descriptors.items()
            },
            "real_flow_or_cache_access": False,
            "forbidden_dataset_access": False,
        }
    )


def _authenticate_synthetic_evidence(
    path: Path,
    *,
    identity: CleanSourceIdentity,
    contract: PreparationContract,
    reexecute_oracles: bool = False,
    expected_size: int | None = None,
    expected_file_sha256: str | None = None,
) -> Mapping[str, Any]:
    raw, snapshot = _json_from_authenticated_snapshot(
        path,
        expected_size=expected_size,
        expected_sha256=expected_file_sha256,
    )
    if snapshot.size_bytes <= 0 or len(snapshot.sha256) != 64:
        raise ValueError("synthetic evidence file identity is invalid")
    observed = _verify_content_sha256(
        raw, name="synthetic oracle evidence"
    )
    if set(observed) != _SYNTHETIC_EVIDENCE_NAMES:
        raise ValueError("synthetic oracle evidence member set changed")
    expected_identity = {
        "schema": SYNTHETIC_EVIDENCE_SCHEMA,
        "experiment": EXPERIMENT,
        "status": "passed",
        **_identity_payload(identity),
        "real_flow_or_cache_access": False,
        "forbidden_dataset_access": False,
    }
    if any(observed.get(name) != value for name, value in expected_identity.items()):
        raise ValueError("synthetic oracle evidence provenance changed")
    records = observed.get("checks")
    if (
        not isinstance(records, list)
        or len(records) != len(SYNTHETIC_CHECK_NAMES)
        or int(observed.get("check_count", -1)) != len(records)
        or observed.get("checks_content_sha256") != canonical_json_sha256(records)
    ):
        raise ValueError("synthetic oracle evidence check population changed")
    for expected_name, record in zip(SYNTHETIC_CHECK_NAMES, records, strict=True):
        if (
            not isinstance(record, Mapping)
            or set(record) != _SYNTHETIC_CHECK_RECORD_NAMES
            or record.get("name") != expected_name
            or record.get("status") != "passed"
            or not isinstance(record.get("detail"), Mapping)
            or record.get("detail_content_sha256")
            != canonical_json_sha256(dict(record["detail"]))
        ):
            raise ValueError(f"synthetic oracle evidence record changed: {expected_name}")
    checks = observed.get("check_results")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != set(SYNTHETIC_CHECK_NAMES)
        or any(checks[name] is not True for name in SYNTHETIC_CHECK_NAMES)
        or observed.get("check_results_content_sha256")
        != canonical_json_sha256(dict(checks))
    ):
        raise ValueError("synthetic oracle evidence result table changed")
    descriptors = composite_descriptor_contracts(identity, contract=contract)
    expected_ids = {
        name: str(value["descriptor_id"]) for name, value in descriptors.items()
    }
    if observed.get("composite_descriptor_ids") != expected_ids:
        raise ValueError("synthetic oracle evidence descriptor binding changed")
    if reexecute_oracles:
        expected = _synthetic_evidence_payload(identity, contract=contract)
        if observed != expected:
            raise ValueError("synthetic oracle evidence does not reproduce production checks")
    frozen = _deep_freeze(observed)
    if not isinstance(frozen, Mapping):
        raise RuntimeError("synthetic oracle evidence is not immutable")
    return frozen


def write_synthetic_pass_marker(
    run_dir: str | Path,
    *,
    identity: CleanSourceIdentity,
    contract: PreparationContract = PRODUCTION_CONTRACT,
) -> Mapping[str, Any]:
    """Execute production oracles, persist their evidence, then write PASS last."""

    _validate_identity_against_contract(identity, contract)
    root = Path(run_dir).resolve()
    marker_path = root / "SYNTHETIC_PASS.json"
    evidence_path = root / "synthetic_oracle_evidence.json"
    if marker_path.exists() or evidence_path.exists():
        raise FileExistsError("immutable synthetic evidence or marker already exists")
    evidence = _synthetic_evidence_payload(identity, contract=contract)
    _atomic_json_no_overwrite(evidence_path, evidence)
    authenticated_evidence = _authenticate_synthetic_evidence(
        evidence_path,
        identity=identity,
        contract=contract,
        reexecute_oracles=True,
    )
    check_results = dict(authenticated_evidence["check_results"])
    evidence_snapshot = _read_authenticated_bytes(evidence_path)
    rows = [
        {
            "relative_path": evidence_path.name,
            "size_bytes": evidence_snapshot.size_bytes,
            "sha256": evidence_snapshot.sha256,
        }
    ]
    descriptors = composite_descriptor_contracts(identity, contract=contract)
    marker = _with_content_sha256(
        {
            "schema": SYNTHETIC_PASS_SCHEMA,
            "experiment": EXPERIMENT,
            "status": "passed",
            **_identity_payload(identity),
            "check_results": dict(check_results),
            "check_results_content_sha256": canonical_json_sha256(dict(check_results)),
            "evidence": rows,
            "evidence_content_sha256": canonical_json_sha256(rows),
            "composite_descriptor_ids": {
                name: str(value["descriptor_id"]) for name, value in descriptors.items()
            },
            "real_flow_or_cache_access": False,
            "forbidden_dataset_access": False,
            "marker_write_order": "last_after_all_synthetic_evidence_was_closed_and_hashed",
        }
    )
    _atomic_json_no_overwrite(marker_path, marker)
    return authenticate_synthetic_pass_marker(
        marker_path,
        expected_file_sha256=sha256_file(marker_path),
        identity=identity,
        contract=contract,
    )


def authenticate_synthetic_pass_marker(
    path: str | Path,
    *,
    expected_file_sha256: str,
    identity: CleanSourceIdentity,
    contract: PreparationContract = PRODUCTION_CONTRACT,
) -> Mapping[str, Any]:
    """Authenticate the marker, all evidence files, and source bindings."""

    _validate_identity_against_contract(identity, contract)
    source = Path(path).resolve()
    if source.name != "SYNTHETIC_PASS.json":
        raise ValueError("synthetic pass marker must be named SYNTHETIC_PASS.json")
    expected_digest = _lower_hex(
        expected_file_sha256, 64, name="synthetic marker file SHA-256"
    )
    raw_marker, _marker_snapshot = _json_from_authenticated_snapshot(
        source,
        expected_sha256=expected_digest,
    )
    marker = _verify_content_sha256(
        raw_marker, name="synthetic pass marker"
    )
    if set(marker) != _SYNTHETIC_MARKER_NAMES:
        raise ValueError("synthetic marker member set changed")
    expected_identity = _identity_payload(identity)
    drift = {
        name: (marker.get(name), value)
        for name, value in {
            "schema": SYNTHETIC_PASS_SCHEMA,
            "experiment": EXPERIMENT,
            "status": "passed",
            **expected_identity,
            "real_flow_or_cache_access": False,
            "forbidden_dataset_access": False,
            "marker_write_order": "last_after_all_synthetic_evidence_was_closed_and_hashed",
        }.items()
        if marker.get(name) != value
    }
    if drift:
        raise ValueError(f"synthetic marker provenance changed: {drift}")
    checks = marker.get("check_results")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != set(SYNTHETIC_CHECK_NAMES)
        or any(checks[name] is not True for name in SYNTHETIC_CHECK_NAMES)
        or marker.get("check_results_content_sha256")
        != canonical_json_sha256(dict(checks))
    ):
        raise ValueError("synthetic marker check population changed")
    evidence = marker.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 1:
        raise ValueError("synthetic marker must bind exactly one oracle evidence report")
    if marker.get("evidence_content_sha256") != canonical_json_sha256(evidence):
        raise ValueError("synthetic evidence row SHA-256 mismatch")
    seen: set[Path] = set()
    evidence_expected_size: int | None = None
    evidence_expected_sha256: str | None = None
    for row in evidence:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("synthetic evidence row contract changed")
        relative = _safe_relative(row["relative_path"], name="synthetic evidence path")
        evidence_path = (source.parent / relative).resolve()
        try:
            evidence_path.relative_to(source.parent)
        except ValueError as error:
            raise ValueError("synthetic evidence resolves outside the run directory") from error
        if evidence_path in seen:
            raise ValueError("synthetic evidence rows are duplicated")
        evidence_expected_size = _positive_integer(
            row["size_bytes"], name="evidence size"
        )
        evidence_expected_sha256 = _lower_hex(
            row["sha256"], 64, name="evidence SHA-256"
        )
        seen.add(evidence_path)
    evidence_path = source.parent / "synthetic_oracle_evidence.json"
    if seen != {evidence_path.resolve()}:
        raise ValueError("synthetic marker evidence path changed")
    authenticated_evidence = _authenticate_synthetic_evidence(
        evidence_path.resolve(),
        identity=identity,
        contract=contract,
        expected_size=evidence_expected_size,
        expected_file_sha256=evidence_expected_sha256,
    )
    if dict(authenticated_evidence["check_results"]) != dict(checks):
        raise ValueError("synthetic marker results disagree with reproduced oracle evidence")
    descriptors = composite_descriptor_contracts(identity, contract=contract)
    expected_ids = {
        name: str(value["descriptor_id"]) for name, value in descriptors.items()
    }
    if marker.get("composite_descriptor_ids") != expected_ids:
        raise ValueError("synthetic marker composite descriptor IDs changed")
    frozen = _deep_freeze(marker)
    if not isinstance(frozen, Mapping):
        raise RuntimeError("authenticated synthetic marker is not immutable")
    return frozen


def _load_json_file_stable(path: Path, *, expected_size: int, expected_sha256: str) -> dict[str, Any]:
    value, _snapshot = _json_from_authenticated_snapshot(
        path,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
    )
    return value


def _expected_keys(contract: PreparationContract) -> tuple[tuple[str, int], ...]:
    return tuple(
        (dataset, ordinal)
        for dataset in contract.datasets
        for ordinal in range(contract.source_count)
    )


def _authenticate_parent_input_manifest(
    path: Path, contract: PreparationContract
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    manifest = _load_json_file_stable(
        path,
        expected_size=contract.parent_input_manifest_size,
        expected_sha256=contract.parent_input_manifest_sha256,
    )
    if set(manifest) != _PARENT_MANIFEST_NAMES:
        raise ValueError("parent train-cache input manifest member set changed")
    expected_top = {
        "schema": PARENT_INPUT_SCHEMA,
        "experiment": "Verify_LongArcHorizon_1.1",
        "parent_experiment": "mainExp_TemplateMatching_3.1",
        "git_commit": contract.parent_cache_builder_commit,
        "main_config_sha256": contract.parent_main_config_sha256,
        "input_scope": "exactly_32_train_cache_shards_and_sidecars",
        "test_dataset_access": False,
        "row_count": contract.row_count,
        "rows_content_sha256": contract.parent_input_rows_sha256,
    }
    drift = {
        name: (manifest.get(name), value)
        for name, value in expected_top.items()
        if manifest.get(name) != value
    }
    if drift:
        raise ValueError(f"parent train-cache input manifest changed: {drift}")
    rows = manifest.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != contract.row_count
        or canonical_json_sha256(rows) != contract.parent_input_rows_sha256
    ):
        raise ValueError("parent train-cache input row population changed")
    expected_keys = _expected_keys(contract)
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for expected_key, raw_row in zip(expected_keys, rows, strict=True):
        if not isinstance(raw_row, Mapping) or set(raw_row) != _PARENT_MANIFEST_ROW_NAMES:
            raise ValueError("parent train-cache input row member set changed")
        row = dict(raw_row)
        key = (str(row.get("dataset", "")), int(row.get("source_ordinal", -1)))
        if key != expected_key or key in indexed:
            raise ValueError("parent train-cache input rows are missing, duplicated, extra, or reordered")
        _nonnegative_integer(row.get("source_index"), name="parent source_index")
        _positive_integer(row.get("cache_size_bytes"), name="parent cache size")
        _lower_hex(row.get("cache_file_sha256"), 64, name="parent cache SHA-256")
        _positive_integer(row.get("sidecar_size_bytes"), name="parent evidence sidecar size")
        _lower_hex(row.get("sidecar_file_sha256"), 64, name="parent evidence sidecar SHA-256")
        _require_no_forbidden_path(row["cache_path"], name="parent cache path")
        _require_no_forbidden_path(row["sidecar_path"], name="parent evidence sidecar path")
        indexed[key] = row
    return manifest, indexed


def _authenticate_portable_marker(
    path: Path,
    contract: PreparationContract,
    *,
    parent_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    expected_marker_sha = contract.train_portable_marker_sha256
    marker, marker_snapshot = _json_from_authenticated_snapshot(
        path,
        expected_sha256=expected_marker_sha,
    )
    size = marker_snapshot.size_bytes
    digest = marker_snapshot.sha256
    parent_evidence = parent_manifest.get("train_portable_population_pass")
    if not isinstance(parent_evidence, Mapping):
        raise ValueError("parent manifest lacks train portable population evidence")
    if (
        str(Path(str(parent_evidence.get("path", ""))).resolve()) != str(path.resolve())
        or int(parent_evidence.get("file_size", -1)) != size
        or parent_evidence.get("file_sha256") != digest
        or parent_evidence.get("access_scope") != "train-only"
    ):
        raise ValueError("parent manifest points to a different portable marker")
    if set(marker) != _PORTABLE_MARKER_NAMES:
        raise ValueError("train portable marker member set changed")
    portable_root = Path(str(marker.get("portable_root", ""))).resolve()
    expected_top = {
        "schema": PARENT_PORTABLE_MARKER_SCHEMA,
        "experiment": "mainExp_TemplateMatching_3.1",
        "status": "passed",
        "access_scope": "train-only",
        "git_commit": contract.parent_cache_builder_commit,
        "worktree_clean": True,
        "config_sha256": contract.parent_main_config_sha256,
        "dataset_count": len(contract.datasets),
        "window_count": contract.row_count,
        "train_coverage_pass_file_sha256": None,
    }
    drift = {
        name: (marker.get(name), value)
        for name, value in expected_top.items()
        if marker.get(name) != value
    }
    if drift:
        raise ValueError(f"train portable marker changed: {drift}")
    rows = marker.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != contract.row_count
        or marker.get("rows_content_sha256") != canonical_json_sha256(rows)
        or parent_evidence.get("rows_content_sha256") != marker.get("rows_content_sha256")
    ):
        raise ValueError("train portable marker row population changed")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for expected_key, raw_row in zip(_expected_keys(contract), rows, strict=True):
        if not isinstance(raw_row, Mapping) or set(raw_row) != _PORTABLE_MARKER_ROW_NAMES:
            raise ValueError("train portable marker row member set changed")
        row = dict(raw_row)
        key = (str(row.get("dataset", "")), int(row.get("source_ordinal", -1)))
        if key != expected_key or key in indexed:
            raise ValueError("portable marker rows are missing, duplicated, extra, or reordered")
        if row.get("split") != "train":
            raise ValueError("portable marker contains a non-train row")
        _nonnegative_integer(row.get("source_start_index"), name="portable source index")
        relative = _safe_relative(row.get("relative_path"), name="portable relative path")
        manifest_relative = _safe_relative(
            row.get("manifest_relative_path"), name="portable manifest relative path"
        )
        _require_no_forbidden_path(relative, name="portable relative path")
        _require_no_forbidden_path(manifest_relative, name="portable manifest path")
        portable_path = (portable_root / relative).resolve()
        dataset_manifest_path = (portable_root / manifest_relative).resolve()
        for resolved, name in (
            (portable_path, "portable path"),
            (dataset_manifest_path, "dataset manifest path"),
        ):
            try:
                resolved.relative_to(portable_root)
            except ValueError as error:
                raise ValueError(f"{name} resolves outside portable_root") from error
        portable_size, portable_sha = _stable_file_identity(portable_path)
        if portable_size != _positive_integer(row.get("file_size"), name="portable size") or portable_sha != _lower_hex(
            row.get("file_sha256"), 64, name="portable SHA-256"
        ):
            raise ValueError("portable file identity changed")
        manifest_size, manifest_sha = _stable_file_identity(dataset_manifest_path)
        if manifest_sha != _lower_hex(
            row.get("manifest_file_sha256"), 64, name="dataset manifest SHA-256"
        ):
            raise ValueError("portable dataset manifest file SHA-256 changed")
        row["__portable_path"] = str(portable_path)
        row["__dataset_manifest_path"] = str(dataset_manifest_path)
        row["__dataset_manifest_size"] = manifest_size
        indexed[key] = row
    return marker, indexed


def _portable_window_evidence(
    marker_row: Mapping[str, Any],
    *,
    dataset: str,
    family: str,
    source_ordinal: int,
    source_index: int,
    contract: PreparationContract,
) -> dict[str, Any]:
    manifest_path = Path(str(marker_row["__dataset_manifest_path"])).resolve()
    manifest_size = int(marker_row["__dataset_manifest_size"])
    manifest_sha = str(marker_row["manifest_file_sha256"])
    dataset_manifest = _load_json_file_stable(
        manifest_path, expected_size=manifest_size, expected_sha256=manifest_sha
    )
    if set(dataset_manifest) != _DATASET_MANIFEST_NAMES:
        raise ValueError("portable dataset manifest member set changed")
    if dataset_manifest.get("schema") != PARENT_DATASET_MANIFEST_SCHEMA:
        raise ValueError("portable dataset manifest schema changed")
    claimed = dataset_manifest.get("manifest_content_sha256")
    digest_payload = dict(dataset_manifest)
    digest_payload.pop("manifest_content_sha256", None)
    if claimed != canonical_json_sha256(digest_payload):
        raise ValueError("portable dataset manifest content SHA-256 changed")
    expected_top = {
        "experiment": "mainExp_TemplateMatching_3.1",
        "config_sha256": contract.parent_main_config_sha256,
        "builder_git_commit": contract.parent_cache_builder_commit,
        "dataset": dataset,
        "physical_family": family,
        "split": "train",
        "window_count": contract.source_count,
    }
    drift = {
        name: (dataset_manifest.get(name), value)
        for name, value in expected_top.items()
        if dataset_manifest.get(name) != value
    }
    if drift:
        raise ValueError(f"portable dataset manifest provenance changed: {drift}")
    windows = dataset_manifest.get("windows")
    if not isinstance(windows, list) or len(windows) != contract.source_count:
        raise ValueError("portable dataset manifest window population changed")
    ordinals = [int(row.get("source_ordinal", -1)) for row in windows if isinstance(row, Mapping)]
    if ordinals != list(range(contract.source_count)):
        raise ValueError("portable dataset manifest rows are missing, duplicated, or reordered")
    window_row = windows[source_ordinal]
    if not isinstance(window_row, Mapping):
        raise ValueError("portable dataset manifest window row is invalid")
    expected_window = {
        "dataset": dataset,
        "physical_family": family,
        "split": "train",
        "source_ordinal": source_ordinal,
        "source_start_index": source_index,
        "experiment": "mainExp_TemplateMatching_3.1",
        "config_sha256": contract.parent_main_config_sha256,
        "builder_git_commit": contract.parent_cache_builder_commit,
        "file_size": int(marker_row["file_size"]),
        "file_sha256": str(marker_row["file_sha256"]),
    }
    window_drift = {
        name: (window_row.get(name), value)
        for name, value in expected_window.items()
        if window_row.get(name) != value
    }
    if window_drift:
        raise ValueError(f"portable window/marker provenance changed: {window_drift}")
    marker_relative = Path(str(marker_row["relative_path"]))
    if str(window_row.get("relative_path", "")) != marker_relative.name:
        raise ValueError("portable marker and dataset manifest resolve different windows")
    array_hashes = window_row.get("array_sha256")
    if not isinstance(array_hashes, Mapping) or set(array_hashes) != set(_PORTABLE_ARRAY_NAMES):
        raise ValueError("portable array SHA-256 member set or order changed")
    normalized_hashes = {
        name: _lower_hex(array_hashes[name], 64, name=f"portable {name} SHA-256")
        for name in _PORTABLE_ARRAY_NAMES
    }
    combined = str(window_row.get("combined_array_sha256", ""))
    if combined != canonical_json_sha256(normalized_hashes):
        raise ValueError("portable combined array SHA-256 changed")
    metadata = {
        key: value
        for key, value in window_row.items()
        if key not in {"relative_path", "file_size", "file_sha256"}
    }
    metadata_sha = canonical_json_sha256(metadata)
    if metadata_sha != marker_row.get("portable_metadata_sha256"):
        raise ValueError("portable metadata SHA-256 differs between marker and manifest")
    return {
        "path": str(Path(str(marker_row["__portable_path"])).resolve()),
        "size_bytes": int(marker_row["file_size"]),
        "file_sha256": str(marker_row["file_sha256"]),
        "schema": PORTABLE_FLOW_SCHEMA,
        "builder_git_commit": contract.parent_cache_builder_commit,
        "config_sha256": contract.parent_main_config_sha256,
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_size_bytes": manifest_size,
        "dataset_manifest_file_sha256": manifest_sha,
        "dataset_manifest_content_sha256": str(claimed),
        "portable_metadata_sha256": metadata_sha,
        "array_sha256": normalized_hashes,
        "combined_array_sha256": combined,
    }


def _narrow_parent_evidence(
    row: Mapping[str, Any], contract: PreparationContract
) -> dict[str, Any]:
    path = Path(str(row["cache_path"])).resolve()
    size = int(row["cache_size_bytes"])
    digest = str(row["cache_file_sha256"])
    snapshot = _read_authenticated_bytes(
        path,
        expected_size=size,
        expected_sha256=digest,
    )
    with np.load(io.BytesIO(snapshot.content), allow_pickle=False) as archive:
        missing = set(PARENT_PROJECTION_MEMBER_NAMES).difference(archive.files)
        if missing:
            raise ValueError(f"parent cache misses allowed projection members: {sorted(missing)}")
        arrays = {
            name: np.asarray(archive[name]) for name in PARENT_PROJECTION_MEMBER_NAMES
        }
    # Construction validates exact dtypes, shapes, time-zero, and row identity.
    from .seed_time_kinematic_sidecar import ParentKinematicProjection

    projection = ParentKinematicProjection(**arrays)
    hashes = {
        name: canonical_array_sha256(np.asarray(getattr(projection, name)))
        for name in PARENT_PROJECTION_MEMBER_NAMES
    }
    return {
        "path": str(path),
        "size_bytes": size,
        "file_sha256": digest,
        "schema": PARENT_CACHE_SCHEMA,
        "builder_git_commit": contract.parent_cache_builder_commit,
        "config_sha256": contract.parent_main_config_sha256,
        "allowed_array_sha256": hashes,
        "opened_members": list(PARENT_PROJECTION_MEMBER_NAMES),
    }


def build_kinematic_input_manifest(
    output_path: str | Path,
    *,
    parent_input_manifest_path: str | Path,
    train_portable_marker_path: str | Path,
    synthetic_pass_path: str | Path,
    synthetic_pass_file_sha256: str,
    identity: CleanSourceIdentity,
    contract: PreparationContract = PRODUCTION_CONTRACT,
) -> Mapping[str, Any]:
    """Freeze the exact 32 input rows without opening any portable NPZ."""

    _validate_identity_against_contract(identity, contract)
    synthetic = authenticate_synthetic_pass_marker(
        synthetic_pass_path,
        expected_file_sha256=synthetic_pass_file_sha256,
        identity=identity,
        contract=contract,
    )
    parent_path = Path(parent_input_manifest_path).resolve()
    portable_marker_path = Path(train_portable_marker_path).resolve()
    _require_no_forbidden_path(parent_path, name="parent input manifest path")
    _require_no_forbidden_path(portable_marker_path, name="portable marker path")
    parent_manifest, parent_rows = _authenticate_parent_input_manifest(parent_path, contract)
    portable_marker, portable_rows = _authenticate_portable_marker(
        portable_marker_path, contract, parent_manifest=parent_manifest
    )
    rows: list[dict[str, Any]] = []
    for dataset, ordinal in _expected_keys(contract):
        parent_row = parent_rows[(dataset, ordinal)]
        portable_row = portable_rows[(dataset, ordinal)]
        source_index = int(parent_row["source_index"])
        if int(portable_row["source_start_index"]) != source_index:
            raise ValueError("parent cache and portable source indices differ")
        family = contract.family_by_dataset[dataset]
        rows.append(
            {
                "dataset": dataset,
                "physical_family": family,
                "split": "train",
                "source_ordinal": ordinal,
                "source_index": source_index,
                "parent_cache": _narrow_parent_evidence(parent_row, contract),
                "portable": _portable_window_evidence(
                    portable_row,
                    dataset=dataset,
                    family=family,
                    source_ordinal=ordinal,
                    source_index=source_index,
                    contract=contract,
                ),
            }
        )
    descriptors = composite_descriptor_contracts(identity, contract=contract)
    value = _with_content_sha256(
        {
            "schema": INPUT_MANIFEST_SCHEMA,
            "experiment": EXPERIMENT,
            "status": "frozen",
            **_identity_payload(identity),
            "synthetic_pass": {
                "path": str(Path(synthetic_pass_path).resolve()),
                "size_bytes": int(Path(synthetic_pass_path).stat().st_size),
                "file_sha256": str(synthetic_pass_file_sha256),
                "content_sha256": str(synthetic["content_sha256"]),
            },
            "parent_input_manifest": {
                "path": str(parent_path),
                "size_bytes": contract.parent_input_manifest_size,
                "file_sha256": contract.parent_input_manifest_sha256,
                "rows_content_sha256": contract.parent_input_rows_sha256,
            },
            "train_portable_population_marker": {
                "path": str(portable_marker_path),
                "size_bytes": int(portable_marker_path.stat().st_size),
                "file_sha256": contract.train_portable_marker_sha256,
                "rows_content_sha256": str(portable_marker["rows_content_sha256"]),
            },
            "parent_descriptor_id": PARENT_DESCRIPTOR_ID,
            "composite_descriptors": _descriptor_json(descriptors),
            "input_scope": "every_and_only_8_train_datasets_times_4_source_windows",
            "portable_npz_opened_during_freeze": False,
            "forbidden_dataset_access": False,
            "row_count": len(rows),
            "rows": rows,
            "rows_content_sha256": canonical_json_sha256(rows),
        }
    )
    output = Path(output_path).resolve()
    _require_no_forbidden_path(output, name="kinematic input manifest output")
    _atomic_json_no_overwrite(output, value)
    return authenticate_kinematic_input_manifest(
        output,
        expected_file_sha256=sha256_file(output),
        identity=identity,
        contract=contract,
        authenticate_all_referenced_rows=True,
    )


def _validate_input_row(
    raw_row: object,
    *,
    expected_key: tuple[str, int],
    contract: PreparationContract,
) -> dict[str, Any]:
    if not isinstance(raw_row, Mapping) or set(raw_row) != _INPUT_ROW_NAMES:
        raise ValueError("kinematic input row member set changed")
    row = dict(raw_row)
    dataset, ordinal = expected_key
    expected_identity = {
        "dataset": dataset,
        "physical_family": contract.family_by_dataset[dataset],
        "split": "train",
        "source_ordinal": ordinal,
    }
    drift = {
        name: (row.get(name), value)
        for name, value in expected_identity.items()
        if row.get(name) != value
    }
    if drift:
        raise ValueError(f"kinematic input row identity changed: {drift}")
    _nonnegative_integer(row.get("source_index"), name="source_index")
    parent = row.get("parent_cache")
    portable = row.get("portable")
    if not isinstance(parent, Mapping) or set(parent) != _PARENT_ROW_NAMES:
        raise ValueError("kinematic parent row member set changed")
    if not isinstance(portable, Mapping) or set(portable) != _PORTABLE_ROW_NAMES:
        raise ValueError("kinematic portable row member set changed")
    if parent.get("schema") != PARENT_CACHE_SCHEMA:
        raise ValueError("kinematic parent cache schema changed")
    if parent.get("builder_git_commit") != contract.parent_cache_builder_commit or parent.get(
        "config_sha256"
    ) != contract.parent_main_config_sha256:
        raise ValueError("kinematic parent cache provenance changed")
    if tuple(parent.get("opened_members", ())) != PARENT_PROJECTION_MEMBER_NAMES:
        raise ValueError("kinematic parent opened-member audit changed")
    parent_hashes = parent.get("allowed_array_sha256")
    if not isinstance(parent_hashes, Mapping) or set(parent_hashes) != set(PARENT_PROJECTION_MEMBER_NAMES):
        raise ValueError("kinematic parent array hash population changed")
    for name in PARENT_PROJECTION_MEMBER_NAMES:
        _lower_hex(parent_hashes[name], 64, name=f"parent {name} SHA-256")
    if portable.get("schema") != PORTABLE_FLOW_SCHEMA:
        raise ValueError("kinematic portable schema changed")
    if portable.get("builder_git_commit") != contract.parent_cache_builder_commit or portable.get(
        "config_sha256"
    ) != contract.parent_main_config_sha256:
        raise ValueError("kinematic portable provenance changed")
    portable_hashes = portable.get("array_sha256")
    if not isinstance(portable_hashes, Mapping) or set(portable_hashes) != set(_PORTABLE_ARRAY_NAMES):
        raise ValueError("kinematic portable array hash population changed")
    normalized = {
        name: _lower_hex(portable_hashes[name], 64, name=f"portable {name} SHA-256")
        for name in _PORTABLE_ARRAY_NAMES
    }
    if portable.get("combined_array_sha256") != canonical_json_sha256(normalized):
        raise ValueError("kinematic portable combined array SHA-256 changed")
    for identity_name, value in (
        ("parent cache", parent),
        ("portable", portable),
    ):
        _positive_integer(value.get("size_bytes"), name=f"{identity_name} size")
        _lower_hex(value.get("file_sha256"), 64, name=f"{identity_name} SHA-256")
        _require_no_forbidden_path(value.get("path", ""), name=f"{identity_name} path")
    return row


def authenticate_kinematic_input_manifest(
    path: str | Path,
    *,
    expected_file_sha256: str,
    identity: CleanSourceIdentity,
    contract: PreparationContract = PRODUCTION_CONTRACT,
    authenticate_all_referenced_rows: bool = True,
) -> Mapping[str, Any]:
    """Authenticate manifest structure and, by default, all referenced files."""

    _validate_identity_against_contract(identity, contract)
    source = Path(path).resolve()
    expected_digest = _lower_hex(
        expected_file_sha256, 64, name="kinematic input manifest SHA-256"
    )
    raw_value, _manifest_snapshot = _json_from_authenticated_snapshot(
        source,
        expected_sha256=expected_digest,
    )
    value = _verify_content_sha256(
        raw_value, name="kinematic input manifest"
    )
    if set(value) != _INPUT_TOP_LEVEL_NAMES:
        raise ValueError("kinematic input manifest member set changed")
    expected_top = {
        "schema": INPUT_MANIFEST_SCHEMA,
        "experiment": EXPERIMENT,
        "status": "frozen",
        **_identity_payload(identity),
        "parent_descriptor_id": PARENT_DESCRIPTOR_ID,
        "input_scope": "every_and_only_8_train_datasets_times_4_source_windows",
        "portable_npz_opened_during_freeze": False,
        "forbidden_dataset_access": False,
        "row_count": contract.row_count,
    }
    drift = {
        name: (value.get(name), expected)
        for name, expected in expected_top.items()
        if value.get(name) != expected
    }
    if drift:
        raise ValueError(f"kinematic input manifest provenance changed: {drift}")
    synthetic_evidence = value.get("synthetic_pass")
    if not isinstance(synthetic_evidence, Mapping) or set(synthetic_evidence) != {
        "path",
        "size_bytes",
        "file_sha256",
        "content_sha256",
    }:
        raise ValueError("kinematic input synthetic-pass binding changed")
    _require_no_forbidden_path(
        synthetic_evidence["path"], name="kinematic input synthetic-pass path"
    )
    _positive_integer(
        synthetic_evidence["size_bytes"], name="kinematic input synthetic-pass size"
    )
    _lower_hex(
        synthetic_evidence["file_sha256"],
        64,
        name="kinematic input synthetic-pass file SHA-256",
    )
    _lower_hex(
        synthetic_evidence["content_sha256"],
        64,
        name="kinematic input synthetic-pass content SHA-256",
    )
    parent_binding = value.get("parent_input_manifest")
    expected_parent_binding = {
        "size_bytes": contract.parent_input_manifest_size,
        "file_sha256": contract.parent_input_manifest_sha256,
        "rows_content_sha256": contract.parent_input_rows_sha256,
    }
    if not isinstance(parent_binding, Mapping) or set(parent_binding) != {
        "path",
        *expected_parent_binding,
    } or any(
        parent_binding.get(name) != expected
        for name, expected in expected_parent_binding.items()
    ):
        raise ValueError("kinematic input parent-manifest binding changed")
    _require_no_forbidden_path(
        parent_binding["path"], name="kinematic input parent-manifest path"
    )
    portable_binding = value.get("train_portable_population_marker")
    if not isinstance(portable_binding, Mapping) or set(portable_binding) != {
        "path",
        "size_bytes",
        "file_sha256",
        "rows_content_sha256",
    }:
        raise ValueError("kinematic input portable-marker binding changed")
    _require_no_forbidden_path(
        portable_binding["path"], name="kinematic input portable-marker path"
    )
    _positive_integer(
        portable_binding["size_bytes"], name="kinematic input portable-marker size"
    )
    if portable_binding.get("file_sha256") != contract.train_portable_marker_sha256:
        raise ValueError("kinematic input portable-marker SHA-256 changed")
    _lower_hex(
        portable_binding["rows_content_sha256"],
        64,
        name="kinematic input portable-marker rows SHA-256",
    )
    descriptors = composite_descriptor_contracts(identity, contract=contract)
    expected_descriptors = _descriptor_json(descriptors)
    if value.get("composite_descriptors") != expected_descriptors:
        raise ValueError("kinematic input composite descriptor contracts changed")
    rows = value.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != contract.row_count
        or value.get("rows_content_sha256") != canonical_json_sha256(rows)
    ):
        raise ValueError("kinematic input row population changed")
    validated_rows = [
        _validate_input_row(raw_row, expected_key=key, contract=contract)
        for key, raw_row in zip(_expected_keys(contract), rows, strict=True)
    ]
    if authenticate_all_referenced_rows:
        bound_files = (
            (
                value["synthetic_pass"],
                "synthetic pass marker",
            ),
            (
                value["parent_input_manifest"],
                "parent train-cache input manifest",
            ),
            (
                value["train_portable_population_marker"],
                "train portable population marker",
            ),
        )
        for binding, binding_name in bound_files:
            bound_size, bound_sha = _stable_file_identity(Path(str(binding["path"])))
            if bound_size != int(binding["size_bytes"]) or bound_sha != str(
                binding["file_sha256"]
            ):
                raise ValueError(f"referenced {binding_name} identity changed")
        for row in validated_rows:
            parent = row["parent_cache"]
            portable = row["portable"]
            load_parent_kinematic_projection(
                parent["path"],
                expected_size_bytes=int(parent["size_bytes"]),
                expected_file_sha256=str(parent["file_sha256"]),
                expected_array_sha256=dict(parent["allowed_array_sha256"]),
            )
            portable_size, portable_sha = _stable_file_identity(Path(portable["path"]))
            if (portable_size, portable_sha) != (
                int(portable["size_bytes"]),
                str(portable["file_sha256"]),
            ):
                raise ValueError("referenced portable file identity changed")
            manifest_size, manifest_sha = _stable_file_identity(
                Path(portable["dataset_manifest_path"])
            )
            if (manifest_size, manifest_sha) != (
                int(portable["dataset_manifest_size_bytes"]),
                str(portable["dataset_manifest_file_sha256"]),
            ):
                raise ValueError("referenced portable dataset manifest identity changed")
    frozen = _deep_freeze(value)
    if not isinstance(frozen, Mapping):
        raise RuntimeError("authenticated kinematic input manifest is not immutable")
    return frozen


def _selected_input_row(
    manifest: Mapping[str, Any], *, dataset: str, source_ordinal: int
) -> Mapping[str, Any]:
    rows = manifest["rows"]
    matches = [
        row
        for row in rows
        if row["dataset"] == dataset and int(row["source_ordinal"]) == source_ordinal
    ]
    if len(matches) != 1:
        raise ValueError("kinematic input row does not resolve uniquely")
    return matches[0]


def sidecar_row_relative_directory(
    dataset: str, source_ordinal: int, source_index: int
) -> Path:
    _require_no_forbidden_path(dataset, name="sidecar dataset")
    if not dataset or Path(dataset).name != dataset:
        raise ValueError("sidecar dataset must be one safe path component")
    ordinal = _nonnegative_integer(source_ordinal, name="source_ordinal")
    index = _nonnegative_integer(source_index, name="source_index")
    return Path(dataset) / f"source_{ordinal:02d}_index_{index:06d}"


def _provenance_bindings_for_row(
    *,
    manifest_path: Path,
    manifest_file_sha256: str,
    manifest: Mapping[str, Any],
    row: Mapping[str, Any],
    identity: CleanSourceIdentity,
    descriptors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    parent = row["parent_cache"]
    portable = row["portable"]
    bindings: dict[str, Any] = {
        "input_manifest_file_and_content_sha256": {
            "path": str(manifest_path),
            "file_sha256": manifest_file_sha256,
            "content_sha256": str(manifest["content_sha256"]),
            "rows_content_sha256": str(manifest["rows_content_sha256"]),
        },
        "parent_cache_path_size_file_sha256_and_allowed_array_hashes": {
            "path": str(parent["path"]),
            "size_bytes": int(parent["size_bytes"]),
            "file_sha256": str(parent["file_sha256"]),
            "allowed_array_sha256": dict(parent["allowed_array_sha256"]),
        },
        "portable_path_size_file_sha256_and_velocity_coordinate_time_array_hashes": {
            "path": str(portable["path"]),
            "size_bytes": int(portable["size_bytes"]),
            "file_sha256": str(portable["file_sha256"]),
            "array_sha256": dict(portable["array_sha256"]),
            "combined_array_sha256": str(portable["combined_array_sha256"]),
        },
        "config_sha256_and_clean_builder_git_commit": {
            "verify_config_sha256": str(manifest["verify_config_sha256"]),
            "git_commit": identity.git_commit,
            "worktree_clean": True,
            "source_file_sha256_content_sha256": identity.source_content_sha256,
        },
        "kinematic_algorithm_source_sha256": {
            "kinematic": identity.source_file_sha256[KINEMATIC_ALGORITHM_SOURCE_PATH],
            "sampler": identity.source_file_sha256[SAMPLER_SOURCE_PATH],
        },
        "composite_descriptor_contract": _descriptor_json(descriptors),
        "dataset_family_source_identity": {
            "dataset": str(row["dataset"]),
            "physical_family": str(row["physical_family"]),
            "source_ordinal": int(row["source_ordinal"]),
            "source_index": int(row["source_index"]),
        },
        "line_order_time_interpolation_and_float_contract": {
            "primitive_order": list(FROZEN_PRIMITIVE_ORDER),
            "sampling_time": "portable_velocity_frame_zero",
            "interpolation": "production_RK4_interp4_quadrilinear_scalar",
            "kinematic_feature_order": list(FROZEN_KINEMATIC_FEATURE_ORDER),
            "velocity_dtype": "float32",
            "derivative_dtype": "float64",
            "serialization_dtype": "float32",
        },
        "all_array_dtype_shape_canonical_sha256_and_combined_sha256": (
            "stored_and_freshly_authenticated_by_seed_time_sidecar_core.v1"
        ),
    }
    if tuple(bindings) != SIDECAR_PROVENANCE_BINDING_NAMES:
        raise RuntimeError("sidecar provenance binding order changed")
    return bindings


def build_one_sidecar_and_completion(
    sidecar_root: str | Path,
    *,
    dataset: str,
    source_ordinal: int,
    input_manifest_path: str | Path,
    input_manifest_file_sha256: str,
    synthetic_pass_path: str | Path,
    synthetic_pass_file_sha256: str,
    identity: CleanSourceIdentity,
    contract: PreparationContract = PRODUCTION_CONTRACT,
) -> Mapping[str, Any]:
    """Build and close exactly one sidecar after both immutable gates pass."""

    _validate_identity_against_contract(identity, contract)
    authenticate_synthetic_pass_marker(
        synthetic_pass_path,
        expected_file_sha256=synthetic_pass_file_sha256,
        identity=identity,
        contract=contract,
    )
    manifest_path = Path(input_manifest_path).resolve()
    manifest = authenticate_kinematic_input_manifest(
        manifest_path,
        expected_file_sha256=input_manifest_file_sha256,
        identity=identity,
        contract=contract,
        authenticate_all_referenced_rows=False,
    )
    if (
        str(Path(synthetic_pass_path).resolve())
        != str(manifest["synthetic_pass"]["path"])
        or str(synthetic_pass_file_sha256)
        != str(manifest["synthetic_pass"]["file_sha256"])
    ):
        raise ValueError("sidecar build supplied a different synthetic-pass marker")
    ordinal = _nonnegative_integer(source_ordinal, name="source_ordinal")
    if dataset not in contract.datasets or ordinal >= contract.source_count:
        raise ValueError("sidecar task identity is outside the frozen 32-row population")
    row = _selected_input_row(
        manifest, dataset=dataset, source_ordinal=ordinal
    )
    root = Path(sidecar_root).resolve()
    _require_no_forbidden_path(root, name="sidecar population root")
    relative_dir = sidecar_row_relative_directory(
        dataset, ordinal, int(row["source_index"])
    )
    row_dir = (root / relative_dir).resolve()
    try:
        row_dir.relative_to(root)
    except ValueError as error:
        raise ValueError("sidecar row directory resolves outside population root") from error
    sidecar_path = row_dir / "seed_time_kinematic4.npz"
    completion_path = row_dir / "SIDECAR_COMPLETE.json"
    if row_dir.exists():
        raise FileExistsError(f"immutable sidecar row directory already exists: {row_dir}")
    row_dir.mkdir(parents=True, exist_ok=False)
    parent_evidence = row["parent_cache"]
    portable_evidence = row["portable"]
    parent = load_parent_kinematic_projection(
        parent_evidence["path"],
        expected_size_bytes=int(parent_evidence["size_bytes"]),
        expected_file_sha256=str(parent_evidence["file_sha256"]),
        expected_array_sha256=dict(parent_evidence["allowed_array_sha256"]),
    )
    portable = load_portable_flow_window(
        portable_evidence["path"],
        expected_dataset=dataset,
        expected_experiment="mainExp_TemplateMatching_3.1",
        expected_config_sha256=contract.parent_main_config_sha256,
        expected_builder_git_commit=contract.parent_cache_builder_commit,
        expected_source_start_index=int(row["source_index"]),
        expected_file_sha256=str(portable_evidence["file_sha256"]),
    )
    expected_portable_hashes = dict(portable_evidence["array_sha256"])
    actual_portable_hashes = {
        "velocity": canonical_array_sha256(portable.window.velocity),
        "x": canonical_array_sha256(portable.window.coordinates_xyz[0]),
        "y": canonical_array_sha256(portable.window.coordinates_xyz[1]),
        "z": canonical_array_sha256(portable.window.coordinates_xyz[2]),
        "time": canonical_array_sha256(portable.window.time),
    }
    if actual_portable_hashes != expected_portable_hashes:
        raise ValueError("opened portable arrays differ from the frozen input manifest")
    descriptors = composite_descriptor_contracts(identity, contract=contract)
    bindings = _provenance_bindings_for_row(
        manifest_path=manifest_path,
        manifest_file_sha256=str(input_manifest_file_sha256),
        manifest=manifest,
        row=row,
        identity=identity,
        descriptors=descriptors,
    )
    payload = build_seed_time_kinematic_sidecar_payload(parent, portable.window)
    loaded = write_seed_time_kinematic_sidecar(
        sidecar_path, payload, provenance_bindings=bindings
    )
    loaded = load_seed_time_kinematic_sidecar(
        sidecar_path,
        expected_file_sha256=loaded.file_sha256,
        expected_provenance_bindings=bindings,
        expected_parent=parent,
    )
    completion = _with_content_sha256(
        {
            "schema": ROW_COMPLETION_SCHEMA,
            "experiment": EXPERIMENT,
            "status": "passed",
            **_identity_payload(identity),
            "dataset": dataset,
            "physical_family": str(row["physical_family"]),
            "split": "train",
            "source_ordinal": ordinal,
            "source_index": int(row["source_index"]),
            "input_manifest_path": str(manifest_path),
            "input_manifest_file_sha256": str(input_manifest_file_sha256),
            "input_manifest_content_sha256": str(manifest["content_sha256"]),
            "synthetic_pass_path": str(Path(synthetic_pass_path).resolve()),
            "synthetic_pass_file_sha256": str(synthetic_pass_file_sha256),
            "sidecar_relative_path": sidecar_path.relative_to(root).as_posix(),
            "sidecar_size_bytes": int(sidecar_path.stat().st_size),
            "sidecar_file_sha256": loaded.file_sha256,
            "sidecar_combined_array_sha256": str(
                loaded.metadata["combined_array_sha256"]
            ),
            "sidecar_row_count": int(len(loaded.payload.valid_assigned_row_index)),
            "composite_descriptor_ids": {
                name: str(value["descriptor_id"]) for name, value in descriptors.items()
            },
            "forbidden_parent_members_opened": [],
            "forbidden_dataset_access": False,
            "marker_write_order": "last_after_sidecar_was_closed_and_freshly_authenticated",
        }
    )
    _atomic_json_no_overwrite(completion_path, completion)
    return authenticate_row_completion(
        completion_path,
        sidecar_root=root,
        expected_file_sha256=sha256_file(completion_path),
        input_manifest=manifest,
        input_manifest_file_sha256=str(input_manifest_file_sha256),
        synthetic_pass_file_sha256=str(synthetic_pass_file_sha256),
        identity=identity,
        contract=contract,
        authenticate_sidecar=True,
    )


def authenticate_row_completion(
    path: str | Path,
    *,
    sidecar_root: str | Path,
    expected_file_sha256: str,
    input_manifest: Mapping[str, Any],
    input_manifest_file_sha256: str,
    synthetic_pass_file_sha256: str,
    identity: CleanSourceIdentity,
    contract: PreparationContract = PRODUCTION_CONTRACT,
    authenticate_sidecar: bool = True,
) -> Mapping[str, Any]:
    """Authenticate one completion marker and its exact input/sidecar join."""

    _validate_identity_against_contract(identity, contract)
    source = Path(path).resolve()
    root = Path(sidecar_root).resolve()
    expected_digest = _lower_hex(
        expected_file_sha256, 64, name="row completion SHA-256"
    )
    raw_value, _completion_snapshot = _json_from_authenticated_snapshot(
        source,
        expected_sha256=expected_digest,
    )
    value = _verify_content_sha256(
        raw_value, name="row completion"
    )
    if set(value) != _ROW_COMPLETION_NAMES:
        raise ValueError("row completion member set changed")
    dataset = str(value.get("dataset", ""))
    ordinal = int(value.get("source_ordinal", -1))
    if dataset not in contract.datasets or not (0 <= ordinal < contract.source_count):
        raise ValueError("row completion identity is outside the frozen population")
    input_row = _selected_input_row(
        input_manifest, dataset=dataset, source_ordinal=ordinal
    )
    expected_relative_dir = sidecar_row_relative_directory(
        dataset, ordinal, int(input_row["source_index"])
    )
    expected_path = (root / expected_relative_dir / "SIDECAR_COMPLETE.json").resolve()
    if source != expected_path:
        raise ValueError("row completion path differs from its frozen identity")
    expected_ids = {
        name: str(payload["descriptor_id"])
        for name, payload in composite_descriptor_contracts(
            identity, contract=contract
        ).items()
    }
    expected = {
        "schema": ROW_COMPLETION_SCHEMA,
        "experiment": EXPERIMENT,
        "status": "passed",
        **_identity_payload(identity),
        "dataset": dataset,
        "physical_family": contract.family_by_dataset[dataset],
        "split": "train",
        "source_ordinal": ordinal,
        "source_index": int(input_row["source_index"]),
        "input_manifest_file_sha256": input_manifest_file_sha256,
        "input_manifest_content_sha256": str(input_manifest["content_sha256"]),
        "synthetic_pass_path": str(input_manifest["synthetic_pass"]["path"]),
        "synthetic_pass_file_sha256": synthetic_pass_file_sha256,
        "sidecar_relative_path": (expected_relative_dir / "seed_time_kinematic4.npz").as_posix(),
        "composite_descriptor_ids": expected_ids,
        "forbidden_parent_members_opened": [],
        "forbidden_dataset_access": False,
        "marker_write_order": "last_after_sidecar_was_closed_and_freshly_authenticated",
    }
    drift = {
        name: (value.get(name), expected_value)
        for name, expected_value in expected.items()
        if value.get(name) != expected_value
    }
    if drift:
        raise ValueError(f"row completion provenance changed: {drift}")
    referenced_input_path = Path(str(value.get("input_manifest_path", ""))).resolve()
    referenced_size, referenced_sha = _stable_file_identity(referenced_input_path)
    del referenced_size
    if referenced_sha != input_manifest_file_sha256:
        raise ValueError("row completion points to a different input manifest file")
    sidecar_path = (root / str(value["sidecar_relative_path"])).resolve()
    try:
        sidecar_path.relative_to(root)
    except ValueError as error:
        raise ValueError("row completion sidecar path escapes population root") from error
    sidecar_size, sidecar_sha = _stable_file_identity(sidecar_path)
    if sidecar_size != _positive_integer(value.get("sidecar_size_bytes"), name="sidecar size") or sidecar_sha != _lower_hex(
        value.get("sidecar_file_sha256"), 64, name="sidecar SHA-256"
    ):
        raise ValueError("row completion sidecar file identity changed")
    if authenticate_sidecar:
        parent_evidence = input_row["parent_cache"]
        parent = load_parent_kinematic_projection(
            parent_evidence["path"],
            expected_size_bytes=int(parent_evidence["size_bytes"]),
            expected_file_sha256=str(parent_evidence["file_sha256"]),
            expected_array_sha256=dict(parent_evidence["allowed_array_sha256"]),
        )
        bindings = _provenance_bindings_for_row(
            manifest_path=Path(str(value["input_manifest_path"])).resolve(),
            manifest_file_sha256=input_manifest_file_sha256,
            manifest=input_manifest,
            row=input_row,
            identity=identity,
            descriptors=composite_descriptor_contracts(identity, contract=contract),
        )
        loaded = load_seed_time_kinematic_sidecar(
            sidecar_path,
            expected_file_sha256=sidecar_sha,
            expected_provenance_bindings=bindings,
            expected_parent=parent,
        )
        if (
            value.get("sidecar_combined_array_sha256")
            != loaded.metadata["combined_array_sha256"]
            or int(value.get("sidecar_row_count", -1))
            != len(loaded.payload.valid_assigned_row_index)
        ):
            raise ValueError("row completion sidecar payload evidence changed")
    frozen = _deep_freeze(value)
    if not isinstance(frozen, Mapping):
        raise RuntimeError("authenticated row completion is not immutable")
    return frozen


def _expected_population_files(
    root: Path, manifest: Mapping[str, Any]
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for row in manifest["rows"]:
        relative_dir = sidecar_row_relative_directory(
            str(row["dataset"]),
            int(row["source_ordinal"]),
            int(row["source_index"]),
        )
        paths.extend(
            (
                (root / relative_dir / "seed_time_kinematic4.npz").resolve(),
                (root / relative_dir / "SIDECAR_COMPLETE.json").resolve(),
            )
        )
    return tuple(paths)


def write_sidecar_population_manifest(
    sidecar_root: str | Path,
    *,
    input_manifest_path: str | Path,
    input_manifest_file_sha256: str,
    synthetic_pass_path: str | Path,
    synthetic_pass_file_sha256: str,
    identity: CleanSourceIdentity,
    contract: PreparationContract = PRODUCTION_CONTRACT,
) -> Mapping[str, Any]:
    """Authenticate every sidecar and publish the exact 32-row population."""

    _validate_identity_against_contract(identity, contract)
    authenticate_synthetic_pass_marker(
        synthetic_pass_path,
        expected_file_sha256=synthetic_pass_file_sha256,
        identity=identity,
        contract=contract,
    )
    input_path = Path(input_manifest_path).resolve()
    input_manifest = authenticate_kinematic_input_manifest(
        input_path,
        expected_file_sha256=input_manifest_file_sha256,
        identity=identity,
        contract=contract,
        authenticate_all_referenced_rows=False,
    )
    if (
        str(Path(synthetic_pass_path).resolve())
        != str(input_manifest["synthetic_pass"]["path"])
        or str(synthetic_pass_file_sha256)
        != str(input_manifest["synthetic_pass"]["file_sha256"])
    ):
        raise ValueError("population build supplied a different synthetic-pass marker")
    root = Path(sidecar_root).resolve()
    population_path = root / "SIDECAR_POPULATION.json"
    if population_path.exists():
        raise FileExistsError(f"immutable sidecar population already exists: {population_path}")
    expected_files = _expected_population_files(root, input_manifest)
    actual_files = tuple(sorted((path.resolve() for path in root.rglob("*") if path.is_file()), key=str))
    if set(actual_files) != set(expected_files) or len(actual_files) != len(expected_files):
        missing = sorted(str(path) for path in set(expected_files) - set(actual_files))
        extra = sorted(str(path) for path in set(actual_files) - set(expected_files))
        raise ValueError(f"sidecar population file set changed: missing={missing}, extra={extra}")
    population_rows: list[dict[str, Any]] = []
    total_rows = 0
    for input_row in input_manifest["rows"]:
        relative_dir = sidecar_row_relative_directory(
            str(input_row["dataset"]),
            int(input_row["source_ordinal"]),
            int(input_row["source_index"]),
        )
        completion_path = root / relative_dir / "SIDECAR_COMPLETE.json"
        completion_sha = sha256_file(completion_path)
        completion = authenticate_row_completion(
            completion_path,
            sidecar_root=root,
            expected_file_sha256=completion_sha,
            input_manifest=input_manifest,
            input_manifest_file_sha256=input_manifest_file_sha256,
            synthetic_pass_file_sha256=synthetic_pass_file_sha256,
            identity=identity,
            contract=contract,
            authenticate_sidecar=True,
        )
        count = int(completion["sidecar_row_count"])
        total_rows += count
        population_rows.append(
            {
                "dataset": str(input_row["dataset"]),
                "physical_family": str(input_row["physical_family"]),
                "source_ordinal": int(input_row["source_ordinal"]),
                "source_index": int(input_row["source_index"]),
                "completion_relative_path": completion_path.relative_to(root).as_posix(),
                "completion_size_bytes": int(completion_path.stat().st_size),
                "completion_file_sha256": completion_sha,
                "sidecar_relative_path": str(completion["sidecar_relative_path"]),
                "sidecar_size_bytes": int(completion["sidecar_size_bytes"]),
                "sidecar_file_sha256": str(completion["sidecar_file_sha256"]),
                "sidecar_combined_array_sha256": str(
                    completion["sidecar_combined_array_sha256"]
                ),
                "sidecar_row_count": count,
            }
        )
    descriptors = composite_descriptor_contracts(identity, contract=contract)
    value = _with_content_sha256(
        {
            "schema": POPULATION_MANIFEST_SCHEMA,
            "experiment": EXPERIMENT,
            "status": "passed",
            **_identity_payload(identity),
            "input_manifest_path": str(input_path),
            "input_manifest_file_sha256": str(input_manifest_file_sha256),
            "input_manifest_content_sha256": str(input_manifest["content_sha256"]),
            "synthetic_pass_path": str(Path(synthetic_pass_path).resolve()),
            "synthetic_pass_file_sha256": str(synthetic_pass_file_sha256),
            "composite_descriptor_ids": {
                name: str(payload["descriptor_id"])
                for name, payload in descriptors.items()
            },
            "sidecar_count": len(population_rows),
            "sidecar_row_count_total": total_rows,
            "rows": population_rows,
            "rows_content_sha256": canonical_json_sha256(population_rows),
            "forbidden_dataset_access": False,
            "manifest_write_order": "last_after_all_32_completion_markers_and_sidecars_were_authenticated",
        }
    )
    _atomic_json_no_overwrite(population_path, value)
    return authenticate_sidecar_population_manifest(
        population_path,
        sidecar_root=root,
        expected_file_sha256=sha256_file(population_path),
        input_manifest_path=input_path,
        input_manifest_file_sha256=input_manifest_file_sha256,
        synthetic_pass_file_sha256=synthetic_pass_file_sha256,
        identity=identity,
        contract=contract,
    )


def authenticate_sidecar_population_manifest(
    path: str | Path,
    *,
    sidecar_root: str | Path,
    expected_file_sha256: str,
    input_manifest_path: str | Path,
    input_manifest_file_sha256: str,
    synthetic_pass_file_sha256: str,
    identity: CleanSourceIdentity,
    contract: PreparationContract = PRODUCTION_CONTRACT,
) -> Mapping[str, Any]:
    """Freshly authenticate the population manifest and every child artifact."""

    _validate_identity_against_contract(identity, contract)
    root = Path(sidecar_root).resolve()
    source = Path(path).resolve()
    if source != root / "SIDECAR_POPULATION.json":
        raise ValueError("population manifest must be SIDECAR_POPULATION.json at population root")
    expected_digest = _lower_hex(
        expected_file_sha256, 64, name="population manifest SHA-256"
    )
    raw_value, _population_snapshot = _json_from_authenticated_snapshot(
        source,
        expected_sha256=expected_digest,
    )
    input_path = Path(input_manifest_path).resolve()
    input_manifest = authenticate_kinematic_input_manifest(
        input_path,
        expected_file_sha256=input_manifest_file_sha256,
        identity=identity,
        contract=contract,
        authenticate_all_referenced_rows=False,
    )
    value = _verify_content_sha256(
        raw_value, name="sidecar population manifest"
    )
    if set(value) != _POPULATION_MANIFEST_NAMES:
        raise ValueError("sidecar population manifest member set changed")
    expected_ids = {
        name: str(payload["descriptor_id"])
        for name, payload in composite_descriptor_contracts(
            identity, contract=contract
        ).items()
    }
    expected = {
        "schema": POPULATION_MANIFEST_SCHEMA,
        "experiment": EXPERIMENT,
        "status": "passed",
        **_identity_payload(identity),
        "input_manifest_path": str(input_path),
        "input_manifest_file_sha256": input_manifest_file_sha256,
        "input_manifest_content_sha256": str(input_manifest["content_sha256"]),
        "synthetic_pass_path": str(input_manifest["synthetic_pass"]["path"]),
        "synthetic_pass_file_sha256": synthetic_pass_file_sha256,
        "composite_descriptor_ids": expected_ids,
        "sidecar_count": contract.row_count,
        "forbidden_dataset_access": False,
        "manifest_write_order": "last_after_all_32_completion_markers_and_sidecars_were_authenticated",
    }
    drift = {
        name: (value.get(name), expected_value)
        for name, expected_value in expected.items()
        if value.get(name) != expected_value
    }
    if drift:
        raise ValueError(f"sidecar population provenance changed: {drift}")
    rows = value.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != contract.row_count
        or value.get("rows_content_sha256") != canonical_json_sha256(rows)
    ):
        raise ValueError("sidecar population row set changed")
    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    expected_files = set(_expected_population_files(root, input_manifest)) | {source}
    if actual_files != expected_files:
        raise ValueError("authenticated population contains missing or extra files")
    total_rows = 0
    for expected_input, population_row in zip(input_manifest["rows"], rows, strict=True):
        if not isinstance(population_row, Mapping) or set(population_row) != _POPULATION_ROW_NAMES:
            raise ValueError("population row member set changed")
        expected_identity = {
            "dataset": str(expected_input["dataset"]),
            "physical_family": str(expected_input["physical_family"]),
            "source_ordinal": int(expected_input["source_ordinal"]),
            "source_index": int(expected_input["source_index"]),
        }
        if any(population_row.get(name) != expected_value for name, expected_value in expected_identity.items()):
            raise ValueError("population rows are missing, duplicated, extra, or reordered")
        completion_path = (root / str(population_row["completion_relative_path"])).resolve()
        completion_size, completion_sha = _stable_file_identity(completion_path)
        if completion_size != int(population_row["completion_size_bytes"]) or completion_sha != str(
            population_row["completion_file_sha256"]
        ):
            raise ValueError("population completion file identity changed")
        completion = authenticate_row_completion(
            completion_path,
            sidecar_root=root,
            expected_file_sha256=completion_sha,
            input_manifest=input_manifest,
            input_manifest_file_sha256=input_manifest_file_sha256,
            synthetic_pass_file_sha256=synthetic_pass_file_sha256,
            identity=identity,
            contract=contract,
            authenticate_sidecar=True,
        )
        for name in (
            "sidecar_relative_path",
            "sidecar_size_bytes",
            "sidecar_file_sha256",
            "sidecar_combined_array_sha256",
            "sidecar_row_count",
        ):
            if population_row.get(name) != completion.get(name):
                raise ValueError(f"population/completion evidence differs for {name}")
        total_rows += int(completion["sidecar_row_count"])
    if int(value.get("sidecar_row_count_total", -1)) != total_rows:
        raise ValueError("population total sidecar row count changed")
    frozen = _deep_freeze(value)
    if not isinstance(frozen, Mapping):
        raise RuntimeError("authenticated population manifest is not immutable")
    return frozen


__all__ = [
    "CleanSourceIdentity",
    "COMPOSITE_DESCRIPTOR_SCHEMA",
    "DATASET_FAMILY_PAIRS",
    "EXPERIMENT",
    "INPUT_MANIFEST_SCHEMA",
    "NUMERICAL_DEPENDENCY_SOURCE_PATHS",
    "POPULATION_MANIFEST_SCHEMA",
    "PRODUCTION_CONTRACT",
    "PreparationContract",
    "REQUIRED_SOURCE_PATHS",
    "ROW_COMPLETION_SCHEMA",
    "SYNTHETIC_CHECK_NAMES",
    "SYNTHETIC_EVIDENCE_SCHEMA",
    "SYNTHETIC_PASS_SCHEMA",
    "authenticate_kinematic_input_manifest",
    "authenticate_row_completion",
    "authenticate_sidecar_population_manifest",
    "authenticate_synthetic_pass_marker",
    "build_kinematic_input_manifest",
    "build_one_sidecar_and_completion",
    "capture_clean_source_identity",
    "composite_descriptor_contracts",
    "sidecar_row_relative_directory",
    "write_sidecar_population_manifest",
    "write_synthetic_pass_marker",
]

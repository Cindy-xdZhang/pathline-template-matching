"""Authenticated assigned-row source-centered kinematic sidecars.

This module is the preparation boundary for
``Verify_SourceCenteredPairedScaleTemplate_1.1``.  It deliberately exposes a
narrow parent-cache projection: six identity/geometry arrays are opened, all
128,000 assigned rows are sampled at portable velocity frame zero, and no
label, IVD, validity-mask, metadata, FMT, or Raw feature member is read.

The saved four-coordinate feature is defined by
``source_centered_seed_time_kinematics``.  Valid-row features are never stored
twice; their canonical hash authenticates the exact order-preserving
projection ``source_centered_seed4[valid_assigned_row_index]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .early_opposite_pair_kinematics import FROZEN_PRIMITIVE_ORDER
from .netcdf_io import FlowWindow3D
from .portable_flow import (
    canonical_array_sha256,
    canonical_json_sha256,
    load_portable_flow_window,
    sha256_file,
)
from .seed_time_kinematic_sidecar import (
    FROZEN_DX_GRID_SCALE_BY_ID,
    physical_dx_by_scale_for_window,
    sample_seed_time_velocity_xyz,
)
from .source_centered_seed_time_kinematics import (
    ASSIGNED_ROWS_PER_SOURCE_BLOCK_DX_LEVEL,
    FROZEN_SOURCE_CENTERED_KINEMATIC_FEATURE_ORDER,
    FROZEN_SOURCE_GROUP_ID_ORDER,
    compute_source_centered_seed_time_kinematics,
)


EXPERIMENT = "Verify_SourceCenteredPairedScaleTemplate_1.1"
INPUT_MANIFEST_SCHEMA = "pathline_template_matching.source_centered_input.v1"
SIDECAR_SCHEMA = (
    "pathline_template_matching.source_centered_seed_time_kinematics_cache.v1"
)
ROW_COMPLETION_SCHEMA = (
    "pathline_template_matching.source_centered_row_complete.v1"
)
POPULATION_MANIFEST_SCHEMA = (
    "pathline_template_matching.source_centered_population.v1"
)

ASSIGNED_ROW_COUNT = 128_000
UNIQUE_CENTER_COUNT = 64_000
SCALE_COUNT = 2_000
SCALE_BLOCK_COUNT = 2
SCALES_PER_BLOCK = 1_000
ROWS_PER_EXACT_SCALE = 64
DX_LEVELS_PER_BLOCK = 10

PARENT_ALLOWED_MEMBER_NAMES = (
    "seeds_xyz",
    "scale_assignment",
    "valid_assigned_row_index",
    "valid_center_seed_index",
    "valid_scale_block_index",
    "valid_scale_id",
)
PARENT_FORBIDDEN_MEMBER_NAMES = frozenset(
    {
        "valid_labels",
        "reference_labels_all",
        "ivd_values_all",
        "ivd_volume",
        "valid_mask",
        "metadata_json",
        "fmt_features",
        "raw_features",
    }
)

SIDECAR_ASSIGNED_ARRAY_NAMES = (
    "assigned_row_index",
    "center_seed_index",
    "scale_block_index",
    "scale_id",
    "seed_velocity_xyz",
    "source_centered_seed4",
    "group_mean_curl_xyz",
    "physical_dx_by_scale",
)
SIDECAR_VALID_IDENTITY_NAMES = (
    "valid_assigned_row_index",
    "valid_center_seed_index",
    "valid_scale_block_index",
    "valid_scale_id",
)
SIDECAR_ARRAY_NAMES = SIDECAR_ASSIGNED_ARRAY_NAMES + SIDECAR_VALID_IDENTITY_NAMES
SIDECAR_ARCHIVE_MEMBER_NAMES = SIDECAR_ARRAY_NAMES + ("metadata_json",)
SIDECAR_PROVENANCE_BINDING_NAMES = (
    "input_manifest_file_content_and_rows_sha256",
    "parent_cache_path_size_file_sha256_and_allowed_array_hashes",
    "portable_path_size_file_sha256_and_array_hashes",
    "config_clean_commit_and_source_hashes",
    "algorithm_source_hashes",
    "dataset_family_source_identity",
    "assigned_group_and_valid_projection_contract",
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
SOURCE_COUNT_PER_DATASET = 4
FORBIDDEN_DATASET_TOKENS = ("tangaroa", "smokebuoyancy", "smoke_buoyancy")

EXPECTED_VERIFY_CONFIG_SHA256 = (
    "15ac5b0e82b30cbaf952475a7fbb6d19dc070c1121bc9aa8db980d75600260cc"
)
EXPECTED_EARLY_INPUT_MANIFEST_SHA256 = (
    "1b9df53a9010c6c3c46345639cfbf1d5ab2fe3a43187c79c7dfa0f7d840b102f"
)
EXPECTED_PARENT_INPUT_MANIFEST_SHA256 = (
    "e57d6b527acffb61da32a993f0c30a0e6435021679c7a3f1541dab8ba877b393"
)

REQUIRED_SOURCE_PATHS = (
    "config/Verify_SourceCenteredPairedScaleTemplate_1.1.yaml",
    "src/pathline_template_matching/source_centered_seed_time_kinematics.py",
    "src/pathline_template_matching/source_centered_sidecar.py",
    "src/pathline_template_matching/seed_time_kinematic_sidecar.py",
    "src/pathline_template_matching/early_opposite_pair_kinematics.py",
    "src/pathline_template_matching/portable_flow.py",
    "src/pathline_template_matching/netcdf_io.py",
    "src/pathline_template_matching/vector_field.py",
    "src/pathline_template_matching/arc_length_primitives.py",
    "scripts/prepare_verify_source_centered_paired_scale_template_1_1.py",
)

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SIDECAR_METADATA_NAMES = frozenset(
    {
        "schema",
        "dataset_index",
        "source_ordinal",
        "primitive_order",
        "group_id_order",
        "kinematic_feature_order",
        "sampling_time",
        "float_contract",
        "array_contract",
        "array_sha256",
        "combined_array_sha256",
        "valid_projection",
        "provenance_bindings",
    }
)


def _freeze(values: object, *, dtype: np.dtype | type) -> np.ndarray:
    selected = np.dtype(dtype)
    copied = np.array(values, dtype=selected, order="C", copy=True)
    result = np.frombuffer(copied.tobytes(order="C"), dtype=selected).reshape(
        copied.shape
    )
    result.setflags(write=False)
    return result


def _exact_array(
    values: object,
    *,
    name: str,
    dtype: np.dtype | type,
    shape: tuple[int, ...],
    finite: bool = False,
) -> np.ndarray:
    raw = np.asarray(values)
    expected = np.dtype(dtype)
    if raw.dtype != expected or raw.shape != shape:
        raise ValueError(
            f"{name} must have dtype {expected} and shape {shape}; "
            f"got dtype {raw.dtype} and shape {raw.shape}"
        )
    if finite and not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain only finite values")
    return _freeze(raw, dtype=expected)


def _hex(value: object, length: int, *, name: str) -> str:
    selected = str(value)
    pattern = _HEX40 if length == 40 else _HEX64
    if pattern.fullmatch(selected) is None:
        raise ValueError(f"{name} must be lowercase {length}-hex")
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


def _canonical_json_copy(value: object) -> Any:
    def mutable(child: object) -> object:
        if isinstance(child, Mapping):
            return {str(key): mutable(item) for key, item in child.items()}
        if isinstance(child, (list, tuple)):
            return [mutable(item) for item in child]
        return child

    try:
        return json.loads(
            json.dumps(
                mutable(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError("artifact must contain finite JSON values") from error


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(child) for child in value)
    return value


def _with_content_sha256(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _canonical_json_copy(value)
    if "content_sha256" in payload:
        raise ValueError("content_sha256 may only be added by the writer")
    payload["content_sha256"] = canonical_json_sha256(payload)
    return payload


def _verify_content_sha256(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
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


def _require_no_forbidden_dataset_path(path: str | Path, *, name: str) -> None:
    for part in Path(path).parts:
        normalized = part.casefold().replace("-", "_").replace(" ", "_")
        if any(token in normalized for token in FORBIDDEN_DATASET_TOKENS):
            raise ValueError(f"{name} contains a forbidden dataset token")


@dataclass(frozen=True)
class _FileIdentity:
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    device: int
    inode: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_FileIdentity":
        return cls(
            size_bytes=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            ctime_ns=int(value.st_ctime_ns),
            device=int(value.st_dev),
            inode=int(value.st_ino),
        )


def _stable_file_identity(
    path: str | Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[int, str, _FileIdentity]:
    source = Path(path).resolve()
    before = _FileIdentity.from_stat(source.stat(follow_symlinks=False))
    digest = sha256_file(source)
    after = _FileIdentity.from_stat(source.stat(follow_symlinks=False))
    if before != after:
        raise RuntimeError(f"file changed while authenticating: {source}")
    if expected_size is not None and before.size_bytes != int(expected_size):
        raise ValueError(f"file size mismatch: {source}")
    if expected_sha256 is not None and digest != _hex(
        expected_sha256, 64, name=f"file SHA-256: {source}"
    ):
        raise ValueError(f"file SHA-256 mismatch: {source}")
    return before.size_bytes, digest, before


def _read_json_authenticated(
    path: str | Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], int, str]:
    source = Path(path).resolve()
    size, digest, identity = _stable_file_identity(
        source, expected_size=expected_size, expected_sha256=expected_sha256
    )
    content = source.read_bytes()
    after = _FileIdentity.from_stat(source.stat(follow_symlinks=False))
    if after != identity or len(content) != size or hashlib.sha256(content).hexdigest() != digest:
        raise RuntimeError(f"JSON file changed while reading: {source}")
    value = json.loads(content.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be a mapping: {source}")
    return value, size, digest


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(
        os.fspath(directory), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json_no_overwrite(path: str | Path, value: object) -> str:
    output = Path(path).resolve()
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
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output, follow_symlinks=False)
        _fsync_directory(output.parent)
        temporary.unlink()
        _fsync_directory(output.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    return sha256_file(output)


@dataclass(frozen=True)
class PreparationContract:
    verify_config_sha256: str
    early_input_manifest_sha256: str
    parent_input_manifest_sha256: str
    dataset_family_pairs: tuple[tuple[str, str], ...] = DATASET_FAMILY_PAIRS
    source_count: int = SOURCE_COUNT_PER_DATASET

    def __post_init__(self) -> None:
        for name in (
            "verify_config_sha256",
            "early_input_manifest_sha256",
            "parent_input_manifest_sha256",
        ):
            object.__setattr__(self, name, _hex(getattr(self, name), 64, name=name))
        pairs = tuple((str(dataset), str(family)) for dataset, family in self.dataset_family_pairs)
        if not pairs or len({dataset for dataset, _ in pairs}) != len(pairs):
            raise ValueError("dataset_family_pairs must be unique and nonempty")
        for dataset, family in pairs:
            if not dataset or not family or Path(dataset).name != dataset:
                raise ValueError("dataset/family identities must be nonempty safe values")
            _require_no_forbidden_dataset_path(dataset, name="dataset")
        object.__setattr__(self, "dataset_family_pairs", pairs)
        object.__setattr__(
            self, "source_count", _positive_integer(self.source_count, name="source_count")
        )

    @property
    def datasets(self) -> tuple[str, ...]:
        return tuple(dataset for dataset, _ in self.dataset_family_pairs)

    @property
    def family_by_dataset(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.dataset_family_pairs))

    @property
    def row_count(self) -> int:
        return len(self.dataset_family_pairs) * self.source_count


PRODUCTION_CONTRACT = PreparationContract(
    verify_config_sha256=EXPECTED_VERIFY_CONFIG_SHA256,
    early_input_manifest_sha256=EXPECTED_EARLY_INPUT_MANIFEST_SHA256,
    parent_input_manifest_sha256=EXPECTED_PARENT_INPUT_MANIFEST_SHA256,
)


@dataclass(frozen=True)
class CleanSourceIdentity:
    git_commit: str
    worktree_clean: bool
    source_file_sha256_items: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "git_commit", _hex(self.git_commit, 40, name="git_commit"))
        if self.worktree_clean is not True:
            raise ValueError("production artifacts require a clean worktree")
        items = tuple((str(path), str(digest)) for path, digest in self.source_file_sha256_items)
        if tuple(path for path, _ in items) != REQUIRED_SOURCE_PATHS:
            raise ValueError("source identity path order changed")
        for path, digest in items:
            _safe_relative(path, name="source path")
            _hex(digest, 64, name=f"source SHA-256 for {path}")
        object.__setattr__(self, "source_file_sha256_items", items)

    @property
    def source_file_sha256(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.source_file_sha256_items))

    @property
    def source_content_sha256(self) -> str:
        return canonical_json_sha256(dict(self.source_file_sha256_items))


def capture_clean_source_identity(project_root: str | Path) -> CleanSourceIdentity:
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
        raise RuntimeError("production artifacts require a clean committed worktree")
    hashes: list[tuple[str, str]] = []
    for relative in REQUIRED_SOURCE_PATHS:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        path = (root / relative).resolve()
        path.relative_to(root)
        hashes.append((relative, sha256_file(path)))
    commit_after = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty_after = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit_after != commit or dirty_after:
        raise RuntimeError("Git identity changed while source hashes were captured")
    return CleanSourceIdentity(commit, True, tuple(hashes))


def _identity_payload(identity: CleanSourceIdentity) -> dict[str, Any]:
    return {
        "git_commit": identity.git_commit,
        "worktree_clean": True,
        "verify_config_sha256": identity.source_file_sha256[REQUIRED_SOURCE_PATHS[0]],
        "source_file_sha256": dict(identity.source_file_sha256_items),
        "source_file_sha256_content_sha256": identity.source_content_sha256,
    }


def _validate_identity(
    identity: CleanSourceIdentity, contract: PreparationContract
) -> None:
    if identity.source_file_sha256[REQUIRED_SOURCE_PATHS[0]] != contract.verify_config_sha256:
        raise ValueError("clean source identity points to a different Verify config")


@dataclass(frozen=True)
class AssignedRowParentProjection:
    """Exactly six allowed members from one 3.1 parent train-cache shard."""

    seeds_xyz: np.ndarray
    scale_assignment: np.ndarray
    valid_assigned_row_index: np.ndarray
    valid_center_seed_index: np.ndarray
    valid_scale_block_index: np.ndarray
    valid_scale_id: np.ndarray
    opened_member_names: tuple[str, ...] = PARENT_ALLOWED_MEMBER_NAMES

    def __post_init__(self) -> None:
        valid_raw = np.asarray(self.valid_assigned_row_index)
        if valid_raw.ndim != 1 or len(valid_raw) == 0:
            raise ValueError("valid_assigned_row_index must be a nonempty vector")
        valid_count = len(valid_raw)
        arrays = {
            "seeds_xyz": _exact_array(
                self.seeds_xyz,
                name="seeds_xyz",
                dtype=np.float64,
                shape=(ASSIGNED_ROW_COUNT, 3),
                finite=True,
            ),
            "scale_assignment": _exact_array(
                self.scale_assignment,
                name="scale_assignment",
                dtype=np.int32,
                shape=(ASSIGNED_ROW_COUNT,),
            ),
            "valid_assigned_row_index": _exact_array(
                valid_raw,
                name="valid_assigned_row_index",
                dtype=np.int64,
                shape=(valid_count,),
            ),
            "valid_center_seed_index": _exact_array(
                self.valid_center_seed_index,
                name="valid_center_seed_index",
                dtype=np.int64,
                shape=(valid_count,),
            ),
            "valid_scale_block_index": _exact_array(
                self.valid_scale_block_index,
                name="valid_scale_block_index",
                dtype=np.int8,
                shape=(valid_count,),
            ),
            "valid_scale_id": _exact_array(
                self.valid_scale_id,
                name="valid_scale_id",
                dtype=np.int32,
                shape=(valid_count,),
            ),
        }
        if tuple(self.opened_member_names) != PARENT_ALLOWED_MEMBER_NAMES:
            raise ValueError("parent opened-member audit changed")
        assigned = arrays["valid_assigned_row_index"]
        if (
            np.any(assigned < 0)
            or np.any(assigned >= ASSIGNED_ROW_COUNT)
            or (len(assigned) > 1 and np.any(np.diff(assigned) <= 0))
        ):
            raise ValueError("valid assigned identities must be ordered, unique, and in range")
        center = arrays["valid_center_seed_index"]
        block = arrays["valid_scale_block_index"]
        scale = arrays["valid_scale_id"]
        if not np.array_equal(center, assigned % UNIQUE_CENTER_COUNT):
            raise ValueError("valid center identity disagrees with assigned row")
        if not np.array_equal(block, (assigned // UNIQUE_CENTER_COUNT).astype(np.int8)):
            raise ValueError("valid block identity disagrees with assigned row")
        assignment = arrays["scale_assignment"]
        if not np.array_equal(scale, assignment[assigned]):
            raise ValueError("valid scale identity disagrees with assigned assignment")
        all_blocks = np.repeat(
            np.arange(SCALE_BLOCK_COUNT, dtype=np.int8), UNIQUE_CENTER_COUNT
        )
        if not np.array_equal(all_blocks, (assignment >= SCALES_PER_BLOCK).astype(np.int8)):
            raise ValueError("assigned scale IDs disagree with block-major row identity")
        if np.any(assignment < 0) or np.any(assignment >= SCALE_COUNT):
            raise ValueError("scale_assignment lies outside 0..1999")
        if not np.array_equal(
            arrays["seeds_xyz"][:UNIQUE_CENTER_COUNT],
            arrays["seeds_xyz"][UNIQUE_CENTER_COUNT:],
        ):
            raise ValueError("legacy and expanded rows do not share the same center grid")
        counts = np.bincount(assignment, minlength=SCALE_COUNT)
        if counts.shape != (SCALE_COUNT,) or np.any(counts != ROWS_PER_EXACT_SCALE):
            raise ValueError("each exact scale must have exactly 64 assigned rows")
        dx_level = (assignment % SCALES_PER_BLOCK) // 100
        group = all_blocks.astype(np.int32) * DX_LEVELS_PER_BLOCK + dx_level
        group_counts = np.bincount(group, minlength=20)
        if np.any(group_counts != ASSIGNED_ROWS_PER_SOURCE_BLOCK_DX_LEVEL):
            raise ValueError("each source x block x dx group must have exactly 6400 rows")
        for name, values in arrays.items():
            object.__setattr__(self, name, values)
        object.__setattr__(self, "opened_member_names", PARENT_ALLOWED_MEMBER_NAMES)


def load_assigned_row_parent_projection(
    path: str | Path,
    *,
    expected_size_bytes: int,
    expected_file_sha256: str,
    expected_array_sha256: Mapping[str, str],
) -> AssignedRowParentProjection:
    """Authenticate one parent file while opening only the six allowed arrays."""

    source = Path(path).resolve()
    expected_hashes = {
        name: _hex(expected_array_sha256[name], 64, name=f"parent {name} SHA-256")
        for name in PARENT_ALLOWED_MEMBER_NAMES
    } if set(expected_array_sha256) == set(PARENT_ALLOWED_MEMBER_NAMES) else None
    if expected_hashes is None:
        raise ValueError("expected_array_sha256 must contain exactly six allowed members")
    _size, _digest, before = _stable_file_identity(
        source,
        expected_size=_positive_integer(expected_size_bytes, name="parent size"),
        expected_sha256=expected_file_sha256,
    )
    with np.load(source, allow_pickle=False) as archive:
        missing = set(PARENT_ALLOWED_MEMBER_NAMES).difference(archive.files)
        if missing:
            raise ValueError(f"parent cache misses allowed members: {sorted(missing)}")
        arrays = {
            name: np.asarray(archive[name]) for name in PARENT_ALLOWED_MEMBER_NAMES
        }
    after = _FileIdentity.from_stat(source.stat(follow_symlinks=False))
    if after != before:
        raise RuntimeError("parent cache changed while allowed members were read")
    final_size, final_digest, final_identity = _stable_file_identity(
        source,
        expected_size=expected_size_bytes,
        expected_sha256=expected_file_sha256,
    )
    if final_size != before.size_bytes or final_digest != expected_file_sha256 or final_identity != before:
        raise RuntimeError("parent cache identity changed during narrow projection")
    projection = AssignedRowParentProjection(**arrays)
    actual = {
        name: canonical_array_sha256(np.asarray(getattr(projection, name)))
        for name in PARENT_ALLOWED_MEMBER_NAMES
    }
    if actual != expected_hashes:
        raise ValueError("parent allowed-array canonical SHA-256 mismatch")
    return projection


def _group_ids(
    assignment: np.ndarray, *, dataset_index: int, source_ordinal: int
) -> np.ndarray:
    groups = np.empty((ASSIGNED_ROW_COUNT, 4), dtype=np.int32)
    groups[:, 0] = np.int32(dataset_index)
    groups[:, 1] = np.int32(source_ordinal)
    groups[:, 2] = (np.arange(ASSIGNED_ROW_COUNT, dtype=np.int64) // UNIQUE_CENTER_COUNT).astype(
        np.int32
    )
    groups[:, 3] = (assignment % SCALES_PER_BLOCK) // 100
    return np.ascontiguousarray(groups)


@dataclass(frozen=True)
class SourceCenteredSidecarPayload:
    assigned_row_index: np.ndarray
    center_seed_index: np.ndarray
    scale_block_index: np.ndarray
    scale_id: np.ndarray
    seed_velocity_xyz: np.ndarray
    source_centered_seed4: np.ndarray
    group_mean_curl_xyz: np.ndarray
    physical_dx_by_scale: np.ndarray
    valid_assigned_row_index: np.ndarray
    valid_center_seed_index: np.ndarray
    valid_scale_block_index: np.ndarray
    valid_scale_id: np.ndarray
    dataset_index: int
    source_ordinal: int

    def __post_init__(self) -> None:
        dataset_index = _nonnegative_integer(self.dataset_index, name="dataset_index")
        source_ordinal = _nonnegative_integer(self.source_ordinal, name="source_ordinal")
        valid_raw = np.asarray(self.valid_assigned_row_index)
        if valid_raw.ndim != 1 or len(valid_raw) == 0:
            raise ValueError("sidecar valid identity must be a nonempty vector")
        valid_count = len(valid_raw)
        contract: dict[str, tuple[np.dtype | type, tuple[int, ...], bool]] = {
            "assigned_row_index": (np.int64, (ASSIGNED_ROW_COUNT,), False),
            "center_seed_index": (np.int64, (ASSIGNED_ROW_COUNT,), False),
            "scale_block_index": (np.int8, (ASSIGNED_ROW_COUNT,), False),
            "scale_id": (np.int32, (ASSIGNED_ROW_COUNT,), False),
            "seed_velocity_xyz": (np.float32, (ASSIGNED_ROW_COUNT, 7, 3), True),
            "source_centered_seed4": (np.float32, (ASSIGNED_ROW_COUNT, 4), True),
            "group_mean_curl_xyz": (np.float64, (20, 3), True),
            "physical_dx_by_scale": (np.float64, (SCALE_COUNT,), True),
            "valid_assigned_row_index": (np.int64, (valid_count,), False),
            "valid_center_seed_index": (np.int64, (valid_count,), False),
            "valid_scale_block_index": (np.int8, (valid_count,), False),
            "valid_scale_id": (np.int32, (valid_count,), False),
        }
        arrays = {
            name: _exact_array(
                getattr(self, name),
                name=name,
                dtype=dtype,
                shape=shape,
                finite=finite,
            )
            for name, (dtype, shape, finite) in contract.items()
        }
        assigned = arrays["assigned_row_index"]
        expected_assigned = np.arange(ASSIGNED_ROW_COUNT, dtype=np.int64)
        if not np.array_equal(assigned, expected_assigned):
            raise ValueError("assigned_row_index must equal arange(128000)")
        if not np.array_equal(arrays["center_seed_index"], assigned % UNIQUE_CENTER_COUNT):
            raise ValueError("assigned center identity changed")
        if not np.array_equal(
            arrays["scale_block_index"],
            (assigned // UNIQUE_CENTER_COUNT).astype(np.int8),
        ):
            raise ValueError("assigned block identity changed")
        scale = arrays["scale_id"]
        if np.any(scale < 0) or np.any(scale >= SCALE_COUNT):
            raise ValueError("assigned scale ID lies outside 0..1999")
        if not np.array_equal(
            arrays["scale_block_index"], (scale >= SCALES_PER_BLOCK).astype(np.int8)
        ):
            raise ValueError("assigned block and scale IDs disagree")
        valid = arrays["valid_assigned_row_index"]
        if (
            np.any(valid < 0)
            or np.any(valid >= ASSIGNED_ROW_COUNT)
            or (len(valid) > 1 and np.any(np.diff(valid) <= 0))
        ):
            raise ValueError("valid assigned identity is not ordered and in range")
        for valid_name, assigned_name in (
            ("valid_center_seed_index", "center_seed_index"),
            ("valid_scale_block_index", "scale_block_index"),
            ("valid_scale_id", "scale_id"),
        ):
            if not np.array_equal(arrays[valid_name], arrays[assigned_name][valid]):
                raise ValueError(f"{valid_name} is not the exact assigned-row projection")
        physical = arrays["physical_dx_by_scale"]
        if np.any(physical <= 0.0):
            raise ValueError("physical_dx_by_scale must be positive")
        ratio = physical / FROZEN_DX_GRID_SCALE_BY_ID
        if not np.allclose(ratio, ratio[0], rtol=1.0e-12, atol=0.0):
            raise ValueError("physical_dx_by_scale is not the frozen 2000-scale table")
        groups = _group_ids(scale, dataset_index=dataset_index, source_ordinal=source_ordinal)
        computed = compute_source_centered_seed_time_kinematics(
            arrays["seed_velocity_xyz"],
            physical[scale],
            groups,
            primitive_order=FROZEN_PRIMITIVE_ORDER,
        )
        if not np.array_equal(arrays["source_centered_seed4"], computed.source_centered_kinematic4):
            raise ValueError("source_centered_seed4 does not reproduce from saved velocities")
        if not np.array_equal(arrays["group_mean_curl_xyz"], computed.group_mean_curl_xyz):
            raise ValueError("group_mean_curl_xyz does not reproduce exactly")
        for name, values in arrays.items():
            object.__setattr__(self, name, values)
        object.__setattr__(self, "dataset_index", dataset_index)
        object.__setattr__(self, "source_ordinal", source_ordinal)

    @property
    def valid_source_centered_seed4(self) -> np.ndarray:
        return _freeze(
            np.asarray(self.source_centered_seed4)[
                np.asarray(self.valid_assigned_row_index)
            ],
            dtype=np.float32,
        )


def validate_parent_sidecar_identity_join(
    parent: AssignedRowParentProjection, payload: SourceCenteredSidecarPayload
) -> None:
    if not np.array_equal(payload.scale_id, parent.scale_assignment):
        raise ValueError("sidecar assigned scale identity differs from parent")
    for name in SIDECAR_VALID_IDENTITY_NAMES:
        if not np.array_equal(np.asarray(getattr(payload, name)), np.asarray(getattr(parent, name))):
            raise ValueError(f"sidecar valid identity differs from parent for {name}")


def build_source_centered_sidecar_payload(
    parent: AssignedRowParentProjection,
    window: FlowWindow3D,
    *,
    dataset_index: int,
    source_ordinal: int,
) -> SourceCenteredSidecarPayload:
    physical = physical_dx_by_scale_for_window(window)
    scale = np.asarray(parent.scale_assignment)
    velocity = sample_seed_time_velocity_xyz(
        window, np.asarray(parent.seeds_xyz), physical[scale]
    )
    groups = _group_ids(
        scale,
        dataset_index=_nonnegative_integer(dataset_index, name="dataset_index"),
        source_ordinal=_nonnegative_integer(source_ordinal, name="source_ordinal"),
    )
    computed = compute_source_centered_seed_time_kinematics(
        velocity,
        physical[scale],
        groups,
        primitive_order=FROZEN_PRIMITIVE_ORDER,
    )
    payload = SourceCenteredSidecarPayload(
        assigned_row_index=np.arange(ASSIGNED_ROW_COUNT, dtype=np.int64),
        center_seed_index=np.arange(ASSIGNED_ROW_COUNT, dtype=np.int64) % UNIQUE_CENTER_COUNT,
        scale_block_index=(
            np.arange(ASSIGNED_ROW_COUNT, dtype=np.int64) // UNIQUE_CENTER_COUNT
        ).astype(np.int8),
        scale_id=scale,
        seed_velocity_xyz=velocity,
        source_centered_seed4=computed.source_centered_kinematic4,
        group_mean_curl_xyz=computed.group_mean_curl_xyz,
        physical_dx_by_scale=physical,
        valid_assigned_row_index=parent.valid_assigned_row_index,
        valid_center_seed_index=parent.valid_center_seed_index,
        valid_scale_block_index=parent.valid_scale_block_index,
        valid_scale_id=parent.valid_scale_id,
        dataset_index=dataset_index,
        source_ordinal=source_ordinal,
    )
    validate_parent_sidecar_identity_join(parent, payload)
    return payload


def _payload_arrays(payload: SourceCenteredSidecarPayload) -> dict[str, np.ndarray]:
    return {name: np.asarray(getattr(payload, name)) for name in SIDECAR_ARRAY_NAMES}


def _validated_bindings(bindings: Mapping[str, Any]) -> dict[str, Any]:
    if set(bindings) != set(SIDECAR_PROVENANCE_BINDING_NAMES):
        raise ValueError("provenance binding member set changed")
    canonical = _canonical_json_copy(bindings)
    if any(canonical[name] is None for name in SIDECAR_PROVENANCE_BINDING_NAMES):
        raise ValueError("provenance binding values may not be null")
    return canonical


def _sidecar_metadata(
    payload: SourceCenteredSidecarPayload,
    provenance_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    arrays = _payload_arrays(payload)
    hashes = {name: canonical_array_sha256(values) for name, values in arrays.items()}
    projected = payload.valid_source_centered_seed4
    return {
        "schema": SIDECAR_SCHEMA,
        "dataset_index": payload.dataset_index,
        "source_ordinal": payload.source_ordinal,
        "primitive_order": list(FROZEN_PRIMITIVE_ORDER),
        "group_id_order": list(FROZEN_SOURCE_GROUP_ID_ORDER),
        "kinematic_feature_order": list(FROZEN_SOURCE_CENTERED_KINEMATIC_FEATURE_ORDER),
        "sampling_time": "portable_velocity_frame_zero",
        "float_contract": {
            "velocity": "float32",
            "gradient_and_group_mean": "float64",
            "feature_serialization": "float32",
            "physical_dx": "float64",
        },
        "array_contract": {
            name: {"dtype": arrays[name].dtype.str, "shape": list(arrays[name].shape)}
            for name in SIDECAR_ARRAY_NAMES
        },
        "array_sha256": hashes,
        "combined_array_sha256": canonical_json_sha256(hashes),
        "valid_projection": {
            "rule": "source_centered_seed4[valid_assigned_row_index]",
            "stored_duplicate_feature": False,
            "dtype": projected.dtype.str,
            "shape": list(projected.shape),
            "canonical_sha256": canonical_array_sha256(projected),
            "identity_array_sha256": {
                name: hashes[name] for name in SIDECAR_VALID_IDENTITY_NAMES
            },
        },
        "provenance_bindings": _validated_bindings(provenance_bindings),
    }


@dataclass(frozen=True)
class LoadedSourceCenteredSidecar:
    payload: SourceCenteredSidecarPayload
    metadata: Mapping[str, Any]
    file_sha256: str


def load_source_centered_sidecar(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
    expected_provenance_bindings: Mapping[str, Any] | None = None,
    expected_parent: AssignedRowParentProjection | None = None,
) -> LoadedSourceCenteredSidecar:
    source = Path(path).resolve()
    size, digest, before = _stable_file_identity(source)
    if expected_file_sha256 is not None and digest != _hex(
        expected_file_sha256, 64, name="sidecar SHA-256"
    ):
        raise ValueError("sidecar file SHA-256 mismatch")
    with np.load(source, allow_pickle=False) as archive:
        if tuple(archive.files) != SIDECAR_ARCHIVE_MEMBER_NAMES:
            raise ValueError("sidecar archive member set or order changed")
        arrays = {name: np.asarray(archive[name]) for name in SIDECAR_ARRAY_NAMES}
        metadata_scalar = np.asarray(archive["metadata_json"])
        if metadata_scalar.ndim != 0 or metadata_scalar.dtype.kind != "U":
            raise ValueError("sidecar metadata_json must be a scalar Unicode string")
        metadata = json.loads(str(metadata_scalar.item()))
    after = _FileIdentity.from_stat(source.stat(follow_symlinks=False))
    if after != before:
        raise RuntimeError("sidecar changed while archive members were read")
    final_size, final_digest, final_identity = _stable_file_identity(source)
    if (final_size, final_digest, final_identity) != (size, digest, before):
        raise RuntimeError("sidecar changed during fresh authentication")
    if not isinstance(metadata, dict) or set(metadata) != _SIDECAR_METADATA_NAMES:
        raise ValueError("sidecar metadata member set changed")
    payload = SourceCenteredSidecarPayload(
        **arrays,
        dataset_index=int(metadata.get("dataset_index", -1)),
        source_ordinal=int(metadata.get("source_ordinal", -1)),
    )
    if expected_parent is not None:
        validate_parent_sidecar_identity_join(expected_parent, payload)
    expected_metadata = _sidecar_metadata(
        payload, metadata.get("provenance_bindings", {})
    )
    if metadata != expected_metadata:
        raise ValueError("sidecar metadata or canonical hashes failed fresh replay")
    if expected_provenance_bindings is not None and metadata["provenance_bindings"] != _validated_bindings(
        expected_provenance_bindings
    ):
        raise ValueError("sidecar provenance bindings mismatch")
    frozen = _deep_freeze(metadata)
    if not isinstance(frozen, Mapping):
        raise RuntimeError("sidecar metadata did not remain immutable")
    return LoadedSourceCenteredSidecar(payload, frozen, digest)


def write_source_centered_sidecar(
    path: str | Path,
    payload: SourceCenteredSidecarPayload,
    *,
    provenance_bindings: Mapping[str, Any],
) -> LoadedSourceCenteredSidecar:
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"sidecar already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = _sidecar_metadata(payload, provenance_bindings)
    arrays = _payload_arrays(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            np.savez_compressed(
                stream,
                **arrays,
                metadata_json=np.asarray(
                    json.dumps(
                        metadata,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                ),
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary_loaded = load_source_centered_sidecar(
            temporary, expected_provenance_bindings=provenance_bindings
        )
        os.link(temporary, output, follow_symlinks=False)
        _fsync_directory(output.parent)
        temporary.unlink()
        _fsync_directory(output.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    return load_source_centered_sidecar(
        output,
        expected_file_sha256=temporary_loaded.file_sha256,
        expected_provenance_bindings=provenance_bindings,
    )


def _expected_keys(contract: PreparationContract) -> tuple[tuple[str, int], ...]:
    return tuple(
        (dataset, ordinal)
        for dataset in contract.datasets
        for ordinal in range(contract.source_count)
    )


def _selected_row(
    manifest: Mapping[str, Any], *, dataset: str, source_ordinal: int
) -> Mapping[str, Any]:
    matches = [
        row
        for row in manifest["rows"]
        if row["dataset"] == dataset and int(row["source_ordinal"]) == source_ordinal
    ]
    if len(matches) != 1:
        raise ValueError("input row does not resolve uniquely")
    return matches[0]


def _narrow_parent_evidence(parent: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(str(parent["path"])).resolve()
    expected_size = _positive_integer(parent["size_bytes"], name="parent size")
    expected_file = _hex(parent["file_sha256"], 64, name="parent SHA-256")
    _size, _digest, before = _stable_file_identity(
        source, expected_size=expected_size, expected_sha256=expected_file
    )
    with np.load(source, allow_pickle=False) as archive:
        missing = set(PARENT_ALLOWED_MEMBER_NAMES).difference(archive.files)
        if missing:
            raise ValueError(f"parent misses allowed members: {sorted(missing)}")
        arrays = {name: np.asarray(archive[name]) for name in PARENT_ALLOWED_MEMBER_NAMES}
    if _FileIdentity.from_stat(source.stat(follow_symlinks=False)) != before:
        raise RuntimeError("parent changed while narrow input evidence was collected")
    projection = AssignedRowParentProjection(**arrays)
    hashes = {
        name: canonical_array_sha256(np.asarray(getattr(projection, name)))
        for name in PARENT_ALLOWED_MEMBER_NAMES
    }
    _stable_file_identity(
        source, expected_size=expected_size, expected_sha256=expected_file
    )
    return {
        "path": str(source),
        "size_bytes": expected_size,
        "file_sha256": expected_file,
        "schema": str(parent.get("schema", "")),
        "builder_git_commit": str(parent.get("builder_git_commit", "")),
        "config_sha256": str(parent.get("config_sha256", "")),
        "allowed_array_sha256": hashes,
        "opened_members": list(PARENT_ALLOWED_MEMBER_NAMES),
        "forbidden_members_opened": [],
    }


def build_source_centered_input_manifest(
    output_path: str | Path,
    *,
    early_input_manifest_path: str | Path,
    identity: CleanSourceIdentity,
    contract: PreparationContract = PRODUCTION_CONTRACT,
) -> Mapping[str, Any]:
    """Freeze all 32 rows while opening no portable NPZ and no forbidden parent member."""

    _validate_identity(identity, contract)
    early_path = Path(early_input_manifest_path).resolve()
    _require_no_forbidden_dataset_path(early_path, name="Early input manifest path")
    early, early_size, early_sha = _read_json_authenticated(
        early_path, expected_sha256=contract.early_input_manifest_sha256
    )
    early_rows = early.get("rows")
    if not isinstance(early_rows, list) or len(early_rows) != contract.row_count:
        raise ValueError("Early input manifest does not contain the frozen 32 rows")
    if early.get("rows_content_sha256") != canonical_json_sha256(early_rows):
        raise ValueError("Early input manifest row hash changed")
    parent_binding = early.get("parent_input_manifest")
    if not isinstance(parent_binding, Mapping) or parent_binding.get("file_sha256") != contract.parent_input_manifest_sha256:
        raise ValueError("Early manifest points to a different parent input manifest")
    parent_path = Path(str(parent_binding.get("path", ""))).resolve()
    parent_size, parent_sha, _ = _stable_file_identity(
        parent_path,
        expected_size=int(parent_binding.get("size_bytes", -1)),
        expected_sha256=contract.parent_input_manifest_sha256,
    )
    rows: list[dict[str, Any]] = []
    for dataset_index, ((dataset, ordinal), raw_row) in enumerate(
        zip(_expected_keys(contract), early_rows, strict=True)
    ):
        if not isinstance(raw_row, Mapping):
            raise ValueError("Early input row must be a mapping")
        expected_family = contract.family_by_dataset[dataset]
        expected_identity = {
            "dataset": dataset,
            "physical_family": expected_family,
            "split": "train",
            "source_ordinal": ordinal,
        }
        if any(raw_row.get(name) != value for name, value in expected_identity.items()):
            raise ValueError("Early input rows are missing, duplicated, extra, or reordered")
        parent = raw_row.get("parent_cache")
        portable = raw_row.get("portable")
        if not isinstance(parent, Mapping) or not isinstance(portable, Mapping):
            raise ValueError("Early input row lacks parent or portable evidence")
        _require_no_forbidden_dataset_path(parent.get("path", ""), name="parent path")
        _require_no_forbidden_dataset_path(portable.get("path", ""), name="portable path")
        portable_copy = _canonical_json_copy(portable)
        if set(portable_copy.get("array_sha256", {})) != {"velocity", "x", "y", "z", "time"}:
            raise ValueError("portable array hash contract changed")
        rows.append(
            {
                "dataset": dataset,
                "dataset_index": dataset_index // contract.source_count,
                "physical_family": expected_family,
                "split": "train",
                "source_ordinal": ordinal,
                "source_index": _nonnegative_integer(raw_row.get("source_index"), name="source_index"),
                "parent_cache": _narrow_parent_evidence(parent),
                "portable": portable_copy,
            }
        )
    value = _with_content_sha256(
        {
            "schema": INPUT_MANIFEST_SCHEMA,
            "experiment": EXPERIMENT,
            "status": "frozen",
            **_identity_payload(identity),
            "early_input_manifest": {
                "path": str(early_path),
                "size_bytes": early_size,
                "file_sha256": early_sha,
                "content_sha256": str(early.get("content_sha256", "")),
                "rows_content_sha256": str(early["rows_content_sha256"]),
            },
            "parent_input_manifest": {
                "path": str(parent_path),
                "size_bytes": parent_size,
                "file_sha256": parent_sha,
                "rows_content_sha256": str(parent_binding.get("rows_content_sha256", "")),
            },
            "input_scope": "every_and_only_8_train_datasets_times_4_source_windows",
            "assigned_rows_per_input": ASSIGNED_ROW_COUNT,
            "portable_npz_opened_during_freeze": False,
            "parent_opened_members": list(PARENT_ALLOWED_MEMBER_NAMES),
            "forbidden_parent_members_opened": [],
            "forbidden_dataset_access": False,
            "row_count": len(rows),
            "rows": rows,
            "rows_content_sha256": canonical_json_sha256(rows),
        }
    )
    output = Path(output_path).resolve()
    _require_no_forbidden_dataset_path(output, name="input manifest output")
    _atomic_json_no_overwrite(output, value)
    return authenticate_source_centered_input_manifest(
        output,
        expected_file_sha256=sha256_file(output),
        identity=identity,
        contract=contract,
        authenticate_all_referenced_rows=True,
    )


def authenticate_source_centered_input_manifest(
    path: str | Path,
    *,
    expected_file_sha256: str,
    identity: CleanSourceIdentity,
    contract: PreparationContract = PRODUCTION_CONTRACT,
    authenticate_all_referenced_rows: bool = True,
) -> Mapping[str, Any]:
    _validate_identity(identity, contract)
    source = Path(path).resolve()
    raw, _size, _digest = _read_json_authenticated(
        source, expected_sha256=expected_file_sha256
    )
    value = _verify_content_sha256(raw, name="source-centered input manifest")
    expected_top = {
        "schema": INPUT_MANIFEST_SCHEMA,
        "experiment": EXPERIMENT,
        "status": "frozen",
        **_identity_payload(identity),
        "input_scope": "every_and_only_8_train_datasets_times_4_source_windows",
        "assigned_rows_per_input": ASSIGNED_ROW_COUNT,
        "portable_npz_opened_during_freeze": False,
        "parent_opened_members": list(PARENT_ALLOWED_MEMBER_NAMES),
        "forbidden_parent_members_opened": [],
        "forbidden_dataset_access": False,
        "row_count": contract.row_count,
    }
    drift = {
        name: (value.get(name), expected)
        for name, expected in expected_top.items()
        if value.get(name) != expected
    }
    if drift:
        raise ValueError(f"input manifest provenance changed: {drift}")
    early = value.get("early_input_manifest")
    parent_binding = value.get("parent_input_manifest")
    if not isinstance(early, Mapping) or early.get("file_sha256") != contract.early_input_manifest_sha256:
        raise ValueError("input manifest Early binding changed")
    if not isinstance(parent_binding, Mapping) or parent_binding.get("file_sha256") != contract.parent_input_manifest_sha256:
        raise ValueError("input manifest parent binding changed")
    rows = value.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != contract.row_count
        or value.get("rows_content_sha256") != canonical_json_sha256(rows)
    ):
        raise ValueError("input manifest row population changed")
    for expected_key, row in zip(_expected_keys(contract), rows, strict=True):
        if not isinstance(row, Mapping):
            raise ValueError("input row is not a mapping")
        dataset, ordinal = expected_key
        dataset_index = contract.datasets.index(dataset)
        expected_identity = {
            "dataset": dataset,
            "dataset_index": dataset_index,
            "physical_family": contract.family_by_dataset[dataset],
            "split": "train",
            "source_ordinal": ordinal,
        }
        if any(row.get(name) != expected for name, expected in expected_identity.items()):
            raise ValueError("input rows are missing, duplicated, extra, or reordered")
        parent = row.get("parent_cache")
        portable = row.get("portable")
        if not isinstance(parent, Mapping) or not isinstance(portable, Mapping):
            raise ValueError("input row lacks parent or portable evidence")
        if tuple(parent.get("opened_members", ())) != PARENT_ALLOWED_MEMBER_NAMES:
            raise ValueError("parent opened-member evidence changed")
        if parent.get("forbidden_members_opened") != []:
            raise ValueError("input manifest reports forbidden parent-member access")
        hashes = parent.get("allowed_array_sha256")
        if not isinstance(hashes, Mapping) or set(hashes) != set(PARENT_ALLOWED_MEMBER_NAMES):
            raise ValueError("parent allowed-array hash population changed")
        for name in PARENT_ALLOWED_MEMBER_NAMES:
            _hex(hashes[name], 64, name=f"parent {name} SHA-256")
        portable_hashes = portable.get("array_sha256")
        if not isinstance(portable_hashes, Mapping) or set(portable_hashes) != {"velocity", "x", "y", "z", "time"}:
            raise ValueError("portable array hash population changed")
        if portable.get("combined_array_sha256") != canonical_json_sha256(dict(portable_hashes)):
            raise ValueError("portable combined array hash changed")
        if authenticate_all_referenced_rows:
            load_assigned_row_parent_projection(
                parent["path"],
                expected_size_bytes=int(parent["size_bytes"]),
                expected_file_sha256=str(parent["file_sha256"]),
                expected_array_sha256=dict(hashes),
            )
            _stable_file_identity(
                portable["path"],
                expected_size=int(portable["size_bytes"]),
                expected_sha256=str(portable["file_sha256"]),
            )
    if authenticate_all_referenced_rows:
        for binding_name in ("early_input_manifest", "parent_input_manifest"):
            binding = value[binding_name]
            _stable_file_identity(
                binding["path"],
                expected_size=int(binding["size_bytes"]),
                expected_sha256=str(binding["file_sha256"]),
            )
    frozen = _deep_freeze(value)
    if not isinstance(frozen, Mapping):
        raise RuntimeError("input manifest did not remain immutable")
    return frozen


def sidecar_row_relative_directory(
    dataset: str, source_ordinal: int, source_index: int
) -> Path:
    if not dataset or Path(dataset).name != dataset:
        raise ValueError("dataset must be one safe path component")
    _require_no_forbidden_dataset_path(dataset, name="sidecar dataset")
    ordinal = _nonnegative_integer(source_ordinal, name="source_ordinal")
    index = _nonnegative_integer(source_index, name="source_index")
    return Path(dataset) / f"source_{ordinal:02d}_index_{index:06d}"


def _provenance_bindings(
    *,
    input_manifest_path: Path,
    input_manifest_file_sha256: str,
    manifest: Mapping[str, Any],
    row: Mapping[str, Any],
    identity: CleanSourceIdentity,
) -> dict[str, Any]:
    parent = row["parent_cache"]
    portable = row["portable"]
    bindings = {
        "input_manifest_file_content_and_rows_sha256": {
            "path": str(input_manifest_path),
            "file_sha256": input_manifest_file_sha256,
            "content_sha256": str(manifest["content_sha256"]),
            "rows_content_sha256": str(manifest["rows_content_sha256"]),
        },
        "parent_cache_path_size_file_sha256_and_allowed_array_hashes": {
            "path": str(parent["path"]),
            "size_bytes": int(parent["size_bytes"]),
            "file_sha256": str(parent["file_sha256"]),
            "allowed_array_sha256": dict(parent["allowed_array_sha256"]),
            "opened_members": list(PARENT_ALLOWED_MEMBER_NAMES),
            "forbidden_members_opened": [],
        },
        "portable_path_size_file_sha256_and_array_hashes": {
            "path": str(portable["path"]),
            "size_bytes": int(portable["size_bytes"]),
            "file_sha256": str(portable["file_sha256"]),
            "array_sha256": dict(portable["array_sha256"]),
            "combined_array_sha256": str(portable["combined_array_sha256"]),
        },
        "config_clean_commit_and_source_hashes": {
            "verify_config_sha256": str(manifest["verify_config_sha256"]),
            "git_commit": identity.git_commit,
            "worktree_clean": True,
            "source_file_sha256_content_sha256": identity.source_content_sha256,
        },
        "algorithm_source_hashes": {
            path: identity.source_file_sha256[path]
            for path in REQUIRED_SOURCE_PATHS[1:9]
        },
        "dataset_family_source_identity": {
            "dataset": str(row["dataset"]),
            "dataset_index": int(row["dataset_index"]),
            "physical_family": str(row["physical_family"]),
            "source_ordinal": int(row["source_ordinal"]),
            "source_index": int(row["source_index"]),
        },
        "assigned_group_and_valid_projection_contract": {
            "assigned_row_count": ASSIGNED_ROW_COUNT,
            "unique_center_count": UNIQUE_CENTER_COUNT,
            "mean_group_fields": [
                "dataset_index",
                "source_ordinal",
                "scale_block_index",
                "dx_level_index",
            ],
            "required_rows_per_mean_group": ASSIGNED_ROWS_PER_SOURCE_BLOCK_DX_LEVEL,
            "valid_projection_rule": "source_centered_seed4[valid_assigned_row_index]",
            "forbidden_parent_members": sorted(PARENT_FORBIDDEN_MEMBER_NAMES),
        },
    }
    if tuple(bindings) != SIDECAR_PROVENANCE_BINDING_NAMES:
        raise RuntimeError("provenance binding order changed")
    return bindings


def build_one_source_centered_sidecar_and_completion(
    sidecar_root: str | Path,
    *,
    dataset: str,
    source_ordinal: int,
    input_manifest_path: str | Path,
    input_manifest_file_sha256: str,
    identity: CleanSourceIdentity,
    contract: PreparationContract = PRODUCTION_CONTRACT,
) -> Mapping[str, Any]:
    _validate_identity(identity, contract)
    manifest_path = Path(input_manifest_path).resolve()
    manifest = authenticate_source_centered_input_manifest(
        manifest_path,
        expected_file_sha256=input_manifest_file_sha256,
        identity=identity,
        contract=contract,
        authenticate_all_referenced_rows=False,
    )
    ordinal = _nonnegative_integer(source_ordinal, name="source_ordinal")
    if dataset not in contract.datasets or ordinal >= contract.source_count:
        raise ValueError("sidecar identity lies outside the frozen population")
    row = _selected_row(manifest, dataset=dataset, source_ordinal=ordinal)
    root = Path(sidecar_root).resolve()
    _require_no_forbidden_dataset_path(root, name="sidecar root")
    relative = sidecar_row_relative_directory(dataset, ordinal, int(row["source_index"]))
    row_dir = (root / relative).resolve()
    row_dir.relative_to(root)
    if row_dir.exists():
        raise FileExistsError(f"immutable sidecar row directory already exists: {row_dir}")
    row_dir.mkdir(parents=True, exist_ok=False)
    sidecar_path = row_dir / "source_centered_seed_time_kinematics.npz"
    completion_path = row_dir / "SIDECAR_COMPLETE.json"
    parent_evidence = row["parent_cache"]
    parent = load_assigned_row_parent_projection(
        parent_evidence["path"],
        expected_size_bytes=int(parent_evidence["size_bytes"]),
        expected_file_sha256=str(parent_evidence["file_sha256"]),
        expected_array_sha256=dict(parent_evidence["allowed_array_sha256"]),
    )
    portable_evidence = row["portable"]
    portable = load_portable_flow_window(
        portable_evidence["path"],
        expected_dataset=dataset,
        expected_experiment="mainExp_TemplateMatching_3.1",
        expected_config_sha256=str(portable_evidence["config_sha256"]),
        expected_builder_git_commit=str(portable_evidence["builder_git_commit"]),
        expected_source_start_index=int(row["source_index"]),
        expected_file_sha256=str(portable_evidence["file_sha256"]),
    )
    actual_portable_hashes = {
        "velocity": canonical_array_sha256(portable.window.velocity),
        "x": canonical_array_sha256(portable.window.coordinates_xyz[0]),
        "y": canonical_array_sha256(portable.window.coordinates_xyz[1]),
        "z": canonical_array_sha256(portable.window.coordinates_xyz[2]),
        "time": canonical_array_sha256(portable.window.time),
    }
    if actual_portable_hashes != dict(portable_evidence["array_sha256"]):
        raise ValueError("opened portable arrays differ from frozen evidence")
    payload = build_source_centered_sidecar_payload(
        parent,
        portable.window,
        dataset_index=int(row["dataset_index"]),
        source_ordinal=ordinal,
    )
    bindings = _provenance_bindings(
        input_manifest_path=manifest_path,
        input_manifest_file_sha256=str(input_manifest_file_sha256),
        manifest=manifest,
        row=row,
        identity=identity,
    )
    loaded = write_source_centered_sidecar(
        sidecar_path, payload, provenance_bindings=bindings
    )
    loaded = load_source_centered_sidecar(
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
            "dataset_index": int(row["dataset_index"]),
            "physical_family": str(row["physical_family"]),
            "split": "train",
            "source_ordinal": ordinal,
            "source_index": int(row["source_index"]),
            "input_manifest_path": str(manifest_path),
            "input_manifest_file_sha256": str(input_manifest_file_sha256),
            "input_manifest_content_sha256": str(manifest["content_sha256"]),
            "sidecar_relative_path": sidecar_path.relative_to(root).as_posix(),
            "sidecar_size_bytes": int(sidecar_path.stat().st_size),
            "sidecar_file_sha256": loaded.file_sha256,
            "sidecar_combined_array_sha256": str(loaded.metadata["combined_array_sha256"]),
            "valid_projection_sha256": str(
                loaded.metadata["valid_projection"]["canonical_sha256"]
            ),
            "assigned_row_count": ASSIGNED_ROW_COUNT,
            "valid_projection_row_count": len(parent.valid_assigned_row_index),
            "forbidden_parent_members_opened": [],
            "forbidden_dataset_access": False,
            "marker_write_order": "last_after_sidecar_was_closed_and_freshly_replayed",
        }
    )
    _atomic_json_no_overwrite(completion_path, completion)
    return authenticate_source_centered_row_completion(
        completion_path,
        sidecar_root=root,
        expected_file_sha256=sha256_file(completion_path),
        input_manifest=manifest,
        input_manifest_file_sha256=str(input_manifest_file_sha256),
        identity=identity,
        contract=contract,
        authenticate_sidecar=True,
    )


def authenticate_source_centered_row_completion(
    path: str | Path,
    *,
    sidecar_root: str | Path,
    expected_file_sha256: str,
    input_manifest: Mapping[str, Any],
    input_manifest_file_sha256: str,
    identity: CleanSourceIdentity,
    contract: PreparationContract = PRODUCTION_CONTRACT,
    authenticate_sidecar: bool = True,
) -> Mapping[str, Any]:
    _validate_identity(identity, contract)
    source = Path(path).resolve()
    root = Path(sidecar_root).resolve()
    raw, _size, _digest = _read_json_authenticated(
        source, expected_sha256=expected_file_sha256
    )
    value = _verify_content_sha256(raw, name="source-centered row completion")
    dataset = str(value.get("dataset", ""))
    ordinal = int(value.get("source_ordinal", -1))
    if dataset not in contract.datasets or not 0 <= ordinal < contract.source_count:
        raise ValueError("completion identity lies outside frozen population")
    row = _selected_row(input_manifest, dataset=dataset, source_ordinal=ordinal)
    relative = sidecar_row_relative_directory(dataset, ordinal, int(row["source_index"]))
    expected_path = (root / relative / "SIDECAR_COMPLETE.json").resolve()
    if source != expected_path:
        raise ValueError("completion path differs from frozen identity")
    expected = {
        "schema": ROW_COMPLETION_SCHEMA,
        "experiment": EXPERIMENT,
        "status": "passed",
        **_identity_payload(identity),
        "dataset": dataset,
        "dataset_index": int(row["dataset_index"]),
        "physical_family": contract.family_by_dataset[dataset],
        "split": "train",
        "source_ordinal": ordinal,
        "source_index": int(row["source_index"]),
        "input_manifest_path": str(Path(str(value.get("input_manifest_path", ""))).resolve()),
        "input_manifest_file_sha256": input_manifest_file_sha256,
        "input_manifest_content_sha256": str(input_manifest["content_sha256"]),
        "sidecar_relative_path": (
            relative / "source_centered_seed_time_kinematics.npz"
        ).as_posix(),
        "assigned_row_count": ASSIGNED_ROW_COUNT,
        "forbidden_parent_members_opened": [],
        "forbidden_dataset_access": False,
        "marker_write_order": "last_after_sidecar_was_closed_and_freshly_replayed",
    }
    drift = {
        name: (value.get(name), expected_value)
        for name, expected_value in expected.items()
        if value.get(name) != expected_value
    }
    if drift:
        raise ValueError(f"completion provenance changed: {drift}")
    manifest_path = Path(str(value["input_manifest_path"])).resolve()
    _stable_file_identity(manifest_path, expected_sha256=input_manifest_file_sha256)
    sidecar_path = (root / str(value["sidecar_relative_path"])).resolve()
    sidecar_path.relative_to(root)
    sidecar_size, sidecar_sha, _ = _stable_file_identity(
        sidecar_path,
        expected_size=_positive_integer(value.get("sidecar_size_bytes"), name="sidecar size"),
        expected_sha256=str(value.get("sidecar_file_sha256", "")),
    )
    del sidecar_size
    if authenticate_sidecar:
        parent_evidence = row["parent_cache"]
        parent = load_assigned_row_parent_projection(
            parent_evidence["path"],
            expected_size_bytes=int(parent_evidence["size_bytes"]),
            expected_file_sha256=str(parent_evidence["file_sha256"]),
            expected_array_sha256=dict(parent_evidence["allowed_array_sha256"]),
        )
        bindings = _provenance_bindings(
            input_manifest_path=manifest_path,
            input_manifest_file_sha256=input_manifest_file_sha256,
            manifest=input_manifest,
            row=row,
            identity=identity,
        )
        loaded = load_source_centered_sidecar(
            sidecar_path,
            expected_file_sha256=sidecar_sha,
            expected_provenance_bindings=bindings,
            expected_parent=parent,
        )
        if value.get("sidecar_combined_array_sha256") != loaded.metadata["combined_array_sha256"]:
            raise ValueError("completion combined array hash changed")
        if value.get("valid_projection_sha256") != loaded.metadata["valid_projection"]["canonical_sha256"]:
            raise ValueError("completion valid projection hash changed")
        if int(value.get("valid_projection_row_count", -1)) != len(parent.valid_assigned_row_index):
            raise ValueError("completion valid projection row count changed")
    frozen = _deep_freeze(value)
    if not isinstance(frozen, Mapping):
        raise RuntimeError("completion did not remain immutable")
    return frozen


def _expected_population_files(
    root: Path, manifest: Mapping[str, Any]
) -> tuple[Path, ...]:
    files: list[Path] = []
    for row in manifest["rows"]:
        relative = sidecar_row_relative_directory(
            str(row["dataset"]), int(row["source_ordinal"]), int(row["source_index"])
        )
        files.extend(
            (
                (root / relative / "source_centered_seed_time_kinematics.npz").resolve(),
                (root / relative / "SIDECAR_COMPLETE.json").resolve(),
            )
        )
    return tuple(files)


def write_source_centered_population_manifest(
    sidecar_root: str | Path,
    *,
    input_manifest_path: str | Path,
    input_manifest_file_sha256: str,
    identity: CleanSourceIdentity,
    contract: PreparationContract = PRODUCTION_CONTRACT,
) -> Mapping[str, Any]:
    _validate_identity(identity, contract)
    input_path = Path(input_manifest_path).resolve()
    manifest = authenticate_source_centered_input_manifest(
        input_path,
        expected_file_sha256=input_manifest_file_sha256,
        identity=identity,
        contract=contract,
        authenticate_all_referenced_rows=False,
    )
    root = Path(sidecar_root).resolve()
    population_path = root / "SIDECAR_POPULATION.json"
    if population_path.exists():
        raise FileExistsError(f"immutable population already exists: {population_path}")
    expected_files = set(_expected_population_files(root, manifest))
    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError(
            "sidecar population file set changed: "
            f"missing={sorted(map(str, expected_files - actual_files))}, "
            f"extra={sorted(map(str, actual_files - expected_files))}"
        )
    rows: list[dict[str, Any]] = []
    assigned_total = 0
    valid_total = 0
    for input_row in manifest["rows"]:
        relative = sidecar_row_relative_directory(
            str(input_row["dataset"]),
            int(input_row["source_ordinal"]),
            int(input_row["source_index"]),
        )
        completion_path = root / relative / "SIDECAR_COMPLETE.json"
        completion_sha = sha256_file(completion_path)
        completion = authenticate_source_centered_row_completion(
            completion_path,
            sidecar_root=root,
            expected_file_sha256=completion_sha,
            input_manifest=manifest,
            input_manifest_file_sha256=input_manifest_file_sha256,
            identity=identity,
            contract=contract,
            authenticate_sidecar=True,
        )
        assigned_count = int(completion["assigned_row_count"])
        valid_count = int(completion["valid_projection_row_count"])
        assigned_total += assigned_count
        valid_total += valid_count
        rows.append(
            {
                "dataset": str(input_row["dataset"]),
                "dataset_index": int(input_row["dataset_index"]),
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
                "valid_projection_sha256": str(completion["valid_projection_sha256"]),
                "assigned_row_count": assigned_count,
                "valid_projection_row_count": valid_count,
            }
        )
    value = _with_content_sha256(
        {
            "schema": POPULATION_MANIFEST_SCHEMA,
            "experiment": EXPERIMENT,
            "status": "passed",
            **_identity_payload(identity),
            "input_manifest_path": str(input_path),
            "input_manifest_file_sha256": input_manifest_file_sha256,
            "input_manifest_content_sha256": str(manifest["content_sha256"]),
            "sidecar_count": len(rows),
            "assigned_row_count_total": assigned_total,
            "valid_projection_row_count_total": valid_total,
            "rows": rows,
            "rows_content_sha256": canonical_json_sha256(rows),
            "forbidden_parent_members_opened": [],
            "forbidden_dataset_access": False,
            "manifest_write_order": "last_after_all_32_sidecars_and_completions_were_freshly_replayed",
        }
    )
    _atomic_json_no_overwrite(population_path, value)
    return authenticate_source_centered_population_manifest(
        population_path,
        sidecar_root=root,
        expected_file_sha256=sha256_file(population_path),
        input_manifest_path=input_path,
        input_manifest_file_sha256=input_manifest_file_sha256,
        identity=identity,
        contract=contract,
    )


def authenticate_source_centered_population_manifest(
    path: str | Path,
    *,
    sidecar_root: str | Path,
    expected_file_sha256: str,
    input_manifest_path: str | Path,
    input_manifest_file_sha256: str,
    identity: CleanSourceIdentity,
    contract: PreparationContract = PRODUCTION_CONTRACT,
) -> Mapping[str, Any]:
    _validate_identity(identity, contract)
    root = Path(sidecar_root).resolve()
    source = Path(path).resolve()
    if source != root / "SIDECAR_POPULATION.json":
        raise ValueError("population manifest must be at the sidecar root")
    raw, _size, _digest = _read_json_authenticated(
        source, expected_sha256=expected_file_sha256
    )
    value = _verify_content_sha256(raw, name="source-centered population manifest")
    input_path = Path(input_manifest_path).resolve()
    manifest = authenticate_source_centered_input_manifest(
        input_path,
        expected_file_sha256=input_manifest_file_sha256,
        identity=identity,
        contract=contract,
        authenticate_all_referenced_rows=False,
    )
    expected = {
        "schema": POPULATION_MANIFEST_SCHEMA,
        "experiment": EXPERIMENT,
        "status": "passed",
        **_identity_payload(identity),
        "input_manifest_path": str(input_path),
        "input_manifest_file_sha256": input_manifest_file_sha256,
        "input_manifest_content_sha256": str(manifest["content_sha256"]),
        "sidecar_count": contract.row_count,
        "assigned_row_count_total": contract.row_count * ASSIGNED_ROW_COUNT,
        "forbidden_parent_members_opened": [],
        "forbidden_dataset_access": False,
        "manifest_write_order": "last_after_all_32_sidecars_and_completions_were_freshly_replayed",
    }
    drift = {
        name: (value.get(name), expected_value)
        for name, expected_value in expected.items()
        if value.get(name) != expected_value
    }
    if drift:
        raise ValueError(f"population provenance changed: {drift}")
    rows = value.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != contract.row_count
        or value.get("rows_content_sha256") != canonical_json_sha256(rows)
    ):
        raise ValueError("population rows changed")
    expected_files = set(_expected_population_files(root, manifest)) | {source}
    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("authenticated population contains missing or extra files")
    valid_total = 0
    for input_row, row in zip(manifest["rows"], rows, strict=True):
        expected_identity = {
            "dataset": input_row["dataset"],
            "dataset_index": input_row["dataset_index"],
            "physical_family": input_row["physical_family"],
            "source_ordinal": input_row["source_ordinal"],
            "source_index": input_row["source_index"],
        }
        if not isinstance(row, Mapping) or any(
            row.get(name) != expected_value for name, expected_value in expected_identity.items()
        ):
            raise ValueError("population rows are missing, duplicated, extra, or reordered")
        completion_path = (root / str(row["completion_relative_path"])).resolve()
        completion_size, completion_sha, _ = _stable_file_identity(
            completion_path,
            expected_size=int(row["completion_size_bytes"]),
            expected_sha256=str(row["completion_file_sha256"]),
        )
        del completion_size
        completion = authenticate_source_centered_row_completion(
            completion_path,
            sidecar_root=root,
            expected_file_sha256=completion_sha,
            input_manifest=manifest,
            input_manifest_file_sha256=input_manifest_file_sha256,
            identity=identity,
            contract=contract,
            authenticate_sidecar=True,
        )
        for name in (
            "sidecar_relative_path",
            "sidecar_size_bytes",
            "sidecar_file_sha256",
            "sidecar_combined_array_sha256",
            "valid_projection_sha256",
            "assigned_row_count",
            "valid_projection_row_count",
        ):
            if row.get(name) != completion.get(name):
                raise ValueError(f"population/completion evidence differs for {name}")
        valid_total += int(row["valid_projection_row_count"])
    if int(value.get("valid_projection_row_count_total", -1)) != valid_total:
        raise ValueError("population valid projection total changed")
    frozen = _deep_freeze(value)
    if not isinstance(frozen, Mapping):
        raise RuntimeError("population manifest did not remain immutable")
    return frozen


__all__ = [
    "ASSIGNED_ROW_COUNT",
    "AssignedRowParentProjection",
    "CleanSourceIdentity",
    "DATASET_FAMILY_PAIRS",
    "EXPERIMENT",
    "INPUT_MANIFEST_SCHEMA",
    "LoadedSourceCenteredSidecar",
    "PARENT_ALLOWED_MEMBER_NAMES",
    "PARENT_FORBIDDEN_MEMBER_NAMES",
    "POPULATION_MANIFEST_SCHEMA",
    "PRODUCTION_CONTRACT",
    "PreparationContract",
    "REQUIRED_SOURCE_PATHS",
    "ROW_COMPLETION_SCHEMA",
    "SIDECAR_ARCHIVE_MEMBER_NAMES",
    "SIDECAR_ARRAY_NAMES",
    "SIDECAR_SCHEMA",
    "SourceCenteredSidecarPayload",
    "authenticate_source_centered_input_manifest",
    "authenticate_source_centered_population_manifest",
    "authenticate_source_centered_row_completion",
    "build_one_source_centered_sidecar_and_completion",
    "build_source_centered_input_manifest",
    "build_source_centered_sidecar_payload",
    "capture_clean_source_identity",
    "load_assigned_row_parent_projection",
    "load_source_centered_sidecar",
    "sidecar_row_relative_directory",
    "validate_parent_sidecar_identity_join",
    "write_source_centered_population_manifest",
    "write_source_centered_sidecar",
]

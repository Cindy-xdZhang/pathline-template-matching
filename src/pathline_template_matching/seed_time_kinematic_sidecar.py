"""Label-free seed-time kinematic sidecar core.

Only the six parent-cache members preregistered by
``Verify_EarlyOppositePairKinematics_1.1`` are projected.  Seven synchronous
seed velocities are sampled from portable frame zero with the exact production
RK4 interpolation primitive, then converted to the frozen four-coordinate
kinematic block.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numba import njit

from .arc_length_primitives import _interp4_quadrilinear_scalar
from .early_opposite_pair_kinematics import (
    FROZEN_KINEMATIC_FEATURE_ORDER,
    FROZEN_PRIMITIVE_ORDER,
    compute_seed_time_kinematic4,
)
from .netcdf_io import FlowWindow3D
from .portable_flow import (
    canonical_array_sha256,
    canonical_json_sha256,
    sha256_file,
)
from .vector_field import UnsteadyVectorField3D


PARENT_PROJECTION_MEMBER_NAMES = (
    "seeds_xyz",
    "valid_assigned_row_index",
    "valid_center_seed_index",
    "valid_scale_block_index",
    "valid_scale_id",
    "center_sample_time",
)
FORBIDDEN_PARENT_MEMBER_NAMES = frozenset(
    {
        "valid_labels",
        "reference_labels_all",
        "ivd_values_all",
        "ivd_volume",
        "metadata_json",
    }
)
SIDECAR_SCHEMA = "pathline_template_matching.seed_time_opposite_pair_kinematics_cache.v1"
SIDECAR_ARRAY_NAMES = (
    "valid_assigned_row_index",
    "valid_center_seed_index",
    "valid_scale_block_index",
    "valid_scale_id",
    "seed_velocity_xyz",
    "seed_kinematic4",
    "physical_dx_by_scale",
)
SIDECAR_ARCHIVE_MEMBER_NAMES = SIDECAR_ARRAY_NAMES + ("metadata_json",)
SIDECAR_PROVENANCE_BINDING_NAMES = (
    "input_manifest_file_and_content_sha256",
    "parent_cache_path_size_file_sha256_and_allowed_array_hashes",
    "portable_path_size_file_sha256_and_velocity_coordinate_time_array_hashes",
    "config_sha256_and_clean_builder_git_commit",
    "kinematic_algorithm_source_sha256",
    "composite_descriptor_contract",
    "dataset_family_source_identity",
    "line_order_time_interpolation_and_float_contract",
    "all_array_dtype_shape_canonical_sha256_and_combined_sha256",
)
_SIDECAR_METADATA_NAMES = frozenset(
    {
        "schema",
        "primitive_order",
        "kinematic_feature_order",
        "sampling_time",
        "gradient_layout",
        "float_contract",
        "array_contract",
        "array_sha256",
        "combined_array_sha256",
        "provenance_bindings",
    }
)
_FORBIDDEN_METADATA_KEY_TOKENS = (
    "label",
    "ivd",
    "metric",
    "prediction",
    "valid_rate",
    "positive",
    "negative",
)

_LEGACY_DX_GRID_SCALE = np.asarray(
    (
        0.250000000000,
        0.361111111111,
        0.472222222222,
        0.583333333333,
        0.694444444444,
        0.805555555556,
        0.916666666667,
        1.027777777778,
        1.138888888889,
        1.250000000000,
    ),
    dtype=np.float64,
)
_EXPANDED_DX_GRID_SCALE = np.asarray(
    (
        0.125000000000,
        0.388888888889,
        0.652777777778,
        0.916666666667,
        1.180555555556,
        1.444444444444,
        1.708333333333,
        1.972222222222,
        2.236111111111,
        2.500000000000,
    ),
    dtype=np.float64,
)
FROZEN_DX_GRID_SCALE_BY_ID = np.ascontiguousarray(
    np.concatenate(
        (
            np.repeat(_LEGACY_DX_GRID_SCALE, 100),
            np.repeat(_EXPANDED_DX_GRID_SCALE, 100),
        )
    ),
    dtype=np.float64,
)
FROZEN_DX_GRID_SCALE_BY_ID = np.frombuffer(
    FROZEN_DX_GRID_SCALE_BY_ID.tobytes(order="C"), dtype=np.float64
)
FROZEN_DX_GRID_SCALE_BY_ID.setflags(write=False)

_OFFSETS_XYZ = np.asarray(
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
_OFFSETS_XYZ = np.frombuffer(
    np.ascontiguousarray(_OFFSETS_XYZ).tobytes(order="C"), dtype=np.float64
).reshape(7, 3)
_OFFSETS_XYZ.setflags(write=False)


def _freeze(values: object, *, dtype: np.dtype | type) -> np.ndarray:
    copied = np.array(values, dtype=dtype, order="C", copy=True)
    result = np.frombuffer(
        copied.tobytes(order="C"), dtype=np.dtype(dtype)
    ).reshape(copied.shape)
    result.setflags(write=False)
    return result


def _deep_freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze_json(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze_json(child) for child in value)
    return value


def _require_exact_array(
    values: object,
    *,
    name: str,
    dtype: np.dtype | type,
    shape: tuple[int, ...],
    finite: bool = False,
) -> np.ndarray:
    raw = np.asarray(values)
    expected_dtype = np.dtype(dtype)
    if raw.dtype != expected_dtype or raw.shape != shape:
        raise ValueError(
            f"{name} must have dtype {expected_dtype} and shape {shape}; "
            f"got dtype {raw.dtype} and shape {raw.shape}"
        )
    if finite and not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain only finite values")
    return _freeze(raw, dtype=expected_dtype)


def _validate_hex_digest(value: object, *, name: str) -> str:
    selected = str(value)
    if len(selected) != 64 or any(character not in "0123456789abcdef" for character in selected):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return selected


def _validate_parent_identity(
    seeds_xyz: np.ndarray,
    assigned: np.ndarray,
    center: np.ndarray,
    block: np.ndarray,
    scale: np.ndarray,
) -> None:
    if len(seeds_xyz) == 0 or len(seeds_xyz) % 2:
        raise ValueError("seeds_xyz must contain two equal phase-3.1 scale blocks")
    if len(assigned) == 0:
        raise ValueError("parent projection must contain at least one valid row")
    if np.any(assigned < 0) or np.any(assigned >= len(seeds_xyz)):
        raise ValueError("valid_assigned_row_index is outside seeds_xyz")
    if len(assigned) > 1 and np.any(np.diff(assigned) <= 0):
        raise ValueError("valid_assigned_row_index must be unique and strictly increasing")
    center_count = len(seeds_xyz) // 2
    if np.any(center < 0) or np.any(center >= center_count):
        raise ValueError("valid_center_seed_index is outside the shared center grid")
    if np.any((block < 0) | (block > 1)):
        raise ValueError("valid_scale_block_index must contain only 0 or 1")
    if not np.array_equal(assigned, block.astype(np.int64) * center_count + center):
        raise ValueError("assigned, center, and scale-block identities disagree")
    if np.any((scale < 0) | (scale >= 2000)):
        raise ValueError("valid_scale_id must lie in 0..1999")
    expected_block = (scale >= 1000).astype(np.int8)
    if not np.array_equal(block, expected_block):
        raise ValueError("valid_scale_id disagrees with valid_scale_block_index")


@dataclass(frozen=True)
class ParentKinematicProjection:
    """Read-only projection of exactly six allowed parent-cache members."""

    seeds_xyz: np.ndarray
    valid_assigned_row_index: np.ndarray
    valid_center_seed_index: np.ndarray
    valid_scale_block_index: np.ndarray
    valid_scale_id: np.ndarray
    center_sample_time: np.ndarray
    opened_member_names: tuple[str, ...] = PARENT_PROJECTION_MEMBER_NAMES

    def __post_init__(self) -> None:
        assigned_raw = np.asarray(self.valid_assigned_row_index)
        if assigned_raw.ndim != 1:
            raise ValueError("valid_assigned_row_index must be one-dimensional")
        count = len(assigned_raw)
        seeds_raw = np.asarray(self.seeds_xyz)
        if seeds_raw.ndim != 2 or seeds_raw.shape[1:] != (3,):
            raise ValueError("seeds_xyz must have shape [assigned_rows,3]")
        arrays = {
            "seeds_xyz": _require_exact_array(
                seeds_raw,
                name="seeds_xyz",
                dtype=np.float64,
                shape=seeds_raw.shape,
                finite=True,
            ),
            "valid_assigned_row_index": _require_exact_array(
                assigned_raw,
                name="valid_assigned_row_index",
                dtype=np.int64,
                shape=(count,),
            ),
            "valid_center_seed_index": _require_exact_array(
                self.valid_center_seed_index,
                name="valid_center_seed_index",
                dtype=np.int64,
                shape=(count,),
            ),
            "valid_scale_block_index": _require_exact_array(
                self.valid_scale_block_index,
                name="valid_scale_block_index",
                dtype=np.int8,
                shape=(count,),
            ),
            "valid_scale_id": _require_exact_array(
                self.valid_scale_id,
                name="valid_scale_id",
                dtype=np.int32,
                shape=(count,),
            ),
            "center_sample_time": _require_exact_array(
                self.center_sample_time,
                name="center_sample_time",
                dtype=np.float32,
                shape=(count, 32),
                finite=True,
            ),
        }
        if tuple(self.opened_member_names) != PARENT_PROJECTION_MEMBER_NAMES:
            raise ValueError("parent projection opened-member audit changed")
        _validate_parent_identity(
            arrays["seeds_xyz"],
            arrays["valid_assigned_row_index"],
            arrays["valid_center_seed_index"],
            arrays["valid_scale_block_index"],
            arrays["valid_scale_id"],
        )
        if np.any(np.abs(arrays["center_sample_time"][:, 0]) > 1.0e-7):
            raise ValueError("center_sample_time[:,0] must equal zero within atol=1e-7")
        for name, values in arrays.items():
            object.__setattr__(self, name, values)
        object.__setattr__(self, "opened_member_names", PARENT_PROJECTION_MEMBER_NAMES)


def load_parent_kinematic_projection(
    path: str | Path,
    *,
    expected_size_bytes: int,
    expected_file_sha256: str,
    expected_array_sha256: Mapping[str, str],
) -> ParentKinematicProjection:
    """Open only the six frozen parent members and authenticate their bytes."""

    source = Path(path)
    if (
        isinstance(expected_size_bytes, (bool, np.bool_))
        or not isinstance(expected_size_bytes, (int, np.integer))
        or int(expected_size_bytes) < 1
    ):
        raise ValueError("expected_size_bytes must be a positive integer")
    snapshot = _read_authenticated_file_snapshot(source)
    if snapshot.identity.size_bytes != int(expected_size_bytes):
        raise ValueError("parent cache file size mismatch")
    expected_file = _validate_hex_digest(
        expected_file_sha256, name="expected_file_sha256"
    )
    if snapshot.sha256 != expected_file:
        raise ValueError("parent cache file SHA-256 mismatch")
    if set(expected_array_sha256) != set(PARENT_PROJECTION_MEMBER_NAMES):
        raise ValueError("expected_array_sha256 must contain exactly six allowed members")
    expected_hashes = {
        name: _validate_hex_digest(expected_array_sha256[name], name=f"{name} SHA-256")
        for name in PARENT_PROJECTION_MEMBER_NAMES
    }
    with np.load(io.BytesIO(snapshot.content), allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("parent cache contains duplicate archive member names")
        missing = set(PARENT_PROJECTION_MEMBER_NAMES).difference(archive.files)
        if missing:
            raise ValueError(f"parent cache misses allowed members: {sorted(missing)}")
        arrays = {
            name: np.asarray(archive[name]) for name in PARENT_PROJECTION_MEMBER_NAMES
        }
    projection = ParentKinematicProjection(**arrays)
    actual_hashes = {
        name: canonical_array_sha256(np.asarray(getattr(projection, name)))
        for name in PARENT_PROJECTION_MEMBER_NAMES
    }
    if actual_hashes != expected_hashes:
        raise ValueError("parent allowed-member canonical SHA-256 mismatch")
    return projection


def _validated_window_arrays(
    window: FlowWindow3D,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(window, FlowWindow3D):
        raise ValueError("window must be a validated FlowWindow3D")
    velocity = np.asarray(window.velocity)
    if (
        velocity.dtype != np.dtype(np.float32)
        or velocity.ndim != 5
        or velocity.shape[-1] != 3
        or velocity.shape[0] < 2
        or min(velocity.shape[1:4]) < 2
        or not np.isfinite(velocity).all()
    ):
        raise ValueError("portable velocity must be finite float32 [T>=2,Z>=2,Y>=2,X>=2,3]")
    x, y, z = (np.asarray(values) for values in window.coordinates_xyz)
    time = np.asarray(window.time)
    expected_shapes = {
        "x": (velocity.shape[3],),
        "y": (velocity.shape[2],),
        "z": (velocity.shape[1],),
        "time": (velocity.shape[0],),
    }
    axes = {"x": x, "y": y, "z": z, "time": time}
    for name, values in axes.items():
        if values.dtype != np.dtype(np.float64) or values.shape != expected_shapes[name]:
            raise ValueError(f"portable {name} must be float64 with shape {expected_shapes[name]}")
        difference = np.diff(values)
        if (
            not np.isfinite(values).all()
            or np.any(difference <= 0.0)
            or not np.allclose(difference, difference[0], rtol=1.0e-4, atol=1.0e-8)
        ):
            raise ValueError(f"portable {name} must be finite, increasing, and uniform")
    return velocity, x, y, z, time


def physical_dx_by_scale_for_window(window: FlowWindow3D) -> np.ndarray:
    """Return all 2000 explicit physical opposite-pair distances."""

    _validated_window_arrays(window)
    # The parent 3.1 cache computes h after FlowWindow3D has been normalized by
    # UnsteadyVectorField3D, whose grid spacing is float32.  Reuse that exact
    # production conversion rather than recomputing h from float64 coordinates.
    vector_field = UnsteadyVectorField3D.from_window(window)
    minimum_spacing = float(np.min(vector_field.grid_interval))
    result = FROZEN_DX_GRID_SCALE_BY_ID * np.float64(minimum_spacing)
    if result.shape != (2000,) or not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError("physical dx table is invalid")
    return _freeze(result, dtype=np.float64)


@njit(cache=True)
def _sample_positions_with_production_interpolator(
    velocity,
    domain_min,
    spacing,
    time_min,
    time_interval,
    positions,
):
    count = positions.shape[0]
    result = np.empty((count, 7, 3), dtype=np.float32)
    time_steps = velocity.shape[0]
    zdim = velocity.shape[1]
    ydim = velocity.shape[2]
    xdim = velocity.shape[3]
    for row in range(count):
        for line in range(7):
            vx, vy, vz = _interp4_quadrilinear_scalar(
                velocity,
                domain_min,
                spacing,
                xdim,
                ydim,
                zdim,
                time_min,
                time_interval,
                time_steps,
                positions[row, line, 0],
                positions[row, line, 1],
                positions[row, line, 2],
                time_min,
            )
            result[row, line, 0] = vx
            result[row, line, 1] = vy
            result[row, line, 2] = vz
    return result


def sample_seed_time_velocity_xyz(
    window: FlowWindow3D,
    center_seed_xyz: object,
    physical_dx: object,
) -> np.ndarray:
    """Sample ``center,x+,x-,y+,y-,z+,z-`` at portable frame zero."""

    _validated_window_arrays(window)
    vector_field = UnsteadyVectorField3D.from_window(window)
    velocity = vector_field.field
    centers_raw = np.asarray(center_seed_xyz)
    if centers_raw.ndim != 2 or centers_raw.shape[1:] != (3,) or len(centers_raw) == 0:
        raise ValueError("center_seed_xyz must have nonempty shape [N,3]")
    centers = _require_exact_array(
        centers_raw,
        name="center_seed_xyz",
        dtype=np.float64,
        shape=centers_raw.shape,
        finite=True,
    )
    dx = _require_exact_array(
        physical_dx,
        name="physical_dx",
        dtype=np.float64,
        shape=(len(centers),),
        finite=True,
    )
    if np.any(dx <= 0.0):
        raise ValueError("physical_dx must be strictly positive")
    positions = centers[:, None, :] + dx[:, None, None] * _OFFSETS_XYZ[None, :, :]
    domain_min = vector_field.domain_min
    domain_max = vector_field.domain_max
    if np.any(positions < domain_min[None, None, :]) or np.any(
        positions > domain_max[None, None, :]
    ):
        raise ValueError("one or more seven-point seed positions lie outside the portable domain")
    sampled = _sample_positions_with_production_interpolator(
        velocity,
        domain_min,
        vector_field.grid_interval,
        np.float64(vector_field.tmin),
        np.float64(vector_field.time_interval),
        np.ascontiguousarray(positions, dtype=np.float64),
    )
    if sampled.shape != (len(centers), 7, 3) or sampled.dtype != np.dtype(np.float32):
        raise RuntimeError("production seed-time sampler returned the wrong contract")
    if not np.isfinite(sampled).all():
        raise ValueError("seed-time sampler produced NaN or Inf")
    return _freeze(sampled, dtype=np.float32)


@dataclass(frozen=True)
class SeedTimeKinematicSidecarPayload:
    """The exact seven-array, valid-row-only sidecar payload."""

    valid_assigned_row_index: np.ndarray
    valid_center_seed_index: np.ndarray
    valid_scale_block_index: np.ndarray
    valid_scale_id: np.ndarray
    seed_velocity_xyz: np.ndarray
    seed_kinematic4: np.ndarray
    physical_dx_by_scale: np.ndarray

    def __post_init__(self) -> None:
        assigned_raw = np.asarray(self.valid_assigned_row_index)
        if assigned_raw.ndim != 1 or len(assigned_raw) == 0:
            raise ValueError("sidecar identity must contain at least one row")
        count = len(assigned_raw)
        arrays = {
            "valid_assigned_row_index": _require_exact_array(
                assigned_raw,
                name="valid_assigned_row_index",
                dtype=np.int64,
                shape=(count,),
            ),
            "valid_center_seed_index": _require_exact_array(
                self.valid_center_seed_index,
                name="valid_center_seed_index",
                dtype=np.int64,
                shape=(count,),
            ),
            "valid_scale_block_index": _require_exact_array(
                self.valid_scale_block_index,
                name="valid_scale_block_index",
                dtype=np.int8,
                shape=(count,),
            ),
            "valid_scale_id": _require_exact_array(
                self.valid_scale_id,
                name="valid_scale_id",
                dtype=np.int32,
                shape=(count,),
            ),
            "seed_velocity_xyz": _require_exact_array(
                self.seed_velocity_xyz,
                name="seed_velocity_xyz",
                dtype=np.float32,
                shape=(count, 7, 3),
                finite=True,
            ),
            "seed_kinematic4": _require_exact_array(
                self.seed_kinematic4,
                name="seed_kinematic4",
                dtype=np.float32,
                shape=(count, 4),
                finite=True,
            ),
            "physical_dx_by_scale": _require_exact_array(
                self.physical_dx_by_scale,
                name="physical_dx_by_scale",
                dtype=np.float64,
                shape=(2000,),
                finite=True,
            ),
        }
        assigned = arrays["valid_assigned_row_index"]
        if len(assigned) > 1 and np.any(np.diff(assigned) <= 0):
            raise ValueError("sidecar assigned identities must be unique and ordered")
        block = arrays["valid_scale_block_index"]
        scale = arrays["valid_scale_id"]
        if np.any((block < 0) | (block > 1)) or np.any((scale < 0) | (scale >= 2000)):
            raise ValueError("sidecar block or scale identity is outside the frozen range")
        if not np.array_equal(block, (scale >= 1000).astype(np.int8)):
            raise ValueError("sidecar block and scale identities disagree")
        physical = arrays["physical_dx_by_scale"]
        if np.any(physical <= 0.0):
            raise ValueError("physical_dx_by_scale must be positive")
        ratio = physical / FROZEN_DX_GRID_SCALE_BY_ID
        if not np.allclose(ratio, ratio[0], rtol=1.0e-12, atol=0.0):
            raise ValueError("physical_dx_by_scale is not the explicit frozen 2000-scale table")
        expected_feature = compute_seed_time_kinematic4(
            arrays["seed_velocity_xyz"],
            physical[scale],
            primitive_order=FROZEN_PRIMITIVE_ORDER,
        )
        if not np.array_equal(arrays["seed_kinematic4"], expected_feature):
            raise ValueError("seed_kinematic4 does not match the frozen velocity transform")
        for name, values in arrays.items():
            object.__setattr__(self, name, values)


def validate_sidecar_identity_join(
    parent: ParentKinematicProjection,
    sidecar: SeedTimeKinematicSidecarPayload,
) -> None:
    """Require exact identity arrays and parent row order."""

    for name in (
        "valid_assigned_row_index",
        "valid_center_seed_index",
        "valid_scale_block_index",
        "valid_scale_id",
    ):
        if not np.array_equal(np.asarray(getattr(parent, name)), np.asarray(getattr(sidecar, name))):
            raise ValueError(f"sidecar identity join mismatch for {name}")


def build_seed_time_kinematic_sidecar_payload(
    parent: ParentKinematicProjection,
    window: FlowWindow3D,
) -> SeedTimeKinematicSidecarPayload:
    """Build the label-free seven-array payload from one parent/window pair."""

    physical_by_scale = physical_dx_by_scale_for_window(window)
    scale = np.asarray(parent.valid_scale_id)
    per_row_dx = physical_by_scale[scale]
    centers = np.asarray(parent.seeds_xyz)[np.asarray(parent.valid_assigned_row_index)]
    velocity = sample_seed_time_velocity_xyz(window, centers, per_row_dx)
    feature = compute_seed_time_kinematic4(
        velocity,
        per_row_dx,
        primitive_order=FROZEN_PRIMITIVE_ORDER,
    )
    payload = SeedTimeKinematicSidecarPayload(
        valid_assigned_row_index=parent.valid_assigned_row_index,
        valid_center_seed_index=parent.valid_center_seed_index,
        valid_scale_block_index=parent.valid_scale_block_index,
        valid_scale_id=parent.valid_scale_id,
        seed_velocity_xyz=velocity,
        seed_kinematic4=feature,
        physical_dx_by_scale=physical_by_scale,
    )
    validate_sidecar_identity_join(parent, payload)
    return payload


def _payload_arrays(payload: SeedTimeKinematicSidecarPayload) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(getattr(payload, name)) for name in SIDECAR_ARRAY_NAMES
    }


def _reject_forbidden_metadata_keys(value: object, *, path: str = "metadata") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            selected = str(key).lower()
            if any(token in selected for token in _FORBIDDEN_METADATA_KEY_TOKENS):
                raise ValueError(f"forbidden label/IVD/metric metadata key at {path}.{key}")
            _reject_forbidden_metadata_keys(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_metadata_keys(child, path=f"{path}[{index}]")


def _validated_provenance_bindings(bindings: Mapping[str, Any]) -> dict[str, Any]:
    if set(bindings) != set(SIDECAR_PROVENANCE_BINDING_NAMES):
        raise ValueError("provenance_bindings must contain every frozen metadata binding")
    _reject_forbidden_metadata_keys(bindings)
    try:
        canonical = json.loads(
            json.dumps(
                bindings,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError("provenance_bindings must be finite JSON values") from error
    if any(canonical[name] is None for name in SIDECAR_PROVENANCE_BINDING_NAMES):
        raise ValueError("provenance_bindings values may not be null")
    return canonical


def _array_contract(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {
        name: {"dtype": np.asarray(arrays[name]).dtype.str, "shape": list(np.asarray(arrays[name]).shape)}
        for name in SIDECAR_ARRAY_NAMES
    }


def _sidecar_metadata(
    payload: SeedTimeKinematicSidecarPayload,
    provenance_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    arrays = _payload_arrays(payload)
    hashes = {name: canonical_array_sha256(values) for name, values in arrays.items()}
    return {
        "schema": SIDECAR_SCHEMA,
        "primitive_order": list(FROZEN_PRIMITIVE_ORDER),
        "kinematic_feature_order": list(FROZEN_KINEMATIC_FEATURE_ORDER),
        "sampling_time": "portable_velocity_frame_zero",
        "gradient_layout": "velocity_component_rows_and_spatial_derivative_columns",
        "float_contract": {
            "velocity": "float32",
            "derivative": "float64",
            "kinematic_serialization": "float32",
            "physical_dx_by_scale": "float64",
        },
        "array_contract": _array_contract(arrays),
        "array_sha256": hashes,
        "combined_array_sha256": canonical_json_sha256(hashes),
        "provenance_bindings": _validated_provenance_bindings(provenance_bindings),
    }


@dataclass(frozen=True)
class LoadedSeedTimeKinematicSidecar:
    payload: SeedTimeKinematicSidecarPayload
    metadata: Mapping[str, Any]
    file_sha256: str


@dataclass(frozen=True)
class _FileIdentity:
    """Filesystem identity used on both sides of sidecar deserialization."""

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


@dataclass(frozen=True)
class _AuthenticatedFileSnapshot:
    content: bytes
    identity: _FileIdentity
    sha256: str


def _read_authenticated_file_snapshot(source: Path) -> _AuthenticatedFileSnapshot:
    """Read one stable byte snapshot and bind it to the opened file identity."""

    path_before = _FileIdentity.from_stat(source.stat())
    with source.open("rb") as stream:
        descriptor_before = _FileIdentity.from_stat(os.fstat(stream.fileno()))
        if descriptor_before != path_before:
            raise ValueError("sidecar path changed before its bytes were opened")
        content = stream.read()
        descriptor_after = _FileIdentity.from_stat(os.fstat(stream.fileno()))
    path_after = _FileIdentity.from_stat(source.stat())
    if descriptor_after != descriptor_before or path_after != path_before:
        raise ValueError("sidecar file identity changed while its bytes were read")
    if len(content) != path_before.size_bytes:
        raise ValueError("sidecar byte count disagrees with its authenticated file size")
    return _AuthenticatedFileSnapshot(
        content=content,
        identity=path_before,
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _reauthenticate_file_snapshot(
    source: Path,
    snapshot: _AuthenticatedFileSnapshot,
) -> None:
    """Require the final path to remain the same size/mtime/inode and SHA-256."""

    path_before_hash = _FileIdentity.from_stat(source.stat())
    if path_before_hash != snapshot.identity:
        raise ValueError("sidecar file identity changed before post-read authentication")
    final_digest = sha256_file(source)
    path_after_hash = _FileIdentity.from_stat(source.stat())
    if path_after_hash != path_before_hash or final_digest != snapshot.sha256:
        raise ValueError("sidecar file size, mtime, identity, or SHA-256 changed while loading")


def _fsync_parent_directory(parent: Path) -> None:
    """Persist directory-entry changes on POSIX, including the Ibex target."""

    if os.name == "nt":
        # Python/Windows cannot open a directory descriptor for os.fsync.  The
        # no-replace hard-link operation remains atomic locally; production
        # publication is on Ibex/POSIX where the directory fsync below is
        # mandatory and any failure propagates.
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(os.fspath(parent), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_seed_time_kinematic_sidecar(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
    expected_provenance_bindings: Mapping[str, Any] | None = None,
    expected_parent: ParentKinematicProjection | None = None,
) -> LoadedSeedTimeKinematicSidecar:
    """Load and authenticate the exact sidecar member, dtype, shape, and hashes."""

    source = Path(path)
    snapshot = _read_authenticated_file_snapshot(source)
    if expected_file_sha256 is not None and snapshot.sha256 != _validate_hex_digest(
        expected_file_sha256, name="expected_file_sha256"
    ):
        raise ValueError("sidecar file SHA-256 mismatch")
    with np.load(io.BytesIO(snapshot.content), allow_pickle=False) as archive:
        if tuple(archive.files) != SIDECAR_ARCHIVE_MEMBER_NAMES:
            raise ValueError("sidecar archive member set or order changed")
        arrays = {name: np.asarray(archive[name]) for name in SIDECAR_ARRAY_NAMES}
        metadata_scalar = np.asarray(archive["metadata_json"])
        if metadata_scalar.ndim != 0 or metadata_scalar.dtype.kind != "U":
            raise ValueError("sidecar metadata_json must be a scalar Unicode string")
        metadata = json.loads(str(metadata_scalar.item()))
    if not isinstance(metadata, dict) or set(metadata) != _SIDECAR_METADATA_NAMES:
        raise ValueError("sidecar metadata member set changed")
    payload = SeedTimeKinematicSidecarPayload(**arrays)
    if expected_parent is not None:
        validate_sidecar_identity_join(expected_parent, payload)
    expected_core = _sidecar_metadata(payload, metadata.get("provenance_bindings", {}))
    if metadata != expected_core:
        raise ValueError("sidecar metadata, dtype, shape, or canonical hash authentication failed")
    if expected_provenance_bindings is not None:
        expected_bindings = _validated_provenance_bindings(expected_provenance_bindings)
        if metadata["provenance_bindings"] != expected_bindings:
            raise ValueError("sidecar provenance bindings mismatch")
    immutable_metadata = _deep_freeze_json(
        json.loads(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    )
    if not isinstance(immutable_metadata, Mapping):
        raise RuntimeError("authenticated sidecar metadata did not remain a mapping")
    _reauthenticate_file_snapshot(source, snapshot)
    return LoadedSeedTimeKinematicSidecar(
        payload=payload,
        metadata=immutable_metadata,
        file_sha256=snapshot.sha256,
    )


def write_seed_time_kinematic_sidecar(
    path: str | Path,
    payload: SeedTimeKinematicSidecarPayload,
    *,
    provenance_bindings: Mapping[str, Any],
) -> LoadedSeedTimeKinematicSidecar:
    """Authenticate a temporary NPZ, then hard-link it with atomic no-replace."""

    output = Path(path)
    if output.exists():
        raise FileExistsError(f"sidecar already exists: {output}")
    metadata = _sidecar_metadata(payload, provenance_bindings)
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    arrays = _payload_arrays(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = -1
            np.savez_compressed(
                destination,
                **arrays,
                metadata_json=np.asarray(metadata_json),
            )
            destination.flush()
            os.fsync(destination.fileno())
        load_seed_time_kinematic_sidecar(
            temporary,
            expected_provenance_bindings=provenance_bindings,
        )
        # A same-directory hard link is one atomic no-replace operation: if a
        # competing writer creates ``output`` after the early existence check,
        # os.link raises FileExistsError and never modifies that winner.
        os.link(temporary, output, follow_symlinks=False)
        _fsync_parent_directory(output.parent)
        temporary.unlink()
        _fsync_parent_directory(output.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    return load_seed_time_kinematic_sidecar(
        output,
        expected_file_sha256=sha256_file(output),
        expected_provenance_bindings=provenance_bindings,
    )


__all__ = [
    "FORBIDDEN_PARENT_MEMBER_NAMES",
    "FROZEN_DX_GRID_SCALE_BY_ID",
    "LoadedSeedTimeKinematicSidecar",
    "PARENT_PROJECTION_MEMBER_NAMES",
    "ParentKinematicProjection",
    "SIDECAR_ARCHIVE_MEMBER_NAMES",
    "SIDECAR_ARRAY_NAMES",
    "SIDECAR_PROVENANCE_BINDING_NAMES",
    "SIDECAR_SCHEMA",
    "SeedTimeKinematicSidecarPayload",
    "build_seed_time_kinematic_sidecar_payload",
    "load_parent_kinematic_projection",
    "load_seed_time_kinematic_sidecar",
    "physical_dx_by_scale_for_window",
    "sample_seed_time_velocity_xyz",
    "validate_sidecar_identity_join",
    "write_seed_time_kinematic_sidecar",
]

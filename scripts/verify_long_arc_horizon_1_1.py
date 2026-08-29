"""Run the synthetic numerical gate for ``Verify_LongArcHorizon_1.1``.

This verifier deliberately opens no raw flow, portable window, primitive cache,
label, feature, prediction, or metric artifact.  The train-only coverage gate is
a separate post-cache process; success here therefore means only that the
synthetic 2,000-scale/H48 numerical contract passed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching.arc_length_primitives import (  # noqa: E402
    ArcLengthPrimitiveResult,
    ArcLengthScaleTable,
    integrate_arc_length_primitives_3d,
)
from pathline_template_matching.phase21_pipeline import (  # noqa: E402
    balanced_scale_assignment,
    load_phase21_plan,
    load_phase31_plan,
)
from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_array_sha256,
    canonical_json_sha256,
    sha256_file,
)
from pathline_template_matching.vector_field import (  # noqa: E402
    UnsteadyVectorField3D,
)


EXPERIMENT = "Verify_LongArcHorizon_1.1"
SYNTHETIC_STATUS = "synthetic_gate_passed_train_only_coverage_not_run"
RESULT_ARRAY_NAMES = (
    "primitives",
    "valid_mask",
    "line_steps",
    "line_travel",
    "line_end_time",
    "line_reached_target",
    "scale_id",
    "dx_grid_scale",
    "ds_frame_scale",
    "arc_length_grid_scale",
    "physical_dx",
    "physical_dt",
    "target_arc_length",
)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _atomic_bytes(path, payload.encode("utf-8"))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"verification output already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    """Persist directory entries where the host supports directory fsync."""

    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError:
        # Windows does not expose POSIX directory fsync.  Files themselves were
        # already fsynced before the atomic replacement above.
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def load_frozen_inputs(
    main_config_path: Path, verify_config_path: Path
) -> tuple[Any, Any, dict[str, Any]]:
    """Load both production plans plus the strict verifier configuration."""

    verify = _load_yaml(verify_config_path)
    if verify.get("experiment") != EXPERIMENT:
        raise ValueError("unexpected verification config")
    if verify.get("parent_experiment") != "mainExp_TemplateMatching_3.1":
        raise ValueError("verification parent experiment drifted")
    plan = load_phase31_plan(main_config_path)
    legacy_path = ROOT / str(verify["scale_union_gate"]["legacy_parent_config"])
    legacy = load_phase21_plan(legacy_path)
    expected_parent_sha = str(
        verify["scale_union_gate"]["legacy_parent_config_sha256"]
    )
    if sha256_file(legacy_path) != expected_parent_sha:
        raise ValueError("legacy parent config SHA-256 drifted")
    return plan, legacy, verify


def _constant_field(
    velocity_xyz: np.ndarray,
    *,
    frame_count: int,
    source_dt: float,
    spacing_xyz: np.ndarray,
    shape_xyz: tuple[int, int, int],
    domain_min_xyz: np.ndarray | None = None,
) -> UnsteadyVectorField3D:
    velocity = np.asarray(velocity_xyz, dtype=np.float32)
    spacing = np.asarray(spacing_xyz, dtype=np.float64)
    xdim, ydim, zdim = (int(value) for value in shape_xyz)
    values = np.empty((frame_count, zdim, ydim, xdim, 3), dtype=np.float32)
    values[...] = velocity
    domain_min = (
        np.zeros(3, dtype=np.float64)
        if domain_min_xyz is None
        else np.asarray(domain_min_xyz, dtype=np.float64)
    )
    domain_max = domain_min + spacing * np.asarray(
        [xdim - 1, ydim - 1, zdim - 1], dtype=np.float64
    )
    return UnsteadyVectorField3D(
        field=values,
        domain_min=domain_min,
        domain_max=domain_max,
        grid_interval=spacing,
        time_interval=float(source_dt),
    )


def _analytic_geometry(
    analytic: dict[str, Any]
) -> tuple[np.ndarray, tuple[int, int, int], np.ndarray, np.ndarray]:
    spacing = np.asarray(analytic["grid_spacing_xyz"], dtype=np.float64)
    shape = tuple(int(value) for value in analytic["grid_shape_xyz"])
    bounds = analytic["grid_bounds_xyz"]
    domain_min = np.asarray([bounds[axis][0] for axis in "xyz"], dtype=np.float64)
    domain_max = np.asarray([bounds[axis][1] for axis in "xyz"], dtype=np.float64)
    expected_max = domain_min + spacing * (np.asarray(shape, dtype=np.float64) - 1.0)
    np.testing.assert_allclose(domain_max, expected_max, atol=1e-12, rtol=0.0)
    return spacing, shape, domain_min, domain_max


def _assert_result_slice_exact(
    reference: ArcLengthPrimitiveResult,
    candidate: ArcLengthPrimitiveResult,
    start: int,
    stop: int,
) -> None:
    if not reference.valid_mask[start:stop].all() or not candidate.valid_mask.all():
        raise AssertionError("batch exactness oracle requires all-valid analytic rows")
    for name in RESULT_ARRAY_NAMES:
        np.testing.assert_array_equal(
            getattr(reference, name)[start:stop], getattr(candidate, name)
        )
    if reference.integration_max_time != candidate.integration_max_time:
        raise AssertionError("external batching changed integration horizon")


def _assert_permuted_result_exact(
    reference: ArcLengthPrimitiveResult,
    candidate: ArcLengthPrimitiveResult,
    inverse: np.ndarray,
) -> None:
    if not reference.valid_mask.all() or not candidate.valid_mask.all():
        raise AssertionError("permutation exactness oracle requires all-valid rows")
    for name in RESULT_ARRAY_NAMES:
        np.testing.assert_array_equal(
            getattr(reference, name), getattr(candidate, name)[inverse]
        )
    if reference.integration_max_time != candidate.integration_max_time:
        raise AssertionError("input permutation changed integration horizon")


def _scale_union_rows(plan: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in plan.effective_scale_blocks:
        for scale_id in range(block.scale_id_start, block.scale_id_stop):
            local = scale_id - block.scale_id_start
            rows.append(
                {
                    "scale_id": int(scale_id),
                    "block_id": block.block_id,
                    "block_local_scale_id": int(local),
                    "dx_index": int(local // 100),
                    "ds_index": int((local // 10) % 10),
                    "arc_index": int(local % 10),
                    "dx_grid_scale": (
                        f"{plan.scale_table.dx_grid_scale[scale_id]:.12f}"
                    ),
                    "ds_frame_scale": (
                        f"{plan.scale_table.ds_frame_scale[scale_id]:.12f}"
                    ),
                    "arc_length_grid_scale": (
                        f"{plan.scale_table.arc_length_grid_scale[scale_id]:.12f}"
                    ),
                }
            )
    return rows


def _legacy_scale_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scale_id": row["scale_id"],
            "dx_grid_scale": row["dx_grid_scale"],
            "ds_frame_scale": row["ds_frame_scale"],
            "arc_length_grid_scale": row["arc_length_grid_scale"],
        }
        for row in rows[:1_000]
    ]


def verify_scale_union_and_assignments(
    plan: Any, legacy: Any, verify: dict[str, Any]
) -> dict[str, Any]:
    gate = verify["scale_union_gate"]
    assignment_gate = verify["assignment_gate"]
    scales = plan.scale_table
    legacy_scales = legacy.scale_table
    required_count = int(gate["required_total_scale_count"])
    if len(scales) != required_count or len(legacy_scales) != 1_000:
        raise AssertionError("scale population drifted")
    if scales.scale_id.dtype != np.dtype(np.int32):
        raise AssertionError("scale IDs are not int32")
    np.testing.assert_array_equal(scales.scale_id, np.arange(2_000, dtype=np.int32))
    for name in (
        "scale_id",
        "dx_grid_scale",
        "ds_frame_scale",
        "arc_length_grid_scale",
    ):
        np.testing.assert_array_equal(
            getattr(scales, name)[:1_000], getattr(legacy_scales, name)
        )
    tuples = list(
        zip(
            scales.dx_grid_scale,
            scales.ds_frame_scale,
            scales.arc_length_grid_scale,
            strict=True,
        )
    )
    if len(set(tuples)) != 2_000 or len(set(tuples[1_000:])) != 1_000:
        raise AssertionError("scale union contains duplicate numeric tuples")
    if set(tuples[:1_000]).intersection(tuples[1_000:]):
        raise AssertionError("expanded block overlaps the legacy block")

    scale_rows = _scale_union_rows(plan)
    if len(scale_rows) != 2_000:
        raise AssertionError("fixed-point scale manifest is not 2,000 rows")
    legacy_rows = scale_rows[:1_000]
    expanded_rows = scale_rows[1_000:]
    legacy_projection = _legacy_scale_projection(scale_rows)
    legacy_subset_hash = canonical_json_sha256(legacy_projection)
    expected_legacy_subset_hash = str(gate["required_legacy_scale_subset_sha256"])
    if legacy_subset_hash != expected_legacy_subset_hash:
        raise AssertionError(
            "fixed-point legacy scale projection differs from the frozen 2.1 hash"
        )

    assignment = plan.primitive_scale_assignment()
    expected_legacy = balanced_scale_assignment(64_000, 1_000, 15_068)
    expected_expanded_local = balanced_scale_assignment(64_000, 1_000, 35_068)
    expected = np.concatenate(
        (expected_legacy, expected_expanded_local + 1_000)
    ).astype(np.int32, copy=False)
    np.testing.assert_array_equal(assignment, expected)
    if assignment.dtype != np.dtype(np.int32) or assignment.shape != (128_000,):
        raise AssertionError("two-block primitive assignment shape/dtype drifted")
    counts = np.bincount(assignment.astype(np.int64), minlength=2_000)
    if not np.all(counts == 64):
        raise AssertionError("a scale does not receive exactly 64 rows")
    if np.array_equal(expected_legacy, expected_expanded_local):
        raise AssertionError("expanded assignment is not independent of legacy assignment")
    legacy_assignment_hash = canonical_array_sha256(assignment[:64_000])
    expected_legacy_assignment_hash = str(
        assignment_gate["required_legacy_assignment_canonical_sha256"]
    )
    if legacy_assignment_hash != expected_legacy_assignment_hash:
        raise AssertionError(
            "legacy 64,000-row assignment differs from the frozen 2.1 hash"
        )

    axis = np.linspace(0.0, 1.0, 40, endpoint=True, dtype=np.float64)
    zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij")
    center_seeds = np.ascontiguousarray(
        np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=-1)
    )
    repeated = plan.repeated_center_seeds(center_seeds)
    if repeated.shape != (128_000, 3):
        raise AssertionError("repeated center population is not 128000x3")
    np.testing.assert_array_equal(repeated[:64_000], center_seeds)
    np.testing.assert_array_equal(repeated[64_000:], center_seeds)

    required_horizon = float(
        verify["long_horizon_identity_gate"]["maximum_future_horizon_frames"]
    )
    required_frames = int(
        verify["long_horizon_identity_gate"]["derived_window_frame_count"]
    )
    if (
        plan.maximum_source_frame_intervals != required_horizon
        or plan.window_frame_count != required_frames
        or required_frames != int(required_horizon) + 1
    ):
        raise AssertionError("H48/49-frame plan identity drifted")

    return {
        "scale_count": len(scales),
        "legacy_scale_count": 1_000,
        "expanded_scale_count": 1_000,
        "legacy_ids_exactly_preserved": True,
        "expanded_tuples_unique_and_disjoint": True,
        "scale_id_dtype": str(scales.scale_id.dtype),
        "legacy_rows_content_sha256": canonical_json_sha256(legacy_rows),
        "legacy_scale_subset_sha256": legacy_subset_hash,
        "expanded_rows_content_sha256": canonical_json_sha256(expanded_rows),
        "scale_rows_content_sha256": canonical_json_sha256(scale_rows),
        "assignment_shape": list(assignment.shape),
        "assignment_dtype": str(assignment.dtype),
        "assignment_count_per_scale_minimum": int(counts.min()),
        "assignment_count_per_scale_maximum": int(counts.max()),
        "legacy_assignment_sha256": legacy_assignment_hash,
        "expanded_assignment_sha256": canonical_array_sha256(assignment[64_000:]),
        "complete_assignment_sha256": canonical_array_sha256(assignment),
        "shared_center_seed_sha256": canonical_array_sha256(center_seeds),
        "repeated_center_seed_sha256": canonical_array_sha256(repeated),
        "maximum_source_frame_intervals": plan.maximum_source_frame_intervals,
        "derived_window_frame_count": plan.window_frame_count,
        "assignment_contract": assignment_gate["algorithm"],
    }


def verify_full_constant_oracle(
    plan: Any, verify: dict[str, Any]
) -> dict[str, Any]:
    analytic = verify["analytic_field"]
    velocity = np.asarray(analytic["velocity_xyz"], dtype=np.float64)
    spacing, shape, domain_min, domain_max = _analytic_geometry(analytic)
    source_dt = float(analytic["source_frame_interval"])
    frame_count = int(analytic["source_frame_count"])
    field = _constant_field(
        velocity,
        frame_count=frame_count,
        source_dt=source_dt,
        spacing_xyz=spacing,
        shape_xyz=shape,
        domain_min_xyz=domain_min,
    )
    np.testing.assert_allclose(field.domain_min, domain_min, atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(field.domain_max, domain_max, atol=1e-7, rtol=0.0)
    # The frozen finite oracle is symmetric about zero and was dimensioned so
    # the maximum dx arm and complete 0.8-unit path remain inside its bounds.
    seeds = np.broadcast_to(
        np.zeros(3, dtype=np.float64), (2_000, 3)
    ).copy()
    assignment = np.arange(2_000, dtype=np.int32)
    result = integrate_arc_length_primitives_3d(
        field,
        seeds,
        0.0,
        plan.scale_table,
        assignment,
        maximum_source_frame_intervals=plan.maximum_source_frame_intervals,
    )
    if not result.valid_mask.all() or result.primitives.shape != (2_000, 7, 32, 4):
        raise AssertionError("constant analytic union did not yield 2000 valid rows")
    if result.line_steps.dtype != np.dtype(np.int32):
        raise AssertionError("line-step diagnostics are not int32")

    speed = float(np.linalg.norm(velocity))
    unit_velocity = velocity / speed
    initial = np.broadcast_to(seeds[:, None, :], (2_000, 7, 3)).copy()
    initial[:, 1, 0] += result.physical_dx
    initial[:, 2, 0] -= result.physical_dx
    initial[:, 3, 1] += result.physical_dx
    initial[:, 4, 1] -= result.physical_dx
    initial[:, 5, 2] += result.physical_dx
    initial[:, 6, 2] -= result.physical_dx
    fractions = np.linspace(0.0, 1.0, 32, dtype=np.float64)
    arc = result.target_arc_length[:, None] * fractions[None, :]
    expected_xyz = (
        initial[:, :, None, :]
        + arc[:, None, :, None] * unit_velocity[None, None, None, :]
    )
    expected_time = np.broadcast_to(arc[:, None, :] / speed, (2_000, 7, 32))
    spatial_tolerance = float(verify["tolerances"]["spatial_absolute"])
    time_tolerance = float(verify["tolerances"]["time_absolute"])
    np.testing.assert_allclose(
        result.primitives[..., :3], expected_xyz, atol=spatial_tolerance, rtol=0.0
    )
    np.testing.assert_allclose(
        result.primitives[..., 3], expected_time, atol=time_tolerance, rtol=0.0
    )
    expected_line_travel = np.broadcast_to(
        result.target_arc_length[:, None], result.line_travel.shape
    )
    np.testing.assert_allclose(
        result.line_travel,
        expected_line_travel,
        atol=spatial_tolerance,
        rtol=0.0,
    )
    expected_line_end_time = np.broadcast_to(
        result.target_arc_length[:, None] / speed, result.line_end_time.shape
    )
    np.testing.assert_allclose(
        result.line_end_time,
        expected_line_end_time,
        atol=time_tolerance,
        rtol=0.0,
    )
    segment = np.linalg.norm(np.diff(result.primitives[..., :3], axis=2), axis=-1)
    expected_segment = np.broadcast_to(
        result.target_arc_length[:, None, None] / 31.0, segment.shape
    )
    np.testing.assert_allclose(
        segment, expected_segment, atol=spatial_tolerance, rtol=0.0
    )
    np.testing.assert_allclose(
        result.physical_dx,
        plan.scale_table.dx_grid_scale * float(field.grid_interval.min()),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.physical_dt,
        plan.scale_table.ds_frame_scale * source_dt,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.target_arc_length,
        plan.scale_table.arc_length_grid_scale * float(field.grid_interval.min()),
        atol=0.0,
        rtol=0.0,
    )
    if result.integration_max_time != 48.0 * source_dt:
        raise AssertionError("analytic result did not record the H48 endpoint")

    # Genuine independent integration calls, not the compatibility chunk_size.
    batch_bounds = ((0, 1_000), (1_000, 2_000))
    for start, stop in batch_bounds:
        candidate = integrate_arc_length_primitives_3d(
            field,
            seeds[start:stop],
            0.0,
            plan.scale_table,
            assignment[start:stop],
            maximum_source_frame_intervals=plan.maximum_source_frame_intervals,
        )
        _assert_result_slice_exact(result, candidate, start, stop)
    permutation = np.random.Generator(np.random.PCG64(45_068)).permutation(2_000)
    permuted = integrate_arc_length_primitives_3d(
        field,
        seeds[permutation],
        0.0,
        plan.scale_table,
        assignment[permutation],
        maximum_source_frame_intervals=plan.maximum_source_frame_intervals,
    )
    _assert_permuted_result_exact(result, permuted, np.argsort(permutation))

    xyz_error = float(np.max(np.abs(result.primitives[..., :3] - expected_xyz)))
    time_error = float(np.max(np.abs(result.primitives[..., 3] - expected_time)))
    travel_error = float(
        np.max(
            np.abs(
                result.line_travel - result.target_arc_length[:, None]
            )
        )
    )
    step_bound = int(
        math.ceil(
            plan.maximum_source_frame_intervals
            / float(np.min(plan.scale_table.ds_frame_scale))
        )
        + 1
    )
    if step_bound != 961 or int(result.line_steps.max()) > step_bound:
        raise AssertionError("H48/ds=0.05 int32 step bound drifted")
    return {
        "valid_count": int(result.valid_mask.sum()),
        "primitive_shape": list(result.primitives.shape),
        "maximum_analytic_xyz_absolute_error": xyz_error,
        "maximum_analytic_time_absolute_error": time_error,
        "maximum_target_travel_absolute_error": travel_error,
        "maximum_observed_line_steps": int(result.line_steps.max()),
        "theoretical_int32_step_bound": step_bound,
        "line_steps_dtype": str(result.line_steps.dtype),
        "external_batch_count": len(batch_bounds),
        "external_batches_exact": True,
        "input_permutation_exact": True,
        "primitive_array_sha256": canonical_array_sha256(result.primitives),
        "valid_mask_sha256": canonical_array_sha256(result.valid_mask),
        "line_steps_sha256": canonical_array_sha256(result.line_steps),
    }


def verify_horizon_boundaries(plan: Any, verify: dict[str, Any]) -> dict[str, Any]:
    analytic = verify["analytic_field"]
    boundary = verify["horizon_boundary_oracles"]
    spacing, shape, domain_min, _ = _analytic_geometry(analytic)
    source_dt = float(boundary["source_frame_interval"])
    frame_count = int(boundary["source_frame_count"])
    target = float(boundary["target_arc_length"])
    arc_factor = target / float(spacing.min())
    scales = ArcLengthScaleTable(
        scale_id=np.asarray([0], dtype=np.int32),
        dx_grid_scale=np.asarray([0.25]),
        ds_frame_scale=np.asarray([0.05]),
        arc_length_grid_scale=np.asarray([arc_factor]),
    )
    seed = np.zeros((1, 3), dtype=np.float64)
    case_rows: list[dict[str, Any]] = []
    full_results: list[ArcLengthPrimitiveResult] = []
    for case in boundary["cases"]:
        field = _constant_field(
            np.asarray(case["velocity_xyz"], dtype=np.float64),
            frame_count=frame_count,
            source_dt=source_dt,
            spacing_xyz=spacing,
            shape_xyz=shape,
            domain_min_xyz=domain_min,
        )
        result = integrate_arc_length_primitives_3d(
            field,
            seed,
            0.0,
            scales,
            np.asarray([0], dtype=np.int32),
            maximum_source_frame_intervals=plan.maximum_source_frame_intervals,
        )
        expected_valid = bool(case["expected_valid"])
        if bool(result.valid_mask[0]) != expected_valid:
            raise AssertionError(f"horizon boundary case failed: {case['id']}")
        h12 = integrate_arc_length_primitives_3d(
            field,
            seed,
            0.0,
            scales,
            np.asarray([0], dtype=np.int32),
            maximum_source_frame_intervals=12.0,
        )
        if h12.valid_mask.any():
            raise AssertionError(f"boundary case incorrectly passed H12: {case['id']}")
        analytic_crossing = float(case["analytic_crossing_time"])
        if expected_valid:
            np.testing.assert_allclose(
                result.line_end_time,
                analytic_crossing,
                atol=float(verify["tolerances"]["time_absolute"]),
                rtol=0.0,
            )
            np.testing.assert_allclose(
                result.line_travel,
                target,
                atol=float(verify["tolerances"]["spatial_absolute"]),
                rtol=0.0,
            )
        else:
            if result.line_reached_target.any():
                raise AssertionError(
                    "target requiring more than H48 reported a reached line"
                )
            np.testing.assert_allclose(
                result.line_end_time,
                plan.maximum_source_frame_intervals * source_dt,
                atol=float(verify["tolerances"]["time_absolute"]),
                rtol=0.0,
            )
        full_results.append(result)
        case_rows.append(
            {
                "id": str(case["id"]),
                "analytic_crossing_time": analytic_crossing,
                "expected_valid": expected_valid,
                "observed_valid": bool(result.valid_mask[0]),
                "H12_observed_valid": bool(h12.valid_mask[0]),
                "line_reached_target_sha256": canonical_array_sha256(
                    result.line_reached_target
                ),
            }
        )

    first_velocity = np.asarray(boundary["cases"][0]["velocity_xyz"], dtype=np.float64)
    thirteen_frame_field = _constant_field(
        first_velocity,
        frame_count=13,
        source_dt=source_dt,
        spacing_xyz=spacing,
        shape_xyz=shape,
        domain_min_xyz=domain_min,
    )
    try:
        integrate_arc_length_primitives_3d(
            thirteen_frame_field,
            seed,
            0.0,
            scales,
            np.asarray([0], dtype=np.int32),
            maximum_source_frame_intervals=48.0,
        )
    except ValueError:
        short_window_rejected = True
    else:
        raise AssertionError("13-frame field was accepted as an H48 primitive source")
    return {
        "cases": case_rows,
        "after_H12_before_H48_valid": case_rows[0]["observed_valid"],
        "exactly_H48_valid": case_rows[1]["observed_valid"],
        "requires_more_than_H48_invalid": not case_rows[2]["observed_valid"],
        "all_cases_invalid_at_H12": not any(
            row["H12_observed_valid"] for row in case_rows
        ),
        "thirteen_frame_H12_window_rejected_for_H48": short_window_rejected,
        "combined_valid_mask_sha256": canonical_array_sha256(
            np.concatenate([item.valid_mask for item in full_results])
        ),
    }


def verify_time_varying_field(plan: Any, verify: dict[str, Any]) -> dict[str, Any]:
    # v_x(t)=a+b*t is represented exactly by linear frame interpolation and by
    # RK4.  The chosen target is reached after frame 12, so this also catches a
    # stale hidden H12 clamp in the time-varying path.
    analytic = verify["analytic_field"]
    oracle = verify["time_linear_velocity_oracle"]
    spacing, shape, domain_min, domain_max = _analytic_geometry(analytic)
    source_dt = float(oracle["source_frame_interval"])
    frame_count = int(oracle["source_frame_count"])
    xdim, ydim, zdim = shape
    a = 0.05
    b = 0.05
    time = np.arange(frame_count, dtype=np.float64) * source_dt
    values = np.zeros((frame_count, zdim, ydim, xdim, 3), dtype=np.float32)
    values[..., 1] = (a + b * time)[:, None, None, None]
    field = UnsteadyVectorField3D(
        field=values,
        domain_min=domain_min,
        domain_max=domain_max,
        grid_interval=spacing,
        time_interval=source_dt,
    )
    scales = ArcLengthScaleTable(
        scale_id=np.asarray([0], dtype=np.int32),
        dx_grid_scale=np.asarray([0.25]),
        ds_frame_scale=np.asarray([0.025]),
        arc_length_grid_scale=np.asarray(
            [float(oracle["target_arc_length"]) / float(spacing.min())]
        ),
    )
    seed = np.zeros((1, 3), dtype=np.float64)
    result = integrate_arc_length_primitives_3d(
        field,
        seed,
        0.0,
        scales,
        np.asarray([0], dtype=np.int32),
        maximum_source_frame_intervals=plan.maximum_source_frame_intervals,
    )
    if not result.valid_mask.all():
        raise AssertionError("time-varying target after H12 did not pass H48")
    target = float(result.target_arc_length[0])
    expected_end_time = (-a + math.sqrt(a * a + 2.0 * b * target)) / b
    if expected_end_time <= 12.0 * source_dt:
        raise AssertionError("time-varying oracle does not actually exercise frames after H12")
    initial = np.broadcast_to(seed[:, None, :], (1, 7, 3)).copy()
    initial[:, 1, 0] += result.physical_dx
    initial[:, 2, 0] -= result.physical_dx
    initial[:, 3, 1] += result.physical_dx
    initial[:, 4, 1] -= result.physical_dx
    initial[:, 5, 2] += result.physical_dx
    initial[:, 6, 2] -= result.physical_dx
    fractions = np.linspace(0.0, 1.0, 32, dtype=np.float64)
    expected_xyz = np.broadcast_to(initial[:, :, None, :], (1, 7, 32, 3)).copy()
    expected_xyz[..., 1] += target * fractions[None, None, :]
    sample_arc = target * fractions
    expected_time = (-a + np.sqrt(a * a + 2.0 * b * sample_arc)) / b
    spatial_tolerance = float(verify["tolerances"]["spatial_absolute"])
    time_tolerance = float(verify["tolerances"]["time_absolute"])
    np.testing.assert_allclose(
        result.primitives[..., :3], expected_xyz, atol=spatial_tolerance, rtol=0.0
    )
    # Time is piecewise-linearly resampled on the numerically exact RK4
    # polyline, so the configured tolerance covers only that last interpolation.
    expected_time_full = np.broadcast_to(
        expected_time[None, None, :], result.primitives[..., 3].shape
    )
    np.testing.assert_allclose(
        result.primitives[..., 3],
        expected_time_full,
        atol=time_tolerance,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.line_end_time,
        expected_end_time,
        atol=time_tolerance,
        rtol=0.0,
    )
    h12 = integrate_arc_length_primitives_3d(
        field,
        seed,
        0.0,
        scales,
        np.asarray([0], dtype=np.int32),
        maximum_source_frame_intervals=12.0,
    )
    if h12.valid_mask.any():
        raise AssertionError("time-varying target incorrectly passed a stale H12 horizon")
    return {
        "field": str(oracle["velocity_xyz_formula"]),
        "analytic_endpoint_time": expected_end_time,
        "H12_endpoint_time": 12.0 * source_dt,
        "uses_frames_after_H12": True,
        "H48_valid": True,
        "H12_invalid": True,
        "maximum_time_absolute_error": float(
            np.max(
                np.abs(
                    result.primitives[..., 3] - expected_time_full
                )
            )
        ),
        "primitive_array_sha256": canonical_array_sha256(result.primitives),
    }


def verify_fail_closed(plan: Any, verify: dict[str, Any]) -> dict[str, Any]:
    analytic = verify["analytic_field"]
    spacing, shape, domain_min, domain_max = _analytic_geometry(analytic)
    scales = ArcLengthScaleTable(
        scale_id=np.asarray([0], dtype=np.int32),
        dx_grid_scale=np.asarray([2.5]),
        ds_frame_scale=np.asarray([0.1]),
        arc_length_grid_scale=np.asarray([13.0]),
    )
    zero = _constant_field(
        np.zeros(3),
        frame_count=49,
        source_dt=0.1,
        spacing_xyz=spacing,
        shape_xyz=shape,
        domain_min_xyz=domain_min,
    )
    seed = np.zeros((1, 3), dtype=np.float64)
    zero_result = integrate_arc_length_primitives_3d(
        zero,
        seed,
        0.0,
        scales,
        np.asarray([0], dtype=np.int32),
        maximum_source_frame_intervals=plan.maximum_source_frame_intervals,
    )
    if zero_result.valid_mask.any() or zero_result.primitives.shape[0] != 0:
        raise AssertionError("zero velocity did not fail closed")

    moving = _constant_field(
        np.asarray([0.0, 0.2, 0.0]),
        frame_count=49,
        source_dt=0.1,
        spacing_xyz=spacing,
        shape_xyz=shape,
        domain_min_xyz=domain_min,
    )
    boundary_result = integrate_arc_length_primitives_3d(
        moving,
        np.asarray([[domain_min[0] + 0.01, 0.0, 0.0]], dtype=np.float64),
        0.0,
        scales,
        np.asarray([0], dtype=np.int32),
        maximum_source_frame_intervals=plan.maximum_source_frame_intervals,
    )
    if boundary_result.valid_mask.any() or boundary_result.primitives.shape[0] != 0:
        raise AssertionError("out-of-domain neighbor did not fail closed")

    try:
        integrate_arc_length_primitives_3d(
            moving,
            np.asarray([[np.nan, 0.0, 0.0]]),
            0.0,
            scales,
            np.asarray([0], dtype=np.int32),
            maximum_source_frame_intervals=plan.maximum_source_frame_intervals,
        )
    except ValueError:
        nonfinite_seed_rejected = True
    else:
        raise AssertionError("nonfinite seed did not fail closed")

    xdim, ydim, zdim = shape
    bad_values = np.zeros((49, zdim, ydim, xdim, 3), dtype=np.float32)
    bad_values[0, 0, 0, 0, 0] = np.nan
    try:
        UnsteadyVectorField3D(
            field=bad_values,
            domain_min=domain_min,
            domain_max=domain_max,
            grid_interval=spacing,
            time_interval=0.1,
        )
    except ValueError:
        nonfinite_field_rejected = True
    else:
        raise AssertionError("nonfinite velocity field did not fail closed")
    return {
        "zero_velocity_invalid": True,
        "boundary_neighbor_invalid": True,
        "nonfinite_seed_rejected": nonfinite_seed_rejected,
        "nonfinite_field_rejected": nonfinite_field_rejected,
    }


def _synthetic_evidence(
    plan: Any, legacy: Any, verify: dict[str, Any]
) -> dict[str, Any]:
    scale_assignment = verify_scale_union_and_assignments(plan, legacy, verify)
    constant = verify_full_constant_oracle(plan, verify)
    horizon = verify_horizon_boundaries(plan, verify)
    time_varying = verify_time_varying_field(plan, verify)
    fail_closed = verify_fail_closed(plan, verify)
    return {
        "schema": "pathline_template_matching.long_arc_synthetic_verification.v1",
        "experiment": EXPERIMENT,
        "phase": "synthetic",
        "status": SYNTHETIC_STATUS,
        "evidence_scope": "synthetic_numeric_only_no_real_flow_access",
        "performance_claim": "forbidden",
        "train_only_coverage_gate_run": False,
        "real_flow_files_opened": False,
        "portable_or_cache_files_opened": False,
        "test_dataset_artifacts_opened": False,
        "scale_union_and_assignment": scale_assignment,
        "constant_velocity_oracle": constant,
        "horizon_boundary_oracle": horizon,
        "time_varying_oracle": time_varying,
        "fail_closed_oracles": fail_closed,
    }


def synthetic_verification(
    main_config_path: Path, verify_config_path: Path
) -> dict[str, Any]:
    plan, legacy, verify = load_frozen_inputs(main_config_path, verify_config_path)
    return _synthetic_evidence(plan, legacy, verify)


def _environment_versions(plan: Any) -> dict[str, Any]:
    numba = __import__("numba")
    return {
        "schema": "pathline_template_matching.long_arc_environment.v1",
        "experiment": EXPERIMENT,
        "phase": "synthetic",
        "device": "cpu",
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": importlib.metadata.version("numpy"),
        "numba": importlib.metadata.version("numba"),
        "numba_threads": int(numba.get_num_threads()),
        "os_cpu_count": os.cpu_count(),
        "maximum_source_frame_intervals": float(
            plan.maximum_source_frame_intervals
        ),
    }


def _scale_union_manifest(
    plan: Any, scale_verification: dict[str, Any], provenance: dict[str, Any]
) -> dict[str, Any]:
    rows = _scale_union_rows(plan)
    legacy_rows = rows[:1_000]
    expanded_rows = rows[1_000:]
    legacy_subset_hash = canonical_json_sha256(_legacy_scale_projection(rows))
    computed = {
        "legacy_rows_content_sha256": canonical_json_sha256(legacy_rows),
        "legacy_scale_subset_sha256": legacy_subset_hash,
        "expanded_rows_content_sha256": canonical_json_sha256(expanded_rows),
        "rows_content_sha256": canonical_json_sha256(rows),
    }
    expected = {
        "legacy_rows_content_sha256": scale_verification[
            "legacy_rows_content_sha256"
        ],
        "legacy_scale_subset_sha256": scale_verification[
            "legacy_scale_subset_sha256"
        ],
        "expanded_rows_content_sha256": scale_verification[
            "expanded_rows_content_sha256"
        ],
        "rows_content_sha256": scale_verification["scale_rows_content_sha256"],
    }
    if computed != expected:
        raise AssertionError("scale manifest hashes disagree with the numeric gate")
    return {
        "schema": "pathline_template_matching.long_arc_scale_union_manifest.v1",
        "experiment": EXPERIMENT,
        "phase": "synthetic",
        "status": "passed",
        "layout": "ordered_union_of_two_10x10x10_cartesian_blocks",
        "block_order": [block.block_id for block in plan.effective_scale_blocks],
        "block_scale_id_ranges": [
            {
                "id": block.block_id,
                "start": int(block.scale_id_start),
                "stop_exclusive": int(block.scale_id_stop),
            }
            for block in plan.effective_scale_blocks
        ],
        "block_local_order": "dx_outer_ds_middle_arc_inner",
        "block_local_scale_id_formula": (
            "((dx_index * 10) + ds_index) * 10 + arc_index"
        ),
        "decimal_places": 12,
        "scale_count": len(rows),
        "legacy_rows_content_sha256": computed[
            "legacy_rows_content_sha256"
        ],
        "legacy_scale_subset_sha256": legacy_subset_hash,
        "expanded_rows_content_sha256": computed[
            "expanded_rows_content_sha256"
        ],
        "rows": rows,
        "rows_content_sha256": computed["rows_content_sha256"],
        "provenance": provenance,
    }


def _assignment_verification_artifact(
    verification: dict[str, Any], provenance: dict[str, Any]
) -> dict[str, Any]:
    fields = (
        "assignment_shape",
        "assignment_dtype",
        "assignment_count_per_scale_minimum",
        "assignment_count_per_scale_maximum",
        "legacy_assignment_sha256",
        "expanded_assignment_sha256",
        "complete_assignment_sha256",
        "shared_center_seed_sha256",
        "repeated_center_seed_sha256",
        "assignment_contract",
    )
    return {
        "schema": "pathline_template_matching.long_arc_assignment_verification.v1",
        "experiment": EXPERIMENT,
        "phase": "synthetic",
        "status": "passed",
        "block_order": ["legacy_2_1", "expanded_3_1"],
        "block_assigned_row_counts": [64_000, 64_000],
        "scale_count_per_block": 1_000,
        "verification": {name: verification[name] for name in fields},
        "provenance": provenance,
    }


def _phase_a_output_names(verify: dict[str, Any]) -> tuple[str, ...]:
    required = tuple(
        str(value)
        for value in verify["execution"]["phase_a_synthetic"]["required_outputs"]
    )
    expected = (
        "frozen_verify_config.yaml",
        "frozen_main_config.yaml",
        "synthetic_verification.json",
        "scale_union_manifest.json",
        "assignment_verification.json",
        "environment_versions.json",
        "SYNTHETIC_PASS.json",
    )
    if required != expected:
        raise ValueError("phase A required output contract drifted")
    return required


def _audit_output_files(run_dir: Path, names: tuple[str, ...]) -> list[dict[str, Any]]:
    entries = list(run_dir.iterdir())
    if any(not item.is_file() for item in entries):
        raise RuntimeError("phase A run directory contains a non-file entry")
    actual = {item.name for item in entries}
    if actual != set(names):
        raise RuntimeError(
            f"phase A evidence file set differs from the frozen contract: {actual}"
        )
    rows: list[dict[str, Any]] = []
    for name in names:
        path = run_dir / name
        size = int(path.stat().st_size)
        if size <= 0:
            raise RuntimeError(f"phase A evidence file is empty: {name}")
        first_hash = sha256_file(path)
        if first_hash != sha256_file(path):
            raise RuntimeError(f"phase A evidence file changed during audit: {name}")
        rows.append({"path": name, "size_bytes": size, "sha256": first_hash})
    return rows


def _write_phase_a_artifacts(
    *,
    run_dir: Path,
    plan: Any,
    verify: dict[str, Any],
    evidence: dict[str, Any],
    main_config_bytes: bytes,
    verify_config_bytes: bytes,
    provenance: dict[str, Any],
    environment: dict[str, Any],
    completed_utc: str,
) -> dict[str, Any]:
    required = _phase_a_output_names(verify)
    evidence_names = required[:-1]
    if not run_dir.is_dir() or any(run_dir.iterdir()):
        raise FileExistsError("phase A run directory must exist and be empty")

    evidence = dict(evidence)
    evidence.update({"completed_utc": completed_utc, "provenance": provenance})
    scale_verification = evidence["scale_union_and_assignment"]
    scale_manifest = _scale_union_manifest(plan, scale_verification, provenance)
    assignment = _assignment_verification_artifact(
        scale_verification, provenance
    )
    environment = dict(environment)
    environment.update({"completed_utc": completed_utc, "provenance": provenance})

    _atomic_bytes(run_dir / evidence_names[0], verify_config_bytes)
    _atomic_bytes(run_dir / evidence_names[1], main_config_bytes)
    _atomic_json(run_dir / evidence_names[2], evidence)
    _atomic_json(run_dir / evidence_names[3], scale_manifest)
    _atomic_json(run_dir / evidence_names[4], assignment)
    _atomic_json(run_dir / evidence_names[5], environment)

    output_rows = _audit_output_files(run_dir, evidence_names)
    output_by_name = {row["path"]: row for row in output_rows}
    if output_by_name["frozen_main_config.yaml"]["sha256"] != (
        provenance["main_config_sha256"]
    ):
        raise RuntimeError("frozen main config copy failed its SHA-256 audit")
    if output_by_name["frozen_verify_config.yaml"]["sha256"] != (
        provenance["verify_config_sha256"]
    ):
        raise RuntimeError("frozen Verify config copy failed its SHA-256 audit")

    marker = {
        "schema": "pathline_template_matching.long_arc_synthetic_pass.v1",
        "experiment": EXPERIMENT,
        "phase": "synthetic",
        "status": SYNTHETIC_STATUS,
        "completed_utc": completed_utc,
        "git_commit": provenance["git_commit"],
        "worktree_clean": True,
        "main_config_sha256": provenance["main_config_sha256"],
        "verify_config_sha256": provenance["verify_config_sha256"],
        "legacy_parent_config_sha256": provenance[
            "legacy_parent_config_sha256"
        ],
        "dataset_registry_sha256": provenance["dataset_registry_sha256"],
        "source_sha256": provenance["source_sha256"],
        "evidence_scope": "synthetic_numeric_only_no_real_flow_access",
        "train_only_coverage_gate_run": False,
        "final_verify_pass": False,
        "final_verify_pass_reason": (
            "phase_b_train_coverage_has_not_run"
        ),
        "outputs": output_rows,
        "outputs_content_sha256": canonical_json_sha256(output_rows),
        "marker_write_order": "last_after_six_audited_fsynced_phase_a_outputs",
    }
    _atomic_json(run_dir / required[-1], marker)
    final_entries = list(run_dir.iterdir())
    if any(not item.is_file() for item in final_entries) or {
        item.name for item in final_entries
    } != set(required):
        raise RuntimeError("phase A marker finalization produced an unexpected file set")
    return marker


def run(
    main_config_path: Path, verify_config_path: Path, run_dir: Path
) -> dict[str, Any]:
    if _git_dirty():
        raise RuntimeError("synthetic verification requires a clean committed worktree")
    git_commit = _git_commit()
    source_paths = {
        "arc_length_primitives.py": (
            ROOT / "src/pathline_template_matching/arc_length_primitives.py"
        ),
        "phase21_pipeline.py": (
            ROOT / "src/pathline_template_matching/phase21_pipeline.py"
        ),
        "portable_flow.py": ROOT / "src/pathline_template_matching/portable_flow.py",
        "vector_field.py": ROOT / "src/pathline_template_matching/vector_field.py",
        "verify_long_arc_horizon_1_1.py": Path(__file__).resolve(),
    }
    source_sha256 = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    main_config_path = main_config_path.resolve()
    verify_config_path = verify_config_path.resolve()
    run_dir = run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"phase A run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    _fsync_directory(run_dir.parent)

    main_config_bytes = main_config_path.read_bytes()
    verify_config_bytes = verify_config_path.read_bytes()
    main_config_sha = sha256_file(main_config_path)
    verify_config_sha = sha256_file(verify_config_path)
    plan, legacy, verify = load_frozen_inputs(main_config_path, verify_config_path)
    evidence = _synthetic_evidence(plan, legacy, verify)
    if (
        sha256_file(main_config_path) != main_config_sha
        or sha256_file(verify_config_path) != verify_config_sha
        or sha256_file(legacy.config_path) != legacy.config_sha256
        or sha256_file(plan.dataset_registry_path) != plan.dataset_registry_sha256
        or {
            name: sha256_file(path) for name, path in source_paths.items()
        }
        != source_sha256
        or _git_commit() != git_commit
        or _git_dirty()
    ):
        raise RuntimeError(
            "a frozen config, source, commit, or clean-worktree invariant changed "
            "during synthetic verification"
        )
    if plan.config_sha256 != main_config_sha:
        raise RuntimeError("production plan config hash disagrees with source bytes")
    completed_utc = datetime.now(timezone.utc).isoformat()
    provenance = {
        "git_commit": git_commit,
        "worktree_clean": True,
        "main_config_path": str(main_config_path),
        "main_config_sha256": main_config_sha,
        "verify_config_path": str(verify_config_path),
        "verify_config_sha256": verify_config_sha,
        "legacy_parent_config_path": str(legacy.config_path),
        "legacy_parent_config_sha256": legacy.config_sha256,
        "dataset_registry_path": str(plan.dataset_registry_path),
        "dataset_registry_sha256": plan.dataset_registry_sha256,
        "source_sha256": source_sha256,
    }
    return _write_phase_a_artifacts(
        run_dir=run_dir,
        plan=plan,
        verify=verify,
        evidence=evidence,
        main_config_bytes=main_config_bytes,
        verify_config_bytes=verify_config_bytes,
        provenance=provenance,
        environment=_environment_versions(plan),
        completed_utc=completed_utc,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("synthetic", "train-coverage"),
        default="synthetic",
        help=(
            "Run the pre-cache synthetic gate.  The train-coverage phase is "
            "reserved for the post-cache coverage API and is not implemented "
            "by this synthetic-only revision."
        ),
    )
    parser.add_argument(
        "--main-config",
        type=Path,
        default=ROOT / "config/mainExp_TemplateMatching_3.1.yaml",
    )
    parser.add_argument(
        "--verify-config",
        type=Path,
        default=ROOT / "config/Verify_LongArcHorizon_1.1.yaml",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=(
            "New immutable phase-A directory. Six evidence files are written "
            "and audited before SYNTHETIC_PASS.json is written last."
        ),
    )
    arguments = parser.parse_args()
    if arguments.phase != "synthetic":
        raise RuntimeError(
            "train-coverage requires the separately frozen post-cache coverage API; "
            "no final verification.json may be produced by the synthetic phase"
        )
    result = run(arguments.main_config, arguments.verify_config, arguments.run_dir)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

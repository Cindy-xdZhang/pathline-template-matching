"""Run the frozen analytic verification for arc-length primitives."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching.arc_length_primitives import (  # noqa: E402
    build_arc_length_scale_table,
    integrate_arc_length_primitives_3d,
)
from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_array_sha256,
    sha256_file,
)
from pathline_template_matching.vector_field import UnsteadyVectorField3D  # noqa: E402


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
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"verification output already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def _assert_result_slice_exact(reference, candidate, start: int, stop: int) -> None:
    """Compare one independently integrated input batch with the full result."""

    array_names = (
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
    for name in array_names:
        np.testing.assert_array_equal(
            getattr(reference, name)[start:stop], getattr(candidate, name)
        )
    if reference.integration_max_time != candidate.integration_max_time:
        raise AssertionError("external batch changed integration_max_time")


def _assert_permuted_result_exact(reference, candidate, inverse: np.ndarray) -> None:
    """Compare a permuted independent call after restoring input seed order."""

    array_names = (
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
    for name in array_names:
        np.testing.assert_array_equal(
            getattr(reference, name), getattr(candidate, name)[inverse]
        )
    if reference.integration_max_time != candidate.integration_max_time:
        raise AssertionError("input order changed integration_max_time")


def run(main_config_path: Path, verify_config_path: Path, output: Path) -> dict:
    if _git_dirty():
        raise RuntimeError("analytic verification requires a clean committed worktree")
    main_config = yaml.safe_load(main_config_path.read_text(encoding="utf-8"))
    verify_config = yaml.safe_load(verify_config_path.read_text(encoding="utf-8"))
    if main_config.get("experiment") != "mainExp_TemplateMatching_2.1":
        raise ValueError("unexpected main experiment config")
    if verify_config.get("experiment") != "Verify_ArcLengthResampling_1.1":
        raise ValueError("unexpected verification config")
    scale_config = main_config["scale_protocol"]
    scales = build_arc_length_scale_table(
        scale_config["dx_grid_scale"]["values"],
        scale_config["ds_frame_scale"]["values"],
        scale_config["arc_length_grid_scale"]["values"],
    )
    if len(scales) != int(verify_config["required_scale_count"]):
        raise AssertionError("scale table count changed")
    if scales.scale_id.dtype != np.dtype(np.int32):
        raise AssertionError("scale_id dtype changed")
    tuples = list(
        zip(
            scales.dx_grid_scale,
            scales.ds_frame_scale,
            scales.arc_length_grid_scale,
            strict=True,
        )
    )
    if len(set(tuples)) != 1000:
        raise AssertionError("scale tuples are not unique")
    dx_values = scale_config["dx_grid_scale"]["values"]
    ds_values = scale_config["ds_frame_scale"]["values"]
    arc_values = scale_config["arc_length_grid_scale"]["values"]
    anchor_expectations = {
        0: (dx_values[0], ds_values[0], arc_values[0]),
        9: (dx_values[0], ds_values[0], arc_values[9]),
        10: (dx_values[0], ds_values[1], arc_values[0]),
        100: (dx_values[1], ds_values[0], arc_values[0]),
        999: (dx_values[9], ds_values[9], arc_values[9]),
    }
    for scale_id, expected in anchor_expectations.items():
        if tuples[scale_id] != tuple(float(value) for value in expected):
            raise AssertionError(f"Cartesian scale order changed at scale_id={scale_id}")
    analytic = verify_config["analytic_field"]
    spacing = np.asarray(analytic["grid_spacing_xyz"], dtype=np.float64)
    source_dt = float(analytic["source_frame_interval"])
    frame_count = int(analytic["source_frame_count"])
    velocity = np.zeros((frame_count, 31, 31, 31, 3), dtype=np.float32)
    velocity[...] = np.asarray(analytic["velocity_xyz"], dtype=np.float32)
    field = UnsteadyVectorField3D(
        field=velocity,
        domain_min=np.zeros(3),
        domain_max=np.full(3, 3.0),
        grid_interval=spacing,
        time_interval=source_dt,
    )
    seeds = np.broadcast_to(np.asarray([0.5, 1.5, 1.5]), (len(scales), 3)).copy()
    assignment = np.arange(len(scales), dtype=np.int32)
    first = integrate_arc_length_primitives_3d(field, seeds, 0.0, scales, assignment)
    if not first.valid_mask.all() or first.primitives.shape != (1000, 7, 32, 4):
        raise AssertionError("constant velocity did not produce 1000 valid 7x32 primitives")

    # ``chunk_size`` is a compatibility argument and the integrator deliberately
    # launches one parallel kernel.  Verify real external batching instead of
    # pretending that two ignored argument values exercise different paths.
    external_batch_size = 137
    for start in range(0, len(seeds), external_batch_size):
        stop = min(start + external_batch_size, len(seeds))
        batch = integrate_arc_length_primitives_3d(
            field, seeds[start:stop], 0.0, scales, assignment[start:stop]
        )
        _assert_result_slice_exact(first, batch, start, stop)

    permutation = np.random.default_rng(15068).permutation(len(seeds))
    permuted = integrate_arc_length_primitives_3d(
        field, seeds[permutation], 0.0, scales, assignment[permutation]
    )
    _assert_permuted_result_exact(first, permuted, np.argsort(permutation))

    spatial_tolerance = float(verify_config["tolerances"]["spatial_absolute"])
    time_tolerance = float(verify_config["tolerances"]["time_absolute"])

    # Complete analytic oracle: every x/y/z/t value of every line and sample.
    # Arc distance ``s`` advances position by ``s * unit_velocity`` and time by
    # ``s / speed``.  This simultaneously detects line-order, direction,
    # sample-time, uniform-resampling, and final-segment truncation errors.
    constant_velocity = np.asarray(analytic["velocity_xyz"], dtype=np.float64)
    speed = float(np.linalg.norm(constant_velocity))
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("analytic verification velocity must have positive speed")
    unit_velocity = constant_velocity / speed
    expected_initial = np.broadcast_to(seeds[:, None, :], (1000, 7, 3)).copy()
    expected_initial[:, 1, 0] += first.physical_dx
    expected_initial[:, 2, 0] -= first.physical_dx
    expected_initial[:, 3, 1] += first.physical_dx
    expected_initial[:, 4, 1] -= first.physical_dx
    expected_initial[:, 5, 2] += first.physical_dx
    expected_initial[:, 6, 2] -= first.physical_dx
    arc_fractions = np.linspace(0.0, 1.0, 32, dtype=np.float64)
    expected_arc = first.target_arc_length[:, None] * arc_fractions[None, :]
    expected_xyz = (
        expected_initial[:, :, None, :]
        + expected_arc[:, None, :, None] * unit_velocity[None, None, None, :]
    )
    expected_time = np.broadcast_to(
        expected_arc[:, None, :] / speed, (1000, 7, 32)
    )
    expected_primitive = np.concatenate(
        (expected_xyz, expected_time[..., None]), axis=-1
    )
    if not np.isfinite(first.primitives).all():
        raise AssertionError("valid analytic primitives contain nonfinite values")
    np.testing.assert_allclose(
        first.primitives[..., :3],
        expected_primitive[..., :3],
        atol=spatial_tolerance,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        first.primitives[..., 3],
        expected_primitive[..., 3],
        atol=time_tolerance,
        rtol=0.0,
    )
    analytic_xyz_error = float(
        np.max(np.abs(first.primitives[..., :3] - expected_primitive[..., :3]))
    )
    analytic_time_error = float(
        np.max(np.abs(first.primitives[..., 3] - expected_primitive[..., 3]))
    )
    endpoint_xyz_error = float(
        np.max(np.abs(first.primitives[:, :, -1, :3] - expected_xyz[:, :, -1]))
    )
    segment_length = np.linalg.norm(
        np.diff(first.primitives[..., :3], axis=2), axis=-1
    )
    expected_segment = first.target_arc_length[:, None, None] / 31.0
    spatial_error = float(np.max(np.abs(segment_length - expected_segment)))
    expected_end_time = first.target_arc_length / speed
    time_error = float(
        np.max(np.abs(first.line_end_time - expected_end_time[:, None]))
    )
    travel_error = float(
        np.max(np.abs(first.line_travel - first.target_arc_length[:, None]))
    )
    initial_cross_error = float(
        np.max(np.abs(first.primitives[:, :, 0, :3] - expected_initial))
    )
    if spatial_error > spatial_tolerance or travel_error > spatial_tolerance:
        raise AssertionError("arc-length resampling exceeds the frozen spatial tolerance")
    if initial_cross_error > spatial_tolerance:
        raise AssertionError("initial seven-line dx offsets or line order changed")
    if time_error > time_tolerance:
        raise AssertionError("constant-velocity endpoint time exceeds tolerance")
    minimum_spacing = float(field.grid_interval.min())
    np.testing.assert_allclose(
        first.physical_dx,
        scales.dx_grid_scale * minimum_spacing,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        first.physical_dt,
        scales.ds_frame_scale * source_dt,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        first.target_arc_length,
        scales.arc_length_grid_scale * minimum_spacing,
        atol=0.0,
        rtol=0.0,
    )
    zero_field = UnsteadyVectorField3D(
        field=np.zeros_like(velocity),
        domain_min=field.domain_min,
        domain_max=field.domain_max,
        grid_interval=field.grid_interval,
        time_interval=field.time_interval,
    )
    zero = integrate_arc_length_primitives_3d(
        zero_field, seeds[:1], 0.0, scales, assignment[:1]
    )
    if zero.valid_mask.any() or zero.primitives.shape[0] != 0:
        raise AssertionError("zero velocity did not fail the arc-length target")
    boundary = integrate_arc_length_primitives_3d(
        field,
        np.asarray([[0.01, 1.5, 1.5]]),
        0.0,
        scales,
        np.asarray([100], dtype=np.int32),
    )
    if boundary.valid_mask.any() or boundary.primitives.shape[0] != 0:
        raise AssertionError("out-of-domain neighbor did not invalidate the primitive")
    result = {
        "experiment": "Verify_ArcLengthResampling_1.1",
        "status": "passed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "main_config_path": str(main_config_path.resolve()),
        "main_config_sha256": sha256_file(main_config_path),
        "verify_config_path": str(verify_config_path.resolve()),
        "verify_config_sha256": sha256_file(verify_config_path),
        "scale_count": len(scales),
        "valid_count": int(first.valid_mask.sum()),
        "primitive_shape": list(first.primitives.shape),
        "scale_id_dtype": str(first.scale_id.dtype),
        "maximum_uniform_segment_length_absolute_error": spatial_error,
        "maximum_target_travel_absolute_error": travel_error,
        "maximum_endpoint_time_absolute_error": time_error,
        "maximum_initial_cross_absolute_error": initial_cross_error,
        "maximum_analytic_xyz_absolute_error": analytic_xyz_error,
        "maximum_analytic_time_absolute_error": analytic_time_error,
        "maximum_truncated_endpoint_xyz_absolute_error": endpoint_xyz_error,
        "scale_anchor_tuples": {
            str(index): [float(value) for value in tuples[index]]
            for index in anchor_expectations
        },
        "external_batch_size": external_batch_size,
        "external_batch_count": int(np.ceil(len(seeds) / external_batch_size)),
        "external_batch_results_exactly_equal": True,
        "input_order_results_exactly_equal": True,
        "zero_velocity_invalid": True,
        "boundary_neighbor_invalid": True,
        "primitive_array_sha256": canonical_array_sha256(first.primitives),
        "valid_mask_sha256": canonical_array_sha256(first.valid_mask),
        "execution": {
            "device": "cpu",
            "hostname": platform.node(),
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "numba": importlib.metadata.version("numba"),
            "numba_threads": int(__import__("numba").get_num_threads()),
            "os_cpu_count": os.cpu_count(),
        },
    }
    _atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-config",
        type=Path,
        default=ROOT / "config/mainExp_TemplateMatching_2.1.yaml",
    )
    parser.add_argument(
        "--verify-config",
        type=Path,
        default=ROOT / "config/Verify_ArcLengthResampling_1.1.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.main_config, arguments.verify_config, arguments.output)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

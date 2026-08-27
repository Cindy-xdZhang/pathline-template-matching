"""Aggregate tables, paired bootstrap, and preregistered development figures."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .development_data import (
    build_input_manifest,
    canonical_json_sha256,
    load_project_specs,
    sha256_file,
)
from .development_experiment import (
    METHOD_FMT,
    METHOD_PCA,
    METHOD_PRIOR,
    METHOD_RAW,
    METHODS,
    METRIC_FIELDS,
    _git_commit,
    _validate_completed_fold,
)
from .ivd import compute_ivd_3d
from .netcdf_io import load_netcdf_window_3d
from .visualization import render_template_matching_triptych


METHOD_LABELS = {
    METHOD_PRIOR: "Library prior",
    METHOD_RAW: "Raw 672D + 1NN",
    METHOD_PCA: "Raw-PCA 161D + 1NN",
    METHOD_FMT: "FMT 161D + 1NN",
}
DATASET_TITLES = {
    "cylinder3d": "Half-cylinder Re160",
    "halfcylinderRe640": "Half-cylinder Re640",
    "halfcylinderRe6400": "Half-cylinder Re6400",
    "tangaroa": "Tangaroa",
    "deltaWing_resampled": "Delta-wing resampled",
    "deltaWing_LBM": "Delta-wing original LBM",
    "f22raptor": "F-22",
    "channel": "Channel observer",
    "boeing747": "Boeing 747",
    "smokeBuoyancy": "Smoke buoyancy",
}
DATASET_VIEWS = {
    "cylinder3d": (22.0, -62.0),
    "halfcylinderRe640": (22.0, -62.0),
    "halfcylinderRe6400": (22.0, -62.0),
    "tangaroa": (23.0, -62.0),
    "deltaWing_resampled": (22.0, -58.0),
    "deltaWing_LBM": (22.0, -58.0),
    "f22raptor": (21.0, -58.0),
    "channel": (22.0, -62.0),
    "boeing747": (21.0, -58.0),
    "smokeBuoyancy": (22.0, -58.0),
}

FLOAT_METRICS = (
    "average_precision",
    "auroc",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale atomic-write file exists: {temporary}")
    _write_json(temporary, value)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    names: list[str] = []
    for row in values:
        for name in row:
            if name not in names:
                names.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=names)
        writer.writeheader()
        writer.writerows(values)


def _concatenate_csv(paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as target:
        wrote_header = False
        expected_header = ""
        for path in paths:
            with path.open("r", encoding="utf-8", newline="") as source:
                header = source.readline()
                if not wrote_header:
                    target.write(header)
                    expected_header = header
                    wrote_header = True
                elif header != expected_header:
                    raise RuntimeError(f"CSV header drift in {path}")
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


def _float(row: dict[str, str], name: str) -> float:
    return float(row[name])


def _require_unique_rows(
    rows: Iterable[dict[str, Any]], fields: tuple[str, ...], label: str
) -> None:
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        if key in seen:
            raise RuntimeError(f"duplicate {label} row: {key}")
        seen.add(key)


def _validate_timeslice_coverage(
    per_timeslice: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    expected: set[tuple[str, str, str, int, str]] = set()
    regime_specs = {
        "seen_scale": config["split"]["library_and_seen_scale_query"],
        "unseen_scale": config["split"]["unseen_scale_query"],
    }
    for family, datasets in config["physical_families"].items():
        for dataset in datasets:
            for regime, spec in regime_specs.items():
                for ordinal in spec["source_ordinals"]:
                    for method in METHODS:
                        expected.add((family, dataset, regime, int(ordinal), method))
    actual: set[tuple[str, str, str, int, str]] = set()
    for row in per_timeslice:
        key = (
            str(row["held_out_family"]),
            str(row["dataset"]),
            str(row["regime"]),
            int(row["source_ordinal"]),
            str(row["method"]),
        )
        if key in actual:
            raise RuntimeError(f"duplicate per-timeslice coverage key: {key}")
        actual.add(key)
        if int(row["positive_count"]) <= 0 or int(row["negative_count"]) <= 0:
            raise RuntimeError(f"single-class main metric timeslice: {key}")
        if not np.isfinite([_float(row, metric) for metric in FLOAT_METRICS]).all():
            raise RuntimeError(f"non-finite main metric timeslice: {key}")
    if actual != expected:
        raise RuntimeError(
            "per-timeslice coverage mismatch: "
            f"missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}"
        )


def _family_timeslice_macro_rows(
    per_timeslice: list[dict[str, str]],
) -> list[dict[str, Any]]:
    _require_unique_rows(
        per_timeslice,
        (
            "held_out_family",
            "dataset",
            "regime",
            "method",
            "source_ordinal",
            "source_start_index",
        ),
        "per-timeslice",
    )
    rows: list[dict[str, Any]] = []
    keys = sorted(
        {
            (row["held_out_family"], row["regime"], row["method"])
            for row in per_timeslice
        }
    )
    for family, regime, method in keys:
        selected = [
            row
            for row in per_timeslice
            if row["held_out_family"] == family
            and row["regime"] == regime
            and row["method"] == method
        ]
        rows.append(
            {
                "held_out_family": family,
                "regime": regime,
                "method": method,
                "within_family_aggregation": "source_timeslice_macro",
                "source_timeslice_count": len(selected),
                "sample_count": int(sum(int(row["sample_count"]) for row in selected)),
                "positive_count": int(sum(int(row["positive_count"]) for row in selected)),
                "negative_count": int(sum(int(row["negative_count"]) for row in selected)),
                **{
                    metric: float(np.mean([_float(row, metric) for row in selected]))
                    for metric in FLOAT_METRICS
                },
            }
        )
    return rows


def _macro_rows(
    per_flow: list[dict[str, str]],
    family_timeslice_macro: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for regime in ("seen_scale", "unseen_scale"):
        for method in METHODS:
            for aggregation, source in (
                ("dataset_macro", per_flow),
                ("physical_family_macro", family_timeslice_macro),
            ):
                selected = [
                    row
                    for row in source
                    if row["regime"] == regime and row["method"] == method
                ]
                expected = 10 if aggregation == "dataset_macro" else 7
                if len(selected) != expected:
                    raise RuntimeError(
                        f"{aggregation}/{regime}/{method}: expected {expected} rows, "
                        f"found {len(selected)}"
                    )
                identity_field = (
                    "dataset" if aggregation == "dataset_macro" else "held_out_family"
                )
                _require_unique_rows(selected, (identity_field,), aggregation)
                rows.append(
                    {
                        "evidence_scope": "development_only_exposed_legacy_task5_cache",
                        "regime": regime,
                        "aggregation": aggregation,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "unit_count": len(selected),
                        "within_unit_aggregation": (
                            "pooled_query_samples"
                            if aggregation == "dataset_macro"
                            else "source_timeslice_macro"
                        ),
                        **{
                            metric: float(np.mean([_float(row, metric) for row in selected]))
                            for metric in FLOAT_METRICS
                        },
                    }
                )
    return rows


def _bootstrap_rows(
    per_timeslice: list[dict[str, str]],
    *,
    seed: int,
    replicates: int,
    percentile_method: str = "linear",
) -> list[dict[str, Any]]:
    comparisons = (METHOD_RAW, METHOD_PCA)
    output: list[dict[str, Any]] = []
    for regime in ("seen_scale", "unseen_scale"):
        regime_rows = [row for row in per_timeslice if row["regime"] == regime]
        families = sorted({row["held_out_family"] for row in regime_rows})
        if len(families) != 7:
            raise RuntimeError(f"bootstrap expected 7 families for {regime}")
        for metric in ("average_precision", "f1"):
            for comparator in comparisons:
                identity = f"{seed}|{regime}|{metric}|{comparator}".encode("utf-8")
                local_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
                rng = np.random.default_rng(np.uint64(local_seed))
                family_pairs: dict[str, np.ndarray] = {}
                for family in families:
                    family_rows = [
                        row for row in regime_rows if row["held_out_family"] == family
                    ]
                    by_method: dict[str, dict[tuple[str, int, int], float]] = {}
                    for method in (METHOD_FMT, comparator):
                        keyed: dict[tuple[str, int, int], float] = {}
                        for row in family_rows:
                            if row["method"] != method:
                                continue
                            key = (
                                row["dataset"],
                                int(row["source_ordinal"]),
                                int(row["source_start_index"]),
                            )
                            if key in keyed:
                                raise RuntimeError(
                                    f"duplicate bootstrap timeslice row: {family}/{method}/{key}"
                                )
                            keyed[key] = _float(row, metric)
                        by_method[method] = keyed
                    if set(by_method[METHOD_FMT]) != set(by_method[comparator]):
                        raise RuntimeError("paired timeslice keys differ across methods")
                    keys = sorted(by_method[METHOD_FMT])
                    family_pairs[family] = np.asarray(
                        [
                            [
                                by_method[METHOD_FMT][key],
                                by_method[comparator][key],
                            ]
                            for key in keys
                        ],
                        dtype=np.float64,
                    )
                draws = np.empty(replicates, dtype=np.float64)
                for replicate in range(replicates):
                    family_differences = []
                    for family in families:
                        pair = family_pairs[family]
                        indices = rng.integers(0, len(pair), size=len(pair))
                        sample = pair[indices]
                        family_differences.append(
                            float(sample[:, 0].mean() - sample[:, 1].mean())
                        )
                    draws[replicate] = float(np.mean(family_differences))
                point = float(
                    np.mean(
                        [
                            pair[:, 0].mean() - pair[:, 1].mean()
                            for pair in family_pairs.values()
                        ]
                    )
                )
                output.append(
                    {
                        "evidence_scope": "development_only_exposed_legacy_task5_cache",
                        "regime": regime,
                        "aggregation": "physical_family_macro_of_timeslice_metrics",
                        "metric": metric,
                        "method": METHOD_FMT,
                        "comparator": comparator,
                        "difference": "FMT_minus_comparator",
                        "point_estimate": point,
                        "ci95_lower": float(
                            np.percentile(draws, 2.5, method=percentile_method)
                        ),
                        "ci95_upper": float(
                            np.percentile(draws, 97.5, method=percentile_method)
                        ),
                        "replicates": int(replicates),
                        "bootstrap_seed": int(seed),
                        "derived_seed": int(local_seed),
                        "paired_unit": "source_timeslice",
                        "stratified_by": "physical_family",
                        "interval": "percentile_95",
                        "numpy_percentile_method": percentile_method,
                        "conclusion_mode": "descriptive_only_pending_user_ci_decision",
                    }
                )
    return output


def _counterexample_rows(
    per_flow: list[dict[str, str]], per_scale: list[dict[str, str]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scope, source, identity_fields in (
        ("flow", per_flow, ("held_out_family", "dataset", "regime")),
        (
            "scale_tuple",
            per_scale,
            ("held_out_family", "dataset", "regime", "scale_name"),
        ),
    ):
        groups: dict[tuple[str, ...], dict[str, dict[str, str]]] = {}
        for row in source:
            key = tuple(row[field] for field in identity_fields)
            groups.setdefault(key, {})[row["method"]] = row
        for key, methods in sorted(groups.items()):
            if any(method not in methods for method in (METHOD_FMT, METHOD_RAW, METHOD_PCA)):
                raise RuntimeError(f"method missing in counterexample group {key}")
            for comparator in (METHOD_RAW, METHOD_PCA):
                for metric in ("average_precision", "f1"):
                    difference = _float(methods[METHOD_FMT], metric) - _float(
                        methods[comparator], metric
                    )
                    if difference <= 0.0:
                        output.append(
                            {
                                "scope": scope,
                                **dict(zip(identity_fields, key)),
                                "metric": metric,
                                "comparator": comparator,
                                "fmt_value": _float(methods[METHOD_FMT], metric),
                                "comparator_value": _float(
                                    methods[comparator], metric
                                ),
                                "fmt_minus_comparator": difference,
                            }
                        )
    return output


def _maximin_indices(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if int(count) <= 0:
        return np.empty(0, dtype=np.int64)
    if count >= len(values):
        return np.arange(len(values), dtype=np.int64)
    rng = np.random.default_rng(np.uint64(seed))
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(rng.integers(0, len(values)))
    minimum = np.sum((values - values[selected[0]]) ** 2, axis=1)
    minimum[selected[0]] = -1.0
    for position in range(1, count):
        selected[position] = int(np.argmax(minimum))
        distance = np.sum((values - values[selected[position]]) ** 2, axis=1)
        minimum = np.minimum(minimum, distance)
        minimum[selected[: position + 1]] = -1.0
    return selected


def _stratified_pathline_indices(
    seeds: np.ndarray,
    reference: np.ndarray,
    scale_id: np.ndarray,
    *,
    count: int,
    base_seed: int,
    identity: str,
) -> np.ndarray:
    count = int(count)
    if len(seeds) < count:
        raise RuntimeError(
            f"requested {count} display pathlines but only {len(seeds)} seeds exist"
        )
    lower = seeds.min(axis=0)
    span = np.maximum(seeds.max(axis=0) - lower, 1e-12)
    normalized = (seeds - lower) / span
    strata = [
        (int(scale), int(label), np.flatnonzero((scale_id == scale) & (reference == bool(label))))
        for scale in sorted(np.unique(scale_id))
        for label in (0, 1)
    ]
    strata = [item for item in strata if len(item[2])]
    if not strata:
        raise RuntimeError("pathline display selection found no non-empty stratum")
    base = count // len(strata)
    remainder = count % len(strata)
    chosen: list[int] = []
    for position, (scale, label, candidates) in enumerate(strata):
        target = min(len(candidates), base + int(position < remainder))
        payload = f"{base_seed}|{identity}|{scale}|{label}".encode("utf-8")
        local_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        local = _maximin_indices(normalized[candidates], target, local_seed)
        chosen.extend(candidates[local].tolist())
    if len(chosen) < count:
        remaining = np.setdiff1d(
            np.arange(len(seeds), dtype=np.int64), np.asarray(chosen, dtype=np.int64)
        )
        payload = f"{base_seed}|{identity}|fallback".encode("utf-8")
        fallback_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        extra = _maximin_indices(normalized[remaining], count - len(chosen), fallback_seed)
        chosen.extend(remaining[extra].tolist())
    selected = np.asarray(chosen[:count], dtype=np.int64)
    if len(np.unique(selected)) != len(selected):
        raise AssertionError("display pathline selection contains duplicate seeds")
    return selected


def _cache_pathlines(
    cache_path: Path,
    *,
    reference: np.ndarray,
    scale_id: np.ndarray,
    seeds: np.ndarray,
    count: int,
    seed: int,
    identity: str,
) -> tuple[list[np.ndarray], np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as data:
        raw = np.asarray(data["raw_features"], dtype=np.float32)
        physical_dt = np.asarray(data["physical_dt"], dtype=np.float32)
        integration_steps = np.asarray(data["integration_steps"], dtype=np.int64)
        cache_seeds = np.asarray(data["seeds"], dtype=np.float32)
        cache_reference = np.asarray(data["reference"], dtype=bool)
        cache_scale = np.asarray(data["scale_id"], dtype=np.int16)
    if not (
        np.array_equal(cache_seeds, seeds)
        and np.array_equal(cache_reference, reference)
        and np.array_equal(cache_scale, scale_id)
    ):
        raise RuntimeError("prediction artifact and source cache seed order disagree")
    primitives = raw.reshape(-1, 7, 32, 3) + seeds[:, None, None, :]
    selected = _stratified_pathline_indices(
        seeds,
        reference,
        scale_id,
        count=count,
        base_seed=seed,
        identity=identity,
    )
    pathlines: list[np.ndarray] = []
    for index in selected:
        steps = int(integration_steps[index])
        sample_indices = np.rint(np.linspace(0, steps, 32)).astype(np.int64)
        time = sample_indices.astype(np.float32) * physical_dt[index]
        pathlines.append(
            np.column_stack((primitives[index, 0], time)).astype(np.float32)
        )
    return pathlines, selected


def _raw_path(registry: dict[str, Any], dataset: str, environment: str) -> Path | None:
    item = next(row for row in registry["datasets"] if str(row["id"]) == dataset)
    if str(item.get("kind", "netcdf")) != "netcdf":
        return None
    for value in item.get("raw_paths", {}).get(environment, []):
        path = Path(value)
        if path.is_file():
            return path
    return None


def _sample_regular_volume(
    volume_zyx: np.ndarray,
    coordinates_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    points_xyz: np.ndarray,
) -> np.ndarray:
    """Trilinearly sample a uniform ZYX volume at physical XYZ points."""

    volume = np.asarray(volume_zyx, dtype=np.float64)
    points = np.asarray(points_xyz, dtype=np.float64)
    x, y, z = (np.asarray(axis, dtype=np.float64) for axis in coordinates_xyz)
    fractional = np.column_stack(
        (
            (points[:, 0] - x[0]) / (x[1] - x[0]),
            (points[:, 1] - y[0]) / (y[1] - y[0]),
            (points[:, 2] - z[0]) / (z[1] - z[0]),
        )
    )
    upper = np.asarray([len(x) - 1, len(y) - 1, len(z) - 1], dtype=np.float64)
    if np.any(fractional < -1e-5) or np.any(fractional > upper + 1e-5):
        raise RuntimeError("visualization seeds fall outside the reconstructed raw field")
    fractional = np.clip(fractional, 0.0, upper)
    lower_index = np.floor(fractional).astype(np.int64)
    upper_index = np.minimum(lower_index + 1, upper.astype(np.int64))
    weight = fractional - lower_index
    x0, y0, z0 = lower_index.T
    x1, y1, z1 = upper_index.T
    wx, wy, wz = weight.T
    return (
        volume[z0, y0, x0] * (1 - wx) * (1 - wy) * (1 - wz)
        + volume[z0, y0, x1] * wx * (1 - wy) * (1 - wz)
        + volume[z0, y1, x0] * (1 - wx) * wy * (1 - wz)
        + volume[z0, y1, x1] * wx * wy * (1 - wz)
        + volume[z1, y0, x0] * (1 - wx) * (1 - wy) * wz
        + volume[z1, y0, x1] * wx * (1 - wy) * wz
        + volume[z1, y1, x0] * (1 - wx) * wy * wz
        + volume[z1, y1, x1] * wx * wy * wz
    ).astype(np.float32)


def _ivd_mesh_and_bounds(
    registry: dict[str, Any],
    *,
    dataset: str,
    environment: str,
    source_start_index: int,
    expected_spatial_shape: tuple[int, int, int],
    cached_ivd_threshold: float,
    seeds: np.ndarray,
    reference: np.ndarray,
    raw_sha256_cache: dict[str, str] | None = None,
) -> tuple[dict[str, np.ndarray | float] | None, np.ndarray | None, dict[str, Any]]:
    path = _raw_path(registry, dataset, environment)
    if path is None:
        return None, None, {
            "available": False,
            "reason": f"no supported NetCDF raw path for environment {environment}",
        }
    window = load_netcdf_window_3d(
        path, start_index=int(source_start_index), frame_count=1, max_spatial_dim=96
    )
    if tuple(window.velocity.shape[1:4]) != tuple(expected_spatial_shape):
        raise RuntimeError(
            f"{dataset}: raw IVD spatial shape {window.velocity.shape[1:4]} != "
            f"cache metadata {expected_spatial_shape}"
        )
    ivd = compute_ivd_3d(window.velocity[0], window.spacing_xyz)
    reconstructed_threshold = float(np.percentile(ivd, 95.0))
    if not np.isclose(
        reconstructed_threshold,
        float(cached_ivd_threshold),
        rtol=2e-4,
        atol=2e-5,
    ):
        raise RuntimeError(
            f"{dataset}: reconstructed IVD p95 {reconstructed_threshold} does not "
            f"match cached threshold {cached_ivd_threshold}"
        )
    sampled_ivd = _sample_regular_volume(ivd, window.coordinates_xyz, seeds)
    reconstructed_reference = sampled_ivd >= float(cached_ivd_threshold)
    mismatch_count = int(np.sum(reconstructed_reference != np.asarray(reference, dtype=bool)))
    if mismatch_count:
        raise RuntimeError(
            f"{dataset}: reconstructed raw IVD labels disagree with cache for "
            f"{mismatch_count}/{len(reference)} seeds"
        )
    from skimage import measure

    dx, dy, dz = (float(value) for value in window.spacing_xyz)
    vertices_zyx, faces, _, _ = measure.marching_cubes(
        ivd,
        level=float(cached_ivd_threshold),
        spacing=(dz, dy, dx),
    )
    x, y, z = window.coordinates_xyz
    vertices = np.empty_like(vertices_zyx, dtype=np.float32)
    vertices[:, 0] = float(x[0]) + vertices_zyx[:, 2]
    vertices[:, 1] = float(y[0]) + vertices_zyx[:, 1]
    vertices[:, 2] = float(z[0]) + vertices_zyx[:, 0]
    mesh = {
        "vertices": vertices,
        "faces": np.asarray(faces, dtype=np.int64),
        "level": float(cached_ivd_threshold),
    }
    bounds = np.asarray(
        [[x[0], y[0], z[0]], [x[-1], y[-1], z[-1]]], dtype=np.float64
    )
    difference = reconstructed_threshold - float(cached_ivd_threshold)
    resolved_path = str(path.resolve())
    if raw_sha256_cache is None:
        raw_digest = sha256_file(path)
    else:
        if resolved_path not in raw_sha256_cache:
            raw_sha256_cache[resolved_path] = sha256_file(path)
        raw_digest = raw_sha256_cache[resolved_path]
    return mesh, bounds, {
        "available": True,
        "path": resolved_path,
        "path_sha256": raw_digest,
        "source_start_index": int(source_start_index),
        "spatial_shape_zyx": list(ivd.shape),
        "cached_ivd_p95_threshold": float(cached_ivd_threshold),
        "reconstructed_ivd_p95_threshold": reconstructed_threshold,
        "threshold_difference": difference,
        "threshold_relative_difference": float(
            difference / max(abs(float(cached_ivd_threshold)), 1e-12)
        ),
        "ivd_finite": bool(np.isfinite(ivd).all()),
        "seed_label_mismatch_count": mismatch_count,
        "seed_label_count": int(len(reference)),
        "isosurface_vertex_count": int(len(vertices)),
        "isosurface_face_count": int(len(faces)),
    }


def _save_scene_artifact(
    path: Path,
    *,
    scene: dict[str, Any],
    selected_indices: np.ndarray,
    source_metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pathlines = np.stack(scene["display_pathlines"], axis=0)
    mesh = scene.get("ivd_mesh")
    mesh_vertices = (
        np.empty((0, 3), dtype=np.float32)
        if mesh is None
        else np.asarray(mesh["vertices"], dtype=np.float32)
    )
    mesh_faces = (
        np.empty((0, 3), dtype=np.int64)
        if mesh is None
        else np.asarray(mesh["faces"], dtype=np.int64)
    )
    mesh_level = np.asarray(
        np.nan if mesh is None else float(mesh["level"]), dtype=np.float64
    )
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int16),
        dataset=np.asarray(scene["dataset"]),
        title=np.asarray(scene["title"]),
        regime=np.asarray(scene["regime"]),
        source_ordinal=np.asarray(scene["source_ordinal"], dtype=np.int16),
        bounds=np.asarray(scene["bounds"], dtype=np.float64),
        seeds=np.asarray(scene["seeds"], dtype=np.float32),
        reference=np.asarray(scene["reference"], dtype=bool),
        prediction=np.asarray(scene["prediction"], dtype=bool),
        display_pathlines=pathlines.astype(np.float32),
        selected_pathline_seed_indices=np.asarray(selected_indices, dtype=np.int64),
        ivd_mesh_vertices=mesh_vertices,
        ivd_mesh_faces=mesh_faces,
        ivd_mesh_level=mesh_level,
        source_metadata_json=np.asarray(json.dumps(source_metadata, sort_keys=True)),
    )


def _render_triptychs(
    development_config_path: str | Path,
    run_dir: Path,
    *,
    environment: str,
    dpi: int,
) -> list[dict[str, Any]]:
    _, config, base_config, registry, _ = load_project_specs(development_config_path)
    visual = config["visualization"]
    ordinal = int(visual["source_ordinal"])
    pathline_count = int(visual["display_center_pathlines"]["count"])
    selection_seed = int(visual["display_center_pathlines"]["seed"])
    artifacts: list[dict[str, Any]] = []
    for family in config["physical_families"]:
        index_path = run_dir / "folds" / family / "prediction_artifacts.json"
        artifacts.extend(json.loads(index_path.read_text(encoding="utf-8")))
    selected_artifacts = [item for item in artifacts if int(item["source_ordinal"]) == ordinal]
    expected_keys = {
        (dataset, regime)
        for datasets in config["physical_families"].values()
        for dataset in datasets
        for regime in visual["regimes"]
    }
    actual_keys = {(item["dataset"], item["regime"]) for item in selected_artifacts}
    if len(selected_artifacts) != len(expected_keys) or actual_keys != expected_keys:
        raise RuntimeError(
            "visualization artifact coverage mismatch: "
            f"missing={sorted(expected_keys-actual_keys)}, "
            f"unexpected={sorted(actual_keys-expected_keys)}, "
            f"duplicates={len(selected_artifacts)-len(actual_keys)}"
        )
    output: list[dict[str, Any]] = []
    raw_sha256_cache: dict[str, str] = {}
    for item in sorted(selected_artifacts, key=lambda row: (row["dataset"], row["regime"])):
        artifact_path = Path(item["path"])
        if not artifact_path.is_file():
            raise RuntimeError(f"prediction artifact is missing: {artifact_path}")
        if int(item.get("size_bytes", -1)) != artifact_path.stat().st_size:
            raise RuntimeError(f"prediction artifact size changed: {artifact_path}")
        if str(item.get("sha256")) != sha256_file(artifact_path):
            raise RuntimeError(f"prediction artifact digest changed: {artifact_path}")
        with np.load(artifact_path, allow_pickle=False) as data:
            dataset = str(data["dataset"])
            regime = str(data["regime"])
            seeds = np.asarray(data["seeds"], dtype=np.float32)
            reference = np.asarray(data["reference"], dtype=bool)
            prediction = np.asarray(data["fmt161_prediction"], dtype=bool)
            scale_id = np.asarray(data["scale_id"], dtype=np.int16)
            canonical_scale_names = tuple(
                json.loads(str(data["canonical_scale_names_json"]))
            )
            metadata = json.loads(str(data["metadata_json"]))
            cache_path = Path(str(data["cache_path"]))
            cache_sha256 = str(data["cache_sha256"])
            source_ordinal = int(data["source_ordinal"])
        if (
            dataset != item["dataset"]
            or regime != item["regime"]
            or source_ordinal != int(item["source_ordinal"])
            or str(cache_path.resolve()) != str(Path(item["cache_path"]).resolve())
            or cache_sha256 != item["cache_sha256"]
        ):
            raise RuntimeError(f"prediction artifact index/content mismatch: {artifact_path}")
        expected_role = "library" if regime == "seen_scale" else "unseen_scale_evaluation"
        expected_scale_names = {
            str(scale["name"]) for scale in base_config["scale_sets"][expected_role]
        }
        if set(canonical_scale_names) != expected_scale_names or set(
            np.unique(scale_id).tolist()
        ) != set(range(len(canonical_scale_names))):
            raise RuntimeError(
                f"{dataset}/{regime}: visualization does not cover every frozen scale tuple"
            )
        if sha256_file(cache_path) != cache_sha256:
            raise RuntimeError(f"visualization source cache digest changed: {cache_path}")
        identity = f"{dataset}|{regime}|{ordinal}"
        pathlines, selected_indices = _cache_pathlines(
            cache_path,
            reference=reference,
            scale_id=scale_id,
            seeds=seeds,
            count=pathline_count,
            seed=selection_seed,
            identity=identity,
        )
        ivd_mesh, raw_bounds, raw_audit = _ivd_mesh_and_bounds(
            registry,
            dataset=dataset,
            environment=environment,
            source_start_index=int(metadata["source_start_index"]),
            expected_spatial_shape=tuple(metadata["loaded_shape_TZYXC"][1:4]),
            cached_ivd_threshold=float(metadata["ivd_threshold"]),
            seeds=seeds,
            reference=reference,
            raw_sha256_cache=raw_sha256_cache,
        )
        all_path_points = np.concatenate([path[:, :3] for path in pathlines], axis=0)
        if raw_bounds is None:
            combined = np.concatenate((seeds, all_path_points), axis=0)
            lower = combined.min(axis=0)
            upper = combined.max(axis=0)
            span = np.maximum(upper - lower, 1e-6)
            bounds = np.stack((lower - 0.018 * span, upper + 0.018 * span))
        else:
            bounds = raw_bounds
        scene = {
            "dataset": dataset,
            "title": DATASET_TITLES[dataset],
            "regime": regime,
            "source_ordinal": ordinal,
            "bounds": bounds,
            "seeds": seeds,
            "reference": reference,
            "prediction": prediction,
            "display_pathlines": pathlines,
            "ivd_points": None,
            "ivd_mesh": ivd_mesh,
        }
        scene_path = run_dir / "figures" / "scenes" / f"{dataset}_{regime}_scene.npz"
        _save_scene_artifact(
            scene_path,
            scene=scene,
            selected_indices=selected_indices,
            source_metadata=metadata,
        )
        image_path = run_dir / "figures" / f"{dataset}_{regime}_triptych.png"
        _, render_metadata = render_template_matching_triptych(
            scene,
            image_path,
            view=DATASET_VIEWS[dataset],
            dpi=int(dpi),
        )
        render_metadata.update(
            {
                "evidence_scope": visual["evidence_scope_label"],
                "metric_based_selection": visual["metric_based_selection"],
                "all_scale_tuples_included": True,
                "canonical_scale_names": list(canonical_scale_names),
                "source_cache_path": str(cache_path),
                "source_cache_sha256": cache_sha256,
                "scene_artifact": str(scene_path.resolve()),
                "scene_artifact_sha256": sha256_file(scene_path),
                "selected_pathline_seed_indices_sha256": hashlib.sha256(
                    np.asarray(selected_indices, dtype="<i8").tobytes()
                ).hexdigest(),
                "display_pathline_selection": {
                    "method": "deterministic_stratified_maximin",
                    "strata": ["scale_tuple", "reference_class"],
                    "seed": selection_seed,
                    "count": len(pathlines),
                    "used_only_for_explanatory_display": True,
                },
                "raw_ivd_audit": raw_audit,
                "image_sha256": sha256_file(image_path),
            }
        )
        output.append(render_metadata)
        print(f"[figure] {dataset}/{regime}: {image_path}", flush=True)
    return output


def _performance_figure(main_rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharey="row")
    colors = ("#8c8c8c", "#4c78a8", "#72b7b2", "#d62728")
    for column, regime in enumerate(("seen_scale", "unseen_scale")):
        for row_index, metric in enumerate(("average_precision", "f1")):
            axis = axes[row_index, column]
            values = []
            for method in METHODS:
                match = next(
                    row
                    for row in main_rows
                    if row["regime"] == regime
                    and row["aggregation"] == "physical_family_macro"
                    and row["method"] == method
                )
                values.append(float(match[metric]))
            axis.bar(np.arange(4), values, color=colors, width=0.72)
            axis.set_xticks(np.arange(4), ["Prior", "Raw\n672D", "Raw-PCA\n161D", "FMT\n161D"])
            axis.set_ylim(0.0, 1.0)
            axis.grid(axis="y", alpha=0.18)
            axis.set_ylabel("Average Precision" if metric == "average_precision" else "F1")
            if row_index == 0:
                axis.set_title("Seen scale" if regime == "seen_scale" else "Unseen scale")
            for index, value in enumerate(values):
                axis.text(index, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(
        "Development-only physical-family macro results\n"
        "exposed legacy Task5 caches; not sealed confirmation",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def _delta_heatmap(per_flow: list[dict[str, str]], path: Path) -> None:
    datasets = sorted({row["dataset"] for row in per_flow})
    columns = [
        ("seen_scale", "average_precision"),
        ("seen_scale", "f1"),
        ("unseen_scale", "average_precision"),
        ("unseen_scale", "f1"),
    ]
    values = np.empty((len(datasets), len(columns)), dtype=np.float64)
    for row_index, dataset in enumerate(datasets):
        for column_index, (regime, metric) in enumerate(columns):
            rows = {
                row["method"]: row
                for row in per_flow
                if row["dataset"] == dataset and row["regime"] == regime
            }
            values[row_index, column_index] = _float(rows[METHOD_FMT], metric) - _float(
                rows[METHOD_PCA], metric
            )
    limit = max(float(np.max(np.abs(values))), 0.01)
    fig, axis = plt.subplots(figsize=(9.5, 6.8))
    image = axis.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_yticks(np.arange(len(datasets)), [DATASET_TITLES[item] for item in datasets])
    axis.set_xticks(
        np.arange(4),
        ["Seen AP", "Seen F1", "Unseen AP", "Unseen F1"],
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:+.3f}", ha="center", va="center", fontsize=8)
    axis.set_title("FMT 161D minus Raw-PCA 161D by held-out flow (development only)")
    fig.colorbar(image, ax=axis, label="metric difference")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def _markdown_table(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> str:
    header = ["Regime", "Aggregation", "Method", *metrics]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        values = [
            str(row["regime"]),
            str(row["aggregation"]),
            str(row["method_label"]),
            *[f"{float(row[metric]):.4f}" for metric in metrics],
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report_markdown(
    path: Path,
    *,
    main_rows: list[dict[str, Any]],
    bootstrap: list[dict[str, Any]],
    counterexamples: list[dict[str, Any]],
    input_manifest: dict[str, Any],
    figure_manifest: list[dict[str, Any]],
    fold_evidence: list[dict[str, Any]],
) -> None:
    bootstrap_lines = [
        "| Regime | Metric | Comparator | FMT−Comparator | 95% CI |",
        "|---|---|---|---:|---:|",
    ]
    for row in bootstrap:
        bootstrap_lines.append(
            f"| {row['regime']} | {row['metric']} | {METHOD_LABELS[row['comparator']]} | "
            f"{row['point_estimate']:+.4f} | [{row['ci95_lower']:+.4f}, {row['ci95_upper']:+.4f}] |"
        )
    library_lines = [
        "| Held-out family | Eligible candidates | Balanced templates | Skipped empty-class strata | Candidates in skipped strata | Pre-balance positive prior |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in fold_evidence:
        library_lines.append(
            f"| {row['held_out_family']} | {row['eligible_library_candidate_count']:,} | "
            f"{row['balanced_library_count']:,} | {row['skipped_library_stratum_count']:,} | "
            f"{row['skipped_library_candidate_count']:,} | "
            f"{row['eligible_library_prior_positive_fraction']:.6f} |"
        )
    raw_count = sum(not item["fallback"]["used"] for item in figure_manifest)
    text = f"""# {input_manifest['experiment']} cache-backed development report

状态：`DEVELOPMENT_COMPLETED_CONFIRMATION_NOT_RUN`。这些数值只来自已经暴露的旧 FMT Task5 cache；历史目录名 `confirmation` 在本项目中只是 unseen-scale development evidence，不是 sealed confirmation。

Git commit：`{input_manifest['git_commit']}`

Development config SHA-256：`{input_manifest['development_config_sha256']}`

Input manifest SHA-256：`{input_manifest['manifest_content_sha256']}`
Cache：{input_manifest['cache_file_count']} files，{input_manifest['cache_total_samples']:,} primitives。

## Library construction audit

{chr(10).join(library_lines)}

`mainExp_TemplateMatching_1.2` 对每个 library flow×source-time×scale stratum 独立平衡。若某一类别为空，则两类都选0个 template，并保留该 stratum 的候选数；pre-balance prior 和 Raw-PCA 拟合仍使用全部合格 library candidates。表中的 skip 不会删除 query。

## Main table

{_markdown_table(main_rows, ('average_precision', 'f1', 'auroc', 'precision', 'recall', 'balanced_accuracy'))}

`dataset_macro` 是10个逐-flow pooled-query 指标的算术平均；`physical_family_macro` 是先在每个 family 内对 source-timeslice 指标宏平均，再对7个 family 宏平均。Seen-scale 和 unseen-scale 始终分开。

## Paired source-timeslice bootstrap

{chr(10).join(bootstrap_lines)}

置信区间采用5000次、physical-family-stratified、paired source-timeslice bootstrap。用户尚未冻结“差值置信区间下界必须大于0”的通过规则，因此本表只作描述，不宣告主命题通过或失败。

## Counterexamples

`counterexamples.csv` 保留所有 flow/scale tuple 中 FMT 的 AP 或 F1 不高于 Raw672/Raw-PCA161 的条目，共 {len(counterexamples)} 条；没有筛掉负结果。

## Figures

固定 source ordinal 2、每个 flow×regime 共 {len(figure_manifest)} 张三联图，不按指标选图。三栏依次为 whole-loaded-volume IVD-p95 等值面+240条缓存中心 pathlines、FMT exact-1NN 类别分配、FMT TP/FP/FN/TN；三栏都显示同一组 evaluated seed coordinates。Ibex 本轮有 {raw_count}/{len(figure_manifest)} 张可读取 raw NetCDF 并审计、重建 IVD-p95 等值面；其余图明确标记 positive-seed reference fallback。每个 scene 都单独保存，可在有 raw field 的机器上复绘。

`figures/method_comparison.png` 给出四方法的 physical-family macro AP/F1；`figures/fmt_minus_raw_pca_by_flow.png` 给出逐 flow 的 FMT−Raw-PCA 差值。图只解释结果，机器可读 CSV 才是统计证据。
"""
    path.write_text(text, encoding="utf-8")


def finalize_development_run(
    development_config_path: str | Path,
    run_dir: str | Path,
    *,
    render_environment: str = "ibex",
    figure_dpi: int = 360,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    result_manifest_path = run_path / "result_manifest.json"
    if result_manifest_path.exists():
        raise FileExistsError(
            f"completed report is immutable; refusing to overwrite {result_manifest_path}"
        )
    project_root, config, _, _, _ = load_project_specs(development_config_path)
    input_manifest = json.loads(
        (run_path / "input_manifest.json").read_text(encoding="utf-8")
    )
    config_path = Path(development_config_path).resolve()
    frozen_paths = {
        "development_config_sha256": config_path,
        "base_config_sha256": project_root / str(config["base_config"]),
        "dataset_registry_sha256": project_root / str(config["dataset_registry"]),
    }
    for key, path in frozen_paths.items():
        current = sha256_file(path)
        if current != input_manifest.get(key):
            raise RuntimeError(f"{key} changed after run preparation; refusing finalize")
    current_input_manifest, _ = build_input_manifest(
        development_config_path,
        environment=str(input_manifest["environment"]),
    )
    stored_files = {
        item["path"]: (item["sha256"], int(item["size_bytes"]))
        for item in input_manifest["files"]
    }
    current_files = {
        item["path"]: (item["sha256"], int(item["size_bytes"]))
        for item in current_input_manifest["files"]
    }
    if current_files != stored_files:
        raise RuntimeError("input cache files changed after computation; refusing finalize")
    if input_manifest.get("git_commit") != "synthetic-smoke" and _git_commit(
        project_root
    ) != input_manifest.get("git_commit"):
        raise RuntimeError("Git commit changed after run preparation; refusing finalize")
    manifest_digest = canonical_json_sha256(
        {
            key: value
            for key, value in input_manifest.items()
            if key != "manifest_content_sha256"
        }
    )
    if manifest_digest != input_manifest.get("manifest_content_sha256"):
        raise RuntimeError("input_manifest.json content digest is invalid")
    fold_dirs = [run_path / "folds" / family for family in config["physical_families"]]
    fold_evidence: list[dict[str, Any]] = []
    for fold_dir in fold_dirs:
        success = _validate_completed_fold(
            fold_dir,
            input_manifest_sha256=str(input_manifest["manifest_content_sha256"]),
        )
        manifest_path = fold_dir / "fold_manifest.json"
        fold_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fold_evidence.append(
            {
                "held_out_family": fold_dir.name,
                "fold_manifest_path": str(manifest_path.resolve()),
                "fold_manifest_file_sha256": sha256_file(manifest_path),
                "fold_manifest_content_sha256": success[
                    "fold_manifest_content_sha256"
                ],
                "success_path": str((fold_dir / "_SUCCESS.json").resolve()),
                "success_file_sha256": sha256_file(fold_dir / "_SUCCESS.json"),
                "output_file_count": int(success["output_file_count"]),
                "eligible_library_candidate_count": int(
                    fold_manifest["eligible_library_candidate_count"]
                ),
                "balanced_library_count": int(
                    fold_manifest["balanced_library_count"]
                ),
                "skipped_library_stratum_count": int(
                    fold_manifest["skipped_library_stratum_count"]
                ),
                "skipped_library_candidate_count": int(
                    fold_manifest["skipped_library_candidate_count"]
                ),
                "eligible_library_prior_positive_fraction": float(
                    fold_manifest["eligible_library_prior_positive_fraction"]
                ),
            }
        )
    for filename in (
        "audit_counts.csv",
        "per_query.csv",
        "per_timeslice.csv",
        "per_flow.csv",
        "per_family.csv",
        "per_scale_tuple.csv",
    ):
        _concatenate_csv([directory / filename for directory in fold_dirs], run_path / filename)
    per_timeslice = _read_csv(run_path / "per_timeslice.csv")
    _validate_timeslice_coverage(per_timeslice, config)
    per_flow = _read_csv(run_path / "per_flow.csv")
    pooled_family = _read_csv(run_path / "per_family.csv")
    family_timeslice_macro = _family_timeslice_macro_rows(per_timeslice)
    _write_csv(run_path / "per_family.csv", [*pooled_family, *family_timeslice_macro])
    per_scale = _read_csv(run_path / "per_scale_tuple.csv")
    main_rows = _macro_rows(per_flow, family_timeslice_macro)
    bootstrap = _bootstrap_rows(
        per_timeslice,
        seed=int(config["bootstrap"]["seed"]),
        replicates=int(config["bootstrap"]["replicates"]),
        percentile_method=str(config["bootstrap"]["numpy_percentile_method"]),
    )
    counterexamples = _counterexample_rows(per_flow, per_scale)
    _write_csv(run_path / "main_table.csv", main_rows)
    _write_csv(run_path / "bootstrap_differences.csv", bootstrap)
    _write_csv(run_path / "counterexamples.csv", counterexamples)
    triptychs = _render_triptychs(
        development_config_path,
        run_path,
        environment=render_environment,
        dpi=int(figure_dpi),
    )
    _write_json(
        run_path / "figures" / "visualization_manifest.json",
        {
            "schema_version": 1,
            "experiment": config["experiment"],
            "phase": config["phase"],
            "evidence_scope": config["evidence_scope"],
            "selection": config["visualization"],
            "render_environment": render_environment,
            "figures": triptychs,
        },
    )
    _performance_figure(main_rows, run_path / "figures" / "method_comparison.png")
    _delta_heatmap(per_flow, run_path / "figures" / "fmt_minus_raw_pca_by_flow.png")
    # The Markdown main table is also written independently for direct inclusion.
    (run_path / "main_table.md").write_text(
        _markdown_table(
            main_rows,
            (
                "average_precision",
                "f1",
                "auroc",
                "precision",
                "recall",
                "balanced_accuracy",
            ),
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report_markdown(
        run_path / "development_report.md",
        main_rows=main_rows,
        bootstrap=bootstrap,
        counterexamples=counterexamples,
        input_manifest=input_manifest,
        figure_manifest=triptychs,
        fold_evidence=fold_evidence,
    )
    required_output_audit = []
    for relative in config["required_outputs"]:
        if str(relative) == "result_manifest.json":
            continue
        output_path = run_path / str(relative)
        if not output_path.is_file():
            raise RuntimeError(f"required output was not created: {relative}")
        required_output_audit.append(
            {
                "relative_path": str(relative),
                "size_bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
            }
        )
    result_manifest = {
        "schema_version": 1,
        "status": "development_completed_confirmation_not_run",
        "experiment": config["experiment"],
        "phase": config["phase"],
        "evidence_scope": config["evidence_scope"],
        "conclusion_mode": config["bootstrap"]["conclusion_mode"],
        "input_manifest_sha256": input_manifest["manifest_content_sha256"],
        "git_commit": input_manifest["git_commit"],
        "completed_utc": _utc_now(),
        "fold_count": len(fold_dirs),
        "method_ids": list(METHODS),
        "triptych_count": len(triptychs),
        "counterexample_count": len(counterexamples),
        "folds": fold_evidence,
        "visualization_artifacts": [
            {
                "dataset": item["dataset"],
                "regime": item["regime"],
                "image_path": item["image"],
                "image_sha256": item["image_sha256"],
                "scene_artifact": item["scene_artifact"],
                "scene_artifact_sha256": item["scene_artifact_sha256"],
            }
            for item in triptychs
        ],
        "required_output_audit_before_completion_marker": required_output_audit,
        "outputs": {
            name: {
                "path": str((run_path / name).resolve()),
                "sha256": sha256_file(run_path / name),
            }
            for name in (
                "input_manifest.json",
                "audit_counts.csv",
                "per_query.csv",
                "per_timeslice.csv",
                "per_flow.csv",
                "per_family.csv",
                "per_scale_tuple.csv",
                "bootstrap_differences.csv",
                "counterexamples.csv",
                "main_table.csv",
                "main_table.md",
                "development_report.md",
                "figures/visualization_manifest.json",
                "figures/method_comparison.png",
                "figures/fmt_minus_raw_pca_by_flow.png",
            )
        },
    }
    result_manifest["manifest_content_sha256"] = hashlib.sha256(
        json.dumps(result_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_json_atomic(result_manifest_path, result_manifest)
    result_manifest_file_sha256 = sha256_file(result_manifest_path)
    required_output_audit.append(
        {
            "relative_path": "result_manifest.json",
            "size_bytes": result_manifest_path.stat().st_size,
            "sha256": result_manifest_file_sha256,
        }
    )
    _write_json_atomic(
        run_path / "run_state.json",
        {
            "status": "development_completed_confirmation_not_run",
            "result_manifest": str(result_manifest_path.resolve()),
            "result_manifest_content_sha256": result_manifest[
                "manifest_content_sha256"
            ],
            "result_manifest_file_sha256": result_manifest_file_sha256,
            "required_output_audit": required_output_audit,
            "evidence_scope": config["evidence_scope"],
            "completed_utc": result_manifest["completed_utc"],
        },
    )
    return result_manifest

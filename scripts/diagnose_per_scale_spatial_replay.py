#!/usr/bin/env python3
"""Label-free portability diagnostic for PerScale spatial-score replay.

The diagnostic opens only the published prediction NPZ and its manifest.  It
never opens an outer reference, label, metric, or result file, and it writes no
artifact.  Its JSON stdout quantifies bitwise replay differences caused by the
host numerical stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import scripts.run_verify_per_scale_negative_metric_1_1 as runner


SCHEMA = "pathline_template_matching.per_scale_spatial_replay_diagnostic.v1"
PREDICTION_NAME = "outer_predictions.npz"
MANIFEST_NAME = "outer_prediction_manifest.json"


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _maximum_ulp_distance(left: np.ndarray, right: np.ndarray) -> int:
    """Return the maximum ULP distance for finite nonnegative float arrays."""

    a = np.ascontiguousarray(left)
    b = np.ascontiguousarray(right)
    _require(a.shape == b.shape, "ULP comparison shape mismatch")
    _require(a.dtype == b.dtype and a.dtype in (np.dtype(np.float32), np.dtype(np.float64)), "ULP comparison dtype mismatch")
    _require(np.isfinite(a).all() and np.isfinite(b).all(), "ULP comparison is nonfinite")
    _require(np.all(a >= 0.0) and np.all(b >= 0.0), "ULP comparison requires nonnegative values")
    unsigned_dtype = np.uint32 if a.dtype == np.dtype(np.float32) else np.uint64
    a_bits = a.view(unsigned_dtype)
    b_bits = b.view(unsigned_dtype)
    distance = np.where(a_bits >= b_bits, a_bits - b_bits, b_bits - a_bits)
    return int(distance.max(initial=unsigned_dtype(0)))


def _difference_summary(stored: np.ndarray, replayed: np.ndarray) -> dict[str, Any]:
    _require(stored.shape == replayed.shape, "difference comparison shape mismatch")
    _require(stored.dtype == replayed.dtype, "difference comparison dtype mismatch")
    stored_nan = np.isnan(stored)
    replayed_nan = np.isnan(replayed)
    nan_mask_different_count = int(np.count_nonzero(stored_nan != replayed_nan))
    finite = np.isfinite(stored) & np.isfinite(replayed)
    _require(np.all(np.isfinite(stored) | stored_nan), "stored comparison contains infinity")
    _require(np.all(np.isfinite(replayed) | replayed_nan), "replayed comparison contains infinity")
    stored_finite = stored[finite]
    replayed_finite = replayed[finite]
    absolute = np.abs(stored_finite - replayed_finite)
    scale = np.maximum(np.abs(stored_finite), np.abs(replayed_finite))
    relative = np.divide(
        absolute,
        scale,
        out=np.zeros_like(absolute, dtype=np.float64),
        where=scale > 0.0,
    )
    return {
        "bitwise_equal": bool(np.array_equal(stored, replayed, equal_nan=True)),
        "different_count": int(np.count_nonzero(stored_finite != replayed_finite))
        + nan_mask_different_count,
        "nan_mask_exact": nan_mask_different_count == 0,
        "nan_mask_different_count": nan_mask_different_count,
        "maximum_absolute_difference": float(absolute.max(initial=0.0)),
        "maximum_relative_difference": float(relative.max(initial=0.0)),
        "maximum_ulp_distance": _maximum_ulp_distance(stored_finite, replayed_finite),
        "allclose_atol_1e_15": bool(
            np.allclose(stored, replayed, rtol=0.0, atol=1.0e-15, equal_nan=True)
        ),
        "allclose_atol_1e_14": bool(
            np.allclose(stored, replayed, rtol=0.0, atol=1.0e-14, equal_nan=True)
        ),
    }


def _full_query_replay(
    plan: runner.Plan,
    run_directory: Path,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    selected: runner.TailCandidateSpec,
) -> dict[str, Any]:
    """Rebuild every label-free query field from authenticated fit artifacts."""

    outer_family = str(manifest["outer_family"])
    git_commit = str(manifest["git_commit"])
    fit_families = [family for family in plan.family_order if family != outer_family]
    scaler_record = manifest.get("final_per_scale_scaler_manifest")
    calibration_record = manifest.get("final_calibration_manifest")
    _require(isinstance(scaler_record, Mapping), "prediction scaler reference is invalid")
    _require(isinstance(calibration_record, Mapping), "prediction calibration reference is invalid")
    scaler = runner.authenticate_and_rebuild_final_scaler(
        run_directory / "final_per_scale_scaler.npz",
        run_directory / "final_per_scale_scaler_manifest.json",
        plan=plan,
        selected=selected,
        outer_family=outer_family,
        fit_families=fit_families,
        git_commit=git_commit,
        expected_manifest_file_sha256=str(scaler_record["file_sha256"]),
    )
    calibration = runner.authenticate_and_rebuild_final_calibration(
        run_directory / "final_tail_calibration.npz",
        run_directory / "final_tail_calibration_manifest.json",
        plan=plan,
        selected=selected,
        scaler=scaler,
        outer_family=outer_family,
        fit_families=fit_families,
        git_commit=git_commit,
        expected_manifest_file_sha256=str(calibration_record["file_sha256"]),
    )
    manifest_rows, _ = runner.load_cache_rows(plan)
    outer_rows = tuple(row for row in manifest_rows if row.family == outer_family)
    projections = tuple(
        runner.load_cache_projection(plan, row, include_labels=False) for row in outer_rows
    )
    runner._validate_label_free_outer_scope(plan, outer_family, projections, outer_rows)
    replayed_arrays, replayed_group_audits = runner.build_outer_prediction_arrays(
        projections,
        calibration.model,
        selected,
        plan,
        device="cpu",
    )
    _require(set(replayed_arrays) == set(arrays), "full replay member set drifted")
    exact_fields = (
        "dataset",
        "source_ordinal",
        "source_index",
        "scale_id",
        "center_seed_index",
        "scale_block_index",
        "assigned_row_index",
        "retrieval_supported",
        "calibration_supported",
        "spatial_imputed",
        "spatial_unimputable",
        "calibration_mode",
        "scaler_mode",
        "prediction",
    )
    exact = {
        name: {
            "exact": bool(np.array_equal(arrays[name], replayed_arrays[name])),
            "different_count": int(np.count_nonzero(arrays[name] != replayed_arrays[name])),
        }
        for name in exact_fields
    }
    floating_fields = (
        "raw_negative_distance",
        "tail_probability",
        "tail_anomaly",
        "spatial_score",
        "spatial_denominator",
    )
    floating = {
        name: _difference_summary(arrays[name], replayed_arrays[name])
        for name in floating_fields
    }
    return {
        "access_scope": "authenticated_fit_artifacts_and_outer_feature_projection_only_no_labels_no_metrics",
        "exact_fields": exact,
        "floating_fields": floating,
        "all_exact_fields_pass": all(record["exact"] for record in exact.values()),
        "classification_replay_pass": exact["prediction"]["exact"],
        "group_audits_exact": runner._json_safe(replayed_group_audits)
        == runner._json_safe(manifest["group_audits"]),
    }


def _load_authenticated_prediction(run_directory: Path) -> tuple[Mapping[str, Any], dict[str, np.ndarray]]:
    manifest_path = run_directory / MANIFEST_NAME
    prediction_path = run_directory / PREDICTION_NAME
    _require(manifest_path.is_file(), f"missing prediction manifest: {manifest_path}")
    _require(prediction_path.is_file(), f"missing prediction NPZ: {prediction_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(manifest, Mapping), "prediction manifest root is invalid")
    runner._authenticate_self_hash(manifest)
    _require(manifest.get("schema") == runner.PREDICTION_MANIFEST_SCHEMA, "prediction manifest schema drifted")
    file_record = manifest.get("prediction_file")
    _require(isinstance(file_record, Mapping), "prediction file record is invalid")
    _require(int(file_record["size_bytes"]) == prediction_path.stat().st_size, "prediction size mismatch")
    _require(str(file_record["sha256"]) == _sha256_file(prediction_path), "prediction file SHA-256 mismatch")
    records = manifest.get("arrays")
    _require(isinstance(records, Mapping), "prediction array records are invalid")
    with np.load(prediction_path, allow_pickle=False) as archive:
        _require(set(archive.files) == set(runner.PREDICTION_ARRAY_DTYPES), "prediction member set drifted")
        arrays = {name: np.array(archive[name], copy=True, order="C") for name in archive.files}
    row_count = int(manifest.get("row_count", -1))
    for name, dtype in runner.PREDICTION_ARRAY_DTYPES.items():
        values = arrays[name]
        record = records.get(name)
        _require(isinstance(record, Mapping), f"missing array record: {name}")
        _require(values.dtype == dtype and values.shape == (row_count,), f"array contract drifted: {name}")
        _require(runner.canonical_array_sha256(values) == record.get("sha256"), f"array SHA-256 mismatch: {name}")
    return manifest, arrays


def diagnose(
    config_path: Path,
    run_directory: Path,
    *,
    full_query_replay: bool = False,
) -> dict[str, Any]:
    plan = runner.load_plan(config_path)
    manifest, arrays = _load_authenticated_prediction(run_directory)
    _require(manifest.get("config_sha256") == plan.sha256, "prediction/config binding drifted")
    payload = manifest.get("selected_candidate")
    _require(isinstance(payload, Mapping), "selected candidate is missing")
    selected = runner.TailCandidateSpec(
        representation=str(payload["representation"]),
        k=int(payload["k"]),
        sigma=float(payload["sigma"]),
        decision_rule=str(payload["decision_rule"]),
        decision_value=float(payload["decision_value"]),
    )
    _require(selected.candidate_id == payload.get("candidate_id"), "candidate identity drifted")
    group_audits = manifest.get("group_audits")
    _require(isinstance(group_audits, list) and group_audits, "group audits are missing")

    group_reports: list[dict[str, Any]] = []
    coverage = np.zeros(len(arrays["spatial_score"]), dtype=np.int8)
    changed_predictions = 0
    score_differences = 0
    denominator_differences = 0
    maximum_score_absolute = 0.0
    maximum_denominator_absolute = 0.0
    maximum_score_ulp = 0
    maximum_denominator_ulp = 0
    for group in group_audits:
        _require(isinstance(group, Mapping), "invalid group audit")
        block_index = runner.BLOCK_NAMES.index(str(group["block"]))
        selected_rows = (
            (arrays["dataset"] == str(group["dataset"]))
            & (arrays["source_ordinal"] == int(group["source_ordinal"]))
            & (arrays["source_index"] == int(group["source_index"]))
            & (arrays["scale_block_index"] == block_index)
        )
        _require(int(selected_rows.sum()) == int(group["sample_count"]), "group row count drifted")
        coverage[selected_rows] += 1
        replayed = runner.spatial_calibrated_tail_scores(
            arrays["tail_anomaly"][selected_rows],
            arrays["calibration_supported"][selected_rows],
            arrays["center_seed_index"][selected_rows],
            sigma=selected.sigma,
            grid_shape=plan.grid_shape,
            truncate=plan.gaussian_truncate,
        )
        replayed_prediction = runner.candidate_predictions(
            selected,
            replayed.scores,
            arrays["center_seed_index"][selected_rows],
            replayed.calibration_supported | replayed.imputed,
        )
        score_summary = _difference_summary(arrays["spatial_score"][selected_rows], replayed.scores)
        denominator_summary = _difference_summary(
            arrays["spatial_denominator"][selected_rows], replayed.denominator
        )
        changed = int(np.count_nonzero(arrays["prediction"][selected_rows] != replayed_prediction))
        _require(np.array_equal(arrays["spatial_imputed"][selected_rows], replayed.imputed), "imputation state drifted")
        _require(
            np.array_equal(arrays["spatial_unimputable"][selected_rows], replayed.unimputable),
            "unimputable state drifted",
        )
        score_differences += int(score_summary["different_count"])
        denominator_differences += int(denominator_summary["different_count"])
        changed_predictions += changed
        maximum_score_absolute = max(maximum_score_absolute, float(score_summary["maximum_absolute_difference"]))
        maximum_denominator_absolute = max(
            maximum_denominator_absolute, float(denominator_summary["maximum_absolute_difference"])
        )
        maximum_score_ulp = max(maximum_score_ulp, int(score_summary["maximum_ulp_distance"]))
        maximum_denominator_ulp = max(
            maximum_denominator_ulp, int(denominator_summary["maximum_ulp_distance"])
        )
        group_reports.append(
            {
                "dataset": str(group["dataset"]),
                "source_ordinal": int(group["source_ordinal"]),
                "source_index": int(group["source_index"]),
                "block": str(group["block"]),
                "row_count": int(selected_rows.sum()),
                "score": score_summary,
                "denominator": denominator_summary,
                "changed_prediction_count": changed,
            }
        )
    _require(np.array_equal(coverage, np.ones_like(coverage)), "groups do not partition prediction rows")
    kernel = runner._gaussian_kernel1d(selected.sigma, plan.gaussian_truncate)
    report = {
        "schema": SCHEMA,
        "access_scope": "outer_prediction_manifest_and_npz_only_no_labels_no_metrics",
        "host": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "run_directory": str(run_directory.resolve()),
        "outer_family": str(manifest["outer_family"]),
        "git_commit": str(manifest["git_commit"]),
        "config_sha256": plan.sha256,
        "candidate": dict(payload),
        "kernel_float_hex": [float(value).hex() for value in kernel],
        "kernel_sha256": runner.canonical_array_sha256(kernel),
        "row_count": len(arrays["spatial_score"]),
        "group_count": len(group_reports),
        "score_different_count": score_differences,
        "score_maximum_absolute_difference": maximum_score_absolute,
        "score_maximum_ulp_distance": maximum_score_ulp,
        "denominator_different_count": denominator_differences,
        "denominator_maximum_absolute_difference": maximum_denominator_absolute,
        "denominator_maximum_ulp_distance": maximum_denominator_ulp,
        "changed_prediction_count": changed_predictions,
        "bitwise_replay_pass": score_differences == 0 and denominator_differences == 0,
        "classification_replay_pass": changed_predictions == 0,
        "groups": group_reports,
    }
    if full_query_replay:
        report["full_query_replay"] = _full_query_replay(
            plan, run_directory, manifest, arrays, selected
        )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--full-query-replay", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(
        json.dumps(
            diagnose(
                args.config,
                args.run_dir,
                full_query_replay=bool(args.full_query_replay),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

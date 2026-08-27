"""Cache-backed leave-one-family-out development experiment."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Iterable

import numpy as np
import torch

from .development_data import (
    CacheSlice,
    build_input_manifest,
    cache_paths,
    canonical_json_sha256,
    load_cache_slice,
    load_project_specs,
    sha256_file,
)
from .development_library import build_balanced_library, query_audit_rows
from .matcher import ExhaustiveMatchResult, ExhaustiveOneNearestNeighbor
from .metrics import binary_metrics
from .pca import DeterministicPCA


METHOD_PRIOR = "eligible_library_candidate_prior_constant_score"
METHOD_RAW = "raw_centered_7x32x3_exact_1nn"
METHOD_PCA = "raw_centered_library_only_pca_161d_exact_1nn"
METHOD_FMT = "fmt_independent_3d_161d_sha256_25fce29499c9089e_exact_1nn"
METHODS = (METHOD_PRIOR, METHOD_RAW, METHOD_PCA, METHOD_FMT)
METHOD_PREFIX = {
    METHOD_PRIOR: "prior",
    METHOD_RAW: "raw672",
    METHOD_PCA: "raw_pca161",
    METHOD_FMT: "fmt161",
}

METRIC_FIELDS = (
    "sample_count",
    "positive_count",
    "negative_count",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
    "average_precision",
    "auroc",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _fold_output_entries(fold_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(value for value in fold_dir.rglob("*") if value.is_file()):
        relative = path.relative_to(fold_dir)
        if relative.as_posix() in {"fold_manifest.json", "_SUCCESS.json"}:
            continue
        entries.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    if not entries:
        raise RuntimeError("fold produced no auditable output files")
    return entries


def _complete_fold(
    fold_dir: Path, fold_manifest: dict[str, Any]
) -> dict[str, Any]:
    manifest = dict(fold_manifest)
    manifest["output_files"] = _fold_output_entries(fold_dir)
    manifest["fold_manifest_content_sha256"] = canonical_json_sha256(manifest)
    manifest_path = fold_dir / "fold_manifest.json"
    _write_json(manifest_path, manifest)
    success = {
        "status": "complete",
        "held_out_family": manifest["held_out_family"],
        "input_manifest_sha256": manifest["input_manifest_sha256"],
        "fold_manifest_content_sha256": manifest[
            "fold_manifest_content_sha256"
        ],
        "fold_manifest_file_sha256": sha256_file(manifest_path),
        "output_file_count": len(manifest["output_files"]),
        "elapsed_seconds": manifest["elapsed_seconds"],
    }
    success["success_content_sha256"] = canonical_json_sha256(success)
    _write_json(fold_dir / "_SUCCESS.json", success)
    return success


def _validate_completed_fold(
    fold_dir: Path, *, input_manifest_sha256: str
) -> dict[str, Any]:
    success_path = fold_dir / "_SUCCESS.json"
    manifest_path = fold_dir / "fold_manifest.json"
    if not success_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"fold completion evidence is missing: {fold_dir}")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    expected_success_digest = canonical_json_sha256(
        {
            key: value
            for key, value in success.items()
            if key != "success_content_sha256"
        }
    )
    if success.get("success_content_sha256") != expected_success_digest:
        raise RuntimeError(f"fold success digest is invalid: {fold_dir}")
    if success.get("input_manifest_sha256") != input_manifest_sha256:
        raise RuntimeError("cannot use fold against a different input manifest")
    if success.get("fold_manifest_file_sha256") != sha256_file(manifest_path):
        raise RuntimeError(f"fold manifest file changed: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_digest = canonical_json_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "fold_manifest_content_sha256"
        }
    )
    if manifest.get("fold_manifest_content_sha256") != expected_manifest_digest:
        raise RuntimeError(f"fold manifest content digest is invalid: {manifest_path}")
    if success.get("fold_manifest_content_sha256") != expected_manifest_digest:
        raise RuntimeError(f"fold success references the wrong manifest: {fold_dir}")
    entries = manifest.get("output_files")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"fold output manifest is empty: {fold_dir}")
    seen: set[str] = set()
    for entry in entries:
        relative_text = str(entry["relative_path"])
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts or relative_text in seen:
            raise RuntimeError(f"invalid or duplicate fold output path: {relative_text}")
        seen.add(relative_text)
        path = fold_dir / relative
        if not path.is_file():
            raise RuntimeError(f"fold output is missing: {path}")
        if int(entry["size_bytes"]) != path.stat().st_size:
            raise RuntimeError(f"fold output size changed: {path}")
        if str(entry["sha256"]) != sha256_file(path):
            raise RuntimeError(f"fold output digest changed: {path}")
    if int(success.get("output_file_count", -1)) != len(entries):
        raise RuntimeError(f"fold output count changed: {fold_dir}")
    return success


def _archive_incomplete(path: Path) -> Path:
    archive_root = path.parent / "_incomplete"
    archive_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = path.name.strip(".").replace(".staging", "") or "fold"
    destination = archive_root / f"{stem}_{timestamp}"
    path.rename(destination)
    return destination


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    materialized = list(rows)
    if fieldnames is None:
        names: list[str] = []
        for row in materialized:
            for name in row:
                if name not in names:
                    names.append(name)
    else:
        names = list(fieldnames)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=names, extrasaction="raise")
        writer.writeheader()
        writer.writerows(materialized)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _matcher_result_prior(count: int, fraction: float) -> ExhaustiveMatchResult:
    prediction = np.full(count, float(fraction) > 0.5, dtype=bool)
    return ExhaustiveMatchResult(
        labels=prediction,
        scores=np.full(count, float(fraction), dtype=np.float32),
        nearest_indices=np.full(count, -1, dtype=np.int64),
        nearest_distances=np.full(count, np.nan, dtype=np.float32),
        nearest_positive_distances=np.full(count, np.nan, dtype=np.float32),
        nearest_negative_distances=np.full(count, np.nan, dtype=np.float32),
    )


def _metrics_row(
    record: CacheSlice,
    *,
    regime: str,
    method: str,
    result: ExhaustiveMatchResult,
) -> dict[str, Any]:
    metrics = binary_metrics(record.reference, result.labels, result.scores)
    return {
        "held_out_family": record.physical_family,
        "dataset": record.dataset,
        "regime": regime,
        "legacy_phase": record.legacy_phase,
        "source_ordinal": record.ordinal,
        "source_start_index": int(record.metadata["source_start_index"]),
        "source_time": float(record.metadata["source_time"]),
        "method": method,
        **metrics.as_dict(),
    }


def _pooled_metric_rows(
    accumulator: dict[tuple[Any, ...], list[tuple[np.ndarray, ExhaustiveMatchResult]]],
    key_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(accumulator, key=lambda value: tuple(str(item) for item in value)):
        entries = accumulator[key]
        labels = np.concatenate([item[0] for item in entries])
        predictions = np.concatenate([item[1].labels for item in entries])
        scores = np.concatenate([item[1].scores for item in entries])
        row = dict(zip(key_names, key))
        row.update(binary_metrics(labels, predictions, scores).as_dict())
        rows.append(row)
    return rows


def _load_record(
    path: Path,
    *,
    dataset: str,
    family: str,
    phase: str,
    expected_config_sha256: str,
    base_config: dict[str, Any],
    digest_by_path: dict[str, str],
) -> CacheSlice:
    resolved = str(path.resolve())
    if resolved not in digest_by_path:
        raise RuntimeError(f"cache path is absent from input manifest: {resolved}")
    current_digest = sha256_file(path)
    if current_digest != digest_by_path[resolved]:
        raise RuntimeError(f"cache changed after input manifest creation: {resolved}")
    return load_cache_slice(
        path,
        expected_dataset=dataset,
        expected_family=family,
        expected_phase=phase,
        expected_config_sha256=expected_config_sha256,
        base_config=base_config,
        file_sha256=digest_by_path[resolved],
    )


def _library_rows_with_ids(rows: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    return [{"library_row_id": index, **row} for index, row in enumerate(rows)]


def _query_csv_fields() -> list[str]:
    fields = [
        "query_id",
        "held_out_family",
        "dataset",
        "regime",
        "legacy_phase",
        "source_ordinal",
        "source_start_index",
        "source_time",
        "cache_local_index",
        "scale_id",
        "scale_name",
        "label",
        "seed_x",
        "seed_y",
        "seed_z",
    ]
    for method in METHODS:
        prefix = METHOD_PREFIX[method]
        fields.extend(
            [
                f"{prefix}_prediction",
                f"{prefix}_score",
                f"{prefix}_nearest_library_row",
                f"{prefix}_nearest_distance",
                f"{prefix}_nearest_positive_distance",
                f"{prefix}_nearest_negative_distance",
            ]
        )
    return fields


def _write_query_rows(
    writer: csv.DictWriter,
    record: CacheSlice,
    *,
    regime: str,
    results: dict[str, ExhaustiveMatchResult],
) -> None:
    for index in range(len(record.reference)):
        scale_index = int(record.scale_id[index])
        row: dict[str, Any] = {
            "query_id": f"{record.dataset}|{regime}|{record.ordinal}|{index}",
            "held_out_family": record.physical_family,
            "dataset": record.dataset,
            "regime": regime,
            "legacy_phase": record.legacy_phase,
            "source_ordinal": record.ordinal,
            "source_start_index": int(record.metadata["source_start_index"]),
            "source_time": float(record.metadata["source_time"]),
            "cache_local_index": index,
            "scale_id": scale_index,
            "scale_name": record.canonical_scale_names[scale_index],
            "label": int(record.reference[index]),
            "seed_x": float(record.seeds[index, 0]),
            "seed_y": float(record.seeds[index, 1]),
            "seed_z": float(record.seeds[index, 2]),
        }
        for method in METHODS:
            prefix = METHOD_PREFIX[method]
            result = results[method]
            nearest = int(result.nearest_indices[index])
            row.update(
                {
                    f"{prefix}_prediction": int(result.labels[index]),
                    f"{prefix}_score": float(result.scores[index]),
                    f"{prefix}_nearest_library_row": nearest if nearest >= 0 else "",
                    f"{prefix}_nearest_distance": (
                        float(result.nearest_distances[index])
                        if np.isfinite(result.nearest_distances[index])
                        else ""
                    ),
                    f"{prefix}_nearest_positive_distance": (
                        float(result.nearest_positive_distances[index])
                        if np.isfinite(result.nearest_positive_distances[index])
                        else ""
                    ),
                    f"{prefix}_nearest_negative_distance": (
                        float(result.nearest_negative_distances[index])
                        if np.isfinite(result.nearest_negative_distances[index])
                        else ""
                    ),
                }
            )
        writer.writerow(row)


def _save_prediction_artifact(
    path: Path,
    record: CacheSlice,
    *,
    reported_path: Path | None = None,
    regime: str,
    results: dict[str, ExhaustiveMatchResult],
    config_sha256: str,
    git_commit: str,
) -> dict[str, Any]:
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(1, dtype=np.int16),
        "dataset": np.asarray(record.dataset),
        "physical_family": np.asarray(record.physical_family),
        "regime": np.asarray(regime),
        "legacy_phase": np.asarray(record.legacy_phase),
        "source_ordinal": np.asarray(record.ordinal, dtype=np.int16),
        "source_start_index": np.asarray(
            int(record.metadata["source_start_index"]), dtype=np.int64
        ),
        "source_time": np.asarray(float(record.metadata["source_time"])),
        "cache_path": np.asarray(str(record.path.resolve())),
        "cache_sha256": np.asarray(record.file_sha256),
        "development_config_sha256": np.asarray(config_sha256),
        "git_commit": np.asarray(git_commit),
        "seeds": record.seeds,
        "reference": record.reference,
        "scale_id": record.scale_id,
        "canonical_scale_names_json": np.asarray(
            json.dumps(record.canonical_scale_names)
        ),
        "metadata_json": np.asarray(json.dumps(record.metadata, sort_keys=True)),
    }
    for method in METHODS:
        prefix = METHOD_PREFIX[method]
        result = results[method]
        payload[f"{prefix}_prediction"] = result.labels
        payload[f"{prefix}_score"] = result.scores
        payload[f"{prefix}_nearest_library_row"] = result.nearest_indices
        payload[f"{prefix}_nearest_distance"] = result.nearest_distances
        payload[f"{prefix}_nearest_positive_distance"] = (
            result.nearest_positive_distances
        )
        payload[f"{prefix}_nearest_negative_distance"] = (
            result.nearest_negative_distances
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return {
        "path": str((reported_path if reported_path is not None else path).resolve()),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "dataset": record.dataset,
        "physical_family": record.physical_family,
        "regime": regime,
        "source_ordinal": record.ordinal,
        "source_start_index": int(record.metadata["source_start_index"]),
        "source_time": float(record.metadata["source_time"]),
        "cache_path": str(record.path.resolve()),
        "cache_sha256": record.file_sha256,
        "sample_count": len(record.reference),
        "positive_count": int(record.reference.sum()),
    }


def evaluate_fold(
    development_config_path: str | Path,
    run_dir: str | Path,
    *,
    held_out_family: str,
    environment: str,
    digest_by_path: dict[str, str],
    input_manifest_sha256: str,
    git_commit: str,
    device: str | torch.device,
    resume: bool = False,
) -> dict[str, Any]:
    started = time.time()
    project_root, config, base_config, registry, family_by_dataset = load_project_specs(
        development_config_path
    )
    if held_out_family not in config["physical_families"]:
        raise ValueError(f"unknown held-out family {held_out_family!r}")
    folds_root = Path(run_dir) / "folds"
    folds_root.mkdir(parents=True, exist_ok=True)
    final_fold_dir = folds_root / held_out_family
    success_path = final_fold_dir / "_SUCCESS.json"
    if success_path.exists():
        if not resume:
            raise FileExistsError(f"fold output already exists: {success_path}")
        return _validate_completed_fold(
            final_fold_dir, input_manifest_sha256=input_manifest_sha256
        )
    if final_fold_dir.exists():
        if not resume:
            raise FileExistsError(f"incomplete fold output exists: {final_fold_dir}")
        archived = _archive_incomplete(final_fold_dir)
        print(f"[resume] preserved incomplete fold at {archived}", flush=True)
    staging_fold_dir = folds_root / f".{held_out_family}.staging"
    if staging_fold_dir.exists():
        if not resume:
            raise FileExistsError(f"fold staging output exists: {staging_fold_dir}")
        archived = _archive_incomplete(staging_fold_dir)
        print(f"[resume] preserved stale fold staging at {archived}", flush=True)
    staging_fold_dir.mkdir(parents=True, exist_ok=False)
    fold_dir = staging_fold_dir
    expected_digest = str(registry["legacy_task5_cache_contract"]["config_sha256"])
    library_records: list[CacheSlice] = []
    for dataset, family in family_by_dataset.items():
        if family == held_out_family:
            continue
        paths = cache_paths(
            config,
            registry,
            environment=environment,
            dataset=dataset,
            legacy_phase="development",
        )
        for path in paths[:4]:
            library_records.append(
                _load_record(
                    path,
                    dataset=dataset,
                    family=family,
                    phase="development",
                    expected_config_sha256=expected_digest,
                    base_config=base_config,
                    digest_by_path=digest_by_path,
                )
            )
    library_config = config["library"]
    maximum = int(library_config["maximum_templates_per_class_per_stratum"])
    library = build_balanced_library(
        library_records,
        held_out_family=held_out_family,
        maximum_per_class_per_stratum=maximum,
        random_seed=int(library_config["sampling_random_seed"]),
    )
    eligible_pca_fit_features = np.ascontiguousarray(
        np.concatenate([record.raw_features for record in library_records]),
        dtype=np.float32,
    )
    if len(eligible_pca_fit_features) != library.eligible_candidate_count:
        raise AssertionError("eligible PCA fit rows do not match pre-balance candidates")
    del library_records
    _write_csv(fold_dir / "library_rows.csv", _library_rows_with_ids(library.rows))
    pca = DeterministicPCA.fit(eligible_pca_fit_features, components=161)
    del eligible_pca_fit_features
    library_pca = pca.transform(library.raw_features)
    matcher_device = torch.device(device)
    matchers = {
        METHOD_RAW: ExhaustiveOneNearestNeighbor(
            library.raw_features, library.labels, device=matcher_device
        ),
        METHOD_PCA: ExhaustiveOneNearestNeighbor(
            library_pca, library.labels, device=matcher_device
        ),
        METHOD_FMT: ExhaustiveOneNearestNeighbor(
            library.fmt_features, library.labels, device=matcher_device
        ),
    }
    query_chunk = int(config.get("execution", {}).get("query_chunk_size", 1024))
    library_chunk = int(config.get("execution", {}).get("library_chunk_size", 8192))
    config_sha256 = str(
        json.loads((Path(run_dir) / "input_manifest.json").read_text(encoding="utf-8"))[
            "development_config_sha256"
        ]
    )
    timeslice_rows: list[dict[str, Any]] = []
    audit_rows = list(library.audit_rows)
    artifact_index: list[dict[str, Any]] = []
    flow_accumulator: dict[
        tuple[Any, ...], list[tuple[np.ndarray, ExhaustiveMatchResult]]
    ] = {}
    family_accumulator: dict[
        tuple[Any, ...], list[tuple[np.ndarray, ExhaustiveMatchResult]]
    ] = {}
    scale_accumulator: dict[
        tuple[Any, ...], list[tuple[np.ndarray, ExhaustiveMatchResult]]
    ] = {}
    query_path = fold_dir / "per_query.csv"
    with query_path.open("w", encoding="utf-8", newline="") as query_target:
        query_writer = csv.DictWriter(query_target, fieldnames=_query_csv_fields())
        query_writer.writeheader()
        for dataset in config["physical_families"][held_out_family]:
            for regime, phase in (
                ("seen_scale", "development"),
                ("unseen_scale", "confirmation"),
            ):
                paths = cache_paths(
                    config,
                    registry,
                    environment=environment,
                    dataset=dataset,
                    legacy_phase=phase,
                )[:4]
                for path in paths:
                    record = _load_record(
                        path,
                        dataset=dataset,
                        family=held_out_family,
                        phase=phase,
                        expected_config_sha256=expected_digest,
                        base_config=base_config,
                        digest_by_path=digest_by_path,
                    )
                    audit_rows.extend(query_audit_rows(record, regime=regime))
                    pca_query = pca.transform(record.raw_features)
                    results = {
                        METHOD_PRIOR: _matcher_result_prior(
                            len(record.reference), library.prior_positive_fraction
                        ),
                        METHOD_RAW: matchers[METHOD_RAW].query(
                            record.raw_features,
                            query_chunk_size=query_chunk,
                            library_chunk_size=library_chunk,
                        ),
                        METHOD_PCA: matchers[METHOD_PCA].query(
                            pca_query,
                            query_chunk_size=query_chunk,
                            library_chunk_size=library_chunk,
                        ),
                        METHOD_FMT: matchers[METHOD_FMT].query(
                            record.fmt_features,
                            query_chunk_size=query_chunk,
                            library_chunk_size=library_chunk,
                        ),
                    }
                    for method, result in results.items():
                        timeslice_rows.append(
                            _metrics_row(
                                record,
                                regime=regime,
                                method=method,
                                result=result,
                            )
                        )
                        flow_accumulator.setdefault(
                            (held_out_family, dataset, regime, method), []
                        ).append((record.reference, result))
                        family_accumulator.setdefault(
                            (held_out_family, regime, method), []
                        ).append((record.reference, result))
                        for scale_index, scale_name in enumerate(
                            record.canonical_scale_names
                        ):
                            mask = record.scale_id == scale_index
                            subset = ExhaustiveMatchResult(
                                labels=result.labels[mask],
                                scores=result.scores[mask],
                                nearest_indices=result.nearest_indices[mask],
                                nearest_distances=result.nearest_distances[mask],
                                nearest_positive_distances=result.nearest_positive_distances[
                                    mask
                                ],
                                nearest_negative_distances=result.nearest_negative_distances[
                                    mask
                                ],
                            )
                            scale_accumulator.setdefault(
                                (
                                    held_out_family,
                                    dataset,
                                    regime,
                                    scale_name,
                                    method,
                                ),
                                [],
                            ).append((record.reference[mask], subset))
                    _write_query_rows(
                        query_writer, record, regime=regime, results=results
                    )
                    artifact = fold_dir / "predictions" / (
                        f"{dataset}_{regime}_ordinal{record.ordinal:02d}.npz"
                    )
                    reported_artifact = final_fold_dir / "predictions" / artifact.name
                    artifact_index.append(
                        _save_prediction_artifact(
                            artifact,
                            record,
                            reported_path=reported_artifact,
                            regime=regime,
                            results=results,
                            config_sha256=config_sha256,
                            git_commit=git_commit,
                        )
                    )
    _write_csv(fold_dir / "audit_counts.csv", audit_rows)
    _write_csv(fold_dir / "per_timeslice.csv", timeslice_rows)
    _write_csv(
        fold_dir / "per_flow.csv",
        _pooled_metric_rows(
            flow_accumulator,
            ("held_out_family", "dataset", "regime", "method"),
        ),
    )
    pooled_family_rows = _pooled_metric_rows(
        family_accumulator,
        ("held_out_family", "regime", "method"),
    )
    for row in pooled_family_rows:
        row["within_family_aggregation"] = "pooled_query_samples"
        row["source_timeslice_count"] = sum(
            1
            for item in timeslice_rows
            if item["held_out_family"] == row["held_out_family"]
            and item["regime"] == row["regime"]
            and item["method"] == row["method"]
        )
    _write_csv(fold_dir / "per_family.csv", pooled_family_rows)
    _write_csv(
        fold_dir / "per_scale_tuple.csv",
        _pooled_metric_rows(
            scale_accumulator,
            ("held_out_family", "dataset", "regime", "scale_name", "method"),
        ),
    )
    _write_json(fold_dir / "prediction_artifacts.json", artifact_index)
    fold_manifest = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "phase": config["phase"],
        "evidence_scope": config["evidence_scope"],
        "held_out_family": held_out_family,
        "library_family_count": len(config["physical_families"]) - 1,
        "library_dataset_count": len(
            [value for value in family_by_dataset.values() if value != held_out_family]
        ),
        "eligible_library_candidate_count": library.eligible_candidate_count,
        "eligible_library_positive_count": library.eligible_positive_count,
        "eligible_library_prior_positive_fraction": library.prior_positive_fraction,
        "balanced_library_count": len(library.labels),
        "balanced_library_positive_count": int(library.labels.sum()),
        "library_raw_sha256": _array_sha256(library.raw_features),
        "library_fmt_sha256": _array_sha256(library.fmt_features),
        "library_labels_sha256": _array_sha256(library.labels),
        "pca": {
            "solver": "numpy.linalg.svd_full_matrices_false",
            "fit_scope": "eligible_pre_balance_library_candidates_only_current_fold",
            "sample_count": pca.sample_count,
            "input_width": pca.input_width,
            "component_count": len(pca.components),
            "mean_sha256": _array_sha256(pca.mean),
            "components_sha256": _array_sha256(pca.components),
            "explained_variance_ratio_sum": float(
                pca.explained_variance_ratio.sum()
            ),
        },
        "matcher": {
            "algorithm": "exhaustive_one_nearest_neighbor",
            "arithmetic": "float32",
            "device": str(matcher_device),
            "query_chunk_size": query_chunk,
            "library_chunk_size": library_chunk,
            "torch_deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "cuda_matmul_allow_tf32": bool(
                torch.backends.cuda.matmul.allow_tf32
            ),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_device_name": (
                torch.cuda.get_device_name(matcher_device)
                if matcher_device.type == "cuda"
                else None
            ),
        },
        "method_ids": list(METHODS),
        "input_manifest_sha256": input_manifest_sha256,
        "git_commit": git_commit,
        "started_utc": _utc_now(),
        "elapsed_seconds": float(time.time() - started),
        "query_artifact_count": len(artifact_index),
        "query_sample_count": int(sum(item["sample_count"] for item in artifact_index)),
    }
    _complete_fold(fold_dir, fold_manifest)
    fold_dir.rename(final_fold_dir)
    return _validate_completed_fold(
        final_fold_dir, input_manifest_sha256=input_manifest_sha256
    )


def prepare_run(
    development_config_path: str | Path,
    run_dir: str | Path,
    *,
    environment: str,
) -> tuple[dict[str, Any], dict[str, str], str]:
    run_path = Path(run_dir)
    if run_path.exists():
        raise FileExistsError(f"run directory already exists: {run_path}")
    run_path.mkdir(parents=True)
    project_root, config, _, _, _ = load_project_specs(development_config_path)
    manifest, digest_by_path = build_input_manifest(
        development_config_path, environment=environment
    )
    git_commit = _git_commit(project_root)
    manifest.update(
        {
            "git_commit": git_commit,
            "created_utc": _utc_now(),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "packages": {
                    name: _package_version(name)
                    for name in (
                        "imageio",
                        "lazy-loader",
                        "matplotlib",
                        "netCDF4",
                        "numpy",
                        "PyYAML",
                        "scikit-image",
                        "scipy",
                        "tifffile",
                        "torch",
                    )
                },
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "cuda_device": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
                "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
                "slurm_node": __import__("os").environ.get("SLURMD_NODENAME"),
            },
        }
    )
    manifest["manifest_content_sha256"] = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
    )
    _write_json(run_path / "input_manifest.json", manifest)
    _write_json(
        run_path / "run_state.json",
        {
            "status": "prepared",
            "experiment": config["experiment"],
            "phase": config["phase"],
            "evidence_scope": config["evidence_scope"],
            "input_manifest_sha256": manifest["manifest_content_sha256"],
            "git_commit": git_commit,
            "created_utc": manifest["created_utc"],
        },
    )
    return manifest, digest_by_path, git_commit


def run_all_folds(
    development_config_path: str | Path,
    run_dir: str | Path,
    *,
    environment: str = "ibex",
    device: str = "auto",
    resume: bool = False,
) -> list[dict[str, Any]]:
    run_path = Path(run_dir)
    if resume:
        manifest = json.loads((run_path / "input_manifest.json").read_text(encoding="utf-8"))
        stored_manifest_digest = canonical_json_sha256(
            {
                key: value
                for key, value in manifest.items()
                if key != "manifest_content_sha256"
            }
        )
        if stored_manifest_digest != manifest.get("manifest_content_sha256"):
            raise RuntimeError("input_manifest.json content digest is invalid")
        project_root, config, _, _, _ = load_project_specs(development_config_path)
        current_manifest, digest_by_path = build_input_manifest(
            development_config_path, environment=environment
        )
        for key in (
            "environment",
            "development_config_sha256",
            "base_config_sha256",
            "dataset_registry_sha256",
            "legacy_task5_config_sha256",
        ):
            if current_manifest[key] != manifest.get(key):
                raise RuntimeError(f"{key} changed; refusing cross-protocol resume")
        stored_files = {
            item["path"]: (item["sha256"], int(item["size_bytes"]))
            for item in manifest["files"]
        }
        current_files = {
            item["path"]: (item["sha256"], int(item["size_bytes"]))
            for item in current_manifest["files"]
        }
        if stored_files != current_files:
            raise RuntimeError("input cache files changed; refusing resume")
        if _git_commit(project_root) != manifest.get("git_commit"):
            raise RuntimeError("Git commit changed; refusing cross-revision resume")
        git_commit = str(manifest["git_commit"])
    else:
        manifest, digest_by_path, git_commit = prepare_run(
            development_config_path, run_path, environment=environment
        )
        _, config, _, _, _ = load_project_specs(development_config_path)
    selected_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else
        "cpu" if device == "auto" else device
    )
    if selected_device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
    results = []
    for family in config["physical_families"]:
        print(f"[mainExp development] held-out family: {family}", flush=True)
        results.append(
            evaluate_fold(
                development_config_path,
                run_path,
                held_out_family=family,
                environment=environment,
                digest_by_path=digest_by_path,
                input_manifest_sha256=str(manifest["manifest_content_sha256"]),
                git_commit=git_commit,
                device=selected_device,
                resume=resume,
            )
        )
    _write_json(
        run_path / "run_state.json",
        {
            "status": "folds_complete_pending_report",
            "experiment": config["experiment"],
            "phase": config["phase"],
            "evidence_scope": config["evidence_scope"],
            "input_manifest_sha256": manifest["manifest_content_sha256"],
            "git_commit": git_commit,
            "folds": results,
            "updated_utc": _utc_now(),
        },
    )
    return results

"""Fail-closed access to the frozen legacy Task5 development evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from .data_access import load_dataset_registry, validate_dataset_registry


@dataclass(frozen=True)
class CacheSlice:
    """One validated cache slice loaded into memory."""

    path: Path
    file_sha256: str
    dataset: str
    physical_family: str
    legacy_phase: str
    ordinal: int
    raw_features: np.ndarray
    fmt_features: np.ndarray
    reference: np.ndarray
    seeds: np.ndarray
    scale_id: np.ndarray
    physical_dt: np.ndarray
    integration_steps: np.ndarray
    metadata: dict[str, Any]
    canonical_scale_names: tuple[str, ...]


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(int(chunk_bytes)):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_development_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("experiment") != "mainExp_TemplateMatching_1.1":
        raise ValueError("development config has an unexpected experiment ID")
    if config.get("phase") != "development_cache_backed":
        raise ValueError("only the frozen cache-backed development phase is supported")
    scope = config.get("evidence_scope", {})
    if scope.get("sealed_confirmation_access") != "forbidden":
        raise ValueError("sealed confirmation must remain forbidden")
    if scope.get("allowed_input") != "legacy_task5_cache_only":
        raise ValueError("development input must be the exposed legacy cache only")
    families = config.get("physical_families", {})
    flattened = [dataset for datasets in families.values() for dataset in datasets]
    if len(families) != 7 or len(flattened) != 10 or len(set(flattened)) != 10:
        raise ValueError("frozen development config must contain 7 families and 10 datasets")
    split = config.get("split", {})
    if split.get("method") != "leave_one_physical_family_out":
        raise ValueError("only leave-one-physical-family-out is supported")
    seen = split.get("library_and_seen_scale_query", {})
    unseen = split.get("unseen_scale_query", {})
    selection = split.get("descriptor_selection_only", {})
    if seen.get("source_ordinals") != [0, 1, 2, 3]:
        raise ValueError("seen-scale ordinals have drifted")
    if unseen.get("source_ordinals") != [0, 1, 2, 3]:
        raise ValueError("unseen-scale ordinals have drifted")
    if selection.get("source_ordinals") != [4, 5] or selection.get(
        "main_metric_access"
    ) != "forbidden":
        raise ValueError("descriptor-selection-only ordinals are not sealed from main metrics")
    return config


def _numeric_scale_key(scale: dict[str, Any]) -> tuple[float, float, int]:
    return (
        round(float(scale["offset_grid_scale"]), 12),
        round(float(scale["dt_scale"]), 12),
        int(scale["integration_steps"]),
    )


def _canonical_scale_lookup(base_config: dict[str, Any], role: str) -> dict[tuple[float, float, int], str]:
    scales = base_config["scale_sets"][role]
    lookup = {_numeric_scale_key(scale): str(scale["name"]) for scale in scales}
    if len(lookup) != len(scales):
        raise ValueError(f"duplicate numeric scale tuple in base config role {role}")
    return lookup


def _role_for_slice(legacy_phase: str, ordinal: int) -> tuple[str, str, str]:
    if legacy_phase == "development" and 0 <= ordinal <= 3:
        return "library_and_seen_scale", "train", "library"
    if legacy_phase == "development" and 4 <= ordinal <= 5:
        return "descriptor_selection_only", "validation", "descriptor_selection_only"
    if legacy_phase == "confirmation" and 0 <= ordinal <= 3:
        return "unseen_scale_development", "confirmation", "unseen_scale_evaluation"
    raise ValueError(f"unsupported legacy cache role phase={legacy_phase} ordinal={ordinal}")


def _registry_maps(
    development_config: dict[str, Any], registry: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    rows = {str(item["id"]): item for item in registry["datasets"]}
    family_by_dataset = {
        str(dataset): str(family)
        for family, datasets in development_config["physical_families"].items()
        for dataset in datasets
    }
    if set(rows) != set(family_by_dataset):
        raise ValueError(
            "dataset registry and frozen development family map disagree: "
            f"registry_only={sorted(set(rows)-set(family_by_dataset))}, "
            f"config_only={sorted(set(family_by_dataset)-set(rows))}"
        )
    for dataset, family in family_by_dataset.items():
        if str(rows[dataset]["physical_family"]) != family:
            raise ValueError(f"physical family mismatch for {dataset}")
    return rows, family_by_dataset


def cache_paths(
    development_config: dict[str, Any],
    registry: dict[str, Any],
    *,
    environment: str,
    dataset: str,
    legacy_phase: str,
) -> list[Path]:
    rows, _ = _registry_maps(development_config, registry)
    root_value = rows[dataset]["legacy_task5_cache"][environment][legacy_phase]
    paths = sorted(Path(root_value).glob("slice_*.npz"))
    expected = 6 if legacy_phase == "development" else 4
    if len(paths) != expected:
        raise ValueError(
            f"{dataset}/{legacy_phase} expected {expected} cache slices, found {len(paths)}"
        )
    return paths


def load_cache_slice(
    path: str | Path,
    *,
    expected_dataset: str,
    expected_family: str,
    expected_phase: str,
    expected_config_sha256: str,
    base_config: dict[str, Any],
    file_sha256: str | None = None,
) -> CacheSlice:
    cache_path = Path(path)
    digest = sha256_file(cache_path) if file_sha256 is None else str(file_sha256)
    with np.load(cache_path, allow_pickle=False) as data:
        required = {
            "raw_features",
            "fmt_features",
            "reference",
            "seeds",
            "scale_id",
            "physical_dt",
            "integration_steps",
            "metadata_json",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{cache_path}: missing keys {sorted(missing)}")
        raw = np.asarray(data["raw_features"], dtype=np.float32)
        fmt = np.asarray(data["fmt_features"], dtype=np.float32)
        reference = np.asarray(data["reference"])
        seeds = np.asarray(data["seeds"], dtype=np.float32)
        scale_id = np.asarray(data["scale_id"])
        physical_dt = np.asarray(data["physical_dt"], dtype=np.float32)
        integration_steps = np.asarray(data["integration_steps"])
        metadata_scalar = np.asarray(data["metadata_json"])
        if metadata_scalar.ndim != 0:
            raise ValueError(f"{cache_path}: metadata_json must be scalar")
        metadata = json.loads(str(metadata_scalar.item()))
    count = len(reference) if reference.ndim == 1 else 0
    if not (
        count > 0
        and raw.shape == (count, 672)
        and fmt.shape == (count, 161)
        and seeds.shape == (count, 3)
        and scale_id.shape == (count,)
        and physical_dt.shape == (count,)
        and integration_steps.shape == (count,)
    ):
        raise ValueError(f"{cache_path}: incompatible cache array shapes")
    arrays = (raw, fmt, reference, seeds, scale_id, physical_dt, integration_steps)
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError(f"{cache_path}: cache contains NaN or Inf")
    if not np.all(np.isin(reference, (0, 1))):
        raise ValueError(f"{cache_path}: reference is not binary")
    if not np.issubdtype(scale_id.dtype, np.integer) or not np.issubdtype(
        integration_steps.dtype, np.integer
    ):
        raise ValueError(f"{cache_path}: scale_id/integration_steps must be integer")
    expected_metadata = {
        "experiment": "mainExp_Task5_3D_1.1",
        "dataset": expected_dataset,
        "phase": expected_phase,
        "config_sha256": expected_config_sha256,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{cache_path}: metadata mismatch {mismatches}")
    ordinal = int(metadata["ordinal"])
    _, expected_scale_set, canonical_role = _role_for_slice(expected_phase, ordinal)
    if str(metadata.get("scale_set")) != expected_scale_set:
        raise ValueError(
            f"{cache_path}: scale_set {metadata.get('scale_set')!r} != {expected_scale_set!r}"
        )
    lookup = _canonical_scale_lookup(base_config, canonical_role)
    table = metadata.get("scale_table")
    if not isinstance(table, list) or len(table) != len(lookup):
        raise ValueError(f"{cache_path}: scale table width has drifted")
    canonical_names: list[str] = []
    for scale in table:
        key = _numeric_scale_key(scale)
        if key not in lookup:
            raise ValueError(f"{cache_path}: unexpected numeric scale tuple {key}")
        canonical_names.append(lookup[key])
    if len(set(canonical_names)) != len(canonical_names):
        raise ValueError(f"{cache_path}: duplicate canonical scale tuple")
    if int(scale_id.min()) < 0 or int(scale_id.max()) >= len(canonical_names):
        raise ValueError(f"{cache_path}: scale_id outside scale table")
    expected_valid = np.asarray(metadata["valid_count_by_scale"], dtype=np.int64)
    actual_valid = np.bincount(scale_id.astype(np.int64), minlength=len(table))
    if not np.array_equal(expected_valid, actual_valid):
        raise ValueError(f"{cache_path}: valid_count_by_scale does not match rows")
    return CacheSlice(
        path=cache_path,
        file_sha256=digest,
        dataset=expected_dataset,
        physical_family=expected_family,
        legacy_phase=expected_phase,
        ordinal=ordinal,
        raw_features=np.ascontiguousarray(raw),
        fmt_features=np.ascontiguousarray(fmt),
        reference=reference.astype(bool, copy=False),
        seeds=np.ascontiguousarray(seeds),
        scale_id=scale_id.astype(np.int16, copy=False),
        physical_dt=physical_dt,
        integration_steps=integration_steps.astype(np.int16, copy=False),
        metadata=metadata,
        canonical_scale_names=tuple(canonical_names),
    )


def build_input_manifest(
    development_config_path: str | Path,
    *,
    environment: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    config_path = Path(development_config_path)
    config = load_development_config(config_path)
    project_root = config_path.resolve().parents[1]
    base_path = project_root / str(config["base_config"])
    registry_path = project_root / str(config["dataset_registry"])
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    registry = load_dataset_registry(registry_path)
    validation = validate_dataset_registry(registry_path, environment)
    if validation["summary"]["cache_accessible_count"] != 10:
        raise RuntimeError("not all ten exposed legacy cache datasets are accessible")
    rows, family_by_dataset = _registry_maps(config, registry)
    expected_digest = str(registry["legacy_task5_cache_contract"]["config_sha256"])
    files: list[dict[str, Any]] = []
    digest_by_path: dict[str, str] = {}
    for dataset in sorted(rows):
        for phase in ("development", "confirmation"):
            for path in cache_paths(
                config,
                registry,
                environment=environment,
                dataset=dataset,
                legacy_phase=phase,
            ):
                digest = sha256_file(path)
                record = load_cache_slice(
                    path,
                    expected_dataset=dataset,
                    expected_family=family_by_dataset[dataset],
                    expected_phase=phase,
                    expected_config_sha256=expected_digest,
                    base_config=base_config,
                    file_sha256=digest,
                )
                role, _, _ = _role_for_slice(phase, record.ordinal)
                digest_by_path[str(path.resolve())] = digest
                files.append(
                    {
                        "path": str(path.resolve()),
                        "sha256": digest,
                        "size_bytes": path.stat().st_size,
                        "dataset": dataset,
                        "physical_family": record.physical_family,
                        "legacy_phase": phase,
                        "ordinal": record.ordinal,
                        "development_role": role,
                        "sample_count": len(record.reference),
                        "positive_count": int(record.reference.sum()),
                        "source_start_index": int(record.metadata["source_start_index"]),
                        "source_time": float(record.metadata["source_time"]),
                        "scale_tuple_count": len(record.canonical_scale_names),
                    }
                )
    manifest = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "phase": config["phase"],
        "evidence_scope": config["evidence_scope"],
        "environment": str(environment),
        "development_config": str(config_path.resolve()),
        "development_config_sha256": sha256_file(config_path),
        "base_config": str(base_path.resolve()),
        "base_config_sha256": sha256_file(base_path),
        "dataset_registry": str(registry_path.resolve()),
        "dataset_registry_sha256": sha256_file(registry_path),
        "legacy_task5_config_sha256": expected_digest,
        "cache_file_count": len(files),
        "cache_total_bytes": int(sum(item["size_bytes"] for item in files)),
        "cache_total_samples": int(sum(item["sample_count"] for item in files)),
        "files": files,
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    return manifest, digest_by_path


def load_project_specs(
    development_config_path: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    config_path = Path(development_config_path).resolve()
    config = load_development_config(config_path)
    project_root = config_path.parents[1]
    base_config = yaml.safe_load(
        (project_root / str(config["base_config"])).read_text(encoding="utf-8")
    )
    registry = load_dataset_registry(project_root / str(config["dataset_registry"]))
    _, family_by_dataset = _registry_maps(config, registry)
    return project_root, config, base_config, registry, family_by_dataset

"""Dataset-registry validation for raw fields and legacy FMT caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .netcdf_io import inspect_netcdf_3d, load_netcdf_window_3d


_REQUIRED_CACHE_KEYS = {
    "raw_features",
    "fmt_features",
    "reference",
    "seeds",
    "scale_id",
    "metadata_json",
}


def load_dataset_registry(path: str | Path) -> dict[str, Any]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not isinstance(spec.get("datasets"), list):
        raise ValueError("dataset registry must contain a datasets list")
    identifiers = [str(item["id"]) for item in spec["datasets"]]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"dataset identifiers must be unique: {identifiers}")
    contract = spec.get("legacy_task5_cache_contract")
    if not isinstance(contract, dict):
        raise ValueError("dataset registry must define legacy_task5_cache_contract")
    digest = str(contract.get("config_sha256", ""))
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("legacy Task5 config_sha256 must be 64 lowercase hexadecimal characters")
    return spec


def _first_existing(candidates: list[str]) -> Path | None:
    for value in candidates:
        path = Path(value)
        if path.is_file():
            return path
    return None


def _validate_cache_root(
    path: Path,
    expected_slices: int,
    *,
    expected_dataset: str,
    expected_phase: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    """Validate every legacy Task5 slice, not merely one representative file."""

    paths = sorted(path.glob("slice_*.npz")) if path.is_dir() else []
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_dir(),
        "expected_slices": int(expected_slices),
        "slice_count": len(paths),
        "valid": False,
    }
    if len(paths) != int(expected_slices):
        return result
    errors: list[str] = []
    total_count = 0
    positive_count = 0
    ordinals: list[int] = []
    experiments: set[str] = set()
    config_hashes: set[str] = set()
    for cache_path in paths:
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                missing = sorted(_REQUIRED_CACHE_KEYS.difference(data.files))
                if missing:
                    errors.append(f"{cache_path.name}: missing keys {missing}")
                    continue
                raw = np.asarray(data["raw_features"])
                fmt = np.asarray(data["fmt_features"])
                reference = np.asarray(data["reference"])
                seeds = np.asarray(data["seeds"])
                scale_id = np.asarray(data["scale_id"])
                metadata_value = np.asarray(data["metadata_json"])
                if metadata_value.ndim != 0:
                    raise ValueError("metadata_json must be a scalar string")
                metadata = json.loads(str(metadata_value.item()))

            count = len(reference) if reference.ndim else 0
            if not (
                count > 0
                and raw.shape == (count, 672)
                and fmt.shape == (count, 161)
                and reference.shape == (count,)
                and seeds.shape == (count, 3)
                and scale_id.shape == (count,)
            ):
                errors.append(
                    f"{cache_path.name}: incompatible shapes raw={raw.shape}, "
                    f"fmt={fmt.shape}, reference={reference.shape}, "
                    f"seeds={seeds.shape}, scale_id={scale_id.shape}"
                )
                continue
            if not (
                np.isfinite(raw).all()
                and np.isfinite(fmt).all()
                and np.isfinite(reference).all()
                and np.isfinite(seeds).all()
                and np.isfinite(scale_id).all()
            ):
                errors.append(f"{cache_path.name}: contains masked, NaN, or Inf values")
                continue
            if not np.all(np.isin(reference, (0, 1, False, True))):
                errors.append(f"{cache_path.name}: reference is not binary")
                continue
            if not np.issubdtype(scale_id.dtype, np.integer):
                errors.append(f"{cache_path.name}: scale_id is not an integer array")
                continue
            scale_table = metadata.get("scale_table")
            if not isinstance(scale_table, list) or not scale_table:
                errors.append(f"{cache_path.name}: metadata has no non-empty scale_table")
                continue
            if int(scale_id.min()) < 0 or int(scale_id.max()) >= len(scale_table):
                errors.append(f"{cache_path.name}: scale_id is outside metadata scale_table")
                continue
            expected_metadata = {
                "experiment": "mainExp_Task5_3D_1.1",
                "dataset": expected_dataset,
                "phase": expected_phase,
            }
            mismatches = {
                key: (metadata.get(key), value)
                for key, value in expected_metadata.items()
                if metadata.get(key) != value
            }
            if mismatches:
                errors.append(f"{cache_path.name}: metadata mismatch {mismatches}")
                continue
            config_sha256 = metadata.get("config_sha256")
            if (
                not isinstance(config_sha256, str)
                or len(config_sha256) != 64
                or any(character not in "0123456789abcdef" for character in config_sha256.lower())
            ):
                errors.append(f"{cache_path.name}: invalid metadata config_sha256")
                continue
            if config_sha256.lower() != expected_config_sha256:
                errors.append(
                    f"{cache_path.name}: config_sha256 {config_sha256.lower()} does not "
                    f"equal canonical {expected_config_sha256}"
                )
                continue
            ordinal = metadata.get("ordinal")
            if not isinstance(ordinal, int):
                errors.append(f"{cache_path.name}: metadata ordinal is not an integer")
                continue
            ordinals.append(ordinal)
            experiments.add(str(metadata["experiment"]))
            config_hashes.add(config_sha256.lower())
            total_count += count
            positive_count += int(reference.astype(bool).sum())
        except Exception as error:  # report a damaged archive without aborting the registry
            errors.append(f"{cache_path.name}: {type(error).__name__}: {error}")

    expected_ordinals = list(range(int(expected_slices)))
    if sorted(ordinals) != expected_ordinals:
        errors.append(
            f"metadata ordinals {sorted(ordinals)} do not equal {expected_ordinals}"
        )
    if len(config_hashes) != 1:
        errors.append(f"cache slices do not share one config_sha256: {sorted(config_hashes)}")
    result.update(
        {
            "checked_slices": len(paths),
            "sample_count_total": total_count,
            "raw_feature_width": 672,
            "fmt_feature_width": 161,
            "reference_positive_fraction": (
                float(positive_count / total_count) if total_count else None
            ),
            "metadata_experiments": sorted(experiments),
            "metadata_config_sha256": sorted(config_hashes),
            "errors": errors,
            "valid": not errors,
        }
    )
    return result


def validate_dataset_registry(
    registry_path: str | Path,
    environment: str,
    *,
    read_raw_sample: bool = False,
    raw_sample_max_spatial_dim: int = 8,
) -> dict[str, Any]:
    """Validate raw and cache access without treating one as the other."""

    spec = load_dataset_registry(registry_path)
    cache_contract = spec["legacy_task5_cache_contract"]
    expected_cache_digest = str(cache_contract["config_sha256"])
    environment = str(environment)
    report: dict[str, Any] = {
        "registry": str(Path(registry_path).resolve()),
        "environment": environment,
        "datasets": [],
    }
    for item in spec["datasets"]:
        dataset_id = str(item["id"])
        row: dict[str, Any] = {
            "id": dataset_id,
            "physical_family": str(item["physical_family"]),
        }
        raw_candidates = [str(value) for value in item.get("raw_paths", {}).get(environment, [])]
        raw_path = _first_existing(raw_candidates)
        raw_report: dict[str, Any] = {
            "candidates": raw_candidates,
            "file_present": raw_path is not None,
            "accessible": False,
        }
        if raw_path is not None:
            raw_report["selected_path"] = str(raw_path)
            if str(item.get("kind", "netcdf")) == "netcdf":
                try:
                    raw_report["inspection"] = inspect_netcdf_3d(raw_path)
                    if read_raw_sample:
                        window = load_netcdf_window_3d(
                            raw_path,
                            start_index=0,
                            frame_count=2,
                            max_spatial_dim=int(raw_sample_max_spatial_dim),
                        )
                        raw_report["sample"] = {
                            **window.metadata(),
                            "finite": bool(np.isfinite(window.velocity).all()),
                        }
                    raw_report["accessible"] = True
                except Exception as error:
                    raw_report["validation_error"] = f"{type(error).__name__}: {error}"
            else:
                raw_report["inspection"] = {
                    "path": str(raw_path.resolve()),
                    "size_bytes": raw_path.stat().st_size,
                    "kind": str(item.get("kind")),
                }
                raw_report["validation_error"] = (
                    "no VTK loader is implemented in this repository; file presence "
                    "does not prove reintegration access"
                )
        row["raw"] = raw_report

        legacy_spec = item.get("legacy_task5_cache", {})
        evidence_scope = str(legacy_spec.get("evidence_scope", ""))
        cache_report: dict[str, Any] = {
            "accessible": False,
            "evidence_scope": evidence_scope,
            "eligible_for_sealed_confirmation": False,
            "phases": {},
        }
        cache_spec = legacy_spec.get(environment, {})
        for phase, expected in (("development", 6), ("confirmation", 4)):
            if phase in cache_spec:
                phase_report = _validate_cache_root(
                    Path(cache_spec[phase]),
                    expected,
                    expected_dataset=dataset_id,
                    expected_phase=phase,
                    expected_config_sha256=expected_cache_digest,
                )
                cache_report["phases"][phase] = phase_report
        phase_hashes = {
            value
            for phase_report in cache_report["phases"].values()
            for value in phase_report.get("metadata_config_sha256", [])
        }
        cache_report["accessible"] = bool(cache_report["phases"]) and bool(
            evidence_scope == "exposed_development_only"
            and set(cache_report["phases"]) == {"development", "confirmation"}
            and all(phase["valid"] for phase in cache_report["phases"].values())
            and len(phase_hashes) == 1
        )
        cache_report["metadata_config_sha256"] = sorted(phase_hashes)
        row["legacy_task5_cache"] = cache_report
        row["usable_for_bootstrap"] = bool(
            raw_report["accessible"] or cache_report["accessible"]
        )
        report["datasets"].append(row)

    cache_config_hashes = {
        value
        for row in report["datasets"]
        for value in row["legacy_task5_cache"].get("metadata_config_sha256", [])
    }
    report["summary"] = {
        "dataset_count": len(report["datasets"]),
        "raw_accessible_count": sum(row["raw"]["accessible"] for row in report["datasets"]),
        "cache_accessible_count": sum(
            row["legacy_task5_cache"]["accessible"] for row in report["datasets"]
        ),
        "bootstrap_usable_count": sum(
            row["usable_for_bootstrap"] for row in report["datasets"]
        ),
        "cache_config_sha256": sorted(cache_config_hashes),
        "cache_config_consistent": len(cache_config_hashes) == 1,
    }
    return report

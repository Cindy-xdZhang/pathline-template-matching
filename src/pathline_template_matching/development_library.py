"""Frozen balanced template-library construction for development folds."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable

import numpy as np

from .development_data import CacheSlice


@dataclass(frozen=True)
class DevelopmentLibrary:
    """Balanced library arrays, provenance rows, and pre-balance prior."""

    raw_features: np.ndarray
    fmt_features: np.ndarray
    labels: np.ndarray
    rows: tuple[dict[str, Any], ...]
    audit_rows: tuple[dict[str, Any], ...]
    eligible_candidate_count: int
    eligible_positive_count: int
    prior_positive_fraction: float
    skipped_stratum_count: int
    skipped_candidate_count: int


def stable_selection_seed(base_seed: int, *identity: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(value) for value in identity)]).encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _sample_sorted(
    candidates: np.ndarray,
    count: int,
    *,
    seed: int,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    if count == len(candidates):
        return candidates.copy()
    rng = np.random.default_rng(np.uint64(seed))
    return np.sort(rng.choice(candidates, size=int(count), replace=False))


def build_balanced_library(
    records: Iterable[CacheSlice],
    *,
    held_out_family: str,
    maximum_per_class_per_stratum: int,
    random_seed: int,
    empty_class_action: str = "fail_and_report_stratum",
) -> DevelopmentLibrary:
    """Balance every flow×source-time×scale stratum independently."""

    maximum = int(maximum_per_class_per_stratum)
    if maximum < 1:
        raise ValueError("maximum templates per class per stratum must be positive")
    if empty_class_action not in {
        "fail_and_report_stratum",
        "skip_both_classes_and_audit",
    }:
        raise ValueError(f"unsupported empty-class action {empty_class_action!r}")
    ordered = sorted(
        records,
        key=lambda item: (item.dataset, item.ordinal, str(item.path)),
    )
    if not ordered:
        raise ValueError("no cache slices were supplied for library construction")
    if any(record.physical_family == held_out_family for record in ordered):
        raise ValueError("held-out physical family leaked into the library candidates")
    if any(record.legacy_phase != "development" or record.ordinal not in range(4) for record in ordered):
        raise ValueError("library candidates must be development ordinals 0-3 only")

    raw_blocks: list[np.ndarray] = []
    fmt_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    eligible_count = 0
    eligible_positive = 0
    skipped_strata = 0
    skipped_candidates = 0
    for record in ordered:
        eligible_count += len(record.reference)
        eligible_positive += int(record.reference.sum())
        assigned_counts = np.asarray(
            record.metadata["assigned_count_by_scale"], dtype=np.int64
        )
        valid_counts = np.asarray(
            record.metadata["valid_count_by_scale"], dtype=np.int64
        )
        for scale_index, scale_name in enumerate(record.canonical_scale_names):
            scale_mask = record.scale_id == scale_index
            class_candidates = {
                class_id: np.flatnonzero(scale_mask & (record.reference == bool(class_id)))
                for class_id in (0, 1)
            }
            selected_count = min(
                maximum,
                len(class_candidates[0]),
                len(class_candidates[1]),
            )
            if selected_count < 1:
                if empty_class_action == "fail_and_report_stratum":
                    raise RuntimeError(
                        f"empty class in library stratum {record.dataset}/ordinal"
                        f"{record.ordinal}/{scale_name}: negative={len(class_candidates[0])}, "
                        f"positive={len(class_candidates[1])}"
                    )
                skipped_strata += 1
                skipped_candidates += len(class_candidates[0]) + len(class_candidates[1])
            stratum_status = (
                "skipped_empty_class" if selected_count < 1 else "selected_balanced"
            )
            audit_rows.append(
                {
                    "population": "library",
                    "held_out_family": held_out_family,
                    "dataset": record.dataset,
                    "physical_family": record.physical_family,
                    "legacy_phase": record.legacy_phase,
                    "source_ordinal": record.ordinal,
                    "source_start_index": int(record.metadata["source_start_index"]),
                    "source_time": float(record.metadata["source_time"]),
                    "scale_id": int(scale_index),
                    "scale_name": scale_name,
                    "class_label": "all",
                    "assigned_count": int(assigned_counts[scale_index]),
                    "valid_count": int(valid_counts[scale_index]),
                    "invalid_count": int(
                        assigned_counts[scale_index] - valid_counts[scale_index]
                    ),
                    "candidate_count": int(scale_mask.sum()),
                    "selected_count": int(2 * selected_count),
                    "selection_seed": "",
                    "stratum_status": stratum_status,
                    "empty_class_action": empty_class_action,
                }
            )
            for class_id in (0, 1):
                seed = stable_selection_seed(
                    random_seed,
                    record.dataset,
                    record.ordinal,
                    scale_name,
                    class_id,
                )
                selected = _sample_sorted(
                    class_candidates[class_id], selected_count, seed=seed
                )
                if len(selected):
                    raw_blocks.append(record.raw_features[selected])
                    fmt_blocks.append(record.fmt_features[selected])
                    label_blocks.append(record.reference[selected])
                for local_index in selected:
                    metadata_rows.append(
                        {
                            "dataset": record.dataset,
                            "physical_family": record.physical_family,
                            "legacy_phase": record.legacy_phase,
                            "source_ordinal": record.ordinal,
                            "source_start_index": int(
                                record.metadata["source_start_index"]
                            ),
                            "source_time": float(record.metadata["source_time"]),
                            "scale_id": int(scale_index),
                            "scale_name": scale_name,
                            "cache_local_index": int(local_index),
                            "label": int(class_id),
                            "cache_path": str(record.path.resolve()),
                            "cache_sha256": record.file_sha256,
                            "selection_seed": int(seed),
                        }
                    )
                audit_rows.append(
                    {
                        "population": "library",
                        "held_out_family": held_out_family,
                        "dataset": record.dataset,
                        "physical_family": record.physical_family,
                        "legacy_phase": record.legacy_phase,
                        "source_ordinal": record.ordinal,
                        "source_start_index": int(record.metadata["source_start_index"]),
                        "source_time": float(record.metadata["source_time"]),
                        "scale_id": int(scale_index),
                        "scale_name": scale_name,
                        "class_label": int(class_id),
                        "assigned_count": "",
                        "valid_count": "",
                        "invalid_count": "",
                        "candidate_count": len(class_candidates[class_id]),
                        "selected_count": int(selected_count),
                        "selection_seed": int(seed),
                        "stratum_status": stratum_status,
                        "empty_class_action": empty_class_action,
                    }
                )
    if not raw_blocks:
        raise RuntimeError("every library stratum was skipped; no templates remain")
    raw = np.ascontiguousarray(np.concatenate(raw_blocks), dtype=np.float32)
    fmt = np.ascontiguousarray(np.concatenate(fmt_blocks), dtype=np.float32)
    labels = np.concatenate(label_blocks).astype(bool, copy=False)
    if not (len(raw) == len(fmt) == len(labels) == len(metadata_rows)):
        raise AssertionError("balanced library arrays and provenance rows disagree")
    if labels.sum() * 2 != len(labels):
        raise AssertionError("balanced library must contain exactly equal class counts")
    return DevelopmentLibrary(
        raw_features=raw,
        fmt_features=fmt,
        labels=labels,
        rows=tuple(metadata_rows),
        audit_rows=tuple(audit_rows),
        eligible_candidate_count=int(eligible_count),
        eligible_positive_count=int(eligible_positive),
        prior_positive_fraction=float(eligible_positive / eligible_count),
        skipped_stratum_count=int(skipped_strata),
        skipped_candidate_count=int(skipped_candidates),
    )


def query_audit_rows(record: CacheSlice, *, regime: str) -> list[dict[str, Any]]:
    if regime not in {"seen_scale", "unseen_scale"}:
        raise ValueError(f"unsupported query regime {regime!r}")
    expected_phase = "development" if regime == "seen_scale" else "confirmation"
    if record.legacy_phase != expected_phase or record.ordinal not in range(4):
        raise ValueError("query cache slice does not match its frozen regime")
    assigned = np.asarray(record.metadata["assigned_count_by_scale"], dtype=np.int64)
    valid = np.asarray(record.metadata["valid_count_by_scale"], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for scale_index, scale_name in enumerate(record.canonical_scale_names):
        mask = record.scale_id == scale_index
        common = {
            "population": "query",
            "held_out_family": record.physical_family,
            "dataset": record.dataset,
            "physical_family": record.physical_family,
            "regime": regime,
            "legacy_phase": record.legacy_phase,
            "source_ordinal": record.ordinal,
            "source_start_index": int(record.metadata["source_start_index"]),
            "source_time": float(record.metadata["source_time"]),
            "scale_id": int(scale_index),
            "scale_name": scale_name,
            "selection_seed": "",
        }
        rows.append(
            {
                **common,
                "class_label": "all",
                "assigned_count": int(assigned[scale_index]),
                "valid_count": int(valid[scale_index]),
                "invalid_count": int(assigned[scale_index] - valid[scale_index]),
                "candidate_count": int(mask.sum()),
                "selected_count": int(mask.sum()),
            }
        )
        for class_id in (0, 1):
            count = int(np.sum(mask & (record.reference == bool(class_id))))
            rows.append(
                {
                    **common,
                    "class_label": class_id,
                    "assigned_count": "",
                    "valid_count": count,
                    "invalid_count": "",
                    "candidate_count": count,
                    "selected_count": count,
                }
            )
    return rows

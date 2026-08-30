#!/usr/bin/env python3
"""Run label-free spatial post-processing of negative-template distances.

The parent CSV files physically contain reference labels, and a CSV reader
must parse each complete row before selecting fields.  The prediction phase's
explicit projection and all downstream prediction logic receive only identity,
grid-index, and score columns.  Immutable predictions are published before a
second, explicit reference projection computes diagnostic metrics.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching.metrics import average_precision, auroc  # noqa: E402
from pathline_template_matching.one_class_spatial import (  # noqa: E402
    high_score_two_means_predictions,
    masked_gaussian_grid_scores,
    rank_scores,
)
from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_json_sha256,
    sha256_file,
)


EXPERIMENT = "Other_NegativeDistanceSpatial_1.1"
PREDICTIONS_NAME = "predictions.csv"
PREDICTION_MANIFEST_NAME = "prediction_manifest.json"

_COLUMN_KEYS = (
    "dataset",
    "source_ordinal",
    "block",
    "center_index",
    "score",
    "label",
)
_PREDICTION_COLUMN_KEYS = _COLUMN_KEYS[:-1]


@dataclass(frozen=True)
class InputSpec:
    input_id: str
    path: Path
    expected_sha256: str
    allowed_datasets: tuple[str, ...]
    columns: Mapping[str, str]


@dataclass(frozen=True)
class SpatialPlan:
    config_path: Path
    config_sha256: str
    config: Mapping[str, Any]
    inputs: tuple[InputSpec, ...]
    grid_shape_zyx: tuple[int, int, int]
    sigmas: tuple[float, ...]
    gaussian_truncate: float
    fixed_top_fraction: float

    @property
    def grid_size(self) -> int:
        return int(np.prod(np.asarray(self.grid_shape_zyx, dtype=np.int64)))


@dataclass
class GroupData:
    input_id: str
    dataset: str
    source_ordinal: int
    block: str
    row_keys: list[str] = field(default_factory=list)
    center_indices: list[int] = field(default_factory=list)
    raw_scores: list[float] = field(default_factory=list)
    scores: dict[str, np.ndarray] = field(default_factory=dict)
    predictions: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    labels: np.ndarray | None = None

    @property
    def key(self) -> tuple[str, str, int, str]:
        return (self.input_id, self.dataset, self.source_ordinal, self.block)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_lower_hex(value: Any, length: int) -> bool:
    text = str(value)
    return len(text) == length and all(char in "0123456789abcdef" for char in text)


def _resolve_input_path(config_path: Path, value: Any) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_plan(config_path: str | Path) -> SpatialPlan:
    """Load and strictly validate the method-defining part of the YAML config."""

    path = Path(config_path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "config root must be a mapping")
    _require(payload.get("experiment") == EXPERIMENT, f"experiment must be {EXPERIMENT}")

    shape_value = payload.get("grid_shape_zyx")
    _require(
        isinstance(shape_value, Sequence)
        and not isinstance(shape_value, (str, bytes))
        and len(shape_value) == 3,
        "grid_shape_zyx must contain exactly three integers",
    )
    shape = tuple(int(value) for value in shape_value)
    _require(
        all(not isinstance(value, bool) and int(value) == value for value in shape_value)
        and min(shape) > 0,
        "grid_shape_zyx must contain positive integers",
    )

    gaussian = payload.get("gaussian")
    _require(isinstance(gaussian, Mapping), "gaussian must be a mapping")
    _require(
        gaussian.get("mask_normalized") is True,
        "gaussian.mask_normalized must be true",
    )
    _require(
        gaussian.get("no_cross_group_smoothing") is True,
        "gaussian.no_cross_group_smoothing must be true",
    )
    sigma_value = gaussian.get("sigma_grid_indices")
    _require(
        isinstance(sigma_value, Sequence)
        and not isinstance(sigma_value, (str, bytes))
        and len(sigma_value) > 0,
        "gaussian.sigma_grid_indices must be a nonempty sequence",
    )
    sigmas = tuple(float(value) for value in sigma_value)
    _require(
        all(np.isfinite(value) and value >= 0.0 for value in sigmas)
        and len(set(sigmas)) == len(sigmas),
        "Gaussian sigmas must be finite, non-negative, and unique",
    )
    truncate = float(gaussian.get("truncate"))
    _require(np.isfinite(truncate) and truncate > 0.0, "gaussian.truncate must be positive")

    top_fraction = float(payload.get("fixed_top_fraction"))
    _require(
        top_fraction == 0.05,
        "fixed_top_fraction is frozen at exactly 0.05 for this experiment",
    )
    _require(
        payload.get("grouping") == ["dataset", "source", "block"],
        "grouping must be exactly [dataset, source, block]",
    )
    _require(
        payload.get("rank_definition")
        == "stable_ascending_score_then_center_index_with_percentile_rank_plus_one_over_n",
        "rank_definition does not match the implemented stable percentile rank",
    )
    _require(
        payload.get("prediction_rules")
        == ["high_score_two_means", "fixed_top_fraction"],
        "prediction_rules must contain the two frozen rules in canonical order",
    )
    _require(
        payload.get("fixed_top_fraction_rule")
        == "rank_greater_than_one_minus_fraction_selecting_ceil_fraction_times_n",
        "fixed_top_fraction_rule does not match the frozen exact-count rule",
    )
    _require(
        payload.get("metrics")
        == [
            "accuracy",
            "average_precision",
            "f1",
            "balanced_accuracy",
            "auroc",
            "precision",
            "recall",
            "coverage",
        ],
        "metrics must contain the frozen metrics in canonical order",
    )
    _require(
        payload.get("aggregation")
        == "equal_weight_dataset_source_block_groups",
        "aggregation must be equal_weight_dataset_source_block_groups",
    )
    oracle = payload.get("oracle_threshold")
    _require(isinstance(oracle, Mapping), "oracle_threshold must be a mapping")
    _require(oracle.get("enabled") is True, "oracle_threshold.enabled must be true")
    _require(
        oracle.get("prediction_rule")
        == "score_greater_than_or_equal_to_threshold",
        "oracle threshold comparison does not match the frozen rule",
    )
    _require(
        oracle.get("tie_break")
        == "highest_f1_then_highest_precision_then_highest_threshold",
        "oracle threshold tie break does not match the frozen rule",
    )
    _require(
        oracle.get("may_select_or_name_main_method") is False,
        "oracle diagnostic must be forbidden from selecting the main method",
    )

    raw_inputs = payload.get("inputs")
    _require(
        isinstance(raw_inputs, Sequence)
        and not isinstance(raw_inputs, (str, bytes))
        and len(raw_inputs) == 2,
        "config must list exactly two exposed input CSV files",
    )
    inputs: list[InputSpec] = []
    input_ids: set[str] = set()
    for input_index, value in enumerate(raw_inputs):
        _require(isinstance(value, Mapping), f"inputs[{input_index}] must be a mapping")
        input_id = str(value.get("id", ""))
        _require(input_id and input_id not in input_ids, "input ids must be nonempty and unique")
        input_ids.add(input_id)
        expected_sha = str(value.get("sha256", ""))
        _require(
            _is_lower_hex(expected_sha, 64),
            f"{input_id}: sha256 must be 64 lowercase hexadecimal characters",
        )
        allowed_value = value.get("allowed_datasets")
        _require(
            isinstance(allowed_value, Sequence)
            and not isinstance(allowed_value, (str, bytes))
            and len(allowed_value) > 0,
            f"{input_id}: allowed_datasets must be nonempty",
        )
        allowed = tuple(str(dataset) for dataset in allowed_value)
        _require(
            all(allowed) and len(set(allowed)) == len(allowed),
            f"{input_id}: allowed_datasets must be unique nonempty strings",
        )
        columns = value.get("columns")
        _require(isinstance(columns, Mapping), f"{input_id}: columns must be a mapping")
        missing = [name for name in _COLUMN_KEYS if name not in columns]
        _require(not missing, f"{input_id}: missing explicit column mappings: {missing}")
        normalized_columns = {name: str(columns[name]) for name in _COLUMN_KEYS}
        _require(
            all(normalized_columns.values())
            and len(set(normalized_columns.values())) == len(normalized_columns),
            f"{input_id}: mapped CSV columns must be nonempty and distinct",
        )
        inputs.append(
            InputSpec(
                input_id=input_id,
                path=_resolve_input_path(path, value.get("path")),
                expected_sha256=expected_sha,
                allowed_datasets=allowed,
                columns=normalized_columns,
            )
        )
    return SpatialPlan(
        config_path=path,
        config_sha256=sha256_file(path),
        config=payload,
        inputs=tuple(inputs),
        grid_shape_zyx=shape,
        sigmas=sigmas,
        gaussian_truncate=truncate,
        fixed_top_fraction=top_fraction,
    )


def _stable_expected_file(spec: InputSpec) -> dict[str, Any]:
    """Authenticate one immutable input and detect mutation during hashing."""

    before = spec.path.stat()
    digest = sha256_file(spec.path)
    after = spec.path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"{spec.input_id}: input changed while its SHA-256 was computed")
    if digest != spec.expected_sha256:
        raise ValueError(
            f"{spec.input_id}: input SHA-256 mismatch: expected "
            f"{spec.expected_sha256}, got {digest}"
        )
    return {
        "input_id": spec.input_id,
        "path": str(spec.path),
        "file_size": int(after.st_size),
        "file_sha256": digest,
    }


def _iter_projected_csv(
    path: Path,
    columns: Sequence[str],
) -> Iterator[tuple[int, dict[str, str]]]:
    """Yield only the explicitly requested CSV projection.

    The caller controls whether the reference column is requested.  The first
    phase never includes it in ``columns``.
    """

    requested = tuple(columns)
    _require(len(set(requested)) == len(requested), "projected CSV columns must be unique")
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"empty input CSV: {path}") from error
        _require(len(set(header)) == len(header), f"duplicate CSV header in {path}")
        missing = [name for name in requested if name not in header]
        _require(not missing, f"{path}: missing required CSV columns: {missing}")
        indices = [header.index(name) for name in requested]
        width = len(header)
        for line_number, row in enumerate(reader, start=2):
            _require(
                len(row) == width,
                f"{path}:{line_number}: CSV row width {len(row)} != header width {width}",
            )
            yield line_number, {name: row[index] for name, index in zip(requested, indices)}


def _strict_int(value: str, *, context: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be an integer, got {value!r}") from error
    if str(parsed) != str(value).strip() and str(value).strip() not in {f"+{parsed}", f"-{abs(parsed)}"}:
        # Reject floating encodings such as 1.0 while allowing ordinary signs/zeros.
        try:
            if float(value) != parsed or any(char in str(value).lower() for char in (".", "e")):
                raise ValueError
        except ValueError as error:
            raise ValueError(f"{context} must be an integer, got {value!r}") from error
    return parsed


def _row_key(
    input_id: str,
    dataset: str,
    source_ordinal: int,
    block: str,
    center_index: int,
) -> str:
    identity = [input_id, dataset, int(source_ordinal), block, int(center_index)]
    payload = json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_prediction_groups(plan: SpatialPlan) -> tuple[list[GroupData], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int, str], GroupData] = {}
    all_row_keys: set[str] = set()
    input_evidence: list[dict[str, Any]] = []
    for spec in plan.inputs:
        evidence = _stable_expected_file(spec)
        observed_datasets: set[str] = set()
        column_names = [spec.columns[name] for name in _PREDICTION_COLUMN_KEYS]
        for line_number, projected in _iter_projected_csv(spec.path, column_names):
            logical = {
                name: projected[spec.columns[name]] for name in _PREDICTION_COLUMN_KEYS
            }
            dataset = logical["dataset"]
            if dataset not in spec.allowed_datasets:
                raise ValueError(
                    f"{spec.input_id}:{line_number}: unexpected dataset {dataset!r}"
                )
            observed_datasets.add(dataset)
            source_ordinal = _strict_int(
                logical["source_ordinal"],
                context=f"{spec.input_id}:{line_number}: source ordinal",
            )
            center_index = _strict_int(
                logical["center_index"],
                context=f"{spec.input_id}:{line_number}: center index",
            )
            block = logical["block"]
            _require(block != "", f"{spec.input_id}:{line_number}: block is empty")
            _require(source_ordinal >= 0, f"{spec.input_id}:{line_number}: source ordinal < 0")
            _require(
                0 <= center_index < plan.grid_size,
                f"{spec.input_id}:{line_number}: center index outside frozen grid",
            )
            try:
                score = float(logical["score"])
            except ValueError as error:
                raise ValueError(
                    f"{spec.input_id}:{line_number}: score is not numeric"
                ) from error
            _require(
                np.isfinite(score), f"{spec.input_id}:{line_number}: score must be finite"
            )
            row_key = _row_key(
                spec.input_id, dataset, source_ordinal, block, center_index
            )
            _require(
                row_key not in all_row_keys,
                f"{spec.input_id}:{line_number}: duplicate stable row identity",
            )
            all_row_keys.add(row_key)
            key = (spec.input_id, dataset, source_ordinal, block)
            group = groups.setdefault(
                key,
                GroupData(spec.input_id, dataset, source_ordinal, block),
            )
            group.row_keys.append(row_key)
            group.center_indices.append(center_index)
            group.raw_scores.append(score)
        if observed_datasets != set(spec.allowed_datasets):
            raise ValueError(
                f"{spec.input_id}: observed datasets {sorted(observed_datasets)} do not "
                f"equal allowed_datasets {sorted(spec.allowed_datasets)}"
            )
        evidence["row_count"] = sum(
            len(group.row_keys) for key, group in groups.items() if key[0] == spec.input_id
        )
        evidence["observed_datasets"] = sorted(observed_datasets)
        evidence["score_column"] = spec.columns["score"]
        input_evidence.append(evidence)
    ordered = [groups[key] for key in sorted(groups)]
    _require(ordered, "input CSV files contain no prediction rows")
    for group in ordered:
        centers = np.asarray(group.center_indices, dtype=np.int64)
        _require(
            len(np.unique(centers)) == len(centers),
            f"{group.key}: duplicate center index within spatial group",
        )
    return ordered, input_evidence


def _sigma_token(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def _score_variants(plan: SpatialPlan) -> tuple[tuple[str, float | None], ...]:
    return (
        ("raw_negative_distance", None),
        ("within_group_rank", None),
        *tuple(
            (f"masked_gaussian_rank_sigma_{_sigma_token(sigma)}", float(sigma))
            for sigma in plan.sigmas
        ),
    )


def _fixed_top_fraction_predictions(
    scores: np.ndarray,
    center_indices: np.ndarray,
    fraction: float,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    centers = np.asarray(center_indices, dtype=np.int64)
    _require(
        values.ndim == 1 and values.shape == centers.shape and len(values) > 0,
        "top-fraction prediction requires aligned nonempty one-dimensional arrays",
    )
    _require(np.isfinite(values).all(), "top-fraction scores must be finite")
    selected_count = max(1, int(math.ceil(float(fraction) * len(values))))
    # np.lexsort uses the last key as primary: descending score, then center index.
    order = np.lexsort((centers, -values))
    result = np.zeros(len(values), dtype=bool)
    result[order[:selected_count]] = True
    return result


def _compute_predictions(plan: SpatialPlan, groups: Sequence[GroupData]) -> None:
    for group in groups:
        raw = np.asarray(group.raw_scores, dtype=np.float64)
        centers = np.asarray(group.center_indices, dtype=np.int64)
        ranked = np.asarray(rank_scores(raw, centers), dtype=np.float64)
        group.scores = {
            "raw_negative_distance": raw,
            "within_group_rank": ranked,
        }
        for sigma in plan.sigmas:
            name = f"masked_gaussian_rank_sigma_{_sigma_token(sigma)}"
            group.scores[name] = np.asarray(
                masked_gaussian_grid_scores(
                    ranked,
                    centers,
                    grid_shape=plan.grid_shape_zyx,
                    sigma=float(sigma),
                    truncate=plan.gaussian_truncate,
                ),
                dtype=np.float64,
            )
        expected_names = {name for name, _ in _score_variants(plan)}
        _require(set(group.scores) == expected_names, "internal score-variant mismatch")
        for name, scores in group.scores.items():
            _require(
                scores.shape == raw.shape and np.isfinite(scores).all(),
                f"{group.key}/{name}: spatial score output is invalid",
            )
            group.predictions[(name, "high_score_two_means")] = np.asarray(
                high_score_two_means_predictions(scores), dtype=bool
            )
            group.predictions[(name, "fixed_top_fraction_0.05")] = (
                _fixed_top_fraction_predictions(
                    scores, centers, plan.fixed_top_fraction
                )
            )
            for rule in ("high_score_two_means", "fixed_top_fraction_0.05"):
                prediction = group.predictions[(name, rule)]
                _require(
                    prediction.shape == raw.shape,
                    f"{group.key}/{name}/{rule}: prediction shape mismatch",
                )


def _fsync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> str:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)
    _fsync_parent(path)
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _atomic_json(path: Path, value: Any) -> str:
    payload = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    return _atomic_bytes(path, payload)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return int(bool(value))
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return "" if not np.isfinite(numeric) else format(numeric, ".12g")
    if value is None:
        return ""
    return value


def _atomic_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> str:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(fieldnames), extrasaction="raise"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)
    _fsync_parent(path)
    return sha256_file(path)


def _prediction_fields(plan: SpatialPlan) -> list[str]:
    fields = [
        "input_id",
        "row_key",
        "dataset",
        "source_ordinal",
        "block",
        "center_index",
    ]
    for score_name, _ in _score_variants(plan):
        fields.extend(
            (
                score_name,
                f"{score_name}__high_score_two_means",
                f"{score_name}__fixed_top_fraction_0.05",
            )
        )
    return fields


def _prediction_rows(
    plan: SpatialPlan, groups: Sequence[GroupData]
) -> Iterator[dict[str, Any]]:
    for group in groups:
        for index, row_key in enumerate(group.row_keys):
            row: dict[str, Any] = {
                "input_id": group.input_id,
                "row_key": row_key,
                "dataset": group.dataset,
                "source_ordinal": group.source_ordinal,
                "block": group.block,
                "center_index": group.center_indices[index],
            }
            for score_name, _ in _score_variants(plan):
                row[score_name] = group.scores[score_name][index]
                row[f"{score_name}__high_score_two_means"] = group.predictions[
                    (score_name, "high_score_two_means")
                ][index]
                row[f"{score_name}__fixed_top_fraction_0.05"] = group.predictions[
                    (score_name, "fixed_top_fraction_0.05")
                ][index]
            yield row


def _prediction_manifest(
    plan: SpatialPlan,
    groups: Sequence[GroupData],
    input_evidence: Sequence[Mapping[str, Any]],
    predictions_sha256: str,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": "pathline_template_matching.label_free_spatial_prediction.v1",
        "experiment": EXPERIMENT,
        "phase": "prediction_complete_before_explicit_reference_projection",
        "config_sha256": plan.config_sha256,
        "input_files": [
            {
                key: value
                for key, value in evidence.items()
                if key
                in {
                    "input_id",
                    "path",
                    "file_size",
                    "file_sha256",
                    "row_count",
                    "observed_datasets",
                    "score_column",
                }
            }
            for evidence in input_evidence
        ],
        "identity_fields": [
            "input_id",
            "dataset",
            "source_ordinal",
            "block",
            "center_index",
        ],
        "grid_shape_zyx": list(plan.grid_shape_zyx),
        "score_variants": [
            {"name": name, "sigma_grid_indices": sigma}
            for name, sigma in _score_variants(plan)
        ],
        "prediction_rules": [
            "high_score_two_means",
            "fixed_top_fraction_0.05",
        ],
        "fixed_top_fraction_count_rule": "max(1,ceil(0.05*N_group))",
        "group_count": len(groups),
        "row_count": sum(len(group.row_keys) for group in groups),
        "prediction_column_count": len(_prediction_fields(plan)),
        "predictions_file": PREDICTIONS_NAME,
        "predictions_file_sha256": predictions_sha256,
        "reference_column_projection_to_prediction_logic": "excluded",
        "source_csv_contains_reference_column": True,
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    return manifest


def _read_reference_rows(plan: SpatialPlan) -> dict[str, int]:
    references: dict[str, int] = {}
    for spec in plan.inputs:
        _stable_expected_file(spec)
        logical_names = ("dataset", "source_ordinal", "block", "center_index", "label")
        column_names = [spec.columns[name] for name in logical_names]
        for line_number, projected in _iter_projected_csv(spec.path, column_names):
            logical = {name: projected[spec.columns[name]] for name in logical_names}
            dataset = logical["dataset"]
            if dataset not in spec.allowed_datasets:
                raise ValueError(
                    f"{spec.input_id}:{line_number}: unexpected dataset during reference pass"
                )
            source_ordinal = _strict_int(
                logical["source_ordinal"],
                context=f"{spec.input_id}:{line_number}: source ordinal",
            )
            center_index = _strict_int(
                logical["center_index"],
                context=f"{spec.input_id}:{line_number}: center index",
            )
            try:
                reference = int(logical["label"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{spec.input_id}:{line_number}: reference must be binary 0/1"
                ) from error
            _require(
                str(reference) == logical["label"].strip() and reference in (0, 1),
                f"{spec.input_id}:{line_number}: reference must be binary 0/1",
            )
            row_key = _row_key(
                spec.input_id,
                dataset,
                source_ordinal,
                logical["block"],
                center_index,
            )
            _require(
                row_key not in references,
                f"{spec.input_id}:{line_number}: duplicate reference row identity",
            )
            references[row_key] = reference
        _stable_expected_file(spec)
    return references


def _join_reference_rows(
    groups: Sequence[GroupData], references: Mapping[str, int]
) -> None:
    expected = {row_key for group in groups for row_key in group.row_keys}
    observed = set(references)
    if expected != observed:
        raise ValueError(
            "prediction/reference row-key mismatch: "
            f"missing={len(expected - observed)}, extra={len(observed - expected)}"
        )
    for group in groups:
        group.labels = np.asarray(
            [references[row_key] for row_key in group.row_keys], dtype=bool
        )


def _group_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    *,
    coverage: float,
) -> dict[str, Any]:
    targets = np.asarray(labels, dtype=bool)
    predicted = np.asarray(predictions, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    _require(
        len(targets) > 0
        and targets.shape == predicted.shape == values.shape
        and np.isfinite(values).all(),
        "metric arrays must be finite, aligned, and nonempty",
    )
    tp = int(np.sum(targets & predicted))
    fp = int(np.sum(~targets & predicted))
    tn = int(np.sum(~targets & ~predicted))
    fn = int(np.sum(targets & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(recall) and precision + recall
        else 0.0
    )
    balanced = (
        0.5 * (recall + specificity)
        if np.isfinite(recall) and np.isfinite(specificity)
        else float("nan")
    )
    single_class = bool(targets.all() or not targets.any())
    ap = float("nan") if single_class else average_precision(targets, values)
    area_under_roc = float("nan") if single_class else auroc(targets, values)
    if single_class:
        balanced = float("nan")
    return {
        "sample_count": len(targets),
        "positive_count": int(targets.sum()),
        "negative_count": int((~targets).sum()),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "single_class_group": single_class,
        "accuracy": (tp + tn) / len(targets),
        "average_precision": ap,
        "auroc": area_under_roc,
        "f1": f1,
        "balanced_accuracy": balanced,
        "precision": precision,
        "recall": recall,
        "coverage": float(coverage),
    }


def _per_group_metric_rows(
    plan: SpatialPlan, groups: Sequence[GroupData]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        _require(group.labels is not None, f"{group.key}: reference rows not joined")
        coverage = len(group.row_keys) / plan.grid_size
        _require(0.0 < coverage <= 1.0, f"{group.key}: invalid grid coverage")
        for score_name, sigma in _score_variants(plan):
            for rule in ("high_score_two_means", "fixed_top_fraction_0.05"):
                rows.append(
                    {
                        "input_id": group.input_id,
                        "dataset": group.dataset,
                        "source_ordinal": group.source_ordinal,
                        "block": group.block,
                        "score_variant": score_name,
                        "sigma_grid_indices": sigma,
                        "prediction_rule": rule,
                        **_group_metrics(
                            group.labels,
                            group.predictions[(score_name, rule)],
                            group.scores[score_name],
                            coverage=coverage,
                        ),
                    }
                )
    return rows


def _best_f1_upper_bound(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return the exact tie-preserving best-F1 threshold and its predictions.

    This function is reference-dependent and is called only in phase 2.  The
    first maximum is retained, so equal-F1 choices prefer the highest threshold
    and fewest positive predictions.
    """

    targets = np.asarray(labels, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    _require(
        len(targets) > 0 and targets.shape == values.shape and np.isfinite(values).all(),
        "oracle upper bound requires aligned finite nonempty arrays",
    )
    order = np.argsort(-values, kind="mergesort")
    ordered_values = values[order]
    ordered_targets = targets[order].astype(np.int64)
    group_ends = np.r_[
        np.flatnonzero(np.diff(ordered_values) != 0), len(ordered_values) - 1
    ]
    true_positive = np.cumsum(ordered_targets)[group_ends]
    predicted_positive = group_ends + 1
    positive_count = int(targets.sum())
    if positive_count == 0:
        return float(np.nextafter(np.max(values), np.inf)), np.zeros(len(values), dtype=bool)
    denominators = positive_count + predicted_positive
    f1_values = np.divide(
        2.0 * true_positive,
        denominators,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=denominators > 0,
    )
    precision_values = true_positive / predicted_positive
    best_f1 = np.max(f1_values)
    f1_candidates = np.flatnonzero(f1_values == best_f1)
    best_precision = np.max(precision_values[f1_candidates])
    best_index = int(
        f1_candidates[np.flatnonzero(precision_values[f1_candidates] == best_precision)[0]]
    )
    threshold = float(ordered_values[group_ends[best_index]])
    return threshold, values >= threshold


def _oracle_upper_bound_rows(
    plan: SpatialPlan, groups: Sequence[GroupData]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        _require(group.labels is not None, f"{group.key}: reference rows not joined")
        coverage = len(group.row_keys) / plan.grid_size
        for score_name, sigma in _score_variants(plan):
            threshold, prediction = _best_f1_upper_bound(
                group.labels, group.scores[score_name]
            )
            rows.append(
                {
                    "diagnostic_only": True,
                    "selection_data": "same_group_reference",
                    "threshold_comparison": "score_greater_than_or_equal",
                    "input_id": group.input_id,
                    "dataset": group.dataset,
                    "source_ordinal": group.source_ordinal,
                    "block": group.block,
                    "score_variant": score_name,
                    "sigma_grid_indices": sigma,
                    "best_f1_threshold": threshold,
                    "predicted_positive_count": int(prediction.sum()),
                    **_group_metrics(
                        group.labels,
                        prediction,
                        group.scores[score_name],
                        coverage=coverage,
                    ),
                }
            )
    return rows


_METRIC_NAMES = (
    "accuracy",
    "average_precision",
    "f1",
    "balanced_accuracy",
    "auroc",
    "precision",
    "recall",
    "coverage",
)


def _finite_macro(rows: Sequence[Mapping[str, Any]], name: str) -> float:
    values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def _finite_count(rows: Sequence[Mapping[str, Any]], name: str) -> int:
    return int(
        np.isfinite(np.asarray([float(row[name]) for row in rows], dtype=np.float64)).sum()
    )


def _aggregate_metric_rows(
    per_group: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    combinations = sorted(
        {
            (str(row["score_variant"]), str(row["prediction_rule"]))
            for row in per_group
        }
    )
    output: list[dict[str, Any]] = []
    for score_name, rule in combinations:
        method_rows = [
            row
            for row in per_group
            if row["score_variant"] == score_name and row["prediction_rule"] == rule
        ]
        scopes: list[tuple[str, str, list[Mapping[str, Any]]]] = [
            ("all_inputs", "all", method_rows)
        ]
        for input_id in sorted({str(row["input_id"]) for row in method_rows}):
            scopes.append(
                (
                    "input",
                    input_id,
                    [row for row in method_rows if row["input_id"] == input_id],
                )
            )
        for input_id, dataset in sorted(
            {(str(row["input_id"]), str(row["dataset"])) for row in method_rows}
        ):
            scopes.append(
                (
                    "dataset",
                    f"{input_id}:{dataset}",
                    [
                        row
                        for row in method_rows
                        if row["input_id"] == input_id and row["dataset"] == dataset
                    ],
                )
            )
        for scope, scope_id, rows in scopes:
            output.append(
                {
                    "scope": scope,
                    "scope_id": scope_id,
                    "aggregation": "equal_weight_dataset_source_block_groups",
                    "score_variant": score_name,
                    "prediction_rule": rule,
                    "group_count": len(rows),
                    "total_sample_count": sum(int(row["sample_count"]) for row in rows),
                    "total_positive_count": sum(int(row["positive_count"]) for row in rows),
                    **{name: _finite_macro(rows, name) for name in _METRIC_NAMES},
                    **{
                        f"{name}_valid_group_count": _finite_count(rows, name)
                        for name in _METRIC_NAMES
                    },
                }
            )
    return output


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Execute the immutable two-phase diagnostic and return its completion marker."""

    plan = load_plan(config_path)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    _atomic_bytes(root / "frozen_config.yaml", plan.config_path.read_bytes())

    # Phase 1: the projection deliberately excludes each input's reference column.
    groups, input_evidence = _read_prediction_groups(plan)
    _compute_predictions(plan, groups)
    prediction_fields = _prediction_fields(plan)
    predictions_sha = _atomic_csv(
        root / PREDICTIONS_NAME,
        prediction_fields,
        _prediction_rows(plan, groups),
    )
    prediction_manifest = _prediction_manifest(
        plan, groups, input_evidence, predictions_sha
    )
    _atomic_json(root / PREDICTION_MANIFEST_NAME, prediction_manifest)

    # Phase 2 starts only after both immutable prediction artifacts are closed.
    references = _read_reference_rows(plan)
    _join_reference_rows(groups, references)
    per_group_rows = _per_group_metric_rows(plan, groups)
    aggregate_rows = _aggregate_metric_rows(per_group_rows)
    oracle_rows = _oracle_upper_bound_rows(plan, groups)

    per_group_fields = [
        "input_id",
        "dataset",
        "source_ordinal",
        "block",
        "score_variant",
        "sigma_grid_indices",
        "prediction_rule",
        "sample_count",
        "positive_count",
        "negative_count",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "single_class_group",
        *_METRIC_NAMES,
    ]
    aggregate_fields = [
        "scope",
        "scope_id",
        "aggregation",
        "score_variant",
        "prediction_rule",
        "group_count",
        "total_sample_count",
        "total_positive_count",
        *_METRIC_NAMES,
        *(f"{name}_valid_group_count" for name in _METRIC_NAMES),
    ]
    oracle_fields = [
        "diagnostic_only",
        "selection_data",
        "threshold_comparison",
        "input_id",
        "dataset",
        "source_ordinal",
        "block",
        "score_variant",
        "sigma_grid_indices",
        "best_f1_threshold",
        "predicted_positive_count",
        "sample_count",
        "positive_count",
        "negative_count",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "single_class_group",
        *_METRIC_NAMES,
    ]
    _atomic_csv(root / "per_group_metrics.csv", per_group_fields, per_group_rows)
    _atomic_csv(root / "aggregate_metrics.csv", aggregate_fields, aggregate_rows)
    _atomic_csv(root / "oracle_upper_bound.csv", oracle_fields, oracle_rows)

    artifact_names = (
        "frozen_config.yaml",
        PREDICTIONS_NAME,
        PREDICTION_MANIFEST_NAME,
        "per_group_metrics.csv",
        "aggregate_metrics.csv",
        "oracle_upper_bound.csv",
    )
    result_manifest: dict[str, Any] = {
        "schema": "pathline_template_matching.negative_distance_spatial_result.v1",
        "experiment": EXPERIMENT,
        "status": "complete",
        "evidence_status": "exposed_development_diagnostic_completed",
        "claim_scope": "unlabeled_postprocessing_diagnostic_only",
        "oracle_threshold_selection": "diagnostic_only_same_group_reference_not_a_method",
        "git_commit": _git_commit(),
        "config_sha256": plan.config_sha256,
        "prediction_manifest_file_sha256": sha256_file(
            root / PREDICTION_MANIFEST_NAME
        ),
        "prediction_phase_completed_before_explicit_reference_projection": True,
        "reference_join": "exact_stable_row_key_no_missing_or_extra_rows",
        "aggregation": "equal_weight_dataset_source_block_groups",
        "group_count": len(groups),
        "row_count": sum(len(group.row_keys) for group in groups),
        "per_group_metric_row_count": len(per_group_rows),
        "aggregate_metric_row_count": len(aggregate_rows),
        "oracle_diagnostic_row_count": len(oracle_rows),
        "artifacts": [
            {
                "path": name,
                "file_size": (root / name).stat().st_size,
                "sha256": sha256_file(root / name),
            }
            for name in artifact_names
        ],
    }
    result_manifest["manifest_content_sha256"] = canonical_json_sha256(
        result_manifest
    )
    _atomic_json(root / "result_manifest.json", result_manifest)

    complete: dict[str, Any] = {
        "schema": "pathline_template_matching.run_complete.v1",
        "experiment": EXPERIMENT,
        "status": "complete",
        "result_manifest_file_sha256": sha256_file(root / "result_manifest.json"),
        "prediction_manifest_file_sha256": sha256_file(
            root / PREDICTION_MANIFEST_NAME
        ),
    }
    complete["marker_content_sha256"] = canonical_json_sha256(complete)
    _atomic_json(root / "RUN_COMPLETE.json", complete)
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Exact one-nearest-neighbor template library with frozen preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


_SCHEMA_VERSION = 1


def _strict_binary_labels(labels: np.ndarray, *, context: str) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError(f"{context} labels must be one-dimensional")
    if values.dtype.kind not in "buif" or not np.isfinite(values).all():
        raise ValueError(f"{context} labels must contain only finite 0/1 values")
    if not np.all(np.isin(values, (0, 1, False, True))):
        raise ValueError(f"{context} labels must contain only 0/1 values")
    return values.astype(bool, copy=False)


@dataclass(frozen=True)
class MatchResult:
    """One-nearest-neighbor labels, retrieval evidence, and a ranking score."""

    labels: np.ndarray
    scores: np.ndarray
    nearest_indices: np.ndarray
    nearest_distances: np.ndarray
    nearest_positive_distances: np.ndarray
    nearest_negative_distances: np.ndarray
    nearest_metadata: tuple[dict[str, object], ...]


class TemplateLibrary:
    """A parameter-free classifier over a labeled FMT feature library.

    Feature mean and standard deviation are fitted on library entries only.
    Prediction is exact Euclidean one-nearest-neighbor after that transform.
    The continuous score is ``distance_to_negative - distance_to_positive``;
    its sign gives the same binary decision as global one-nearest-neighbor.
    """

    def __init__(
        self,
        standardized_features: np.ndarray,
        labels: np.ndarray,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        metadata: Iterable[Mapping[str, object]],
        *,
        descriptor_id: str,
    ) -> None:
        self.features = np.ascontiguousarray(standardized_features, dtype=np.float32)
        self.labels = _strict_binary_labels(labels, context="stored")
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float32)
        self.metadata = tuple(dict(item) for item in metadata)
        self.descriptor_id = str(descriptor_id)
        self._validate()

    @classmethod
    def build(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        metadata: Iterable[Mapping[str, object]] | None = None,
        *,
        descriptor_id: str,
        minimum_scale: float = 1e-12,
    ) -> "TemplateLibrary":
        """Build a library; query or confirmation data are never accepted here."""

        values = np.asarray(features, dtype=np.float64)
        targets = _strict_binary_labels(labels, context="library")
        if values.ndim != 2 or len(values) != len(targets) or len(values) < 2:
            raise ValueError("features must be [N,D] with one label per row and N>=2")
        if not np.isfinite(values).all():
            raise ValueError("library features contain NaN or Inf")
        if np.unique(targets).size != 2:
            raise ValueError("a binary library must contain both vortex and non-vortex templates")
        rows = tuple({} for _ in range(len(values))) if metadata is None else tuple(metadata)
        if len(rows) != len(values):
            raise ValueError("metadata must contain one mapping per library row")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < float(minimum_scale)] = 1.0
        standardized = (values - mean) / scale
        return cls(
            standardized,
            targets,
            mean,
            scale,
            rows,
            descriptor_id=descriptor_id,
        )

    def _validate(self) -> None:
        if self.features.ndim != 2 or len(self.features) != len(self.labels):
            raise ValueError("stored features and labels disagree")
        if self.features.shape[1:] != self.feature_mean.shape or self.feature_mean.shape != self.feature_scale.shape:
            raise ValueError("stored normalization width disagrees with feature width")
        if len(self.metadata) != len(self.features):
            raise ValueError("stored metadata row count disagrees with features")
        if not np.isfinite(self.features).all() or not np.isfinite(self.feature_mean).all():
            raise ValueError("stored library contains NaN or Inf")
        if not np.isfinite(self.feature_scale).all() or np.any(self.feature_scale <= 0):
            raise ValueError("stored feature scales must be positive and finite")
        if np.unique(self.labels).size != 2:
            raise ValueError("stored library must contain both labels")

    def transform_queries(self, features: np.ndarray, *, descriptor_id: str) -> np.ndarray:
        if str(descriptor_id) != self.descriptor_id:
            raise ValueError(
                "query/library descriptor mismatch: "
                f"query={descriptor_id!r}, library={self.descriptor_id!r}"
            )
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.features.shape[1]:
            raise ValueError(
                f"query features must be [N,{self.features.shape[1]}], got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("query features contain NaN or Inf")
        return np.ascontiguousarray(
            (values - self.feature_mean) / self.feature_scale,
            dtype=np.float32,
        )

    def query(
        self,
        features: np.ndarray,
        *,
        descriptor_id: str,
        query_chunk_size: int = 1024,
        library_chunk_size: int = 8192,
    ) -> MatchResult:
        """Return exact matches while bounding the temporary distance matrix."""

        queries = self.transform_queries(features, descriptor_id=descriptor_id)
        query_chunk_size = int(query_chunk_size)
        library_chunk_size = int(library_chunk_size)
        if query_chunk_size <= 0 or library_chunk_size <= 0:
            raise ValueError("chunk sizes must be positive")

        count = len(queries)
        class_squared = np.full((count, 2), np.inf, dtype=np.float64)
        class_indices = np.full((count, 2), -1, dtype=np.int64)
        for query_start in range(0, count, query_chunk_size):
            query_stop = min(count, query_start + query_chunk_size)
            query_block = queries[query_start:query_stop].astype(np.float64, copy=False)
            query_norm = np.einsum("ij,ij->i", query_block, query_block)[:, None]
            for library_start in range(0, len(self.features), library_chunk_size):
                library_stop = min(len(self.features), library_start + library_chunk_size)
                library_block = self.features[library_start:library_stop].astype(
                    np.float64, copy=False
                )
                library_norm = np.einsum("ij,ij->i", library_block, library_block)[None]
                squared = query_norm + library_norm - 2.0 * query_block @ library_block.T
                np.maximum(squared, 0.0, out=squared)

                block_labels = self.labels[library_start:library_stop]
                target_rows = np.arange(query_start, query_stop)
                for class_id in (0, 1):
                    mask = block_labels == bool(class_id)
                    if mask.any():
                        class_columns = np.flatnonzero(mask)
                        class_block = squared[:, class_columns]
                        local_columns = class_block.argmin(axis=1)
                        local_class = class_block[
                            np.arange(len(query_block)), local_columns
                        ]
                        improve = local_class < class_squared[target_rows, class_id]
                        improved_rows = target_rows[improve]
                        class_squared[improved_rows, class_id] = local_class[improve]
                        class_indices[improved_rows, class_id] = (
                            library_start + class_columns[local_columns[improve]]
                        )

        if np.any(class_indices < 0) or not np.isfinite(class_squared).all():
            raise RuntimeError("nearest-neighbor search did not cover both classes")
        class_distances = np.sqrt(class_squared)
        scores = class_distances[:, 0] - class_distances[:, 1]
        predicted = scores > 0.0
        nearest_indices = class_indices[np.arange(count), predicted.astype(np.int8)]
        nearest_distances = class_distances[
            np.arange(count), predicted.astype(np.int8)
        ]
        return MatchResult(
            labels=predicted,
            scores=scores.astype(np.float32),
            nearest_indices=nearest_indices,
            nearest_distances=nearest_distances.astype(np.float32),
            nearest_positive_distances=class_distances[:, 1].astype(np.float32),
            nearest_negative_distances=class_distances[:, 0].astype(np.float32),
            nearest_metadata=tuple(self.metadata[index] for index in nearest_indices),
        )

    def save(self, path: str | Path) -> Path:
        """Save without pickle; metadata is canonical JSON."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata_json = json.dumps(self.metadata, sort_keys=True, separators=(",", ":"))
        np.savez_compressed(
            path,
            schema_version=np.asarray(_SCHEMA_VERSION, dtype=np.int16),
            descriptor_id=np.asarray(self.descriptor_id),
            standardized_features=self.features,
            labels=self.labels,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            metadata_json=np.asarray(metadata_json),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TemplateLibrary":
        with np.load(Path(path), allow_pickle=False) as data:
            version = int(data["schema_version"])
            if version != _SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported library schema {version}; expected {_SCHEMA_VERSION}"
                )
            metadata = json.loads(str(data["metadata_json"]))
            return cls(
                data["standardized_features"],
                data["labels"],
                data["feature_mean"],
                data["feature_scale"],
                metadata,
                descriptor_id=str(data["descriptor_id"]),
            )

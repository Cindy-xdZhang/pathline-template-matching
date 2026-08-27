"""Exhaustive class-aware one-nearest-neighbor search on CPU or CUDA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class ExhaustiveMatchResult:
    """Prediction and nearest-template evidence for every query row."""

    labels: np.ndarray
    scores: np.ndarray
    nearest_indices: np.ndarray
    nearest_distances: np.ndarray
    nearest_positive_distances: np.ndarray
    nearest_negative_distances: np.ndarray


class ExhaustiveOneNearestNeighbor:
    """Library-normalized exhaustive 1NN with a CUDA implementation.

    "Exact" means that every query is compared with every eligible template;
    no approximate index or candidate pruning is used. Arithmetic is float32,
    matching the cached feature precision. Equal class distances are assigned
    to non-vortex exactly as in :class:`TemplateLibrary`.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        device: str | torch.device = "cpu",
        minimum_scale: float = 1e-12,
    ) -> None:
        values = np.asarray(features, dtype=np.float32)
        targets = np.asarray(labels)
        if values.ndim != 2 or len(values) < 2 or targets.shape != (len(values),):
            raise ValueError("library features must be [N,D] with one label per row")
        if not np.isfinite(values).all() or not np.all(np.isin(targets, (0, 1))):
            raise ValueError("library values must be finite and labels binary")
        targets = targets.astype(bool, copy=False)
        if np.unique(targets).size != 2:
            raise ValueError("library must contain both binary classes")
        # Compute the library statistics accurately, then freeze them at the
        # float32 precision used by both cache features and CUDA inference.
        # Library and query rows must pass through exactly the same arithmetic.
        values64 = values.astype(np.float64)
        self.feature_mean = values64.mean(axis=0).astype(np.float32)
        scale = values64.std(axis=0)
        scale[scale < float(minimum_scale)] = 1.0
        self.feature_scale = scale.astype(np.float32)
        standardized = np.ascontiguousarray(
            (values - self.feature_mean) / self.feature_scale,
            dtype=np.float32,
        )
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA matcher requested but torch.cuda.is_available() is false")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        self.width = standardized.shape[1]
        self.count = len(standardized)
        self._class_features: dict[int, torch.Tensor] = {}
        self._class_indices: dict[int, torch.Tensor] = {}
        for class_id in (0, 1):
            indices = np.flatnonzero(targets == bool(class_id)).astype(np.int64)
            self._class_features[class_id] = torch.from_numpy(
                standardized[indices]
            ).to(self.device)
            self._class_indices[class_id] = torch.from_numpy(indices).to(self.device)

    def transform(self, queries: np.ndarray) -> np.ndarray:
        values = np.asarray(queries, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.width:
            raise ValueError(f"query features must be [N,{self.width}], got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("query features contain NaN or Inf")
        return np.ascontiguousarray(
            (values - self.feature_mean) / self.feature_scale,
            dtype=np.float32,
        )

    @staticmethod
    def _distance(query: torch.Tensor, library: torch.Tensor) -> torch.Tensor:
        # The matrix-expansion identity ||q||^2 + ||x||^2 - 2 q.x suffers
        # catastrophic cancellation for identical or nearly identical 672D
        # rows in float32.  The direct Euclidean kernel preserves exact zero
        # self-distance while still exhaustively evaluating every pair.
        return torch.cdist(
            query,
            library,
            p=2.0,
            compute_mode="donot_use_mm_for_euclid_dist",
        )

    def query(
        self,
        queries: np.ndarray,
        *,
        query_chunk_size: int = 1024,
        library_chunk_size: int = 8192,
    ) -> ExhaustiveMatchResult:
        transformed = self.transform(queries)
        query_chunk_size = int(query_chunk_size)
        library_chunk_size = int(library_chunk_size)
        if query_chunk_size < 1 or library_chunk_size < 1:
            raise ValueError("query and library chunk sizes must be positive")
        class_distances = np.full((len(transformed), 2), np.inf, dtype=np.float32)
        class_indices = np.full((len(transformed), 2), -1, dtype=np.int64)
        with torch.inference_mode():
            for query_start in range(0, len(transformed), query_chunk_size):
                query_stop = min(len(transformed), query_start + query_chunk_size)
                query_tensor = torch.from_numpy(
                    transformed[query_start:query_stop]
                ).to(self.device)
                for class_id in (0, 1):
                    features = self._class_features[class_id]
                    indices = self._class_indices[class_id]
                    best_distance = torch.full(
                        (len(query_tensor),),
                        torch.inf,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    best_indices = torch.full(
                        (len(query_tensor),),
                        -1,
                        dtype=torch.int64,
                        device=self.device,
                    )
                    for library_start in range(0, len(features), library_chunk_size):
                        library_stop = min(
                            len(features), library_start + library_chunk_size
                        )
                        distance = self._distance(
                            query_tensor, features[library_start:library_stop]
                        )
                        local_distance, local_index = torch.min(distance, dim=1)
                        improve = local_distance < best_distance
                        best_distance[improve] = local_distance[improve]
                        global_indices = indices[library_start:library_stop][local_index]
                        best_indices[improve] = global_indices[improve]
                    class_distances[query_start:query_stop, class_id] = (
                        best_distance.cpu().numpy()
                    )
                    class_indices[query_start:query_stop, class_id] = (
                        best_indices.cpu().numpy()
                    )
        if np.any(class_indices < 0) or not np.isfinite(class_distances).all():
            raise RuntimeError("exhaustive search failed to cover both classes")
        scores = class_distances[:, 0] - class_distances[:, 1]
        predicted = scores > 0.0
        rows = np.arange(len(predicted))
        nearest_indices = class_indices[rows, predicted.astype(np.int8)]
        nearest_distances = class_distances[rows, predicted.astype(np.int8)]
        return ExhaustiveMatchResult(
            labels=predicted,
            scores=scores,
            nearest_indices=nearest_indices,
            nearest_distances=nearest_distances,
            nearest_positive_distances=class_distances[:, 1],
            nearest_negative_distances=class_distances[:, 0],
        )

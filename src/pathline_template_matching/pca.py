"""Library-only deterministic Principal Component Analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeterministicPCA:
    """Principal Component Analysis fitted by a full deterministic SVD."""

    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    sample_count: int
    input_width: int

    @classmethod
    def fit(cls, library_features: np.ndarray, components: int) -> "DeterministicPCA":
        values = np.asarray(library_features, dtype=np.float64)
        count = int(components)
        if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("PCA library features must be a finite [N,D] matrix with N>=2")
        if not 1 <= count <= min(values.shape):
            raise ValueError(
                f"PCA component count must be in [1,{min(values.shape)}], got {count}"
            )
        mean = values.mean(axis=0)
        centered = values - mean
        _, singular_values, right_vectors = np.linalg.svd(
            centered, full_matrices=False
        )
        selected = right_vectors[:count].copy()
        # Freeze the otherwise arbitrary SVD sign by making the largest-magnitude
        # loading in every component non-negative.
        pivots = np.argmax(np.abs(selected), axis=1)
        signs = np.sign(selected[np.arange(count), pivots])
        signs[signs == 0] = 1.0
        selected *= signs[:, None]
        variances = singular_values * singular_values
        total = float(variances.sum())
        explained = variances[:count] / total if total > 0 else np.zeros(count)
        return cls(
            mean=mean.astype(np.float32),
            components=selected.astype(np.float32),
            singular_values=singular_values[:count].astype(np.float64),
            explained_variance_ratio=explained.astype(np.float64),
            sample_count=len(values),
            input_width=values.shape[1],
        )

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.input_width:
            raise ValueError(
                f"PCA input must be [N,{self.input_width}], got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("PCA input contains NaN or Inf")
        return np.ascontiguousarray(
            (values - self.mean) @ self.components.T,
            dtype=np.float32,
        )

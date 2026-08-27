"""Frozen, per-primitive FMT descriptor used by the first retrieval baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import torch

from .fmt_descriptor import pathline_dft_features_3d


@dataclass(frozen=True)
class IndependentFMT3DConfig:
    """Configuration for the 161-dimensional independent 3D FMT descriptor.

    The default has no learned parameters and no cross-sample statistics. One
    primitive therefore receives the same descriptor alone or inside a batch.
    """

    algorithm_source_sha256: str = (
        "7ff09c1b578d0bd0927ccf0b771de30a01e0b79fff1b7b40576cbca57ce1e6c1"
    )
    num_freq: int = 6
    neighbor_weight: float = 1.0
    neighbor_scale: float = 1.0
    neighbor_pool: str = "sort"
    mode: str = "gram"
    include_chirality: bool = True

    @property
    def descriptor_id(self) -> str:
        """Content-derived identity; any numeric recipe change gets a new ID."""

        identity = {
            "input_contract": {"line_count": 7, "sampled_points": 32, "channels": ["x", "y", "z"]},
            **asdict(self),
        }
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"fmt_independent_3d_{self.feature_width()}d_sha256_{digest}"

    def feature_width(self) -> int:
        if self.mode == "gram":
            block_width = 3 * self.num_freq
        elif self.mode == "magnitude":
            block_width = self.num_freq
        else:
            raise ValueError(f"unsupported mode: {self.mode!r}")
        if self.include_chirality:
            block_width += self.num_freq - 1
        if self.neighbor_pool == "sort":
            return 7 * block_width
        if self.neighbor_pool in {"mean", "max"}:
            return 2 * block_width
        if self.neighbor_pool == "none":
            return 7 * block_width
        raise ValueError(f"unsupported neighbor_pool: {self.neighbor_pool!r}")

    def to_metadata(self) -> dict[str, object]:
        return {
            "descriptor_id": self.descriptor_id,
            "line_count": 7,
            "sampled_points": 32,
            "coordinate_channels": ["x", "y", "z"],
            **asdict(self),
        }


def encode_independent_fmt_3d(
    primitives: np.ndarray | torch.Tensor,
    config: IndependentFMT3DConfig | None = None,
) -> np.ndarray:
    """Encode ``[N,7,L,3 or 4]`` primitives without batch-dependent statistics."""

    config = config or IndependentFMT3DConfig()
    shape = tuple(primitives.shape)
    if len(shape) != 4 or shape[1] != 7 or shape[2] != 32 or shape[3] not in (3, 4):
        raise ValueError(
            "the frozen descriptor requires [N,7,32,3 or 4] primitives, "
            f"got {shape}"
        )
    features = pathline_dft_features_3d(
        primitives,
        num_freq=config.num_freq,
        neighbor_weight=config.neighbor_weight,
        neighbor_scale=config.neighbor_scale,
        neighbor_pool=config.neighbor_pool,
        mode=config.mode,
        include_chirality=config.include_chirality,
        return_numpy=True,
    )
    features = np.asarray(features, dtype=np.float32)
    expected = config.feature_width()
    if features.ndim != 2 or features.shape[1] != expected:
        raise RuntimeError(
            f"descriptor width changed: expected {expected}, got {features.shape}"
        )
    if not np.isfinite(features).all():
        raise ValueError("FMT descriptor contains NaN or Inf")
    return features

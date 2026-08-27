"""Run a deterministic, non-scientific end-to-end descriptor/retrieval smoke test."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching import (
    IndependentFMT3DConfig,
    TemplateLibrary,
    encode_independent_fmt_3d,
)


def _primitives() -> np.ndarray:
    rng = np.random.default_rng(20260827)
    increments = rng.normal(size=(12, 7, 31, 3)).astype(np.float32)
    points = np.concatenate(
        (np.zeros((12, 7, 1, 3), dtype=np.float32), increments), axis=2
    ).cumsum(axis=2)
    offsets = np.asarray(
        [
            [0, 0, 0], [0.2, 0, 0], [-0.2, 0, 0],
            [0, 0.2, 0], [0, -0.2, 0], [0, 0, 0.2], [0, 0, -0.2],
        ],
        dtype=np.float32,
    )
    return points + offsets[None, :, None, :]


def main() -> None:
    config = IndependentFMT3DConfig()
    features = encode_independent_fmt_3d(_primitives(), config)
    labels = np.asarray([False] * 6 + [True] * 6)
    metadata = ({"template_id": int(index)} for index in range(10))
    library = TemplateLibrary.build(
        features[:10], labels[:10], metadata, descriptor_id=config.descriptor_id
    )
    result = library.query(
        features[10:],
        descriptor_id=config.descriptor_id,
        query_chunk_size=1,
        library_chunk_size=3,
    )
    payload = {
        "descriptor_id": config.descriptor_id,
        "feature_shape": list(features.shape),
        "query_count": len(result.labels),
        "finite_scores": bool(np.isfinite(result.scores).all()),
        "nearest_indices": result.nearest_indices.tolist(),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

import hashlib
from pathlib import Path

import numpy as np
import torch
import yaml

from pathline_template_matching.encoder import (
    IndependentFMT3DConfig,
    encode_independent_fmt_3d,
)
from pathline_template_matching.fmt_descriptor import pathline_dft_features_3d


ROOT = Path(__file__).resolve().parents[1]


def _random_rotation(seed: int = 0) -> torch.Tensor:
    matrix, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1
    return torch.tensor(matrix, dtype=torch.float64)


def _primitives(count: int = 8, length: int = 32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(4)
    velocity = torch.randn(
        count, 7, length - 1, 3, generator=generator, dtype=torch.float64
    )
    points = torch.cat(
        (torch.zeros(count, 7, 1, 3, dtype=torch.float64), velocity), dim=2
    ).cumsum(dim=2)
    offsets = torch.tensor(
        [
            [0, 0, 0], [0.2, 0, 0], [-0.2, 0, 0],
            [0, 0.2, 0], [0, -0.2, 0], [0, 0, 0.2], [0, 0, -0.2],
        ],
        dtype=torch.float64,
    )
    return points + offsets[None, :, None, :]


def test_default_descriptor_is_161d_and_rigid_motion_invariant():
    primitives = _primitives()
    rotation = _random_rotation(17)
    translation = torch.tensor([4.0, -2.0, 7.0], dtype=torch.float64)
    base = encode_independent_fmt_3d(primitives)
    moved = encode_independent_fmt_3d(primitives @ rotation.T + translation)
    assert base.shape == (8, 161)
    np.testing.assert_allclose(base, moved, rtol=2e-5, atol=2e-5)


def test_descriptor_does_not_depend_on_query_batch_composition_or_chunks():
    primitives = _primitives(count=7)
    alone = encode_independent_fmt_3d(primitives[:1])
    mixed = encode_independent_fmt_3d(primitives)[:1]
    chunks = np.concatenate(
        [encode_independent_fmt_3d(primitives[:3]), encode_independent_fmt_3d(primitives[3:])]
    )
    np.testing.assert_array_equal(alone, mixed)
    np.testing.assert_array_equal(chunks, encode_independent_fmt_3d(primitives))


def test_descriptor_metadata_and_width_are_frozen():
    config = IndependentFMT3DConfig()
    assert config.feature_width() == 161
    assert config.to_metadata()["descriptor_id"] == config.descriptor_id
    assert config.neighbor_weight == 1.0
    assert config.neighbor_scale == 1.0
    spec = yaml.safe_load(
        (ROOT / "config/mainExp_TemplateMatching_1.1.yaml").read_text(encoding="utf-8")
    )
    assert spec["descriptor"]["id"] == config.descriptor_id
    assert spec["descriptor"]["source_sha256"] == config.algorithm_source_sha256
    copied_hash = hashlib.sha256(
        (ROOT / "src/pathline_template_matching/fmt_descriptor.py").read_bytes()
    ).hexdigest()
    assert copied_hash == config.algorithm_source_sha256


def test_frozen_descriptor_rejects_wrong_tensor_contract():
    for primitive in (_primitives(length=31), _primitives()[:, :6]):
        try:
            encode_independent_fmt_3d(primitive)
        except ValueError as error:
            assert "[N,7,32" in str(error)
        else:
            raise AssertionError("an incompatible primitive tensor was accepted")


def test_default_numeric_recipe_matches_legacy_task5_cache_encoder():
    primitives = _primitives(count=2).float()
    wrapped = encode_independent_fmt_3d(primitives)
    legacy = pathline_dft_features_3d(
        primitives,
        num_freq=6,
        neighbor_weight=1.0,
        neighbor_scale=1.0,
        neighbor_pool="sort",
        mode="gram",
        include_chirality=True,
        return_numpy=True,
    )
    np.testing.assert_array_equal(wrapped, legacy)

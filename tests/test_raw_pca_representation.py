from __future__ import annotations

import hashlib
from io import BytesIO

import numpy as np

from pathline_template_matching.raw_pca_representation import (
    RAW_INPUT_WIDTH,
    RAW_OUTPUT_WIDTH,
    RAW_PCA_ARRAY_NAMES,
    RawPCARepresentation,
    deserialize_raw_pca,
    fit_raw_pca,
    serialize_raw_pca,
)


def _expect_value_error(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


_FITTED_CASE = None


def _fitted_case():
    global _FITTED_CASE
    if _FITTED_CASE is not None:
        return _FITTED_CASE
    generator = np.random.default_rng(161672)
    values = generator.normal(size=(181, RAW_INPUT_WIDTH)).astype(np.float32)
    values *= np.linspace(0.4, 2.5, RAW_INPUT_WIDTH, dtype=np.float32)
    values[:, 3] += np.float32(0.35) * values[:, 0]
    values[:, 19] -= np.float32(0.22) * values[:, 7]
    blocks = (values[:37], values[37:109], values[109:])
    for block in blocks:
        block.setflags(write=False)
    snapshots = tuple(block.copy() for block in blocks)

    def block_factory():
        return iter(blocks)

    model = fit_raw_pca(block_factory)
    for block, snapshot in zip(blocks, snapshots, strict=True):
        np.testing.assert_array_equal(block, snapshot)
        assert not block.flags.writeable
    _FITTED_CASE = values, blocks, model
    return _FITTED_CASE


def test_raw_pca_matches_full_right_svd_and_freezes_component_sign():
    values, _blocks, model = _fitted_case()
    centered = values.astype(np.float64) - values.astype(np.float64).mean(axis=0)
    _left, singular_values, right_vectors = np.linalg.svd(
        centered, full_matrices=False
    )
    reference = right_vectors[:RAW_OUTPUT_WIDTH].copy()
    pivots = np.argmax(np.abs(reference), axis=1)
    signs = np.sign(reference[np.arange(RAW_OUTPUT_WIDTH), pivots])
    signs[signs == 0.0] = 1.0
    reference *= signs[:, None]

    assert model.sample_count == len(values)
    assert model.input_width == RAW_INPUT_WIDTH
    assert model.output_width == RAW_OUTPUT_WIDTH
    assert model.whitening is False
    np.testing.assert_allclose(
        model.singular_values,
        singular_values[:RAW_OUTPUT_WIDTH],
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        model.components,
        reference.astype(np.float32),
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    model_pivots = np.argmax(np.abs(model.components), axis=1)
    assert np.all(
        model.components[np.arange(RAW_OUTPUT_WIDTH), model_pivots] >= 0.0
    )
    assert not model.mean.flags.writeable
    assert not model.components.flags.writeable


def test_raw_pca_transform_is_separate_unwhitened_float32_and_input_read_only():
    values, _blocks, model = _fitted_case()
    query = values[5:12].copy()
    query.setflags(write=False)
    snapshot = query.copy()
    transformed = model.transform(query)
    expected = np.ascontiguousarray(
        (query.astype(np.float32) - model.mean) @ model.components.T,
        dtype=np.float32,
    )
    np.testing.assert_array_equal(transformed, expected)
    np.testing.assert_array_equal(query, snapshot)
    assert not query.flags.writeable
    assert transformed.dtype == np.float32
    assert transformed.flags.c_contiguous
    assert transformed.shape == (len(query), RAW_OUTPUT_WIDTH)


def test_raw_pca_repeated_fit_is_bitwise_deterministic():
    _values, blocks, first = _fitted_case()

    def block_factory():
        return iter(blocks)

    second = fit_raw_pca(block_factory)
    for name in RAW_PCA_ARRAY_NAMES:
        np.testing.assert_array_equal(
            first.export_arrays()[name], second.export_arrays()[name]
        )


def test_raw_pca_serialization_round_trip_is_bitwise_and_rejects_tamper():
    _values, _blocks, model = _fitted_case()
    serialized = serialize_raw_pca(model)
    restored = deserialize_raw_pca(
        serialized.payload,
        expected_npz_sha256=serialized.npz_sha256,
        expected_array_audits=serialized.array_audits,
    )
    for name in RAW_PCA_ARRAY_NAMES:
        np.testing.assert_array_equal(
            model.export_arrays()[name], restored.export_arrays()[name]
        )

    damaged_bytes = bytearray(serialized.payload)
    damaged_bytes[len(damaged_bytes) // 2] ^= 1
    _expect_value_error(
        deserialize_raw_pca,
        bytes(damaged_bytes),
        expected_npz_sha256=serialized.npz_sha256,
        expected_array_audits=serialized.array_audits,
    )

    tampered_arrays = model.export_arrays()
    tampered_arrays["mean_float32"][0] += np.float32(1.0)
    buffer = BytesIO()
    np.savez_compressed(buffer, **tampered_arrays)
    tampered_payload = buffer.getvalue()
    _expect_value_error(
        deserialize_raw_pca,
        tampered_payload,
        expected_npz_sha256=hashlib.sha256(tampered_payload).hexdigest(),
        expected_array_audits=serialized.array_audits,
    )


def test_raw_pca_rejects_schema_shape_dtype_finite_and_transform_errors():
    _values, _blocks, model = _fitted_case()
    arrays = model.export_arrays()

    missing = dict(arrays)
    missing.pop("mean_float32")
    _expect_value_error(RawPCARepresentation.from_arrays, missing)

    wrong_shape = {name: value.copy() for name, value in arrays.items()}
    wrong_shape["components_float32"] = wrong_shape["components_float32"][:, :-1]
    _expect_value_error(RawPCARepresentation.from_arrays, wrong_shape)

    wrong_dtype = {name: value.copy() for name, value in arrays.items()}
    wrong_dtype["mean_float32"] = wrong_dtype["mean_float32"].astype(np.float64)
    _expect_value_error(RawPCARepresentation.from_arrays, wrong_dtype)

    nonfinite = {name: value.copy() for name, value in arrays.items()}
    nonfinite["singular_values_float64"][0] = np.nan
    _expect_value_error(RawPCARepresentation.from_arrays, nonfinite)

    _expect_value_error(
        model.transform,
        np.zeros((2, RAW_INPUT_WIDTH - 1), dtype=np.float32),
    )
    bad_query = np.zeros((2, RAW_INPUT_WIDTH), dtype=np.float32)
    bad_query[0, 0] = np.inf
    _expect_value_error(model.transform, bad_query)


def test_raw_pca_fit_rejects_nonfinite_shape_and_changed_second_pass_count():
    good = np.zeros((RAW_OUTPUT_WIDTH, RAW_INPUT_WIDTH), dtype=np.float32)
    bad_shape = np.zeros((RAW_OUTPUT_WIDTH, RAW_INPUT_WIDTH - 1), dtype=np.float32)
    _expect_value_error(fit_raw_pca, lambda: iter((bad_shape,)))

    nonfinite = good.copy()
    nonfinite[0, 0] = np.nan
    _expect_value_error(fit_raw_pca, lambda: iter((nonfinite,)))

    call_count = 0

    def changing_factory():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return iter((good,))
        return iter((good[:-1],))

    _expect_value_error(fit_raw_pca, changing_factory)


def test_raw_pca_accepts_frozen_two_row_minimum_and_rejects_fit_dtype_drift():
    two_rows = np.zeros((2, RAW_INPUT_WIDTH), dtype=np.float32)
    model = fit_raw_pca(lambda: iter((two_rows,)))
    assert model.sample_count == 2
    assert model.components.shape == (RAW_OUTPUT_WIDTH, RAW_INPUT_WIDTH)

    _expect_value_error(
        fit_raw_pca,
        lambda: iter((two_rows.astype(np.float64),)),
    )
    _expect_value_error(
        fit_raw_pca,
        lambda: iter((np.zeros_like(two_rows, dtype=np.int32),)),
    )

    model = fit_raw_pca(lambda: iter((two_rows,)))
    _expect_value_error(model.transform, two_rows.astype(np.float64))
    _expect_value_error(
        model.transform,
        np.zeros_like(two_rows, dtype=np.int32),
    )


def test_authenticated_model_arrays_cannot_be_made_writeable():
    _values, _blocks, model = _fitted_case()
    for array in (
        model.mean,
        model.components,
        model.singular_values,
        model.explained_variance_ratio,
    ):
        assert not array.flags.writeable
        try:
            array.setflags(write=True)
        except ValueError:
            pass
        else:
            raise AssertionError("authenticated Raw-PCA array became writeable")

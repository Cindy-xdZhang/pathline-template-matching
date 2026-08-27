import numpy as np
import tempfile
from pathlib import Path

from pathline_template_matching.library import TemplateLibrary


def _library():
    features = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]],
        dtype=np.float32,
    )
    labels = np.asarray([False, False, True, True])
    metadata = [{"id": index} for index in range(4)]
    return TemplateLibrary.build(
        features, labels, metadata, descriptor_id="test_descriptor"
    )


def test_exact_one_nearest_neighbor_and_signed_class_margin():
    library = _library()
    result = library.query(
        np.asarray([[0.1, 0.2], [9.9, 10.2]], dtype=np.float32),
        descriptor_id="test_descriptor",
        query_chunk_size=1,
        library_chunk_size=1,
    )
    np.testing.assert_array_equal(result.labels, [False, True])
    assert result.scores[0] < 0 < result.scores[1]
    assert result.nearest_metadata == ({"id": 0}, {"id": 2})


def test_query_does_not_refit_library_normalization():
    library = _library()
    before_mean = library.feature_mean.copy()
    before_scale = library.feature_scale.copy()
    library.query(
        np.asarray([[1e9, -1e9]], dtype=np.float32),
        descriptor_id="test_descriptor",
    )
    np.testing.assert_array_equal(library.feature_mean, before_mean)
    np.testing.assert_array_equal(library.feature_scale, before_scale)


def test_cross_class_distance_tie_is_deterministically_non_vortex():
    library = TemplateLibrary.build(
        np.asarray([[1.0], [-1.0]], dtype=np.float32),
        np.asarray([True, False]),
        [{"id": "positive"}, {"id": "negative"}],
        descriptor_id="tie_test",
    )
    result = library.query(
        np.asarray([[0.0]], dtype=np.float32),
        descriptor_id="tie_test",
        library_chunk_size=1,
    )
    assert not result.labels[0]
    assert result.scores[0] == 0.0
    assert result.nearest_metadata[0]["id"] == "negative"


def test_library_round_trip_without_pickle():
    with tempfile.TemporaryDirectory() as directory:
        original = _library()
        path = original.save(Path(directory) / "library.npz")
        restored = TemplateLibrary.load(path)
        assert restored.descriptor_id == original.descriptor_id
        np.testing.assert_array_equal(restored.features, original.features)
        np.testing.assert_array_equal(restored.labels, original.labels)
        assert restored.metadata == original.metadata


def test_query_descriptor_mismatch_and_nonbinary_labels_are_rejected():
    library = _library()
    try:
        library.query(np.asarray([[0.0, 0.0]]), descriptor_id="wrong")
    except ValueError as error:
        assert "descriptor mismatch" in str(error)
    else:
        raise AssertionError("descriptor mismatch was accepted")

    try:
        TemplateLibrary.build(
            np.asarray([[0.0], [1.0]]),
            np.asarray([0, 2]),
            descriptor_id="bad_labels",
        )
    except ValueError as error:
        assert "0/1" in str(error)
    else:
        raise AssertionError("non-binary labels were accepted")

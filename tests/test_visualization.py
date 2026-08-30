from __future__ import annotations

from pathlib import Path
import struct
import tempfile

from matplotlib import image as mpl_image
import numpy as np

from pathline_template_matching.visualization import (
    confusion_masks,
    render_template_matching_triptych,
    validate_scene,
)


def _scene(*, include_ivd_points: bool = True) -> dict[str, object]:
    seeds = np.asarray(
        [
            [-0.75, -0.75, -0.75],
            [0.75, -0.75, -0.25],
            [-0.50, 0.50, 0.25],
            [0.60, 0.60, 0.75],
        ],
        dtype=np.float32,
    )
    ivd_points = None
    if include_ivd_points:
        coordinates = np.linspace(-0.9, 0.9, 20, dtype=np.float32)
        ivd_points = np.column_stack(
            (coordinates, coordinates[::-1], np.zeros(20), np.arange(20))
        ).astype(np.float32)
    return {
        "dataset": "synthetic",
        "title": "Synthetic audit scene",
        "regime": "confirmation",
        "source_ordinal": 2,
        "bounds": np.asarray([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
        "seeds": seeds,
        "reference": np.asarray([False, False, True, True]),
        "prediction": np.asarray([False, True, False, True]),
        "display_pathlines": [
            np.asarray(
                [[-0.8, -0.8, -0.8, 10.0], [0.0, 0.0, 0.0, 10.5]],
                dtype=np.float32,
            ),
            np.asarray(
                [[0.8, 0.8, 0.8, 20.0], [0.4, 0.2, 0.0, 20.25]],
                dtype=np.float32,
            ),
        ],
        "ivd_points": ivd_points,
    }


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    assert header[12:16] == b"IHDR"
    return struct.unpack(">II", header[16:24])


def test_triptych_writes_png_counts_and_identical_camera_metadata():
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "triptych.png"
        path, metadata = render_template_matching_triptych(
            _scene(), output, dpi=24, view=(20.0, -60.0)
        )

        assert path == output
        assert path.exists()
        assert _png_dimensions(path) == (21 * 24, 5 * 24)
        pixels = mpl_image.imread(path)
        assert np.all(pixels[0, :, :3] >= 0.99)
        assert metadata["layout"]["top_margin_fraction"] == 0.08
        assert metadata["panel_order"][1] == "FMT exact-1NN class assignment"
        assert metadata["prediction_semantics"] == (
            "precomputed FMT exact-1NN binary assignment"
        )
        assert metadata["counts"]["sample_count"] == 4
        assert metadata["counts"]["true_negative"] == 1
        assert metadata["counts"]["true_positive"] == 1
        assert metadata["counts"]["false_positive"] == 1
        assert metadata["counts"]["false_negative"] == 1
        assert metadata["counts"]["ivd_input_point_count"] == 20
        assert metadata["counts"]["ivd_rendered_high_point_count"] == 1
        assert metadata["fallback"]["used"] is False
        assert metadata["render_sampling"]["original_seed_count"] == 4
        assert metadata["render_sampling"]["panel_one_context_seed_count"] == 4
        assert metadata["render_sampling"][
            "rendered_seed_count_per_assignment_panel"
        ] == 4
        cameras = metadata["camera"]["panels"]
        assert metadata["camera"]["identical_across_panels"] is True
        assert cameras[0] == cameras[1] == cameras[2]
        assert cameras[0]["projection"] == "orthographic"
        assert cameras[0]["elevation_degrees"] == 20.0
        assert cameras[0]["azimuth_degrees"] == -60.0


def test_triptych_optional_panel_and_semantics_override_is_explicit_and_default_safe():
    titles = (
        "IVD p95 + center pathlines",
        "FMT negative-distance + spatial top-5%",
        "TP / FP / FN / TN against IVD p95",
    )
    semantics = "fixed negative-distance spatial top-5-percent assignment"
    with tempfile.TemporaryDirectory() as directory:
        _, metadata = render_template_matching_triptych(
            _scene(),
            Path(directory) / "overridden.png",
            dpi=20,
            panel_titles=titles,
            prediction_semantics=semantics,
        )
    assert metadata["panel_order"] == list(titles)
    assert metadata["prediction_semantics"] == semantics

    with tempfile.TemporaryDirectory() as directory:
        _, default = render_template_matching_triptych(
            _scene(), Path(directory) / "default.png", dpi=20
        )
    assert default["panel_order"][1] == "FMT exact-1NN class assignment"
    assert default["prediction_semantics"] == (
        "precomputed FMT exact-1NN binary assignment"
    )


def test_triptych_uses_positive_reference_seeds_when_ivd_points_are_absent():
    with tempfile.TemporaryDirectory() as directory:
        _, metadata = render_template_matching_triptych(
            _scene(include_ivd_points=False),
            Path(directory) / "fallback.png",
            dpi=20,
        )

        assert metadata["fallback"] == {
            "used": True,
            "reason": "ivd_points is None",
            "source": "positive reference seeds",
        }
        assert metadata["counts"]["ivd_input_point_count"] == 0
        assert metadata["counts"]["ivd_rendered_high_point_count"] == 2


def test_triptych_renders_audited_ivd_isosurface_without_fallback():
    scene = _scene(include_ivd_points=False)
    scene["ivd_mesh"] = {
        "vertices": np.asarray(
            [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]],
            dtype=np.float32,
        ),
        "faces": np.asarray([[0, 1, 2]], dtype=np.int64),
        "level": 0.75,
    }
    with tempfile.TemporaryDirectory() as directory:
        _, metadata = render_template_matching_triptych(
            scene, Path(directory) / "mesh.png", dpi=20
        )
    assert metadata["fallback"]["used"] is False
    assert metadata["fallback"]["source"] == "whole-loaded-volume IVD p95 isosurface"
    assert metadata["counts"]["ivd_input_point_count"] == 3
    assert metadata["counts"]["ivd_rendered_high_point_count"] == 1
    assert metadata["render_sampling"]["ivd_threshold"] == 0.75


def test_scene_validation_rejects_invalid_shapes_and_mismatched_seed_copies():
    invalid_seeds = _scene()
    invalid_seeds["seeds"] = np.zeros((4, 2), dtype=np.float32)
    try:
        validate_scene(invalid_seeds)
    except ValueError as error:
        assert "shape [N,3]" in str(error)
    else:
        raise AssertionError("invalid seed shape was accepted")

    invalid_prediction = _scene()
    invalid_prediction["prediction"] = np.asarray([False, True, False])
    try:
        validate_scene(invalid_prediction)
    except ValueError as error:
        assert "one assignment per seed" in str(error)
    else:
        raise AssertionError("short prediction array was accepted")

    mismatched_copy = _scene()
    prediction_seeds = np.asarray(mismatched_copy["seeds"]).copy()
    prediction_seeds[0, 0] += 0.1
    mismatched_copy["prediction_seeds"] = prediction_seeds
    try:
        validate_scene(mismatched_copy)
    except ValueError as error:
        assert "exactly equal" in str(error)
    else:
        raise AssertionError("a prediction on different seeds was accepted")


def test_confusion_masks_are_mutually_exclusive_and_exhaustive():
    reference = np.asarray([False, False, True, True])
    prediction = np.asarray([False, True, False, True])
    masks = confusion_masks(reference, prediction)

    assert {name: int(mask.sum()) for name, mask in masks.items()} == {
        "true_negative": 1,
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
    }
    memberships = np.stack(tuple(masks.values())).sum(axis=0)
    np.testing.assert_array_equal(memberships, np.ones(4, dtype=np.int64))
    assert not np.any(masks["false_positive"] & reference)
    assert not np.any(masks["false_negative"] & ~reference)

    try:
        confusion_masks(reference, prediction[:-1])
    except ValueError as error:
        assert "same-length" in str(error)
    else:
        raise AssertionError("different-length confusion inputs were accepted")

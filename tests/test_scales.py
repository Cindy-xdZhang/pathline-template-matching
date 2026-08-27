from pathlib import Path

import numpy as np
import yaml

from pathline_template_matching.scales import (
    balanced_scale_assignment,
    parse_scale_table,
)


ROOT = Path(__file__).resolve().parents[1]


def test_balanced_assignment_is_reproducible_and_shuffled():
    first = balanced_scale_assignment(101, 9, 123)
    second = balanced_scale_assignment(101, 9, 123)
    np.testing.assert_array_equal(first, second)
    counts = np.bincount(first, minlength=9)
    assert counts.max() - counts.min() == 1
    assert not np.array_equal(first, np.arange(101) % 9)


def test_frozen_scale_sets_are_valid_and_disjoint():
    spec = yaml.safe_load(
        (ROOT / "config/mainExp_TemplateMatching_1.1.yaml").read_text(encoding="utf-8")
    )
    parsed = {
        name: parse_scale_table(rows, spec["primitive"]["sampled_points_per_line"])
        for name, rows in spec["scale_sets"].items()
    }
    assert {name: len(rows) for name, rows in parsed.items()} == {
        "library": 18,
        "descriptor_selection_only": 6,
        "unseen_scale_evaluation": 9,
    }
    tuples = {name: {row.tuple for row in rows} for name, rows in parsed.items()}
    assert tuples["library"].isdisjoint(tuples["descriptor_selection_only"])
    assert tuples["library"].isdisjoint(tuples["unseen_scale_evaluation"])
    assert tuples["descriptor_selection_only"].isdisjoint(
        tuples["unseen_scale_evaluation"]
    )


def test_scale_table_rejects_insufficient_integration_steps():
    rows = [
        {
            "name": "bad",
            "offset_grid_scale": 1.0,
            "dt_scale": 0.25,
            "integration_steps": 20,
        }
    ]
    try:
        parse_scale_table(rows, sampled_steps=32)
    except ValueError as error:
        assert "cannot provide" in str(error)
    else:
        raise AssertionError("invalid scale table was accepted")


def test_scale_table_rejects_duplicate_numeric_tuple():
    rows = [
        {"name": "first", "offset_grid_scale": 1, "dt_scale": 0.25, "integration_steps": 32},
        {"name": "alias", "offset_grid_scale": 1, "dt_scale": 0.25, "integration_steps": 32},
    ]
    try:
        parse_scale_table(rows, sampled_steps=32)
    except ValueError as error:
        assert "duplicate numeric" in str(error)
    else:
        raise AssertionError("duplicate numeric scale tuple was accepted")

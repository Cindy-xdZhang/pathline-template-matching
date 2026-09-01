from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import (
    run_verify_source_centered_rank_likelihood_template_1_1 as runner,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "Verify_SourceCenteredRankLikelihoodTemplate_1.1.yaml"


def test_rank_likelihood_plan_freezes_540_primary_90_control_and_18_files() -> None:
    plan = runner.load_plan(CONFIG)
    primary = runner.candidate_specs(plan)
    controls = runner.control_specs(plan)
    assert len(primary) == 540
    assert len(controls) == 90
    assert len({item.candidate_id for item in primary}) == 540
    assert len({item.candidate_id for item in controls}) == 90
    assert {runner._candidate_payload(item)["arm"] for item in primary} == {
        "dual_histogram_llr"
    }
    assert {runner._control_payload(item)["arm"] for item in controls} == {
        "negative_ecdf"
    }
    assert len(plan.required_fold_files) == 18
    assert set(plan.required_fold_files) == set(
        plan.raw["outer_label_gate"]["required_fold_files"]
    )


class _Archive:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.arrays = arrays
        self.files = [
            *arrays,
            "fmt_features",
            "raw_features",
            "reference_labels_all",
            "metadata_json",
        ]
        self.opened: list[str] = []

    def __enter__(self) -> "_Archive":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __getitem__(self, name: str) -> np.ndarray:
        self.opened.append(name)
        if name not in self.arrays:
            raise AssertionError(f"forbidden member was opened: {name}")
        return self.arrays[name]


@contextmanager
def _opened() -> object:
    yield SimpleNamespace(stream=object())


def test_minimal_parent_loader_keeps_fmt_raw_reference_and_label_opaque() -> None:
    arrays = {
        "valid_scale_id": np.asarray([0, 1000], dtype=np.int32),
        "valid_center_seed_index": np.asarray([2, 3], dtype=np.int64),
        "valid_scale_block_index": np.asarray([0, 1], dtype=np.int8),
        "valid_assigned_row_index": np.asarray([2, 64003], dtype=np.int64),
        "valid_labels": np.asarray([False, True], dtype=np.bool_),
    }
    archive = _Archive(arrays)
    row = SimpleNamespace(
        path=Path("opaque_parent.npz"), size_bytes=1, sha256="a" * 64
    )
    with (
        patch.object(
            runner.source_runner.early,
            "_authenticated_open_file",
            return_value=_opened(),
        ),
        patch.object(runner.np, "load", return_value=archive),
    ):
        observed, opened = runner._load_parent_minimal(  # type: ignore[arg-type]
            row, include_labels=False
        )
    assert set(opened) == {
        "valid_scale_id",
        "valid_center_seed_index",
        "valid_scale_block_index",
        "valid_assigned_row_index",
    }
    assert set(archive.opened) == set(opened)
    assert "valid_labels" not in observed
    assert not {
        "fmt_features",
        "raw_features",
        "reference_labels_all",
        "metadata_json",
    }.intersection(archive.opened)


def test_family_batch_source_ids_distinguish_dataset_sources_with_same_index() -> None:
    caches = [
        SimpleNamespace(
            row=SimpleNamespace(
                family="family_a",
                dataset=dataset,
                source_ordinal=0,
                source_index=7,
            ),
            center_indices=np.asarray([0, 1], dtype=np.int64),
            labels=np.asarray([False, True], dtype=np.bool_),
            sidecar_file_sha256="a" * 64,
            parent_members_opened=("valid_labels",),
            sidecar_members_opened=("source_centered_seed4",),
        )
        for dataset in ("flow_a", "flow_b")
    ]
    paired = SimpleNamespace(
        paired_rank=np.asarray([0.2, 0.8], dtype=np.float64),
        combined_valid=np.asarray([True, True], dtype=np.bool_),
    )
    with (
        patch.object(runner, "paired_center_ranks", return_value=paired),
        patch.object(
            runner,
            "_center_labels",
            return_value=np.asarray([False, True], dtype=np.bool_),
        ),
    ):
        batches, audits = runner.family_rank_batches(  # type: ignore[arg-type]
            caches, 0.5
        )
    source_ids = batches["family_a"].source_ids
    assert set(source_ids.tolist()) == {0, 1}
    assert [item["stable_loo_source_id"] for item in audits] == [0, 1]


def test_assigned_rank_memoization_reuses_the_exact_read_only_array() -> None:
    cache = SimpleNamespace(
        assigned_rank_cache=None,
        assigned_center_indices=np.asarray([0, 1], dtype=np.int64),
        assigned_block_indices=np.asarray([0, 1], dtype=np.int8),
        assigned_scale_ids=np.asarray([0, 1000], dtype=np.int32),
        assigned_source_centered_seed4=np.zeros((2, 4), dtype=np.float32),
    )
    expected = np.asarray([0.25, 0.75], dtype=np.float64)
    expected.flags.writeable = False
    with patch.object(
        runner, "assigned_block_dx_midranks", return_value=expected
    ) as compute:
        first = runner._assigned_ranks(cache)  # type: ignore[arg-type]
        second = runner._assigned_ranks(cache)  # type: ignore[arg-type]
    assert first is second is expected
    assert compute.call_count == 1


def test_cached_inner_ranking_metrics_are_bitwise_row_equivalent() -> None:
    cache = SimpleNamespace(
        row=SimpleNamespace(
            dataset="flow", family="family", source_ordinal=0, source_index=7
        ),
        labels=np.asarray([False, True], dtype=np.bool_),
        center_indices=np.asarray([0, 1], dtype=np.int64),
    )
    paired = SimpleNamespace(
        combined_valid=np.asarray([True, True], dtype=np.bool_),
        legacy_valid=np.asarray([True, False], dtype=np.bool_),
        expanded_valid=np.asarray([False, True], dtype=np.bool_),
    )
    scores = np.asarray([0.2, 0.8], dtype=np.float64)
    predictions = np.asarray([False, True], dtype=np.bool_)
    payload = runner._candidate_payload(
        runner.CandidateSpec(0.5, 64, 0.5, 0.0, 0.95)
    )
    direct = runner._inner_row(
        outer_family="half_cylinder",
        inner_family="delta_wing",
        cache=cache,  # type: ignore[arg-type]
        paired=paired,  # type: ignore[arg-type]
        payload=payload,
        center_scores=scores,
        center_predictions=predictions,
    )
    cached = runner._inner_row(
        outer_family="half_cylinder",
        inner_family="delta_wing",
        cache=cache,  # type: ignore[arg-type]
        paired=paired,  # type: ignore[arg-type]
        payload=payload,
        center_scores=scores,
        center_predictions=predictions,
        ranking_metrics=runner._inner_ranking_metrics(  # type: ignore[arg-type]
            cache, scores
        ),
    )
    assert direct == cached


def test_negative_control_query_is_strict_less_and_ignores_positive_templates() -> None:
    arrays = {
        "control_family_order_unicode": np.asarray(["a", "b"], dtype="<U1"),
        "control_negative_rank_reference_values": np.asarray(
            [0.2, 0.5, 0.1, 0.5], dtype=np.float64
        ),
        "control_negative_rank_reference_offsets": np.asarray(
            [0, 2, 4], dtype=np.int64
        ),
        "control_weight_float64": np.asarray(0.5, dtype=np.float64),
    }
    observed = runner.query_negative_control_arrays(
        arrays, np.asarray([0.5], dtype=np.float64)
    )
    # Each family has exactly one reference strictly below the tied query;
    # 1/(2+1), averaged equally across families.
    np.testing.assert_array_equal(observed, np.asarray([1.0 / 3.0]))


def test_negative_control_fit_does_not_require_or_construct_positive_templates() -> None:
    caches = [
        SimpleNamespace(
            row=SimpleNamespace(
                family=family,
                dataset=f"flow_{family}",
                source_ordinal=0,
                source_index=index,
            ),
            sidecar_file_sha256=str(index + 1) * 64,
            parent_members_opened=("valid_labels",),
            sidecar_members_opened=("source_centered_seed4",),
        )
        for index, family in enumerate(("family_a", "family_b"))
    ]
    paired = SimpleNamespace(
        paired_rank=np.asarray([0.2, 0.8], dtype=np.float64),
        combined_valid=np.asarray([True, True], dtype=np.bool_),
    )
    with (
        patch.object(runner, "paired_center_ranks", return_value=paired),
        patch.object(
            runner,
            "_center_labels",
            return_value=np.asarray([False, False], dtype=np.bool_),
        ),
        patch.object(
            runner,
            "FamilySourceRankBatch",
            side_effect=AssertionError("control must not construct primary batches"),
        ),
    ):
        arrays, audits = runner.fit_negative_control(caches, weight=0.5)  # type: ignore[arg-type]
    assert arrays["control_family_order_unicode"].tolist() == [
        "family_a",
        "family_b",
    ]
    np.testing.assert_array_equal(
        arrays["control_negative_rank_reference_offsets"], [0, 2, 4]
    )
    assert all(item["uses_positive_templates"] is False for item in audits)
    assert all(
        item["positive_center_count_observed_but_not_templated"] == 0
        for item in audits
    )


def test_split_final_library_and_negative_control_freshly_reconstruct() -> None:
    plan = runner.load_plan(CONFIG)
    source_plan = replace(
        plan.source_plan,
        sidecar_root=Path("synthetic_sidecars"),
        sidecar_population_path=Path("synthetic_sidecars/SIDECAR_POPULATION.json"),
        sidecar_population={"rows": []},
    )
    plan = replace(
        plan,
        source_plan=source_plan,
        source_evidence={"synthetic": True},
        parent_binding_path=Path("synthetic_binding/parent_sidecar_binding.json"),
        parent_binding_file_sha256="1" * 64,
        parent_binding_content_sha256="2" * 64,
        binding_completion_path=Path("synthetic_binding/BINDING_COMPLETE.json"),
        binding_completion_file_sha256="3" * 64,
    )
    batches = {
        "family_a": runner.FamilySourceRankBatch(
            ranks=np.asarray([0.10, 0.80, 0.20, 0.90], dtype=np.float64),
            labels=np.asarray([False, True, False, True], dtype=np.bool_),
            source_ids=np.asarray([0, 0, 1, 1], dtype=np.int64),
        ),
        "family_b": runner.FamilySourceRankBatch(
            ranks=np.asarray([0.05, 0.70, 0.30, 0.95], dtype=np.float64),
            labels=np.asarray([False, True, False, True], dtype=np.bool_),
            source_ids=np.asarray([2, 2, 3, 3], dtype=np.int64),
        ),
    }
    model = runner.FamilyBalancedRankLikelihoodModel(
        batches, bin_count=64, beta=0.5
    )
    primary = runner.CandidateSpec(0.5, 64, 0.5, 0.0, 0.95)
    control = runner.ControlSpec(0.25, 0.0, 0.95)
    control_arrays = {
        "control_family_order_unicode": np.asarray(
            ["family_a", "family_b"], dtype="<U8"
        ),
        "control_negative_rank_reference_values": np.asarray(
            [0.10, 0.20, 0.05, 0.30], dtype=np.float64
        ),
        "control_negative_rank_reference_offsets": np.asarray(
            [0, 2, 4], dtype=np.int64
        ),
        "control_weight_float64": np.asarray(0.25, dtype=np.float64),
    }
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary)
        paths = runner.write_final_artifacts(
            destination,
            model,
            control_arrays,
            [],
            [],
            plan=plan,
            primary=primary,
            control=control,
            outer_family="half_cylinder",
            fit_families=("family_a", "family_b"),
            git_commit="4" * 40,
        )
        verified = runner.authenticate_final_artifacts(
            destination,
            plan=plan,
            primary=primary,
            control=control,
            outer_family="half_cylinder",
            fit_families=("family_a", "family_b"),
            git_commit="4" * 40,
            expected_manifest_sha256={
                name: record[2] for name, record in paths.items()
            },
        )
        assert tuple(verified.model.family_order) == ("family_a", "family_b")
        assert float(verified.arrays["control_weight_float64"]) == 0.25
        for name, expected in model.export_arrays().items():
            np.testing.assert_array_equal(verified.arrays[name], expected)
        query = np.asarray([0.10, 0.50, 0.90], dtype=np.float64)
        np.testing.assert_array_equal(
            verified.model.query(query).dual_template_score,
            model.query(query).dual_template_score,
        )
        np.testing.assert_array_equal(
            runner.query_negative_control(verified, query),
            runner.query_negative_control_arrays(control_arrays, query),
        )

        wrong_control = runner.ControlSpec(0.5, 0.0, 0.95)
        try:
            runner.authenticate_final_artifacts(
                destination,
                plan=plan,
                primary=primary,
                control=wrong_control,
                outer_family="half_cylinder",
                fit_families=("family_a", "family_b"),
                git_commit="4" * 40,
                expected_manifest_sha256={
                    name: record[2] for name, record in paths.items()
                },
            )
        except ValueError as error:
            assert "provenance drifted" in str(error)
        else:
            raise AssertionError("selected control weight tamper must fail closed")

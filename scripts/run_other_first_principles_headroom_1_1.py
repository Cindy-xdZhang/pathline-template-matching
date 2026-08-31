#!/usr/bin/env python3
"""Diagnose Early score-ordering headroom versus decision calibration.

This is an explicitly post-hoc, exposed-development diagnostic.  The first
three decision arms never receive an outer reference (the prevalence arm may
use nonouter inner labels).  The fourth arm uses each outer group's labels to
compute an optimistic max-F1 threshold and is emitted only as a non-deployable
ranking upper bound.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import io
import json
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pathline_template_matching.nested_scale_validation import (  # noqa: E402
    fixed_top_fraction_predictions,
)
from pathline_template_matching.one_class_spatial import (  # noqa: E402
    high_score_two_means_predictions,
)
from pathline_template_matching.portable_flow import (  # noqa: E402
    canonical_array_sha256,
    sha256_file,
)
from scripts import (  # noqa: E402
    aggregate_verify_early_opposite_pair_kinematics_1_1 as early_aggregate,
)
from scripts import (  # noqa: E402
    run_verify_early_opposite_pair_kinematics_1_1 as early_runner,
)


EXPERIMENT = "Other_FirstPrinciplesHeadroom_1.1"
EXPECTED_CONFIG_SHA256 = (
    "a76ae95710f72a6432e4d392606fe4ca5ad4c0fb89b8d50e6d3868f546117477"
)
EXPECTED_EARLY_COMMIT = "2c3774dca0d81db8edd5645e63576526b9e276f7"
EXPECTED_REPORTING_PROJECT_ROOT = "/home/zhanx0o/pathline-template-matching-headroom"
EXPECTED_EARLY_CONFIG_ABSOLUTE_PATH = (
    "/home/zhanx0o/pathline-template-matching-early-kinematics/"
    "config/Verify_EarlyOppositePairKinematics_1.1.yaml"
)
SUPERSEDED_DRAFT_CONFIG_SHA256 = (
    "f02120bbf4e67a69c3ba60d1850aeb5c8b22d50477adc84f1ff4533bc9f22020"
)
EXPECTED_EARLY_CONFIG_SHA256 = early_runner.EXPECTED_CONFIG_SHA256
EXPECTED_EARLY_RUNNER_SHA256 = (
    "e999960ac06d3fedd355e1d6135d9e69316bfe1e798318a22dadf5a8e2063796"
)
EXPECTED_EARLY_AGGREGATOR_SHA256 = (
    "631909159387cba854f471b3179ff0f0cd97404905e29b74589b2b8cf71f089e"
)
CONFIG_PATH = ROOT / "config" / "Other_FirstPrinciplesHeadroom_1.1.yaml"
EARLY_CONFIG_PATH = EXPECTED_EARLY_CONFIG_ABSOLUTE_PATH

FAMILY_ORDER = early_runner.FAMILY_ORDER
BLOCK_NAMES = early_runner.BLOCK_NAMES
METHOD_ORDER = (
    "current_selected_prediction",
    "inner_prevalence_top_fraction",
    "label_free_exact_1d_two_means",
    "outer_group_oracle_max_f1",
)
DEPLOYABLE_WITHOUT_OUTER_LABEL = MappingProxyType(
    {
        "current_selected_prediction": True,
        "inner_prevalence_top_fraction": True,
        "label_free_exact_1d_two_means": True,
        "outer_group_oracle_max_f1": False,
    }
)
OUTER_LABEL_USE = MappingProxyType(
    {
        "current_selected_prediction": False,
        "inner_prevalence_top_fraction": False,
        "label_free_exact_1d_two_means": False,
        "outer_group_oracle_max_f1": True,
    }
)

INPUT_SCHEMA = "pathline_template_matching.first_principles_headroom_input.v1"
SUMMARY_SCHEMA = "pathline_template_matching.first_principles_headroom_summary.v1"
RESULT_SCHEMA = "pathline_template_matching.first_principles_headroom_result.v1"
COMPLETE_SCHEMA = "pathline_template_matching.first_principles_headroom_complete.v1"

PREVALENCE_FIELDS = (
    "outer_family",
    "block",
    "selected_candidate_id",
    "inner_family_count",
    "group_count",
    "estimated_positive_fraction",
    "minimum_inner_family_fraction",
    "maximum_inner_family_fraction",
    "inner_family_fractions_json",
    "inner_group_metrics_sha256",
    "outer_label_used",
)
GROUP_FIELDS = (
    "outer_family",
    "dataset",
    "source_ordinal",
    "source_index",
    "block",
    "method",
    "selected_candidate_id",
    "decision_parameter",
    "outer_label_used",
    "legal_without_outer_label",
    "sample_count",
    "positive_count",
    "negative_count",
    "eligible_count",
    "predicted_positive_count",
    "predicted_positive_fraction",
    "reference_positive_fraction",
    "inner_prevalence_estimate",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
    "accuracy",
    "average_precision",
    "f1",
    "balanced_accuracy",
    "auroc",
    "precision",
    "recall",
    "oracle_f1_minus_current_f1",
    "oracle_f1_minus_inner_prevalence_f1",
    "oracle_f1_minus_two_means_f1",
)
MACRO_FIELDS = (
    "outer_family",
    "block",
    "method",
    "group_count",
    "sample_count_sum",
    "accuracy",
    "average_precision",
    "f1",
    "balanced_accuracy",
    "auroc",
    "precision",
    "recall",
    "predicted_positive_fraction",
    "reference_positive_fraction",
    "oracle_f1_minus_current_f1",
    "oracle_f1_minus_inner_prevalence_f1",
    "oracle_f1_minus_two_means_f1",
    "outer_label_used",
    "legal_without_outer_label",
)
MACRO_METRICS = (
    "accuracy",
    "average_precision",
    "f1",
    "balanced_accuracy",
    "auroc",
    "precision",
    "recall",
    "predicted_positive_fraction",
    "reference_positive_fraction",
)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class HeadroomPlan:
    path: Path
    sha256: str
    raw: Mapping[str, Any]
    allowed_datasets: tuple[str, ...]
    forbidden_datasets: tuple[str, ...]
    family_datasets: Mapping[str, tuple[str, ...]]
    reporting_project_root: str
    early_config_absolute_path: str
    expected_fold_basenames: Mapping[str, str]
    parent_family_macro_f1: float
    parent_family_macro_f1_tolerance: float
    expected_fold_files: tuple[str, ...]
    target_f1_values: tuple[float, ...]


@dataclass(frozen=True)
class CompactPrediction:
    outer_family: str
    manifest_file_sha256: str
    prediction_file_sha256: str
    selected_candidate_id: str
    dataset_names: tuple[str, ...]
    dataset_code: np.ndarray
    source_ordinal: np.ndarray
    source_index: np.ndarray
    scale_id: np.ndarray
    center_seed_index: np.ndarray
    scale_block_index: np.ndarray
    assigned_row_index: np.ndarray
    spatial_score: np.ndarray
    calibration_supported: np.ndarray
    spatial_imputed: np.ndarray
    spatial_unimputable: np.ndarray
    prediction: np.ndarray

    @property
    def count(self) -> int:
        return len(self.prediction)

    @property
    def eligible(self) -> np.ndarray:
        return self.calibration_supported | self.spatial_imputed


@dataclass(frozen=True)
class Decision:
    predictions: np.ndarray
    parameter: str


def load_plan(config_path: str | Path = CONFIG_PATH) -> HeadroomPlan:
    """Load the immutable diagnostic definition before any real result access."""

    path = Path(config_path).resolve()
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    _require(digest == EXPECTED_CONFIG_SHA256, "frozen headroom config SHA-256 drifted")
    raw = yaml.safe_load(content.decode("utf-8"))
    _require(isinstance(raw, Mapping), "headroom config root is invalid")
    _require(raw.get("experiment") == EXPERIMENT, "headroom experiment identity drifted")
    _require(raw.get("status") == "frozen_pre_run_not_run", "freeze-history status drifted")
    _require(
        raw.get("freeze_timing")
        == "before_opening_any_spatial_score_or_outer_reference_array_for_this_diagnostic",
        "config/result freeze boundary drifted",
    )
    _require(
        raw.get("real_result_access_during_design_and_implementation")
        == (
            "previously_exposed_metadata_and_public_summary_only_"
            "no_spatial_score_or_outer_reference_array"
        ),
        "pre-run result-access boundary drifted",
    )
    superseded = raw.get("superseded_pre_run_draft")
    _require(isinstance(superseded, Mapping), "superseded draft audit is missing")
    assert isinstance(superseded, Mapping)
    _require(
        superseded.get("config_sha256") == SUPERSEDED_DRAFT_CONFIG_SHA256
        and superseded.get("status")
        == "superseded_before_commit_and_before_any_real_run",
        "superseded pre-run draft identity drifted",
    )
    evidence = raw.get("evidence_scope")
    parent = raw.get("parent_early_evidence")
    authentication = raw.get("fold_authentication")
    decisions = raw.get("decision_arms")
    metrics = raw.get("metrics")
    output = raw.get("output")
    _require(
        all(isinstance(value, Mapping) for value in (evidence, parent, authentication, decisions, metrics, output)),
        "headroom config sections are incomplete",
    )
    assert isinstance(evidence, Mapping)
    assert isinstance(parent, Mapping)
    assert isinstance(authentication, Mapping)
    assert isinstance(decisions, Mapping)
    assert isinstance(metrics, Mapping)
    assert isinstance(output, Mapping)
    reporting_project_root = str(authentication.get("reporting_project_root", ""))
    _require(
        reporting_project_root == EXPECTED_REPORTING_PROJECT_ROOT
        and authentication.get(
            "reporting_project_root_must_differ_from_parent_early_checkout"
        )
        is True,
        "reporting project root contract drifted",
    )
    allowed = tuple(str(value) for value in evidence.get("allowed_datasets", ()))
    forbidden = tuple(str(value) for value in evidence.get("forbidden_datasets", ()))
    _require(len(allowed) == len(set(allowed)) == 8, "exactly eight train datasets are required")
    _require(forbidden == ("tangaroa", "smokeBuoyancy"), "forbidden datasets drifted")
    _require(not set(allowed) & set(forbidden), "allowed and forbidden datasets overlap")
    _require(
        parent.get("numerical_git_commit") == EXPECTED_EARLY_COMMIT
        and parent.get("config_sha256") == EXPECTED_EARLY_CONFIG_SHA256
        and parent.get("runner_sha256") == EXPECTED_EARLY_RUNNER_SHA256
        and parent.get("aggregator_sha256") == EXPECTED_EARLY_AGGREGATOR_SHA256,
        "authenticated Early parent identity drifted",
    )
    early_config_absolute_path = str(parent.get("config_absolute_path", ""))
    _require(
        early_config_absolute_path == EXPECTED_EARLY_CONFIG_ABSOLUTE_PATH,
        "authenticated Early absolute config path drifted",
    )
    _require(
        reporting_project_root
        != early_config_absolute_path.rsplit("/config/", maxsplit=1)[0],
        "reporting project root must not reuse the frozen Early checkout",
    )
    parent_family_macro_f1 = float(parent.get("parent_family_macro_f1", float("nan")))
    parent_family_macro_f1_tolerance = float(
        parent.get("parent_family_macro_f1_reproduction_absolute_tolerance", float("nan"))
    )
    _require(
        parent_family_macro_f1 == 0.6391632765825263
        and np.isfinite(parent_family_macro_f1_tolerance)
        and 0.0 < parent_family_macro_f1_tolerance <= 1.0e-12,
        "authenticated parent F1 reproduction contract drifted",
    )
    _require(tuple(parent.get("fold_order", ())) == FAMILY_ORDER, "Early fold order drifted")
    family_raw = parent.get("family_datasets")
    _require(isinstance(family_raw, Mapping), "family dataset mapping is missing")
    family_datasets = {
        family: tuple(str(value) for value in family_raw.get(family, ()))
        for family in FAMILY_ORDER
    }
    _require(
        tuple(dataset for family in FAMILY_ORDER for dataset in family_datasets[family])
        == allowed,
        "family mapping does not exactly reproduce the eight allowed datasets",
    )
    expected_jobs = parent.get("expected_fold_jobs")
    _require(isinstance(expected_jobs, Mapping), "expected Early fold jobs are missing")
    assert isinstance(expected_jobs, Mapping)
    _require(tuple(expected_jobs) == FAMILY_ORDER, "expected Early fold job order drifted")
    expected_fold_basenames: dict[str, str] = {}
    for expected_task, family in enumerate(FAMILY_ORDER):
        record = expected_jobs.get(family)
        _require(isinstance(record, Mapping), f"expected Early fold job is invalid: {family}")
        assert isinstance(record, Mapping)
        job_id = int(record.get("job_id", -1))
        task_id = int(record.get("array_task_id", -1))
        basename = str(record.get("directory_basename", ""))
        _require(job_id > 0 and task_id == expected_task, f"expected Early job/task drifted: {family}")
        _require(
            basename
            == f"slurm_{job_id}_{task_id}_{EXPECTED_EARLY_COMMIT[:12]}_outer_{family}",
            f"expected Early fold basename drifted: {family}",
        )
        expected_fold_basenames[family] = basename
    expected_files = tuple(str(value) for value in authentication.get("exact_fold_files", ()))
    _require(expected_files == early_aggregate.EXPECTED_FOLD_FILES, "Early fold file contract drifted")
    _require(authentication.get("device") == "cpu", "this frozen diagnostic requires CPU replay")
    _require(tuple(decisions.get("order", ())) == METHOD_ORDER, "decision-arm order drifted")
    oracle = decisions.get("outer_group_oracle_max_f1")
    _require(isinstance(oracle, Mapping), "oracle definition is missing")
    _require(
        oracle.get("outer_label_use") is True
        and oracle.get("diagnostic_only") is True
        and oracle.get("deployable") is False,
        "oracle/reporting boundary drifted",
    )
    targets = tuple(float(value) for value in metrics.get("target_f1_values", ()))
    _require(targets == (0.70, 0.80), "target F1 values drifted")
    _require(output.get("overwrite") == "forbidden", "output overwrite policy drifted")
    return HeadroomPlan(
        path=path,
        sha256=digest,
        raw=MappingProxyType(dict(raw)),
        allowed_datasets=allowed,
        forbidden_datasets=forbidden,
        family_datasets=MappingProxyType(family_datasets),
        reporting_project_root=reporting_project_root,
        early_config_absolute_path=early_config_absolute_path,
        expected_fold_basenames=MappingProxyType(expected_fold_basenames),
        parent_family_macro_f1=parent_family_macro_f1,
        parent_family_macro_f1_tolerance=parent_family_macro_f1_tolerance,
        expected_fold_files=expected_files,
        target_f1_values=targets,
    )


def _source_sha256(relative_path: str) -> str:
    return sha256_file(ROOT / relative_path)


def require_exact_reporting_project_root(
    plan: HeadroomPlan,
    project_root: str | Path,
) -> str:
    """Keep reporting code separate from the immutable Early producer checkout."""

    supplied = str(project_root)
    _require(
        supplied == plan.reporting_project_root,
        "reporting project root must equal the frozen dedicated headroom checkout",
    )
    _require(
        supplied
        != plan.early_config_absolute_path.rsplit("/config/", maxsplit=1)[0],
        "reporting project root must differ from the frozen Early checkout",
    )
    return supplied


def require_exact_early_config_path(
    plan: HeadroomPlan,
    early_config_path: str | Path,
) -> str:
    """Reject a clone-relative config path that cannot authenticate old folds."""

    supplied = str(early_config_path)
    _require(
        supplied == plan.early_config_absolute_path,
        "Early config path must equal the frozen producer-checkout absolute path",
    )
    return supplied


def _bind_authenticated_early_plan(
    plan: HeadroomPlan,
    early_config_path: str | Path,
    *,
    expected_reporting_commit: str,
    kinematic_input_manifest_path: str | Path,
    kinematic_input_manifest_file_sha256: str,
    synthetic_pass_path: str | Path,
    synthetic_pass_file_sha256: str,
    sidecar_root: str | Path,
    sidecar_population_manifest_path: str | Path,
    sidecar_population_manifest_file_sha256: str,
) -> early_runner.Plan:
    """Bind unchanged Early sources, then pin replay provenance to its producer."""

    _require(_lower_hex(expected_reporting_commit, 40), "reporting commit must be a full SHA-1")
    require_exact_reporting_project_root(plan, ROOT)
    authenticated_early_config_path = require_exact_early_config_path(
        plan, early_config_path
    )
    reporting_commit, dirty = early_runner._git_identity()
    _require(not dirty, "headroom diagnostic requires a clean committed worktree")
    _require(reporting_commit == expected_reporting_commit, "reporting checkout commit drifted")
    parent = plan.raw["parent_early_evidence"]
    _require(
        _source_sha256(str(parent["runner_path"])) == EXPECTED_EARLY_RUNNER_SHA256,
        "Early runner source changed after the authenticated five-fold run",
    )
    _require(
        _source_sha256(str(parent["aggregator_path"])) == EXPECTED_EARLY_AGGREGATOR_SHA256,
        "Early aggregator source changed after the authenticated five-fold run",
    )
    early_plan = early_runner.load_plan(authenticated_early_config_path)
    _require(early_plan.sha256 == EXPECTED_EARLY_CONFIG_SHA256, "Early config drifted")
    bound = early_runner.bind_early_evidence(
        early_plan,
        kinematic_input_manifest_path=kinematic_input_manifest_path,
        kinematic_input_manifest_file_sha256=kinematic_input_manifest_file_sha256,
        synthetic_pass_path=synthetic_pass_path,
        synthetic_pass_file_sha256=synthetic_pass_file_sha256,
        sidecar_root=sidecar_root,
        sidecar_population_manifest_path=sidecar_population_manifest_path,
        sidecar_population_manifest_file_sha256=sidecar_population_manifest_file_sha256,
    )
    _require(bound.source_identity is not None, "clean Early source identity was not bound")
    pinned_identity = replace(bound.source_identity, git_commit=EXPECTED_EARLY_COMMIT)
    pinned = replace(bound, source_identity=pinned_identity)
    _require(tuple(pinned.families) == FAMILY_ORDER, "Early family order changed")
    _require(
        tuple(dataset for family in FAMILY_ORDER for dataset in pinned.families[family])
        == plan.allowed_datasets,
        "Early plan no longer contains exactly the allowed eight train datasets",
    )
    return pinned


def order_exact_fold_directories(
    plan: HeadroomPlan,
    fold_directories: Sequence[str | Path],
) -> tuple[Path, ...]:
    """Bind the five CLI directories to the exact authenticated job/task names."""

    paths = tuple(Path(value).resolve() for value in fold_directories)
    _require(
        len(paths) == 5 and len(set(paths)) == 5,
        "exactly five unique fold directories are required",
    )
    by_basename = {path.name: path for path in paths}
    expected = tuple(plan.expected_fold_basenames[family] for family in FAMILY_ORDER)
    _require(
        len(by_basename) == 5 and set(by_basename) == set(expected),
        "fold directory basenames differ from the frozen Early job/task population",
    )
    return tuple(by_basename[basename] for basename in expected)


def authenticate_folds(
    plan: HeadroomPlan,
    early_plan: early_runner.Plan,
    fold_directories: Sequence[str | Path],
) -> tuple[early_aggregate.AuthenticatedFold, ...]:
    """Fresh-replay and strictly authenticate every Early outer-family fold."""

    paths = order_exact_fold_directories(plan, fold_directories)
    folds = tuple(
        early_aggregate._authenticate_fold(
            early_plan,
            path,
            device="cpu",
            expected_fold_commit=EXPECTED_EARLY_COMMIT,
        )
        for path in paths
    )
    _require(
        tuple(fold.outer_family for fold in folds) == FAMILY_ORDER,
        "exact fold basename resolved to the wrong outer family",
    )
    for fold in folds:
        _require(fold.config_sha256 == EXPECTED_EARLY_CONFIG_SHA256, "fold config drifted")
        _require(fold.numerical_git_commit == EXPECTED_EARLY_COMMIT, "fold commit drifted")
        _require(set(fold.artifact_identities) == set(early_aggregate.EXPECTED_RESULT_ARTIFACTS), "fold artifact population drifted")
        _require(
            set(plan.family_datasets[fold.outer_family]).isdisjoint(plan.forbidden_datasets),
            "fold resolves to a forbidden dataset",
        )
    return folds


def _freeze_array(values: np.ndarray) -> np.ndarray:
    return early_runner._deep_freeze(np.ascontiguousarray(values))


def authenticate_prediction_artifact(
    prediction_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_prediction_file_sha256: str,
    expected_manifest_file_sha256: str,
    expected_outer_family: str,
    dataset_names: Sequence[str],
) -> CompactPrediction:
    """Authenticate all 19 arrays while retaining a compact diagnostic view."""

    prediction = Path(prediction_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest, observed_manifest_sha = early_aggregate._load_self_hashed_json(
        manifest_file,
        expected_file_sha256=expected_manifest_file_sha256,
    )
    _require(manifest.get("schema") == early_runner.PREDICTION_MANIFEST_SCHEMA, "prediction manifest schema drifted")
    _require(manifest.get("prediction_schema") == early_runner.PREDICTION_SCHEMA, "prediction schema drifted")
    _require(manifest.get("experiment") == early_runner.EXPERIMENT, "prediction experiment drifted")
    _require(manifest.get("config_sha256") == EXPECTED_EARLY_CONFIG_SHA256, "prediction config drifted")
    _require(manifest.get("git_commit") == EXPECTED_EARLY_COMMIT, "prediction commit drifted")
    _require(manifest.get("outer_family") == expected_outer_family, "prediction outer family drifted")
    file_record = manifest.get("prediction_file")
    records = manifest.get("arrays")
    _require(isinstance(file_record, Mapping), "prediction file identity is missing")
    _require(isinstance(records, Mapping), "prediction array identities are missing")
    assert isinstance(file_record, Mapping)
    assert isinstance(records, Mapping)
    _require(file_record.get("path") == prediction.name, "prediction file name drifted")
    _require(file_record.get("sha256") == expected_prediction_file_sha256, "prediction/fold hash binding drifted")
    _require(set(records) == set(early_runner.PREDICTION_ARRAY_DTYPES), "prediction array set drifted")
    row_count = int(manifest.get("row_count", -1))
    _require(row_count > 0 and int(manifest.get("array_count", -1)) == 19, "prediction population is invalid")
    allowed_names = tuple(str(value) for value in dataset_names)
    _require(len(allowed_names) == len(set(allowed_names)) and allowed_names, "dataset codebook is invalid")
    keep_names = {
        "source_ordinal",
        "source_index",
        "scale_id",
        "center_seed_index",
        "scale_block_index",
        "assigned_row_index",
        "spatial_score",
        "calibration_supported",
        "spatial_imputed",
        "spatial_unimputable",
        "prediction",
    }
    retained: dict[str, np.ndarray] = {}
    dataset_code: np.ndarray | None = None
    with early_runner._authenticated_open_file(
        prediction,
        expected_size=int(file_record["size_bytes"]),
        expected_sha256=str(file_record["sha256"]),
    ) as opened:
        _require(opened.sha256 == expected_prediction_file_sha256, "prediction SHA-256 drifted")
        with np.load(opened.stream, allow_pickle=False) as archive:
            _require(set(archive.files) == set(records), "prediction NPZ members drifted")
            for name, dtype in early_runner.PREDICTION_ARRAY_DTYPES.items():
                values = np.array(archive[name], copy=True, order="C")
                record = records[name]
                _require(isinstance(record, Mapping), f"prediction array record is invalid: {name}")
                assert isinstance(record, Mapping)
                _require(values.dtype == dtype and values.shape == (row_count,), f"prediction array contract drifted: {name}")
                _require(values.dtype.str == record.get("dtype") and list(values.shape) == record.get("shape"), f"prediction array manifest drifted: {name}")
                _require(canonical_array_sha256(values) == record.get("sha256"), f"prediction array SHA-256 mismatch: {name}")
                if name == "dataset":
                    codes = np.full(row_count, -1, dtype=np.int8)
                    for code, dataset in enumerate(allowed_names):
                        codes[values == dataset] = code
                    _require(np.all(codes >= 0), "prediction contains a dataset outside the allowed outer family")
                    dataset_code = codes
                elif name in keep_names:
                    retained[name] = values
    _require(dataset_code is not None and set(retained) == keep_names, "compact prediction projection is incomplete")
    blocks = retained["scale_block_index"]
    scales = retained["scale_id"]
    centers = retained["center_seed_index"]
    assigned = retained["assigned_row_index"]
    scores = retained["spatial_score"]
    calibration = retained["calibration_supported"]
    imputed = retained["spatial_imputed"]
    unimputable = retained["spatial_unimputable"]
    predictions = retained["prediction"]
    _require(np.array_equal(blocks, (scales >= 1000).astype(np.int8)), "scale/block identity drifted")
    _require(np.array_equal(assigned, blocks.astype(np.int64) * 64000 + centers), "assigned-row identity drifted")
    _require(np.isfinite(scores).all() and np.all((scores >= 0.0) & (scores <= 1.0)), "spatial score is invalid")
    _require(np.array_equal(calibration | imputed | unimputable, np.ones(row_count, dtype=bool)), "spatial states do not partition rows")
    _require(not np.any((calibration & imputed) | (calibration & unimputable) | (imputed & unimputable)), "spatial states overlap")
    _require(not np.any(predictions[unimputable]), "unimputable rows must remain negative")
    source_ordinals = retained["source_ordinal"]
    _require(set(np.unique(source_ordinals).tolist()) == {0, 1, 2, 3}, "prediction source ordinals are incomplete")
    candidate = manifest.get("selected_candidate")
    _require(isinstance(candidate, Mapping), "selected candidate is missing from prediction")
    assert isinstance(candidate, Mapping)
    candidate_id = str(candidate.get("candidate_id"))
    _require(candidate_id, "selected candidate ID is empty")
    return CompactPrediction(
        outer_family=expected_outer_family,
        manifest_file_sha256=observed_manifest_sha,
        prediction_file_sha256=expected_prediction_file_sha256,
        selected_candidate_id=candidate_id,
        dataset_names=allowed_names,
        dataset_code=_freeze_array(dataset_code),
        source_ordinal=_freeze_array(source_ordinals),
        source_index=_freeze_array(retained["source_index"]),
        scale_id=_freeze_array(scales),
        center_seed_index=_freeze_array(centers),
        scale_block_index=_freeze_array(blocks),
        assigned_row_index=_freeze_array(assigned),
        spatial_score=_freeze_array(scores),
        calibration_supported=_freeze_array(calibration),
        spatial_imputed=_freeze_array(imputed),
        spatial_unimputable=_freeze_array(unimputable),
        prediction=_freeze_array(predictions),
    )


def estimate_inner_prevalence_from_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    outer_family: str,
    family_datasets: Mapping[str, Sequence[str]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Hierarchically estimate block prevalence without any outer label."""

    inner_families = tuple(family for family in FAMILY_ORDER if family != outer_family)
    expected_keys = {
        (family, dataset, source_ordinal, block)
        for family in inner_families
        for dataset in family_datasets[family]
        for source_ordinal in range(4)
        for block in BLOCK_NAMES
    }
    observed: dict[tuple[str, str, int, str], float] = {}
    for record in records:
        key = (
            str(record["inner_family"]),
            str(record["dataset"]),
            int(record["source_ordinal"]),
            str(record["block"]),
        )
        _require(key in expected_keys and key not in observed, "inner prevalence group identity drifted")
        sample_count = int(record["sample_count"])
        positive_count = int(record["positive_count"])
        _require(sample_count > 0 and 0 <= positive_count <= sample_count, "inner prevalence counts are invalid")
        observed[key] = positive_count / sample_count
    _require(set(observed) == expected_keys, "inner prevalence groups are incomplete")
    family_values: dict[str, dict[str, float]] = {block: {} for block in BLOCK_NAMES}
    estimates: dict[str, float] = {}
    for block in BLOCK_NAMES:
        for family in inner_families:
            values = [
                value
                for (row_family, _dataset, _source, row_block), value in observed.items()
                if row_family == family and row_block == block
            ]
            _require(values, f"inner family has no prevalence groups: {family}/{block}")
            family_values[block][family] = float(np.mean(values, dtype=np.float64))
        estimates[block] = float(
            np.mean(list(family_values[block].values()), dtype=np.float64)
        )
        _require(0.0 <= estimates[block] <= 1.0, "inner prevalence estimate left [0,1]")
    return estimates, family_values


def load_inner_prevalence(
    plan: HeadroomPlan,
    fold: early_aggregate.AuthenticatedFold,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Read only selected-candidate inner rows from an authenticated CSV."""

    identity = fold.artifact_identities["inner_group_metrics.csv"]
    content = early_runner._read_authenticated_bytes(
        fold.path / "inner_group_metrics.csv",
        expected_sha256=str(identity["sha256"]),
    )
    reader = csv.DictReader(io.StringIO(content.decode("utf-8"), newline=""))
    _require(tuple(reader.fieldnames or ()) == early_runner.METRIC_FIELDS, "inner metric CSV header drifted")
    candidate_id = str(fold.selected_candidate["candidate_id"])
    selected_rows: list[dict[str, Any]] = []
    for row in reader:
        if row["candidate_id"] != candidate_id:
            continue
        _require(row["outer_family"] == fold.outer_family, "inner row outer family drifted")
        selected_rows.append(
            {
                "inner_family": row["inner_family"],
                "dataset": row["dataset"],
                "source_ordinal": int(row["source_ordinal"]),
                "block": row["block"],
                "sample_count": int(row["sample_count"]),
                "positive_count": int(row["positive_count"]),
            }
        )
    estimates, family_values = estimate_inner_prevalence_from_rows(
        selected_rows,
        outer_family=fold.outer_family,
        family_datasets=plan.family_datasets,
    )
    rows: list[dict[str, Any]] = []
    for block in BLOCK_NAMES:
        components = family_values[block]
        rows.append(
            {
                "outer_family": fold.outer_family,
                "block": block,
                "selected_candidate_id": candidate_id,
                "inner_family_count": len(components),
                "group_count": sum(
                    1 for row in selected_rows if row["block"] == block
                ),
                "estimated_positive_fraction": estimates[block],
                "minimum_inner_family_fraction": min(components.values()),
                "maximum_inner_family_fraction": max(components.values()),
                "inner_family_fractions_json": json.dumps(
                    components, sort_keys=True, separators=(",", ":")
                ),
                "inner_group_metrics_sha256": str(identity["sha256"]),
                "outer_label_used": False,
            }
        )
    return estimates, rows


def inner_prevalence_top_fraction_decision(
    scores: np.ndarray,
    center_indices: np.ndarray,
    eligible: np.ndarray,
    *,
    fraction: float,
) -> Decision:
    """Label-free top fraction using a nonouter inner-prevalence estimate."""

    _require(np.isfinite(fraction) and 0.0 <= fraction <= 1.0, "prevalence fraction is invalid")
    predictions = fixed_top_fraction_predictions(
        scores,
        center_indices,
        eligible,
        fraction=float(fraction),
    )
    return Decision(
        predictions=np.ascontiguousarray(predictions),
        parameter=f"inner_prevalence_fraction={fraction:.17g}",
    )


def label_free_two_means_decision(
    scores: np.ndarray,
    eligible: np.ndarray,
) -> Decision:
    """Exact one-dimensional two-means; no reference argument exists."""

    values = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    _require(values.ndim == mask.ndim == 1 and values.shape == mask.shape, "two-means inputs are misaligned")
    _require(np.isfinite(values).all(), "two-means score is nonfinite")
    predictions = np.zeros(len(values), dtype=bool)
    eligible_rows = np.flatnonzero(mask)
    if len(eligible_rows):
        predictions[eligible_rows] = high_score_two_means_predictions(values[eligible_rows])
    if predictions.any() and (~predictions & mask).any():
        low_mean = float(values[mask & ~predictions].mean(dtype=np.float64))
        high_mean = float(values[predictions].mean(dtype=np.float64))
        _require(high_mean > low_mean, "two-means high cluster is not strictly higher")
        parameter = f"low_mean={low_mean:.17g}|high_mean={high_mean:.17g}"
    else:
        parameter = "no_distinct_two_cluster_solution_all_negative"
    return Decision(predictions=predictions, parameter=parameter)


def oracle_max_f1_decision(
    scores: np.ndarray,
    eligible: np.ndarray,
    labels: np.ndarray,
) -> Decision:
    """Tie-aware per-group outer-label oracle; diagnostic upper bound only."""

    values = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    targets = np.asarray(labels, dtype=bool)
    _require(values.ndim == 1 and values.shape == mask.shape == targets.shape and len(values) > 0, "oracle arrays are misaligned")
    _require(np.isfinite(values).all(), "oracle score is nonfinite")
    candidate_rows = np.flatnonzero(mask & (values > 0.0))
    predictions = np.zeros(len(values), dtype=bool)
    if len(candidate_rows) == 0 or not targets.any():
        return Decision(predictions=predictions, parameter="all_negative")
    order = candidate_rows[np.argsort(-values[candidate_rows], kind="mergesort")]
    ordered_scores = values[order]
    ordered_labels = targets[order].astype(np.int64)
    group_ends = np.r_[
        np.flatnonzero(np.diff(ordered_scores) != 0), len(order) - 1
    ]
    true_positive = np.cumsum(ordered_labels, dtype=np.int64)[group_ends]
    predicted_positive = group_ends + 1
    false_positive = predicted_positive - true_positive
    false_negative = int(targets.sum()) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 0,
    )
    best_f1 = float(f1.max(initial=0.0))
    if best_f1 <= 0.0:
        return Decision(predictions=predictions, parameter="all_negative")
    # group_ends are increasing, so the first best value has the fewest
    # predicted positives; its threshold is consequently the highest.
    best_position = int(np.flatnonzero(f1 == best_f1)[0])
    threshold = float(ordered_scores[group_ends[best_position]])
    predictions = mask & (values > 0.0) & (values >= threshold)
    return Decision(
        predictions=np.ascontiguousarray(predictions),
        parameter=(
            f"diagnostic_only_outer_label_threshold={threshold:.17g}"
            f"|predicted_positive_count={int(predictions.sum())}"
        ),
    )


def _classification_and_ranking_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    counts = dict(early_runner._classification_counts(labels, predictions))
    targets = np.asarray(labels, dtype=bool)
    if targets.any() and (~targets).any():
        average_precision, auroc = early_runner._ranking_metrics_one_sort(targets, scores)
    else:
        average_precision = auroc = float("nan")
    counts["average_precision"] = float(average_precision)
    counts["auroc"] = float(auroc)
    return counts


def _finite_mean(values: Sequence[float]) -> float:
    numeric = np.asarray(values, dtype=np.float64)
    finite = numeric[np.isfinite(numeric)]
    return float(np.mean(finite, dtype=np.float64)) if len(finite) else float("nan")


def diagnose_group(
    *,
    outer_family: str,
    dataset: str,
    source_ordinal: int,
    source_index: int,
    block: str,
    selected_candidate_id: str,
    scores: np.ndarray,
    center_indices: np.ndarray,
    eligible: np.ndarray,
    current_prediction: np.ndarray,
    labels: np.ndarray,
    inner_prevalence: float,
) -> list[dict[str, Any]]:
    """Compute the three legal arms and one clearly marked oracle upper bound."""

    values = np.asarray(scores, dtype=np.float64)
    centers = np.asarray(center_indices, dtype=np.int64)
    mask = np.asarray(eligible, dtype=bool)
    current = np.asarray(current_prediction, dtype=bool)
    targets = np.asarray(labels, dtype=bool)
    _require(values.shape == centers.shape == mask.shape == current.shape == targets.shape and values.ndim == 1 and len(values) > 0, "group arrays are misaligned")
    decisions = {
        "current_selected_prediction": Decision(
            predictions=np.ascontiguousarray(current),
            parameter="authenticated_parent_selected_candidate",
        ),
        "inner_prevalence_top_fraction": inner_prevalence_top_fraction_decision(
            values, centers, mask, fraction=inner_prevalence
        ),
        "label_free_exact_1d_two_means": label_free_two_means_decision(values, mask),
        "outer_group_oracle_max_f1": oracle_max_f1_decision(values, mask, targets),
    }
    metrics_by_method = {
        method: _classification_and_ranking_metrics(
            targets, decision.predictions, values
        )
        for method, decision in decisions.items()
    }
    oracle_f1 = float(metrics_by_method["outer_group_oracle_max_f1"]["f1"])
    oracle_gaps = {
        "oracle_f1_minus_current_f1": oracle_f1
        - float(metrics_by_method["current_selected_prediction"]["f1"]),
        "oracle_f1_minus_inner_prevalence_f1": oracle_f1
        - float(metrics_by_method["inner_prevalence_top_fraction"]["f1"]),
        "oracle_f1_minus_two_means_f1": oracle_f1
        - float(metrics_by_method["label_free_exact_1d_two_means"]["f1"]),
    }
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        decision = decisions[method]
        metric = metrics_by_method[method]
        predicted_count = int(decision.predictions.sum())
        rows.append(
            {
                "outer_family": outer_family,
                "dataset": dataset,
                "source_ordinal": int(source_ordinal),
                "source_index": int(source_index),
                "block": block,
                "method": method,
                "selected_candidate_id": selected_candidate_id,
                "decision_parameter": decision.parameter,
                "outer_label_used": OUTER_LABEL_USE[method],
                "legal_without_outer_label": DEPLOYABLE_WITHOUT_OUTER_LABEL[method],
                "sample_count": len(values),
                "positive_count": int(targets.sum()),
                "negative_count": int((~targets).sum()),
                "eligible_count": int(mask.sum()),
                "predicted_positive_count": predicted_count,
                "predicted_positive_fraction": predicted_count / len(values),
                "reference_positive_fraction": float(targets.mean()),
                "inner_prevalence_estimate": float(inner_prevalence),
                "true_positive": int(metric["true_positive"]),
                "false_positive": int(metric["false_positive"]),
                "true_negative": int(metric["true_negative"]),
                "false_negative": int(metric["false_negative"]),
                "accuracy": float(metric["accuracy"]),
                "average_precision": float(metric["average_precision"]),
                "f1": float(metric["f1"]),
                "balanced_accuracy": float(metric["balanced_accuracy"]),
                "auroc": float(metric["auroc"]),
                "precision": float(metric["precision"]),
                "recall": float(metric["recall"]),
                **oracle_gaps,
            }
        )
    return rows


def _join_and_diagnose_fold(
    plan: HeadroomPlan,
    early_plan: early_runner.Plan,
    fold: early_aggregate.AuthenticatedFold,
    prediction: CompactPrediction,
    prevalence: Mapping[str, float],
    manifest_rows: Sequence[early_runner.CacheRow],
) -> list[dict[str, Any]]:
    """Open fresh references only after strict replay and exact prediction auth."""

    expected_datasets = plan.family_datasets[fold.outer_family]
    expected_rows = tuple(row for row in manifest_rows if row.family == fold.outer_family)
    _require(len(expected_rows) == 4 * len(expected_datasets), "outer reference row population is incomplete")
    expected_keys = {
        (dataset, source_ordinal)
        for dataset in expected_datasets
        for source_ordinal in range(4)
    }
    _require(
        {(row.dataset, row.source_ordinal) for row in expected_rows} == expected_keys,
        "outer reference dataset/source population drifted",
    )
    rows: list[dict[str, Any]] = []
    coverage = np.zeros(prediction.count, dtype=np.int8)
    for cache_row in expected_rows:
        _require(cache_row.dataset not in plan.forbidden_datasets, "forbidden reference dataset access")
        projection = early_runner.load_early_cache_projection(
            early_plan, cache_row, include_labels=True
        )
        _require(projection.labels is not None, "fresh outer reference labels are unavailable")
        dataset_code = prediction.dataset_names.index(cache_row.dataset)
        selected = (
            (prediction.dataset_code == dataset_code)
            & (prediction.source_ordinal == cache_row.source_ordinal)
            & (prediction.source_index == cache_row.source_index)
        )
        selected_indices = np.flatnonzero(selected)
        _require(len(selected_indices) == projection.count, "prediction/reference row count mismatch")
        expected_order = np.concatenate(
            [np.flatnonzero(projection.block_indices == block) for block in (0, 1)]
        )
        _require(np.array_equal(prediction.scale_id[selected_indices], projection.scale_ids[expected_order]), "prediction/reference scale join drifted")
        _require(np.array_equal(prediction.center_seed_index[selected_indices], projection.center_indices[expected_order]), "prediction/reference center join drifted")
        _require(np.array_equal(prediction.scale_block_index[selected_indices], projection.block_indices[expected_order]), "prediction/reference block join drifted")
        _require(np.array_equal(prediction.assigned_row_index[selected_indices], projection.assigned_row_indices[expected_order]), "prediction/reference assigned-row join drifted")
        labels = np.asarray(projection.labels[expected_order], dtype=bool)
        for block_index, block_name in enumerate(BLOCK_NAMES):
            within = prediction.scale_block_index[selected_indices] == block_index
            group_indices = selected_indices[within]
            group_labels = labels[within]
            _require(len(group_indices) > 0 and len(group_indices) == len(group_labels), "outer group is empty or misaligned")
            coverage[group_indices] += 1
            rows.extend(
                diagnose_group(
                    outer_family=fold.outer_family,
                    dataset=cache_row.dataset,
                    source_ordinal=cache_row.source_ordinal,
                    source_index=cache_row.source_index,
                    block=block_name,
                    selected_candidate_id=prediction.selected_candidate_id,
                    scores=prediction.spatial_score[group_indices],
                    center_indices=prediction.center_seed_index[group_indices],
                    eligible=prediction.eligible[group_indices],
                    current_prediction=prediction.prediction[group_indices],
                    labels=group_labels,
                    inner_prevalence=float(prevalence[block_name]),
                )
            )
    _require(np.array_equal(coverage, np.ones(prediction.count, dtype=np.int8)), "reference groups do not exactly partition the prediction")
    return rows


def family_block_macro_rows(group_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Equal-weight dataset/source groups within every family and block."""

    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for family in FAMILY_ORDER:
        for block in BLOCK_NAMES:
            for method in METHOD_ORDER:
                selected = [
                    row
                    for row in group_rows
                    if row["outer_family"] == family
                    and row["block"] == block
                    and row["method"] == method
                ]
                _require(selected, f"macro group is empty: {family}/{block}/{method}")
                row = {
                    "outer_family": family,
                    "block": block,
                    "method": method,
                    "group_count": len(selected),
                    "sample_count_sum": sum(int(value["sample_count"]) for value in selected),
                    **{
                        metric: _finite_mean(
                            [float(value[metric]) for value in selected]
                        )
                        for metric in MACRO_METRICS
                    },
                    "outer_label_used": OUTER_LABEL_USE[method],
                    "legal_without_outer_label": DEPLOYABLE_WITHOUT_OUTER_LABEL[method],
                }
                lookup[(family, block, method)] = row
                rows.append(row)
    for family in FAMILY_ORDER:
        for block in BLOCK_NAMES:
            oracle_f1 = float(
                lookup[(family, block, "outer_group_oracle_max_f1")]["f1"]
            )
            gaps = {
                "oracle_f1_minus_current_f1": oracle_f1
                - float(lookup[(family, block, "current_selected_prediction")]["f1"]),
                "oracle_f1_minus_inner_prevalence_f1": oracle_f1
                - float(lookup[(family, block, "inner_prevalence_top_fraction")]["f1"]),
                "oracle_f1_minus_two_means_f1": oracle_f1
                - float(lookup[(family, block, "label_free_exact_1d_two_means")]["f1"]),
            }
            for method in METHOD_ORDER:
                lookup[(family, block, method)].update(gaps)
    return rows


def aggregate_summary(
    plan: HeadroomPlan,
    family_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize block-specific and all-block five-family macro headroom."""

    by_block: dict[str, dict[str, Any]] = {}
    for block in BLOCK_NAMES:
        method_records: dict[str, Any] = {}
        for method in METHOD_ORDER:
            selected = [
                row for row in family_rows if row["block"] == block and row["method"] == method
            ]
            _require(len(selected) == 5, f"complete five-family block macro is missing: {block}/{method}")
            method_records[method] = {
                metric: _finite_mean([float(row[metric]) for row in selected])
                for metric in MACRO_METRICS
            }
            method_records[method].update(
                {
                    "outer_label_used": OUTER_LABEL_USE[method],
                    "legal_without_outer_label": DEPLOYABLE_WITHOUT_OUTER_LABEL[method],
                }
            )
        oracle_f1 = method_records["outer_group_oracle_max_f1"]["f1"]
        current_f1 = method_records["current_selected_prediction"]["f1"]
        inner_prevalence_f1 = method_records["inner_prevalence_top_fraction"]["f1"]
        two_means_f1 = method_records["label_free_exact_1d_two_means"]["f1"]
        by_block[block] = {
            "methods": method_records,
            "oracle_f1_minus_current_f1": oracle_f1 - current_f1,
            "oracle_f1_minus_inner_prevalence_f1": oracle_f1
            - inner_prevalence_f1,
            "oracle_f1_minus_two_means_f1": oracle_f1 - two_means_f1,
            "oracle_reaches_f1_0_70": oracle_f1 >= 0.70,
            "oracle_reaches_f1_0_80": oracle_f1 >= 0.80,
        }
    all_block: dict[str, Any] = {}
    for method in METHOD_ORDER:
        selected = [row for row in family_rows if row["method"] == method]
        _require(len(selected) == 10, f"all-block family macro is incomplete: {method}")
        all_block[method] = {
            metric: _finite_mean([float(row[metric]) for row in selected])
            for metric in MACRO_METRICS
        }
        all_block[method].update(
            {
                "outer_label_used": OUTER_LABEL_USE[method],
                "legal_without_outer_label": DEPLOYABLE_WITHOUT_OUTER_LABEL[method],
            }
        )
    oracle_f1 = float(all_block["outer_group_oracle_max_f1"]["f1"])
    current_f1 = float(all_block["current_selected_prediction"]["f1"])
    inner_prevalence_f1 = float(all_block["inner_prevalence_top_fraction"]["f1"])
    two_means_f1 = float(all_block["label_free_exact_1d_two_means"]["f1"])
    parent_f1_delta = current_f1 - plan.parent_family_macro_f1
    parent_f1_reproduced = bool(
        np.isfinite(parent_f1_delta)
        and abs(parent_f1_delta) <= plan.parent_family_macro_f1_tolerance
    )
    _require(
        parent_f1_reproduced,
        (
            "current arm failed to reproduce authenticated parent family-macro F1: "
            f"observed={current_f1:.17g}, expected={plan.parent_family_macro_f1:.17g}, "
            f"delta={parent_f1_delta:.17g}"
        ),
    )
    if oracle_f1 >= 0.70:
        answer = (
            "posthoc_group_threshold_headroom_reaches_0_70_so_calibration_remains_"
            "plausible_but_is_not_proven_and_the_oracle_is_not_deployable"
        )
    else:
        answer = (
            "even_the_posthoc_group_specific_oracle_is_below_0_70_so_spatial_score_"
            "ordering_is_insufficient_for_a_calibration_only_fix"
        )
    return {
        "schema": SUMMARY_SCHEMA,
        "experiment": EXPERIMENT,
        "status": "completed",
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "evidence_scope": "exposed_development_posthoc_diagnostic_only",
        "formal_confirmation": False,
        "oracle_is_deployable": False,
        "oracle_definition": "outer_label_group_specific_tie_aware_max_f1_upper_bound",
        "family_macro_definition": "equal_dataset_source_groups_within_family_block_then_equal_five_families",
        "authenticated_parent_family_macro_f1": plan.parent_family_macro_f1,
        "current_selected_prediction_family_macro_f1": current_f1,
        "parent_f1_reproduction_absolute_tolerance": plan.parent_family_macro_f1_tolerance,
        "parent_f1_reproduction_delta": parent_f1_delta,
        "parent_f1_reproduced": parent_f1_reproduced,
        "block_results": by_block,
        "all_block_family_macro": all_block,
        "all_block_oracle_f1_minus_current_f1": oracle_f1 - current_f1,
        "all_block_oracle_f1_minus_inner_prevalence_f1": oracle_f1
        - inner_prevalence_f1,
        "all_block_oracle_f1_minus_two_means_f1": oracle_f1 - two_means_f1,
        "all_block_oracle_reaches_f1_0_70": oracle_f1 >= 0.70,
        "all_block_oracle_reaches_f1_0_80": oracle_f1 >= 0.80,
        "diagnostic_answer": answer,
        "next_method_rule": (
            "only_inner_prevalence_or_label_free_two_means_may_be_carried_"
            "forward_and_requires_new_pre_registered_nested_family_validation"
        ),
    }


def _artifact(path: Path, sha256: str) -> dict[str, Any]:
    return {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256}


def _atomic_bytes(path: Path, content: bytes) -> str:
    return early_runner._publish_no_replace(
        path, lambda stream: stream.write(content), text_mode=False
    )


def begin_output(
    output_directory: str | Path,
    plan: HeadroomPlan,
    input_payload: Mapping[str, Any],
) -> tuple[Path, dict[str, dict[str, Any]]]:
    """Freeze exact diagnostic inputs before this runner's reference projection."""

    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(f"immutable output directory exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    config_path = destination / "frozen_config.yaml"
    config_sha = _atomic_bytes(config_path, plan.path.read_bytes())
    _require(config_sha == plan.sha256, "frozen config copy drifted")
    input_manifest = early_runner._manifest_with_self_hash(dict(input_payload))
    input_path = destination / "input_manifest.json"
    input_sha = early_runner._atomic_json(input_path, input_manifest)
    return destination, {
        config_path.name: _artifact(config_path, config_sha),
        input_path.name: _artifact(input_path, input_sha),
    }


def finish_output(
    destination: Path,
    initial_artifacts: Mapping[str, Mapping[str, Any]],
    *,
    reporting_commit: str,
    prevalence_rows: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    summary_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    artifacts = {name: dict(record) for name, record in initial_artifacts.items()}
    csv_outputs = (
        ("inner_prevalence_estimates.csv", PREVALENCE_FIELDS, prevalence_rows),
        ("group_metrics.csv", GROUP_FIELDS, group_rows),
        ("family_block_macro_metrics.csv", MACRO_FIELDS, family_rows),
    )
    for name, fields, rows in csv_outputs:
        path = destination / name
        digest = early_runner._atomic_csv(path, fields, rows)
        artifacts[name] = _artifact(path, digest)
    summary = early_runner._manifest_with_self_hash(dict(summary_payload))
    summary_path = destination / "aggregate_summary.json"
    summary_sha = early_runner._atomic_json(summary_path, summary)
    artifacts[summary_path.name] = _artifact(summary_path, summary_sha)
    result = early_runner._manifest_with_self_hash(
        {
            "schema": RESULT_SCHEMA,
            "experiment": EXPERIMENT,
            "status": "completed",
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "reporting_git_commit": reporting_commit,
            "early_numerical_git_commit": EXPECTED_EARLY_COMMIT,
            "early_config_sha256": EXPECTED_EARLY_CONFIG_SHA256,
            "group_count": len(group_rows) // len(METHOD_ORDER),
            "method_group_row_count": len(group_rows),
            "family_block_method_row_count": len(family_rows),
            "oracle_is_deployable": False,
            "formal_confirmation": False,
            "artifacts": artifacts,
        }
    )
    result_path = destination / "result_manifest.json"
    result_sha = early_runner._atomic_json(result_path, result)
    _require(
        {path.name for path in destination.iterdir()}
        == {
            "frozen_config.yaml",
            "input_manifest.json",
            "inner_prevalence_estimates.csv",
            "group_metrics.csv",
            "family_block_macro_metrics.csv",
            "aggregate_summary.json",
            "result_manifest.json",
        },
        "pre-completion output file population drifted",
    )
    completion = early_runner._manifest_with_self_hash(
        {
            "schema": COMPLETE_SCHEMA,
            "experiment": EXPERIMENT,
            "status": "completed",
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "reporting_git_commit": reporting_commit,
            "early_numerical_git_commit": EXPECTED_EARLY_COMMIT,
            "result_manifest_file": result_path.name,
            "result_manifest_file_sha256": result_sha,
            "result_manifest_content_sha256": result["content_sha256"],
            "completed_utc": early_runner._utc_now(),
        }
    )
    completion_path = destination / "RUN_COMPLETE.json"
    completion_sha = early_runner._atomic_json(completion_path, completion)
    _require(
        {path.name for path in destination.iterdir()}
        == {
            "frozen_config.yaml",
            "input_manifest.json",
            "inner_prevalence_estimates.csv",
            "group_metrics.csv",
            "family_block_macro_metrics.csv",
            "aggregate_summary.json",
            "result_manifest.json",
            "RUN_COMPLETE.json",
        },
        "completed output file population drifted",
    )
    return {
        "summary": summary,
        "result_manifest_file_sha256": result_sha,
        "completion_file_sha256": completion_sha,
        "output_directory": str(destination),
    }


def run(
    config_path: str | Path,
    early_config_path: str | Path,
    fold_directories: Sequence[str | Path],
    output_directory: str | Path,
    *,
    expected_reporting_commit: str,
    kinematic_input_manifest_path: str | Path,
    kinematic_input_manifest_file_sha256: str,
    synthetic_pass_path: str | Path,
    synthetic_pass_file_sha256: str,
    sidecar_root: str | Path,
    sidecar_population_manifest_path: str | Path,
    sidecar_population_manifest_file_sha256: str,
) -> Mapping[str, Any]:
    """Authenticate five folds, freeze inputs, then compute the diagnostic."""

    plan = load_plan(config_path)
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(f"immutable output directory exists: {destination}")
    early_plan = _bind_authenticated_early_plan(
        plan,
        early_config_path,
        expected_reporting_commit=expected_reporting_commit,
        kinematic_input_manifest_path=kinematic_input_manifest_path,
        kinematic_input_manifest_file_sha256=kinematic_input_manifest_file_sha256,
        synthetic_pass_path=synthetic_pass_path,
        synthetic_pass_file_sha256=synthetic_pass_file_sha256,
        sidecar_root=sidecar_root,
        sidecar_population_manifest_path=sidecar_population_manifest_path,
        sidecar_population_manifest_file_sha256=sidecar_population_manifest_file_sha256,
    )
    folds = authenticate_folds(plan, early_plan, fold_directories)
    manifest_rows, input_identity = early_runner.load_cache_rows(early_plan)
    _require(
        set(row.dataset for row in manifest_rows) == set(plan.allowed_datasets),
        "cache manifest does not contain exactly the eight allowed datasets",
    )
    _require(
        all(row.dataset not in plan.forbidden_datasets for row in manifest_rows),
        "cache manifest contains a forbidden dataset",
    )
    input_payload = {
        "schema": INPUT_SCHEMA,
        "experiment": EXPERIMENT,
        "status": "frozen_before_diagnostic_reference_projection",
        "config_path": str(plan.path),
        "config_sha256": plan.sha256,
        "reporting_project_root": plan.reporting_project_root,
        "reporting_git_commit": expected_reporting_commit,
        "reporting_worktree_clean": True,
        "early_numerical_git_commit": EXPECTED_EARLY_COMMIT,
        "early_config_absolute_path": plan.early_config_absolute_path,
        "early_config_sha256": EXPECTED_EARLY_CONFIG_SHA256,
        "early_runner_sha256": EXPECTED_EARLY_RUNNER_SHA256,
        "early_aggregator_sha256": EXPECTED_EARLY_AGGREGATOR_SHA256,
        "cache_input_manifest": early_runner._json_safe(input_identity),
        "strict_fold_fresh_replay_and_reference_authentication_completed": True,
        "diagnostic_reference_projection_started_after_this_manifest": True,
        "folds": [
            {
                "outer_family": fold.outer_family,
                "run_directory": str(fold.path),
                "required_directory_basename": plan.expected_fold_basenames[
                    fold.outer_family
                ],
                "numerical_git_commit": fold.numerical_git_commit,
                "config_sha256": fold.config_sha256,
                "selected_candidate": early_runner._json_safe(fold.selected_candidate),
                "completion_file_sha256": fold.completion_file_sha256,
                "completion_content_sha256": fold.completion_content_sha256,
                "result_manifest_file_sha256": fold.result_manifest_file_sha256,
                "result_manifest_content_sha256": fold.result_manifest_content_sha256,
                "artifacts": early_runner._json_safe(fold.artifact_identities),
            }
            for fold in folds
        ],
        "allowed_datasets": list(plan.allowed_datasets),
        "forbidden_datasets": list(plan.forbidden_datasets),
        "oracle_is_deployable": False,
        "formal_confirmation": False,
    }
    destination, initial_artifacts = begin_output(destination, plan, input_payload)
    prevalence_by_family: dict[str, Mapping[str, float]] = {}
    prevalence_rows: list[dict[str, Any]] = []
    for fold in folds:
        estimates, rows = load_inner_prevalence(plan, fold)
        prevalence_by_family[fold.outer_family] = MappingProxyType(estimates)
        prevalence_rows.extend(rows)

    group_rows: list[dict[str, Any]] = []
    for fold in folds:
        prediction_identity = fold.artifact_identities["outer_predictions.npz"]
        manifest_identity = fold.artifact_identities["outer_prediction_manifest.json"]
        prediction = authenticate_prediction_artifact(
            fold.path / "outer_predictions.npz",
            fold.path / "outer_prediction_manifest.json",
            expected_prediction_file_sha256=str(prediction_identity["sha256"]),
            expected_manifest_file_sha256=str(manifest_identity["sha256"]),
            expected_outer_family=fold.outer_family,
            dataset_names=plan.family_datasets[fold.outer_family],
        )
        group_rows.extend(
            _join_and_diagnose_fold(
                plan,
                early_plan,
                fold,
                prediction,
                prevalence_by_family[fold.outer_family],
                manifest_rows,
            )
        )
        del prediction
    expected_group_count = len(plan.allowed_datasets) * 4 * len(BLOCK_NAMES)
    _require(len(group_rows) == expected_group_count * len(METHOD_ORDER), "diagnostic group population drifted")
    family_rows = family_block_macro_rows(group_rows)
    summary = aggregate_summary(plan, family_rows)
    return finish_output(
        destination,
        initial_artifacts,
        reporting_commit=expected_reporting_commit,
        prevalence_rows=prevalence_rows,
        group_rows=group_rows,
        family_rows=family_rows,
        summary_payload=summary,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--early-config", default=str(EARLY_CONFIG_PATH))
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-reporting-commit", required=True)
    parser.add_argument("--kinematic-input-manifest", required=True)
    parser.add_argument("--kinematic-input-manifest-sha256", required=True)
    parser.add_argument("--synthetic-pass", required=True)
    parser.add_argument("--synthetic-pass-sha256", required=True)
    parser.add_argument("--sidecar-root", required=True)
    parser.add_argument("--sidecar-population-manifest", required=True)
    parser.add_argument("--sidecar-population-manifest-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_args(argv)
    result = run(
        arguments.config,
        arguments.early_config,
        arguments.run_dir,
        arguments.output_dir,
        expected_reporting_commit=arguments.expected_reporting_commit,
        kinematic_input_manifest_path=arguments.kinematic_input_manifest,
        kinematic_input_manifest_file_sha256=arguments.kinematic_input_manifest_sha256,
        synthetic_pass_path=arguments.synthetic_pass,
        synthetic_pass_file_sha256=arguments.synthetic_pass_sha256,
        sidecar_root=arguments.sidecar_root,
        sidecar_population_manifest_path=arguments.sidecar_population_manifest,
        sidecar_population_manifest_file_sha256=arguments.sidecar_population_manifest_sha256,
    )
    print(json.dumps(early_runner._json_safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

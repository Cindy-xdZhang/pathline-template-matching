"""Raw-reintegration pipeline for ``mainExp_TemplateMatching_2.1``.

The production entry point in this module reads the frozen YAML configuration,
builds one immutable cache per dataset/source-time window, constructs the
balanced cross-scale template library, and evaluates the prior plus three exact
one-nearest-neighbour arms.  Flow staging is deliberately injected through
callbacks so the numerical experiment never guesses a machine-specific raw
data path.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml

from .arc_length_primitives import (
    ArcLengthScaleTable,
    build_arc_length_scale_table,
    build_arc_length_scale_union,
    integrate_arc_length_primitives_3d,
)
from .encoder import IndependentFMT3DConfig, encode_independent_fmt_3d
from .ivd import ivd_p95_reference_at_seeds
from .matcher import ExhaustiveMatchResult, ExhaustiveOneNearestNeighbor
from .metrics import average_precision, auroc
from .netcdf_io import FlowWindow3D
from .portable_flow import (
    PortableFlowWindow,
    canonical_array_sha256,
    canonical_json_sha256,
    load_portable_flow_window,
    sha256_file,
)
from .primitives import centered_xyz
from .vector_field import UnsteadyVectorField3D


EXPERIMENT = "mainExp_TemplateMatching_2.1"
EXPERIMENT31 = "mainExp_TemplateMatching_3.1"
SUPPORTED_EXPERIMENTS = (EXPERIMENT, EXPERIMENT31)
LEGACY_DX_VALUES = (
    0.250000000000,
    0.361111111111,
    0.472222222222,
    0.583333333333,
    0.694444444444,
    0.805555555556,
    0.916666666667,
    1.027777777778,
    1.138888888889,
    1.250000000000,
)
LEGACY_DS_VALUES = (
    0.125000000000,
    0.144444444444,
    0.163888888889,
    0.183333333333,
    0.202777777778,
    0.222222222222,
    0.241666666667,
    0.261111111111,
    0.280555555556,
    0.300000000000,
)
LEGACY_ARC_VALUES = (
    4.000000000000,
    4.888888888889,
    5.777777777778,
    6.666666666667,
    7.555555555556,
    8.444444444444,
    9.333333333333,
    10.222222222222,
    11.111111111111,
    12.000000000000,
)
EXPANDED_DX_VALUES = (
    0.125000000000,
    0.388888888889,
    0.652777777778,
    0.916666666667,
    1.180555555556,
    1.444444444444,
    1.708333333333,
    1.972222222222,
    2.236111111111,
    2.500000000000,
)
EXPANDED_DS_VALUES = (
    0.050000000000,
    0.100000000000,
    0.150000000000,
    0.200000000000,
    0.250000000000,
    0.300000000000,
    0.350000000000,
    0.400000000000,
    0.450000000000,
    0.500000000000,
)
EXPANDED_ARC_VALUES = (
    13.000000000000,
    20.444444444444,
    27.888888888889,
    35.333333333333,
    42.777777777778,
    50.222222222222,
    57.666666666667,
    65.111111111111,
    72.555555555556,
    80.000000000000,
)
METHOD_PRIOR = "eligible_train_candidate_prior_constant_score"
METHOD_RAW = "raw_centered_7x32x3_global_exact_1nn"
METHOD_PCA = "raw_centered_train_only_pca_161d_global_exact_1nn"
METHOD_FMT = (
    "fmt_independent_3d_161d_sha256_25fce29499c9089e_global_exact_1nn"
)
METHODS = (METHOD_PRIOR, METHOD_RAW, METHOD_PCA, METHOD_FMT)
ONE_NEAREST_NEIGHBOUR_METHODS = (METHOD_RAW, METHOD_PCA, METHOD_FMT)
METRIC_NAMES = (
    "accuracy",
    "average_precision",
    "f1",
    "balanced_accuracy",
    "auroc",
    "precision",
    "recall",
)


def configure_deterministic_execution() -> dict[str, Any]:
    """Enable the frozen deterministic PyTorch arithmetic contract."""

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


@dataclass(frozen=True)
class ScaleAssignmentBlock:
    """One contiguous scale block and its independently restarted assignment."""

    block_id: str
    scale_id_start: int
    scale_count: int
    assignment_seed: int

    @property
    def scale_id_stop(self) -> int:
        return int(self.scale_id_start + self.scale_count)


@dataclass(frozen=True)
class Phase21Plan:
    """Validated numerical plan for the frozen 2.1 or 3.1 experiment."""

    config_path: Path
    config_sha256: str
    dataset_registry_path: Path
    dataset_registry_sha256: str
    config: dict[str, Any]
    experiment: str
    output_root: str
    train_datasets: tuple[str, ...]
    test_datasets: tuple[str, ...]
    family_by_dataset: dict[str, str]
    source_count: int
    window_frame_count: int
    seed_shape_xyz: tuple[int, int, int]
    scale_table: ArcLengthScaleTable
    assignment_seed: int
    library_seed: int
    pca_components: int
    bootstrap_seed: int
    bootstrap_replicates: int
    descriptor_config: IndependentFMT3DConfig
    method_ids: tuple[str, ...]
    required_outputs: tuple[str, ...]
    maximum_source_frame_intervals: float = 12.0
    assignment_count_per_seed: int = 1
    scale_blocks: tuple[ScaleAssignmentBlock, ...] = ()
    maximum_library_templates: int = 64_000

    @property
    def datasets(self) -> tuple[str, ...]:
        return self.train_datasets + self.test_datasets

    @property
    def assigned_seed_count(self) -> int:
        return int(np.prod(self.seed_shape_xyz, dtype=np.int64))

    @property
    def assigned_primitive_count(self) -> int:
        return int(self.assigned_seed_count * self.assignment_count_per_seed)

    @property
    def artifact_tag(self) -> str:
        return "phase31" if self.experiment == EXPERIMENT31 else "phase21"

    @property
    def cache_schema(self) -> str:
        if self.experiment == EXPERIMENT31:
            return "pathline_template_matching.phase31_cache.v1"
        return "pathline_template_matching.phase21_cache.v2"

    @property
    def effective_scale_blocks(self) -> tuple[ScaleAssignmentBlock, ...]:
        if self.scale_blocks:
            return self.scale_blocks
        return (
            ScaleAssignmentBlock(
                block_id="legacy_2_1",
                scale_id_start=0,
                scale_count=len(self.scale_table),
                assignment_seed=self.assignment_seed,
            ),
        )

    def primitive_scale_assignment(self) -> np.ndarray:
        """Return block-major scale IDs for every assigned primitive row."""

        blocks = []
        for block in self.effective_scale_blocks:
            local = balanced_scale_assignment(
                self.assigned_seed_count, block.scale_count, block.assignment_seed
            )
            blocks.append(local + int(block.scale_id_start))
        return np.ascontiguousarray(np.concatenate(blocks), dtype=np.int32)

    def repeated_center_seeds(self, seeds_xyz: np.ndarray) -> np.ndarray:
        seeds = np.ascontiguousarray(np.asarray(seeds_xyz, dtype=np.float64))
        if seeds.shape != (self.assigned_seed_count, 3):
            raise ValueError("center seed table disagrees with the frozen seed grid")
        return np.ascontiguousarray(
            np.concatenate([seeds for _ in self.effective_scale_blocks]),
            dtype=np.float64,
        )

    def split_for(self, dataset: str) -> str:
        if dataset in self.train_datasets:
            return "train"
        if dataset in self.test_datasets:
            return "test"
        raise KeyError(f"dataset is outside the frozen split: {dataset}")

    def source_indices(self, native_frame_count: int) -> tuple[int, ...]:
        """Apply the config's field-independent source-index formula."""

        frames = int(native_frame_count)
        if frames < self.window_frame_count:
            raise ValueError(
                f"native frame count {frames} is smaller than the "
                f"{self.window_frame_count}-frame window"
            )
        maximum_start = frames - self.window_frame_count
        if self.source_count == 1:
            result = (0,)
        else:
            denominator = self.source_count - 1
            result = tuple(
                int(math.floor(index * maximum_start / denominator))
                for index in range(self.source_count)
            )
        if len(set(result)) != self.source_count:
            raise ValueError(
                f"source-index formula did not produce {self.source_count} unique "
                f"indices for {frames} frames: {result}"
            )
        return result

    def validate_production_contract(self) -> None:
        """Reject any silent drift from the committed 2.1/3.1 protocols."""

        if self.experiment not in SUPPORTED_EXPERIMENTS:
            raise ValueError(f"unsupported production experiment {self.experiment}")
        if len(self.train_datasets) != 8 or len(self.test_datasets) != 2:
            raise ValueError("production split must contain exactly eight train and two test flows")
        train_families = {self.family_by_dataset[name] for name in self.train_datasets}
        test_families = {self.family_by_dataset[name] for name in self.test_datasets}
        if train_families.intersection(test_families):
            raise ValueError("a physical family appears in both train and test")
        expected_window = 49 if self.experiment == EXPERIMENT31 else 13
        expected_horizon = 48.0 if self.experiment == EXPERIMENT31 else 12.0
        if self.source_count != 4 or self.window_frame_count != expected_window:
            raise ValueError(
                f"{self.experiment} requires four source times and "
                f"{expected_window}-frame windows"
            )
        if self.maximum_source_frame_intervals != expected_horizon:
            raise ValueError(
                f"{self.experiment} integration horizon must equal {expected_horizon:g}"
            )
        if self.seed_shape_xyz != (40, 40, 40) or self.assigned_seed_count != 64_000:
            raise ValueError("production seed grid must be exactly 40x40x40")
        expected_scales = 2_000 if self.experiment == EXPERIMENT31 else 1_000
        expected_assignments = 2 if self.experiment == EXPERIMENT31 else 1
        if len(self.scale_table) != expected_scales:
            raise ValueError(
                f"production scale table must contain exactly {expected_scales} tuples"
            )
        if self.assignment_count_per_seed != expected_assignments:
            raise ValueError(
                f"{self.experiment} requires {expected_assignments} assignment(s) per center seed"
            )
        assignment = self.primitive_scale_assignment()
        scale_counts = np.bincount(assignment, minlength=len(self.scale_table))
        if not np.all(scale_counts == 64):
            raise ValueError("production assignment must give every scale exactly 64 seeds")
        expected_blocks = (
            (
                ScaleAssignmentBlock("legacy_2_1", 0, 1_000, 15068),
                ScaleAssignmentBlock("expanded_3_1", 1_000, 1_000, 35068),
            )
            if self.experiment == EXPERIMENT31
            else (ScaleAssignmentBlock("legacy_2_1", 0, 1_000, 15068),)
        )
        if self.effective_scale_blocks != expected_blocks:
            raise ValueError("scale block IDs, ranges, or assignment seeds drifted")
        if self.experiment == EXPERIMENT31:
            assignment_config = _as_mapping(
                self.config.get("scale_assignment"), name="scale_assignment"
            )
            legacy_assignment_hash = canonical_array_sha256(
                np.ascontiguousarray(assignment[: self.assigned_seed_count])
            )
            if assignment_config.get("legacy_assignment_canonical_sha256") != (
                legacy_assignment_hash
            ):
                raise ValueError(
                    "3.1 legacy assignment canonical SHA-256 differs from the frozen 2.1 mapping"
                )
        if self.assignment_seed != 15068 or self.library_seed != 15068:
            raise ValueError("production assignment and library seeds must both equal 15068")
        expected_library_maximum = 128_000 if self.experiment == EXPERIMENT31 else 64_000
        if self.maximum_library_templates != expected_library_maximum:
            raise ValueError("maximum template population drifted")
        if self.pca_components != 161:
            raise ValueError("production Raw PCA width must equal 161")
        if self.bootstrap_seed != 25068 or self.bootstrap_replicates != 5000:
            raise ValueError("production bootstrap must use seed 25068 and 5000 replicates")
        if self.descriptor_config.descriptor_id != (
            "fmt_independent_3d_161d_sha256_25fce29499c9089e"
        ):
            raise ValueError("FMT descriptor identity drifted from the frozen config")
        if self.method_ids != METHODS:
            raise ValueError("method list or method order drifted from the frozen config")


def _validate_strict_cuda_workspace(
    plan: Phase21Plan, *, selected_device: str, strict_protocol: bool
) -> None:
    """Require the config-frozen deterministic cuBLAS workspace on CUDA."""

    if not strict_protocol or selected_device != "cuda":
        return
    execution = _as_mapping(plan.config.get("execution"), name="execution")
    expected = execution.get("cublas_workspace_config")
    if not isinstance(expected, str) or not expected:
        raise ValueError("config does not freeze cublas_workspace_config")
    actual = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if actual != expected:
        raise RuntimeError(
            "strict CUDA evaluation requires CUBLAS_WORKSPACE_CONFIG="
            f"{expected!r}, got {actual!r}"
        )


@dataclass(frozen=True)
class ResolvedFlowInput:
    """One canonical flow window plus staging provenance."""

    window: FlowWindow3D
    provenance: dict[str, Any]


@dataclass(frozen=True)
class StreamingCovariancePCA:
    """Train-only two-pass PCA from a 672x672 scatter matrix.

    The method is algebraically equivalent to right singular vectors of the
    centered sample matrix but never materializes that matrix or its left
    singular vectors.  It is therefore suitable for millions of candidates.
    """

    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    sample_count: int
    input_width: int
    solver: str = "deterministic_two_pass_streaming_covariance_eigendecomposition"

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.input_width:
            raise ValueError(
                f"PCA input must be [N,{self.input_width}], got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("PCA input contains NaN or Inf")
        return np.ascontiguousarray(
            (values - self.mean) @ self.components.T, dtype=np.float32
        )


def _pca_from_statistics(
    *,
    sample_count: int,
    feature_sum: np.ndarray,
    centered_scatter: np.ndarray,
    components: int,
) -> StreamingCovariancePCA:
    width = int(len(feature_sum))
    count = int(sample_count)
    component_count = int(components)
    if count < 2 or not 1 <= component_count <= min(count, width):
        raise ValueError("PCA sample/component counts are incompatible")
    feature_sum = np.asarray(feature_sum, dtype=np.float64)
    scatter = np.asarray(centered_scatter, dtype=np.float64)
    if feature_sum.shape != (width,) or scatter.shape != (width, width):
        raise ValueError("PCA sufficient-statistic shapes are incompatible")
    if not np.isfinite(feature_sum).all() or not np.isfinite(scatter).all():
        raise ValueError("PCA sufficient statistics contain NaN or Inf")
    scatter = 0.5 * (scatter + scatter.T)
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    order = np.argsort(-eigenvalues, kind="stable")
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    tolerance = max(1e-12, float(max(0.0, eigenvalues[0])) * 1e-10)
    if float(eigenvalues[-1]) < -tolerance:
        raise ValueError(
            "streaming PCA scatter has a materially negative eigenvalue: "
            f"{eigenvalues[-1]}"
        )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    selected = np.ascontiguousarray(
        eigenvectors[:, :component_count].T, dtype=np.float64
    )
    pivots = np.argmax(np.abs(selected), axis=1)
    signs = np.sign(selected[np.arange(component_count), pivots])
    signs[signs == 0.0] = 1.0
    selected *= signs[:, None]
    total = float(eigenvalues.sum())
    explained = (
        eigenvalues[:component_count] / total
        if total > 0.0
        else np.zeros(component_count, dtype=np.float64)
    )
    return StreamingCovariancePCA(
        mean=np.asarray(feature_sum / count, dtype=np.float32),
        components=np.asarray(selected, dtype=np.float32),
        singular_values=np.sqrt(eigenvalues[:component_count]).astype(np.float64),
        explained_variance_ratio=np.asarray(explained, dtype=np.float64),
        sample_count=count,
        input_width=width,
    )


def fit_streaming_covariance_pca(
    block_factory: Callable[[], Iterable[np.ndarray]],
    *,
    input_width: int,
    components: int,
) -> StreamingCovariancePCA:
    """Fit exact covariance PCA in two bounded-memory passes over train blocks."""

    width = int(input_width)
    if width < 1:
        raise ValueError("input_width must be positive")
    feature_sum = np.zeros(width, dtype=np.float64)
    sample_count = 0
    for raw_block in block_factory():
        block = np.asarray(raw_block, dtype=np.float32)
        if block.ndim != 2 or block.shape[1] != width or not np.isfinite(block).all():
            raise ValueError(f"PCA block must be finite [N,{width}], got {block.shape}")
        feature_sum += block.astype(np.float64).sum(axis=0)
        sample_count += len(block)
    if sample_count < 2:
        raise ValueError("streaming PCA requires at least two train candidates")
    mean = feature_sum / sample_count
    scatter = np.zeros((width, width), dtype=np.float64)
    second_pass_count = 0
    for raw_block in block_factory():
        block = np.asarray(raw_block, dtype=np.float32)
        if block.ndim != 2 or block.shape[1] != width or not np.isfinite(block).all():
            raise ValueError(f"PCA block must be finite [N,{width}], got {block.shape}")
        for start in range(0, len(block), 8192):
            centered = block[start : start + 8192].astype(np.float64) - mean
            scatter += centered.T @ centered
        second_pass_count += len(block)
    if second_pass_count != sample_count:
        raise ValueError("PCA block factory returned different populations across passes")
    return _pca_from_statistics(
        sample_count=sample_count,
        feature_sum=feature_sum,
        centered_scatter=scatter,
        components=components,
    )


@dataclass(frozen=True)
class CacheBuildSummary:
    """Manifest and audit evidence returned by one cache build."""

    cache_row: dict[str, Any]
    raw_input_row: dict[str, Any]
    derived_window_row: dict[str, Any]
    assignment_row: dict[str, Any]
    label_row: dict[str, Any]
    primitive_row: dict[str, Any]
    audit_rows: tuple[dict[str, Any], ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_verify_marker_outputs(
    marker_path: Path,
    marker: Mapping[str, Any],
    *,
    expected_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    if marker_path.name != expected_names[-1]:
        raise ValueError(f"verification marker must be named {expected_names[-1]}")
    root = marker_path.parent
    entries = list(root.iterdir())
    if any(not entry.is_file() for entry in entries) or {
        entry.name for entry in entries
    } != set(expected_names):
        raise ValueError("verification marker directory has an unexpected file set")
    rows = marker.get("outputs")
    if not isinstance(rows, list) or len(rows) != len(expected_names) - 1:
        raise ValueError("verification marker output audit is incomplete")
    expected_evidence = expected_names[:-1]
    if [str(row.get("path")) for row in rows] != list(expected_evidence):
        raise ValueError("verification marker output order or names changed")
    for row, name in zip(rows, expected_evidence, strict=True):
        path = root / name
        if (
            int(row.get("size_bytes", -1)) != path.stat().st_size
            or row.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"verification evidence hash/size changed: {name}")
    if marker.get("outputs_content_sha256") != canonical_json_sha256(rows):
        raise ValueError("verification marker output-list SHA-256 changed")
    return [dict(row) for row in rows]


def validate_phase31_synthetic_pass(
    plan: Phase21Plan,
    marker_path: str | Path,
    *,
    verify_config_path: str | Path,
    current_git_commit: str,
) -> dict[str, Any]:
    """Validate Phase A completely before any real-flow path is opened."""

    if plan.experiment != EXPERIMENT31:
        raise ValueError("synthetic-pass marker is defined only for mainExp 3.1")
    source = Path(marker_path).resolve()
    marker = json.loads(source.read_text(encoding="utf-8"))
    verify_path = Path(verify_config_path).resolve()
    expected = {
        "schema": "pathline_template_matching.long_arc_synthetic_pass.v1",
        "experiment": "Verify_LongArcHorizon_1.1",
        "phase": "synthetic",
        "status": "synthetic_gate_passed_train_only_coverage_not_run",
        "git_commit": current_git_commit,
        "worktree_clean": True,
        "main_config_sha256": plan.config_sha256,
        "verify_config_sha256": sha256_file(verify_path),
        "dataset_registry_sha256": plan.dataset_registry_sha256,
        "train_only_coverage_gate_run": False,
        "final_verify_pass": False,
    }
    drift = {
        name: (marker.get(name), value)
        for name, value in expected.items()
        if marker.get(name) != value
    }
    if drift:
        raise ValueError(f"invalid Phase A synthetic marker: {drift}")
    names = (
        "frozen_verify_config.yaml",
        "frozen_main_config.yaml",
        "synthetic_verification.json",
        "scale_union_manifest.json",
        "assignment_verification.json",
        "environment_versions.json",
        "SYNTHETIC_PASS.json",
    )
    output_rows = _validate_verify_marker_outputs(
        source, marker, expected_names=names
    )
    if sha256_file(source.parent / "frozen_main_config.yaml") != plan.config_sha256:
        raise ValueError("Phase A frozen main config differs from the current plan")
    if sha256_file(source.parent / "frozen_verify_config.yaml") != sha256_file(
        verify_path
    ):
        raise ValueError("Phase A frozen Verify config differs from the current file")
    return {
        "path": str(source),
        "file_size": int(source.stat().st_size),
        "file_sha256": sha256_file(source),
        "git_commit": current_git_commit,
        "main_config_sha256": plan.config_sha256,
        "verify_config_sha256": sha256_file(verify_path),
        "dataset_registry_sha256": plan.dataset_registry_sha256,
        "outputs": output_rows,
    }


def validate_phase31_train_coverage_pass(
    plan: Phase21Plan,
    marker_path: str | Path,
    *,
    synthetic_pass_path: str | Path,
    verify_config_path: str | Path,
    current_git_commit: str,
) -> dict[str, Any]:
    """Validate Phase B and its recorded Phase A marker before test access."""

    synthetic = validate_phase31_synthetic_pass(
        plan,
        synthetic_pass_path,
        verify_config_path=verify_config_path,
        current_git_commit=current_git_commit,
    )
    source = Path(marker_path).resolve()
    marker = json.loads(source.read_text(encoding="utf-8"))
    expected = {
        "schema": "pathline_template_matching.long_arc_train_coverage_pass.v1",
        "experiment": "Verify_LongArcHorizon_1.1",
        "phase": "train_coverage",
        "status": "passed",
        "git_commit": current_git_commit,
        "worktree_clean": True,
        "main_config_sha256": plan.config_sha256,
        "verify_config_sha256": synthetic["verify_config_sha256"],
        "dataset_registry_sha256": plan.dataset_registry_sha256,
        "synthetic_pass_file_sha256": synthetic["file_sha256"],
        "final_verify_pass": True,
    }
    drift = {
        name: (marker.get(name), value)
        for name, value in expected.items()
        if marker.get(name) != value
    }
    if drift:
        raise ValueError(f"invalid Phase B train-coverage marker: {drift}")
    names = (
        "frozen_verify_config.yaml",
        "frozen_main_config.yaml",
        "train_cache_input_manifest.json",
        "train_only_coverage_diagnostics.csv",
        "train_only_coverage_summary.json",
        "environment_versions.json",
        "verification.json",
        "TRAIN_COVERAGE_PASS.json",
    )
    output_rows = _validate_verify_marker_outputs(
        source, marker, expected_names=names
    )
    if sha256_file(source.parent / "frozen_main_config.yaml") != plan.config_sha256:
        raise ValueError("Phase B frozen main config differs from the current plan")
    if sha256_file(source.parent / "frozen_verify_config.yaml") != synthetic[
        "verify_config_sha256"
    ]:
        raise ValueError("Phase B frozen Verify config differs from the current file")
    verification_path = source.parent / "verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    portable_evidence_value = verification.get("train_portable_population_pass")
    if not isinstance(portable_evidence_value, Mapping):
        raise ValueError("Phase B verification lacks train portable-population evidence")
    portable_evidence = dict(portable_evidence_value)
    portable_marker_path = Path(str(portable_evidence.get("path", ""))).resolve()
    portable_marker_sha = str(portable_evidence.get("file_sha256", ""))
    if (
        portable_evidence.get("access_scope") != "train-only"
        or int(portable_evidence.get("file_size", -1)) < 1
        or not portable_marker_path.is_file()
        or portable_marker_path.stat().st_size
        != int(portable_evidence["file_size"])
        or sha256_file(portable_marker_path) != portable_marker_sha
        or marker.get("train_portable_population_pass_file_sha256")
        != portable_marker_sha
        or verification.get("train_portable_population_pass_file_sha256")
        != portable_marker_sha
    ):
        raise ValueError("Phase B train portable-population evidence changed")
    cache_input = json.loads(
        (source.parent / "train_cache_input_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if cache_input.get("train_portable_population_pass") != portable_evidence:
        raise ValueError(
            "Phase B input manifest and verification portable evidence differ"
        )
    if (
        verification.get("status") != "passed"
        or verification.get("final_verify_pass") is not True
        or verification.get("train_only") is not True
        or verification.get("no_test_dataset_access") is not True
        or verification.get("synthetic_pass_file_sha256")
        != synthetic["file_sha256"]
        or verification.get("git_commit") != current_git_commit
        or verification.get("main_config_sha256") != plan.config_sha256
        or verification.get("verify_config_sha256")
        != synthetic["verify_config_sha256"]
    ):
        raise ValueError("Phase B verification.json does not prove the frozen pass")
    if marker.get("verification_file_sha256") != sha256_file(verification_path):
        raise ValueError("Phase B marker verification.json SHA-256 changed")
    return {
        "path": str(source),
        "file_size": int(source.stat().st_size),
        "file_sha256": sha256_file(source),
        "git_commit": current_git_commit,
        "main_config_sha256": plan.config_sha256,
        "verify_config_sha256": synthetic["verify_config_sha256"],
        "dataset_registry_sha256": plan.dataset_registry_sha256,
        "synthetic_pass_file_sha256": synthetic["file_sha256"],
        "synthetic_pass": synthetic,
        "train_portable_population_pass": portable_evidence,
        "outputs": output_rows,
    }


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _as_string_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(str(item) for item in value)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique non-empty values")
    return result


def _load_phase_plan(
    config_path: str | Path, *, expected_experiment: str
) -> Phase21Plan:
    """Strictly parse a supported raw-reintegration experiment config."""

    path = Path(config_path).resolve()
    payload = path.read_bytes()
    parsed = yaml.safe_load(payload)
    config = _as_mapping(parsed, name="config")
    _require(expected_experiment in SUPPORTED_EXPERIMENTS, "unsupported experiment profile")
    _require(config.get("experiment") == expected_experiment, "wrong experiment config")
    expected_phase = (
        "development_raw_reintegration_long_arc_horizon"
        if expected_experiment == EXPERIMENT31
        else "development_raw_reintegration"
    )
    _require(config.get("phase") == expected_phase, f"wrong {expected_experiment} phase")
    registry_entry = config.get("dataset_registry")
    _require(isinstance(registry_entry, str) and registry_entry, "dataset_registry is missing")
    raw_registry_path = Path(registry_entry)
    if raw_registry_path.is_absolute():
        registry_path = raw_registry_path.resolve()
    else:
        candidates = (
            (path.parent.parent / raw_registry_path).resolve(),
            (Path.cwd() / raw_registry_path).resolve(),
        )
        existing = list(dict.fromkeys(candidate for candidate in candidates if candidate.is_file()))
        _require(len(existing) == 1, f"dataset_registry cannot be resolved uniquely: {candidates}")
        registry_path = existing[0]
    registry_payload = registry_path.read_bytes()

    split = _as_mapping(config.get("split"), name="split")
    train = _as_string_tuple(split.get("train_datasets"), name="train_datasets")
    test = _as_string_tuple(split.get("test_datasets"), name="test_datasets")
    _require(not set(train).intersection(test), "dataset appears in both train and test")
    family_by_dataset: dict[str, str] = {}
    for section_name in ("train_physical_families", "test_physical_families"):
        families = _as_mapping(split.get(section_name), name=section_name)
        for family, members_value in families.items():
            members = _as_string_tuple(members_value, name=f"{section_name}.{family}")
            for dataset in members:
                _require(dataset not in family_by_dataset, f"duplicate family membership: {dataset}")
                family_by_dataset[dataset] = str(family)
    _require(
        set(family_by_dataset) == set(train + test),
        "physical-family maps do not exactly cover the frozen dataset split",
    )

    source_times = _as_mapping(config.get("source_times"), name="source_times")
    source_loading = _as_mapping(config.get("source_loading"), name="source_loading")
    seed_grid = _as_mapping(config.get("seed_grid"), name="seed_grid")
    scale_protocol = _as_mapping(config.get("scale_protocol"), name="scale_protocol")
    scale_assignment = _as_mapping(config.get("scale_assignment"), name="scale_assignment")
    descriptor = _as_mapping(config.get("descriptor"), name="descriptor")
    library = _as_mapping(config.get("library"), name="library")
    methods = config.get("methods")
    bootstrap = _as_mapping(config.get("bootstrap"), name="bootstrap")

    shape = tuple(int(value) for value in seed_grid.get("shape_xyz", ()))
    _require(len(shape) == 3 and min(shape) >= 2, "seed_grid.shape_xyz must have three values >=2")
    shape_count = int(np.prod(shape, dtype=np.int64))
    if expected_experiment == EXPERIMENT31:
        _require(
            int(seed_grid.get("unique_center_seed_count_per_source_time", -1))
            == shape_count
            and int(seed_grid.get("assignment_blocks_per_center_seed", -1)) == 2
            and int(seed_grid.get("assigned_primitive_count_per_source_time", -1))
            == 2 * shape_count
            and seed_grid.get("identical_center_coordinates_shared_by_both_blocks")
            is True
            and float(seed_grid.get("maximum_dx_grid_scale", np.nan)) == 2.5,
            "3.1 center-seed and two-block primitive counts drifted",
        )
    else:
        _require(
            int(seed_grid.get("assigned_seed_count_per_source_time", -1))
            == shape_count,
            "assigned seed count disagrees with seed grid shape",
        )
    _require(
        seed_grid.get("seed_index_order") == "z_outer_y_middle_x_inner",
        "unsupported seed-index order",
    )

    if expected_experiment == EXPERIMENT:
        dx = _as_mapping(
            scale_protocol.get("dx_grid_scale"), name="dx_grid_scale"
        ).get("values")
        ds = _as_mapping(
            scale_protocol.get("ds_frame_scale"), name="ds_frame_scale"
        ).get("values")
        arc = _as_mapping(
            scale_protocol.get("arc_length_grid_scale"), name="arc_length_grid_scale"
        ).get("values")
        scales = build_arc_length_scale_table(dx, ds, arc)
        scale_blocks = (
            ScaleAssignmentBlock("legacy_2_1", 0, 1_000, 15068),
        )
        _require(
            scale_protocol.get("cartesian_order")
            == "dx_outer_ds_middle_arc_inner",
            "scale Cartesian order drifted",
        )
    else:
        raw_blocks = scale_protocol.get("blocks")
        _require(
            isinstance(raw_blocks, list) and len(raw_blocks) == 2,
            "3.1 scale_protocol.blocks must contain exactly two blocks",
        )
        union_blocks: list[dict[str, object]] = []
        block_ids: list[str] = []
        block_starts: list[int] = []
        for ordinal, raw_value in enumerate(raw_blocks):
            block = _as_mapping(raw_value, name=f"scale_protocol.blocks[{ordinal}]")
            block_id = str(block.get("id", ""))
            start = int(block.get("scale_id_start", -1))
            block_ids.append(block_id)
            block_starts.append(start)
            union_blocks.append(
                {
                    "scale_id_start": start,
                    "dx_grid_scale": _as_mapping(
                        block.get("dx_grid_scale"),
                        name=f"scale_protocol.blocks[{ordinal}].dx_grid_scale",
                    ).get("values"),
                    "ds_frame_scale": _as_mapping(
                        block.get("ds_frame_scale"),
                        name=f"scale_protocol.blocks[{ordinal}].ds_frame_scale",
                    ).get("values"),
                    "arc_length_grid_scale": _as_mapping(
                        block.get("arc_length_grid_scale"),
                        name=f"scale_protocol.blocks[{ordinal}].arc_length_grid_scale",
                    ).get("values"),
                }
            )
        _require(
            block_ids == ["legacy_2_1", "expanded_3_1"]
            and block_starts == [0, 1_000],
            "3.1 scale block order or ID ranges drifted",
        )
        _require(
            scale_protocol.get("block_order")
            == ["legacy_2_1", "expanded_3_1"],
            "3.1 scale_protocol.block_order drifted",
        )
        scales = build_arc_length_scale_union(union_blocks)
        expected_scales = build_arc_length_scale_union(
            (
                {
                    "scale_id_start": 0,
                    "dx_grid_scale": LEGACY_DX_VALUES,
                    "ds_frame_scale": LEGACY_DS_VALUES,
                    "arc_length_grid_scale": LEGACY_ARC_VALUES,
                },
                {
                    "scale_id_start": 1_000,
                    "dx_grid_scale": EXPANDED_DX_VALUES,
                    "ds_frame_scale": EXPANDED_DS_VALUES,
                    "arc_length_grid_scale": EXPANDED_ARC_VALUES,
                },
            )
        )
        for field in (
            "scale_id",
            "dx_grid_scale",
            "ds_frame_scale",
            "arc_length_grid_scale",
        ):
            _require(
                np.array_equal(getattr(scales, field), getattr(expected_scales, field)),
                f"3.1 explicit {field} values drifted",
            )
        scale_blocks = ()
    _require(
        int(scale_protocol.get("expected_unique_tuple_count", -1)) == len(scales),
        "scale tuple count disagrees with explicit config values",
    )
    _require(
        int(scale_protocol.get("decimal_places", -1)) == 12,
        "scale decimal-place contract drifted",
    )

    if expected_experiment == EXPERIMENT31:
        assignment_values = scale_assignment.get("blocks")
        _require(
            isinstance(assignment_values, list) and len(assignment_values) == 2,
            "3.1 scale_assignment.blocks must contain exactly two blocks",
        )
        assignment_by_id: dict[str, int] = {}
        for ordinal, raw_value in enumerate(assignment_values):
            item = _as_mapping(raw_value, name=f"scale_assignment.blocks[{ordinal}]")
            block_id = str(item.get("id", ""))
            seed = item.get("seed")
            _require(
                block_id not in assignment_by_id
                and isinstance(seed, int)
                and not isinstance(seed, bool),
                "3.1 assignment block IDs and seeds must be unique integers",
            )
            assignment_by_id[block_id] = int(seed)
        _require(
            list(assignment_by_id) == ["legacy_2_1", "expanded_3_1"]
            and assignment_by_id
            == {"legacy_2_1": 15068, "expanded_3_1": 35068},
            "3.1 assignment block order or seeds drifted",
        )
        scale_blocks = (
            ScaleAssignmentBlock("legacy_2_1", 0, 1_000, 15068),
            ScaleAssignmentBlock("expanded_3_1", 1_000, 1_000, 35068),
        )
        assignment_seed = 15068
        assignment_count_per_seed = int(
            scale_assignment.get("assignment_count_per_seed", -1)
        )
    else:
        assignment_seed = int(scale_assignment.get("seed", -1))
        assignment_count_per_seed = int(
            scale_assignment.get("assignment_count_per_seed", -1)
        )

    horizon = float(source_times.get("maximum_future_horizon_frames", np.nan))
    _require(
        np.isfinite(horizon)
        and horizon > 0.0
        and int(source_loading.get("derived_window_frame_count", -1))
        == int(horizon) + 1
        and float(int(horizon)) == horizon,
        "source horizon and portable window frame count disagree",
    )

    descriptor_config = IndependentFMT3DConfig()
    _require(descriptor.get("id") == descriptor_config.descriptor_id, "descriptor ID mismatch")
    _require(
        descriptor.get("source_sha256") == descriptor_config.algorithm_source_sha256,
        "descriptor source SHA-256 mismatch",
    )
    _require(int(descriptor.get("feature_width", -1)) == 161, "descriptor width mismatch")
    _require(
        isinstance(methods, list) and all(isinstance(item, Mapping) for item in methods),
        "methods must be a list of mappings",
    )
    method_ids = tuple(str(item.get("id")) for item in methods)
    pca_method = next((dict(item) for item in methods if item.get("id") == METHOD_PCA), None)
    _require(pca_method is not None, "Raw PCA method is missing")
    _require(
        pca_method.get("pca_solver")
        == "deterministic_two_pass_streaming_covariance_eigendecomposition",
        "Raw PCA solver drifted from the bounded-memory method",
    )
    pca_width = int(pca_method.get("feature_width", -1))

    plan = Phase21Plan(
        config_path=path,
        config_sha256=_sha256_bytes(payload),
        dataset_registry_path=registry_path,
        dataset_registry_sha256=_sha256_bytes(registry_payload),
        config=config,
        experiment=str(config["experiment"]),
        output_root=str(config.get("output_root", "")),
        train_datasets=train,
        test_datasets=test,
        family_by_dataset=family_by_dataset,
        source_count=int(source_times.get("count_per_dataset", -1)),
        window_frame_count=int(source_loading.get("derived_window_frame_count", -1)),
        seed_shape_xyz=shape,
        scale_table=scales,
        assignment_seed=assignment_seed,
        library_seed=int(library.get("sampling_random_seed", -1)),
        pca_components=pca_width,
        bootstrap_seed=int(bootstrap.get("seed", -1)),
        bootstrap_replicates=int(bootstrap.get("replicates", -1)),
        descriptor_config=descriptor_config,
        method_ids=method_ids,
        required_outputs=tuple(str(value) for value in config.get("required_outputs", ())),
        maximum_source_frame_intervals=horizon,
        assignment_count_per_seed=assignment_count_per_seed,
        scale_blocks=scale_blocks,
        maximum_library_templates=int(
            library.get("maximum_global_template_count", -1)
        ),
    )
    plan.validate_production_contract()
    _require(
        int(source_times.get("minimum_required_frame_count", -1))
        >= plan.window_frame_count + plan.source_count - 1,
        "minimum source frame count drifted",
    )
    _require(source_times.get("selection_dependencies") == ["time_axis_length_only"],
             "source selection must depend only on time-axis length")
    _require(int(source_loading.get("max_spatial_dim", -1)) == 96,
             "loaded spatial maximum must equal 96")
    _require(library.get("maximum_templates_per_class_per_stratum") == 1,
             "library must select at most one row per class and stratum")
    _require(library.get("empty_class_action") == "skip_both_classes_and_audit",
             "empty-class action drifted")
    _require(
        library.get("sampling_rule")
        == "one_global_generator_draws_negative_then_positive_only_for_each_two_class_nonempty_stratum",
        "library sampling rule drifted",
    )
    visualization = config.get("visualization")
    _require(isinstance(visualization, Mapping), "visualization contract is missing")
    _require(int(visualization.get("source_ordinal", -1)) == 2,
             "visualization source ordinal drifted")
    _require(visualization.get("prediction_method") == METHOD_FMT,
             "visualization prediction method drifted")
    _require(
        visualization.get("metric_based_or_prediction_based_scene_selection")
        == "forbidden",
        "visualization scene selection must be performance-independent",
    )
    display_pathlines = visualization.get("display_center_pathlines", {})
    _require(
        isinstance(display_pathlines, Mapping)
        and int(display_pathlines.get("count", -1)) == 240
        and int(display_pathlines.get("negative_count", -1)) == 120
        and int(display_pathlines.get("positive_count", -1)) == 120
        and int(display_pathlines.get("selection_seed", -1)) == 15068,
        "visualization display-pathline contract drifted",
    )
    _require(int(visualization.get("png_dpi", -1)) == 360,
             "visualization PNG DPI drifted")
    _require(float(visualization.get("panel_alignment_tolerance_points", -1)) == 1.5,
             "visualization panel-alignment tolerance drifted")
    if expected_experiment == EXPERIMENT31:
        expected_exports = [
            "scene_npz",
            "svg_with_editable_text_and_rasterized_3d_marks",
            "pdf_with_editable_text_and_rasterized_3d_marks",
            "png_360dpi",
            "panel_alignment_json",
        ]
        _require(
            visualization.get("scale_blocks")
            == ["legacy_2_1", "expanded_3_1"]
            and visualization.get("figure_unit")
            == "one_test_dataset_by_one_scale_block"
            and int(visualization.get("expected_figure_count", -1)) == 4
            and visualization.get("visualization_manifest_unique_key")
            == ["dataset", "scale_block_id"]
            and visualization.get(
                "cross_block_aggregation_majority_vote_or_overplotting"
            )
            == "forbidden",
            "3.1 dataset-by-scale-block visualization contract drifted",
        )
        _require(
            visualization.get("exports") == expected_exports
            and visualization.get("global_manifest")
            == "visualization_manifest_json"
            and visualization.get("global_manifest_file_sha256_source")
            == "final_result_manifest"
            and visualization.get("visualization_manifest_required_file_fields")
            == ["relative_path", "export_kind", "size_bytes", "sha256"]
            and visualization.get(
                "every_required_export_must_have_file_sha256"
            )
            is True,
            "3.1 visualization export evidence contract drifted",
        )
    return plan


def load_phase21_plan(config_path: str | Path) -> Phase21Plan:
    """Strictly parse and validate ``mainExp_TemplateMatching_2.1``."""

    return _load_phase_plan(config_path, expected_experiment=EXPERIMENT)


def load_phase31_plan(config_path: str | Path) -> Phase21Plan:
    """Strictly parse and validate ``mainExp_TemplateMatching_3.1``."""

    return _load_phase_plan(config_path, expected_experiment=EXPERIMENT31)


def balanced_scale_assignment(count: int, scale_count: int, seed: int) -> np.ndarray:
    """Return the frozen PCG64 permutation/modulo assignment."""

    count = int(count)
    scale_count = int(scale_count)
    if count < 1 or scale_count < 1:
        raise ValueError("count and scale_count must be positive")
    permutation = np.random.Generator(np.random.PCG64(int(seed))).permutation(count)
    assignment = np.empty(count, dtype=np.int32)
    assignment[permutation] = np.arange(count, dtype=np.int64) % scale_count
    return assignment


def generate_phase21_seeds(
    vector_field: UnsteadyVectorField3D,
    shape_xyz: Sequence[int],
    maximum_dx_physical: float,
) -> np.ndarray:
    """Generate the endpoint-inclusive scale-independent interior seed grid."""

    shape = tuple(int(value) for value in shape_xyz)
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError("shape_xyz must contain three integers >=2")
    margin = float(maximum_dx_physical)
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("maximum_dx_physical must be positive and finite")
    low = vector_field.domain_min.astype(np.float64) + margin
    high = vector_field.domain_max.astype(np.float64) - margin
    if np.any(high <= low):
        raise ValueError("maximum dx leaves no scale-independent interior seed domain")
    x = np.linspace(low[0], high[0], shape[0], endpoint=True, dtype=np.float64)
    y = np.linspace(low[1], high[1], shape[1], endpoint=True, dtype=np.float64)
    z = np.linspace(low[2], high[2], shape[2], endpoint=True, dtype=np.float64)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return np.ascontiguousarray(
        np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=-1),
        dtype=np.float64,
    )


def _atomic_json(path: Path, value: Any) -> str:
    payload = json.dumps(
        _json_safe(value), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    _atomic_bytes(path, payload)
    return _sha256_bytes(payload)


def _fsync_parent_directory(path: Path) -> None:
    """Persist a completed rename on POSIX; Windows has no directory fd contract."""

    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)
    _fsync_parent_directory(path)


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("xb") as destination:
        np.savez_compressed(destination, **arrays)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)
    _fsync_parent_directory(path)
    return sha256_file(path)


def _atomic_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)
    _fsync_parent_directory(path)
    return sha256_file(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return int(bool(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return "" if not np.isfinite(numeric) else f"{numeric:.12g}"
    if value is None:
        return ""
    return value


def _coerce_resolved_input(
    value: Any,
    *,
    plan: Phase21Plan,
    dataset: str,
    source_index: int,
    strict_evidence: bool,
) -> ResolvedFlowInput:
    if isinstance(value, ResolvedFlowInput):
        resolved = value
    elif isinstance(value, PortableFlowWindow):
        resolved = ResolvedFlowInput(
            window=value.window,
            provenance={
                **dict(value.metadata),
                "portable_path": str(value.path.resolve()),
                "portable_file_sha256": value.file_sha256,
                "portable_file_size": int(value.path.stat().st_size),
            },
        )
    elif isinstance(value, FlowWindow3D):
        resolved = ResolvedFlowInput(window=value, provenance=value.metadata())
    elif isinstance(value, (str, Path)):
        portable = load_portable_flow_window(
            value,
            expected_dataset=dataset,
            expected_experiment=plan.experiment,
            expected_config_sha256=plan.config_sha256,
            expected_source_start_index=source_index,
        )
        return _coerce_resolved_input(
            portable,
            plan=plan,
            dataset=dataset,
            source_index=source_index,
            strict_evidence=strict_evidence,
        )
    elif isinstance(value, Mapping) and isinstance(value.get("window"), FlowWindow3D):
        resolved = ResolvedFlowInput(
            window=value["window"], provenance=dict(value.get("provenance", {}))
        )
    else:
        raise TypeError(
            "window resolver must return FlowWindow3D, PortableFlowWindow, "
            "ResolvedFlowInput, a portable NPZ path, or {'window','provenance'}"
        )

    window = resolved.window
    if int(window.source_start_index) != int(source_index):
        raise ValueError(
            f"resolved source index mismatch for {dataset}: "
            f"{window.source_start_index} != {source_index}"
        )
    if window.velocity.shape[0] != plan.window_frame_count:
        raise ValueError(
            f"{dataset}/{source_index} has {window.velocity.shape[0]} frames; "
            f"expected {plan.window_frame_count}"
        )
    if max(window.velocity.shape[1:4]) > 96 and strict_evidence:
        raise ValueError("production loaded spatial dimension exceeds maxdim96")
    provenance = dict(resolved.provenance)
    if strict_evidence:
        expected = {
            "dataset": dataset,
            "experiment": plan.experiment,
            "config_sha256": plan.config_sha256,
            "source_start_index": int(source_index),
            "split": plan.split_for(dataset),
            "dataset_registry_sha256": plan.dataset_registry_sha256,
        }
        drift = {
            key: (provenance.get(key), wanted)
            for key, wanted in expected.items()
            if provenance.get(key) != wanted
        }
        if drift:
            raise ValueError(f"portable provenance drift: {drift}")
        for field in (
            "source_file",
            "source_file_size",
            "source_file_sha256",
            "dataset_registry_sha256",
            "builder_git_commit",
            "coordinate_audit",
            "array_sha256",
            "combined_array_sha256",
            "portable_file_sha256",
        ):
            if field not in provenance:
                raise ValueError(f"portable provenance is missing {field}")
    return ResolvedFlowInput(window=window, provenance=provenance)


def _encode_fmt_in_chunks(
    primitives: np.ndarray,
    descriptor: IndependentFMT3DConfig,
    chunk_size: int,
) -> np.ndarray:
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("encoding chunk size must be positive")
    blocks = [
        encode_independent_fmt_3d(primitives[start : start + chunk_size], descriptor)
        for start in range(0, len(primitives), chunk_size)
    ]
    if not blocks:
        return np.empty((0, descriptor.feature_width()), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(blocks), dtype=np.float32)


def build_phase21_cache_slice(
    plan: Phase21Plan,
    *,
    dataset: str,
    source_ordinal: int,
    source_index: int,
    resolved_input: Any,
    cache_path: str | Path,
    integration_chunk_size: int = 2048,
    encoding_chunk_size: int = 4096,
    strict_evidence: bool = True,
    cache_builder_git_commit: str = "unrecorded_nonproduction",
) -> CacheBuildSummary:
    """Build one immutable 672D Raw + 161D FMT cache shard."""

    split = plan.split_for(dataset)
    execution_contract = configure_deterministic_execution()
    if strict_evidence and (
        len(cache_builder_git_commit) != 40
        or any(character not in "0123456789abcdef" for character in cache_builder_git_commit)
    ):
        raise ValueError("production cache build requires a full lowercase Git commit SHA")
    resolved = _coerce_resolved_input(
        resolved_input,
        plan=plan,
        dataset=dataset,
        source_index=int(source_index),
        strict_evidence=strict_evidence,
    )
    window = resolved.window
    provenance = resolved.provenance
    vector_field = UnsteadyVectorField3D.from_window(window)
    minimum_spacing = float(np.min(vector_field.grid_interval))
    maximum_dx = float(np.max(plan.scale_table.dx_grid_scale) * minimum_spacing)
    center_seeds = generate_phase21_seeds(
        vector_field, plan.seed_shape_xyz, maximum_dx
    )
    if len(center_seeds) != plan.assigned_seed_count:
        raise RuntimeError("seed generator changed the assigned population")
    seeds = plan.repeated_center_seeds(center_seeds)
    assignment = plan.primitive_scale_assignment()
    if len(seeds) != plan.assigned_primitive_count or assignment.shape != (
        plan.assigned_primitive_count,
    ):
        raise RuntimeError("block assignment changed the primitive population")

    center_reference_labels, center_ivd_values, ivd_threshold, ivd_mask = (
        ivd_p95_reference_at_seeds(
        window.velocity[0],
        window.spacing_xyz,
        window.coordinates_xyz,
        center_seeds,
        )
    )
    reference_labels = np.ascontiguousarray(
        np.concatenate(
            [center_reference_labels for _ in plan.effective_scale_blocks]
        ),
        dtype=np.bool_,
    )
    ivd_values = np.ascontiguousarray(
        np.concatenate([center_ivd_values for _ in plan.effective_scale_blocks]),
        dtype=np.float32,
    )
    # Preserve the exact IVD volume used for the p95 labels so later figures can
    # reconstruct the reference isosurface from this immutable cache alone.
    from .ivd import compute_ivd_3d

    ivd_volume = compute_ivd_3d(window.velocity[0], window.spacing_xyz)
    result = integrate_arc_length_primitives_3d(
        vector_field,
        seeds,
        vector_field.tmin,
        plan.scale_table,
        assignment,
        chunk_size=integration_chunk_size,
        maximum_source_frame_intervals=plan.maximum_source_frame_intervals,
    )
    valid_seed_index = result.valid_seed_indices.astype(np.int64, copy=False)
    valid_primitives = result.primitives
    center_sample_time = np.ascontiguousarray(
        valid_primitives[:, 0, :, 3], dtype=np.float32
    )
    centered = centered_xyz(valid_primitives)
    raw_features = np.ascontiguousarray(centered.reshape(len(centered), -1), dtype=np.float32)
    fmt_features = _encode_fmt_in_chunks(
        centered, plan.descriptor_config, encoding_chunk_size
    )
    if raw_features.shape != (len(valid_seed_index), 672):
        raise RuntimeError(f"Raw feature shape drifted: {raw_features.shape}")
    if fmt_features.shape != (len(valid_seed_index), 161):
        raise RuntimeError(f"FMT feature shape drifted: {fmt_features.shape}")
    valid_labels = np.ascontiguousarray(reference_labels[result.valid_mask], dtype=np.bool_)
    valid_scale_id = np.ascontiguousarray(assignment[result.valid_mask], dtype=np.int32)

    cache_arrays = {
        "raw_features": np.ascontiguousarray(raw_features, dtype=np.float32),
        "fmt_features": np.ascontiguousarray(fmt_features, dtype=np.float32),
        "valid_labels": np.ascontiguousarray(valid_labels, dtype=np.bool_),
        "valid_seed_index": np.ascontiguousarray(valid_seed_index, dtype=np.int64),
        "valid_scale_id": np.ascontiguousarray(valid_scale_id, dtype=np.int32),
        "center_sample_time": center_sample_time,
        "seeds_xyz": np.ascontiguousarray(seeds, dtype=np.float64),
        "reference_labels_all": np.ascontiguousarray(reference_labels, dtype=np.bool_),
        "ivd_values_all": np.ascontiguousarray(ivd_values, dtype=np.float32),
        "ivd_volume": np.ascontiguousarray(ivd_volume, dtype=np.float32),
        "scale_assignment": np.ascontiguousarray(assignment, dtype=np.int32),
        "valid_mask": np.ascontiguousarray(result.valid_mask, dtype=np.bool_),
        "line_steps": np.ascontiguousarray(result.line_steps, dtype=np.int32),
        "line_travel": np.ascontiguousarray(result.line_travel, dtype=np.float32),
        "line_end_time": np.ascontiguousarray(result.line_end_time, dtype=np.float32),
        "line_reached_target": np.ascontiguousarray(
            result.line_reached_target, dtype=np.bool_
        ),
    }
    if plan.experiment == EXPERIMENT31:
        # ``result.valid_seed_indices`` indexes the 128k block-major assigned
        # primitive rows.  It is not the shared 40^3 center-seed identity.
        # Store both identities explicitly so downstream matching and figures
        # cannot silently confuse the second block with new spatial seeds.
        valid_assigned_row_index = np.ascontiguousarray(
            valid_seed_index, dtype=np.int64
        )
        valid_center_seed_index = np.ascontiguousarray(
            valid_assigned_row_index % plan.assigned_seed_count, dtype=np.int64
        )
        valid_scale_block_index = np.ascontiguousarray(
            valid_assigned_row_index // plan.assigned_seed_count, dtype=np.int8
        )
        expected_block_index = np.empty(len(valid_scale_id), dtype=np.int8)
        for block_index, block in enumerate(plan.effective_scale_blocks):
            within = (valid_scale_id >= block.scale_id_start) & (
                valid_scale_id < block.scale_id_stop
            )
            expected_block_index[within] = block_index
        if not np.array_equal(valid_scale_block_index, expected_block_index):
            raise RuntimeError("3.1 valid row/block/scale identity is inconsistent")
        cache_arrays.update(
            {
                "valid_assigned_row_index": valid_assigned_row_index,
                "valid_center_seed_index": valid_center_seed_index,
                "valid_scale_block_index": valid_scale_block_index,
            }
        )
    array_hashes = {
        name: canonical_array_sha256(values) for name, values in cache_arrays.items()
    }
    metadata = {
        "schema": plan.cache_schema,
        "experiment": plan.experiment,
        "config_sha256": plan.config_sha256,
        "dataset_registry_path": str(plan.dataset_registry_path),
        "dataset_registry_sha256": resolved.provenance.get("dataset_registry_sha256"),
        "portable_builder_git_commit": resolved.provenance.get("builder_git_commit"),
        "cache_builder_git_commit": cache_builder_git_commit,
        "execution_contract": execution_contract,
        "descriptor_id": plan.descriptor_config.descriptor_id,
        "dataset": dataset,
        "physical_family": plan.family_by_dataset[dataset],
        "split": split,
        "source_ordinal": int(source_ordinal),
        "source_index": int(source_index),
        "source_time": float(window.time[0]),
        "source_frame_interval": float(vector_field.time_interval),
        "loaded_shape_TZYXC": list(window.velocity.shape),
        "spacing_xyz": window.spacing_xyz.tolist(),
        "domain_min_xyz": vector_field.domain_min.astype(float).tolist(),
        "domain_max_xyz": vector_field.domain_max.astype(float).tolist(),
        "spatial_strides": {key: int(value) for key, value in window.spatial_strides.items()},
        "seed_shape_xyz": list(plan.seed_shape_xyz),
        "assigned_count": int(len(seeds)),
        "valid_count": int(result.valid_mask.sum()),
        "invalid_count": int((~result.valid_mask).sum()),
        "reference_positive_assigned": int(reference_labels.sum()),
        "reference_positive_valid": int(valid_labels.sum()),
        "ivd_percentile": 95.0,
        "ivd_threshold": float(ivd_threshold),
        "ivd_volume_positive_fraction": float(ivd_mask.mean()),
        "ivd_seed_positive_fraction": float(reference_labels.mean()),
        "minimum_loaded_spacing": minimum_spacing,
        "maximum_dx_physical": maximum_dx,
        "integration_max_time_relative": float(result.integration_max_time),
        "array_sha256": array_hashes,
        "combined_array_sha256": canonical_json_sha256(array_hashes),
        "window_provenance": _json_safe(resolved.provenance),
    }
    if plan.experiment == EXPERIMENT31:
        block_rows = []
        for ordinal, block in enumerate(plan.effective_scale_blocks):
            start = ordinal * plan.assigned_seed_count
            stop = start + plan.assigned_seed_count
            block_assignment = np.ascontiguousarray(assignment[start:stop])
            block_rows.append(
                {
                    "id": block.block_id,
                    "scale_id_start": int(block.scale_id_start),
                    "scale_id_stop_exclusive": int(block.scale_id_stop),
                    "assignment_seed": int(block.assignment_seed),
                    "primitive_row_start": int(start),
                    "primitive_row_stop_exclusive": int(stop),
                    "assignment_sha256": canonical_array_sha256(block_assignment),
                }
            )
        metadata.update(
            {
                "unique_center_seed_count": int(len(center_seeds)),
                "assigned_primitive_count": int(len(seeds)),
                "assignment_count_per_seed": int(plan.assignment_count_per_seed),
                "center_seed_repetition_order": "block_major_then_center_seed_index",
                "valid_seed_index_semantics": (
                    "legacy_alias_of_valid_assigned_row_index_not_center_seed_index"
                ),
                "valid_identity_fields": [
                    "valid_assigned_row_index",
                    "valid_center_seed_index",
                    "valid_scale_block_index",
                ],
                "scale_block_ids_by_index": [
                    block.block_id for block in plan.effective_scale_blocks
                ],
                "center_seed_xyz_sha256": canonical_array_sha256(center_seeds),
                "maximum_source_frame_intervals": float(
                    plan.maximum_source_frame_intervals
                ),
                "scale_assignment_blocks": block_rows,
            }
        )
    cache = Path(cache_path)
    cache_sha = _atomic_npz(
        cache,
        {
            **cache_arrays,
            "metadata_json": np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ),
        },
    )
    cache_row = {
        "dataset": dataset,
        "physical_family": plan.family_by_dataset[dataset],
        "split": split,
        "source_ordinal": int(source_ordinal),
        "source_index": int(source_index),
        "path": str(cache.resolve()),
        "file_size": int(cache.stat().st_size),
        "file_sha256": cache_sha,
        "config_sha256": plan.config_sha256,
        "dataset_registry_sha256": provenance.get("dataset_registry_sha256"),
        "portable_builder_git_commit": provenance.get("builder_git_commit"),
        "cache_builder_git_commit": cache_builder_git_commit,
        "assigned_count": int(len(seeds)),
        "valid_count": int(result.valid_mask.sum()),
        "invalid_count": int((~result.valid_mask).sum()),
    }
    raw_input_row = {
        "dataset": dataset,
        "split": split,
        "registered_path": provenance.get("source_file", window.source_path),
        "size_bytes": provenance.get("source_file_size"),
        "sha256": provenance.get("source_file_sha256"),
        "kind": provenance.get("source_kind", "flow_source"),
        "config_sha256": plan.config_sha256,
        "dataset_registry_sha256": provenance.get("dataset_registry_sha256"),
        "portable_builder_git_commit": provenance.get("builder_git_commit"),
        "cache_builder_git_commit": cache_builder_git_commit,
    }
    derived_window_row = {
        "dataset": dataset,
        "split": split,
        "source_index": int(source_index),
        "frame_index_range": [int(source_index), int(source_index + plan.window_frame_count - 1)],
        "source_time": float(window.time[0]),
        "canonical_array_sha256s": provenance.get("array_sha256"),
        "combined_sha256": provenance.get("combined_array_sha256"),
        "portable_path": provenance.get("portable_path"),
        "portable_file_sha256": provenance.get("portable_file_sha256"),
        "coordinate_audit": provenance.get("coordinate_audit"),
        "config_sha256": plan.config_sha256,
        "dataset_registry_sha256": provenance.get("dataset_registry_sha256"),
        "portable_builder_git_commit": provenance.get("builder_git_commit"),
        "cache_builder_git_commit": cache_builder_git_commit,
    }
    assignment_row = {
        "dataset": dataset,
        "split": split,
        "source_index": int(source_index),
        "seed_count": int(len(seeds)),
        "seed_xyz_sha256": array_hashes["seeds_xyz"],
        "assignment_sha256": array_hashes["scale_assignment"],
        "assignment_seed": int(plan.assignment_seed),
        "scale_count": int(len(plan.scale_table)),
        "config_sha256": plan.config_sha256,
        "dataset_registry_sha256": provenance.get("dataset_registry_sha256"),
        "portable_builder_git_commit": provenance.get("builder_git_commit"),
        "cache_builder_git_commit": cache_builder_git_commit,
        "minimum_count_per_scale": int(np.bincount(assignment).min()),
        "maximum_count_per_scale": int(np.bincount(assignment).max()),
    }
    if plan.experiment == EXPERIMENT31:
        assignment_row.update(
            {
                "unique_center_seed_count": int(len(center_seeds)),
                "assigned_primitive_count": int(len(seeds)),
                "assignment_count_per_seed": int(plan.assignment_count_per_seed),
                "center_seed_repetition_order": "block_major_then_center_seed_index",
                "center_seed_xyz_sha256": canonical_array_sha256(center_seeds),
                "maximum_source_frame_intervals": float(
                    plan.maximum_source_frame_intervals
                ),
                "blocks": metadata["scale_assignment_blocks"],
            }
        )
    label_row = {
        "dataset": dataset,
        "split": split,
        "source_index": int(source_index),
        "ivd_threshold": float(ivd_threshold),
        "ivd_volume_sha256": array_hashes["ivd_volume"],
        "ivd_volume_storage": "stored_in_cache",
        "reference_labels_sha256": array_hashes["reference_labels_all"],
        "assigned_positive_count": int(reference_labels.sum()),
        "assigned_positive_fraction": float(reference_labels.mean()),
        "valid_positive_count": int(valid_labels.sum()),
        "config_sha256": plan.config_sha256,
        "dataset_registry_sha256": provenance.get("dataset_registry_sha256"),
        "portable_builder_git_commit": provenance.get("builder_git_commit"),
        "cache_builder_git_commit": cache_builder_git_commit,
    }
    primitive_row = {
        "dataset": dataset,
        "split": split,
        "source_index": int(source_index),
        "assigned_count": int(len(seeds)),
        "valid_count": int(result.valid_mask.sum()),
        "invalid_count": int((~result.valid_mask).sum()),
        "valid_mask_sha256": array_hashes["valid_mask"],
        "raw_features_sha256": array_hashes["raw_features"],
        "fmt_features_sha256": array_hashes["fmt_features"],
        "cache_file_sha256": cache_sha,
        "config_sha256": plan.config_sha256,
        "dataset_registry_sha256": provenance.get("dataset_registry_sha256"),
        "portable_builder_git_commit": provenance.get("builder_git_commit"),
        "cache_builder_git_commit": cache_builder_git_commit,
    }
    if plan.experiment == EXPERIMENT31:
        identity = {
            "experiment": plan.experiment,
            "maximum_source_frame_intervals": float(
                plan.maximum_source_frame_intervals
            ),
            "assignment_count_per_seed": int(plan.assignment_count_per_seed),
            "unique_center_seed_count": int(len(center_seeds)),
        }
        cache_row.update(identity)
        label_row.update(identity)
        primitive_row.update(identity)

    codes = assignment.astype(np.int64) * 2 + reference_labels.astype(np.int64)
    assigned_counts = np.bincount(codes, minlength=2 * len(plan.scale_table))
    valid_counts = np.bincount(
        codes[result.valid_mask], minlength=2 * len(plan.scale_table)
    )
    audit_rows: list[dict[str, Any]] = []
    for scale_id in range(len(plan.scale_table)):
        for class_id in (0, 1):
            offset = scale_id * 2 + class_id
            assigned_count = int(assigned_counts[offset])
            valid_count = int(valid_counts[offset])
            audit_rows.append(
                {
                    "dataset": dataset,
                    "physical_family": plan.family_by_dataset[dataset],
                    "split": split,
                    "source_ordinal": int(source_ordinal),
                    "source_index": int(source_index),
                    "scale_id": scale_id,
                    "dx_grid_scale": float(plan.scale_table.dx_grid_scale[scale_id]),
                    "ds_frame_scale": float(plan.scale_table.ds_frame_scale[scale_id]),
                    "arc_length_grid_scale": float(
                        plan.scale_table.arc_length_grid_scale[scale_id]
                    ),
                    "reference_class": class_id,
                    "assigned_count": assigned_count,
                    "valid_count": valid_count,
                    "invalid_count": assigned_count - valid_count,
                }
            )
            if plan.experiment == EXPERIMENT31:
                block_index = next(
                    index
                    for index, block in enumerate(plan.effective_scale_blocks)
                    if block.scale_id_start <= scale_id < block.scale_id_stop
                )
                audit_rows[-1].update(
                    {
                        "scale_block_index": int(block_index),
                        "scale_block_id": plan.effective_scale_blocks[
                            block_index
                        ].block_id,
                    }
                )
    return CacheBuildSummary(
        cache_row=cache_row,
        raw_input_row=raw_input_row,
        derived_window_row=derived_window_row,
        assignment_row=assignment_row,
        label_row=label_row,
        primitive_row=primitive_row,
        audit_rows=tuple(audit_rows),
    )


def _manifest_payload(
    kind: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment: str = EXPERIMENT,
) -> dict[str, Any]:
    safe_rows = [_json_safe(dict(row)) for row in rows]
    return {
        "schema": f"pathline_template_matching.{kind}.v1",
        "experiment": experiment,
        "row_count": len(safe_rows),
        "rows": safe_rows,
        "rows_content_sha256": canonical_json_sha256(safe_rows),
    }


def _scale_manifest(plan: Phase21Plan) -> dict[str, Any]:
    if plan.experiment == EXPERIMENT31:
        rows = []
        for block in plan.effective_scale_blocks:
            for index in range(block.scale_id_start, block.scale_id_stop):
                local = index - block.scale_id_start
                rows.append(
                    {
                        "scale_id": int(index),
                        "block_id": block.block_id,
                        "block_local_scale_id": int(local),
                        "dx_index": int(local // 100),
                        "ds_index": int((local // 10) % 10),
                        "arc_index": int(local % 10),
                        "dx_grid_scale": f"{plan.scale_table.dx_grid_scale[index]:.12f}",
                        "ds_frame_scale": f"{plan.scale_table.ds_frame_scale[index]:.12f}",
                        "arc_length_grid_scale": (
                            f"{plan.scale_table.arc_length_grid_scale[index]:.12f}"
                        ),
                    }
                )
        legacy_block = plan.effective_scale_blocks[0]
        expanded_block = plan.effective_scale_blocks[1]
        legacy_rows = rows[
            legacy_block.scale_id_start : legacy_block.scale_id_stop
        ]
        expanded_rows = rows[
            expanded_block.scale_id_start : expanded_block.scale_id_stop
        ]
        legacy_projection = [
            {
                "scale_id": row["scale_id"],
                "dx_grid_scale": row["dx_grid_scale"],
                "ds_frame_scale": row["ds_frame_scale"],
                "arc_length_grid_scale": row["arc_length_grid_scale"],
            }
            for row in legacy_rows
        ]
        legacy_subset_hash = canonical_json_sha256(legacy_projection)
        provenance_value = plan.config.get("provenance")
        if provenance_value is None and plan.config.get("profile") == "tiny_test_only":
            expected_legacy_hash = legacy_subset_hash
        else:
            provenance = _as_mapping(provenance_value, name="provenance")
            expected_legacy_hash = provenance.get(
                "parent_scale_manifest_rows_content_sha256"
            )
        if legacy_subset_hash != expected_legacy_hash:
            raise ValueError(
                "3.1 legacy scale projection differs from the frozen 2.1 rows hash"
            )
        return {
            "schema": "pathline_template_matching.phase31_scale_manifest.v1",
            "experiment": plan.experiment,
            "config_sha256": plan.config_sha256,
            "layout": "ordered_union_of_two_10x10x10_cartesian_blocks",
            "block_order": [block.block_id for block in plan.effective_scale_blocks],
            "block_scale_id_ranges": [
                {
                    "id": block.block_id,
                    "start": int(block.scale_id_start),
                    "stop_exclusive": int(block.scale_id_stop),
                }
                for block in plan.effective_scale_blocks
            ],
            "block_local_order": "dx_outer_ds_middle_arc_inner",
            "block_local_scale_id_formula": "((dx_index * 10) + ds_index) * 10 + arc_index",
            "legacy_2_1_scale_ids_preserved": True,
            "legacy_rows_content_sha256": canonical_json_sha256(legacy_rows),
            "legacy_scale_subset_sha256": legacy_subset_hash,
            "parent_scale_manifest_rows_content_sha256": expected_legacy_hash,
            "expanded_rows_content_sha256": canonical_json_sha256(expanded_rows),
            "maximum_source_frame_intervals": float(
                plan.maximum_source_frame_intervals
            ),
            "decimal_places": 12,
            "scale_count": len(rows),
            "rows": rows,
            "rows_content_sha256": canonical_json_sha256(rows),
        }
    rows = [
        {
            "scale_id": int(index),
            "dx_grid_scale": f"{plan.scale_table.dx_grid_scale[index]:.12f}",
            "ds_frame_scale": f"{plan.scale_table.ds_frame_scale[index]:.12f}",
            "arc_length_grid_scale": (
                f"{plan.scale_table.arc_length_grid_scale[index]:.12f}"
            ),
        }
        for index in range(len(plan.scale_table))
    ]
    return {
        "schema": "pathline_template_matching.phase21_scale_manifest.v1",
        "experiment": plan.experiment,
        "config_sha256": plan.config_sha256,
        "cartesian_order": "dx_outer_ds_middle_arc_inner",
        "scale_id_formula": "((dx_index * 10) + ds_index) * 10 + arc_index",
        "decimal_places": 12,
        "scale_count": len(rows),
        "rows": rows,
        "rows_content_sha256": canonical_json_sha256(rows),
    }


def build_phase21_caches(
    plan: Phase21Plan,
    run_dir: str | Path,
    *,
    frame_count_resolver: Callable[[str], int],
    window_resolver: Callable[[str, int, int], Any],
    integration_chunk_size: int = 2048,
    encoding_chunk_size: int = 4096,
    strict_evidence: bool = True,
    cache_builder_git_commit: str = "unrecorded_nonproduction",
) -> list[dict[str, Any]]:
    """Resolve all frozen 8:2 windows and build forty immutable cache shards."""

    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {root}")
    if (root / "cache_manifest.json").exists():
        raise FileExistsError("cache stage is immutable and already completed")

    summaries: list[CacheBuildSummary] = []
    for dataset in plan.datasets:
        native_frame_count = int(frame_count_resolver(dataset))
        source_indices = plan.source_indices(native_frame_count)
        split = plan.split_for(dataset)
        for ordinal, source_index in enumerate(source_indices):
            resolved = window_resolver(dataset, source_index, plan.window_frame_count)
            cache_path = (
                root
                / "cache"
                / split
                / dataset
                / f"source_{source_index:06d}.npz"
            )
            summary = build_phase21_cache_slice(
                plan,
                dataset=dataset,
                source_ordinal=ordinal,
                source_index=source_index,
                resolved_input=resolved,
                cache_path=cache_path,
                integration_chunk_size=integration_chunk_size,
                encoding_chunk_size=encoding_chunk_size,
                strict_evidence=strict_evidence,
                cache_builder_git_commit=cache_builder_git_commit,
            )
            summaries.append(summary)

    expected_count = len(plan.datasets) * plan.source_count
    if len(summaries) != expected_count:
        raise RuntimeError(f"built {len(summaries)} caches, expected {expected_count}")
    cache_rows = [summary.cache_row for summary in summaries]

    # A raw source commonly supplies four windows.  Deduplicate it while
    # refusing contradictory size/hash evidence.
    raw_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for summary in summaries:
        row = summary.raw_input_row
        key = (str(row["dataset"]), str(row["registered_path"]))
        previous = raw_by_key.get(key)
        if previous is not None and previous != row:
            raise ValueError(f"raw input provenance changed across windows: {key}")
        raw_by_key[key] = row

    manifest_files = {
        "cache_manifest.json": _manifest_payload(
            f"{plan.artifact_tag}_cache_manifest",
            cache_rows,
            experiment=plan.experiment,
        ),
        "raw_input_manifest.json": _manifest_payload(
            f"{plan.artifact_tag}_raw_input_manifest",
            list(raw_by_key.values()),
            experiment=plan.experiment,
        ),
        "derived_window_manifest.json": _manifest_payload(
            f"{plan.artifact_tag}_derived_window_manifest",
            [summary.derived_window_row for summary in summaries],
            experiment=plan.experiment,
        ),
        "seed_and_scale_assignment_manifest.json": _manifest_payload(
            f"{plan.artifact_tag}_seed_assignment_manifest",
            [summary.assignment_row for summary in summaries],
            experiment=plan.experiment,
        ),
        "label_manifest.json": _manifest_payload(
            f"{plan.artifact_tag}_label_manifest",
            [summary.label_row for summary in summaries],
            experiment=plan.experiment,
        ),
        "primitive_manifest.json": _manifest_payload(
            f"{plan.artifact_tag}_primitive_manifest",
            [summary.primitive_row for summary in summaries],
            experiment=plan.experiment,
        ),
    }
    manifest_hashes = {
        name: _atomic_json(root / name, value) for name, value in manifest_files.items()
    }
    audit_rows = [row for summary in summaries for row in summary.audit_rows]
    audit_fields = (
        "dataset",
        "physical_family",
        "split",
        "source_ordinal",
        "source_index",
        "scale_id",
        "dx_grid_scale",
        "ds_frame_scale",
        "arc_length_grid_scale",
        "reference_class",
        "assigned_count",
        "valid_count",
        "invalid_count",
    )
    if plan.experiment == EXPERIMENT31:
        audit_fields = audit_fields[:5] + (
            "scale_block_index",
            "scale_block_id",
        ) + audit_fields[5:]
    audit_sha = _atomic_csv(root / "audit_counts.csv", audit_rows, audit_fields)
    input_manifest = {
        "schema": f"pathline_template_matching.{plan.artifact_tag}_input_manifest.v1",
        "experiment": plan.experiment,
        "config_sha256": plan.config_sha256,
        "dataset_registry_path": str(plan.dataset_registry_path),
        "dataset_registry_sha256": plan.dataset_registry_sha256,
        "dataset_order": list(plan.datasets),
        "train_datasets": list(plan.train_datasets),
        "test_datasets": list(plan.test_datasets),
        "cache_count": len(cache_rows),
        "manifest_file_sha256": manifest_hashes,
        "audit_counts_file_sha256": audit_sha,
        "cache_file_sha256": {
            str(Path(row["path"]).relative_to(root)): row["file_sha256"]
            for row in cache_rows
        },
    }
    if plan.experiment == EXPERIMENT31:
        input_manifest.update(
            {
                "maximum_source_frame_intervals": float(
                    plan.maximum_source_frame_intervals
                ),
                "assignment_count_per_seed": int(plan.assignment_count_per_seed),
                "unique_center_seed_count_per_source_time": int(
                    plan.assigned_seed_count
                ),
                "assigned_primitive_count_per_source_time": int(
                    plan.assigned_primitive_count
                ),
            }
        )
    _atomic_json(root / "input_manifest.json", input_manifest)
    return cache_rows


def _load_cache(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f"cache file SHA-256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        base_required = {
            "raw_features",
            "fmt_features",
            "valid_labels",
            "valid_seed_index",
            "valid_scale_id",
            "center_sample_time",
            "seeds_xyz",
            "reference_labels_all",
            "ivd_values_all",
            "ivd_volume",
            "scale_assignment",
            "valid_mask",
            "line_steps",
            "line_travel",
            "line_end_time",
            "line_reached_target",
            "metadata_json",
        }
        phase31_identity = {
            "valid_assigned_row_index",
            "valid_center_seed_index",
            "valid_scale_block_index",
        }
        actual = set(archive.files)
        if actual not in (base_required, base_required | phase31_identity):
            raise ValueError(
                f"cache keys disagree for {path}: "
                f"base_missing={sorted(base_required-actual)}, "
                f"extra={sorted(actual-base_required-phase31_identity)}"
            )
        result = {
            name: np.asarray(archive[name]) for name in actual - {"metadata_json"}
        }
        metadata_scalar = np.asarray(archive["metadata_json"])
        if metadata_scalar.ndim != 0:
            raise ValueError(f"cache metadata_json is not scalar: {path}")
        result["metadata"] = json.loads(str(metadata_scalar.item()))
    is_phase31 = result["metadata"].get("experiment") == EXPERIMENT31
    has_phase31_identity = phase31_identity.issubset(result)
    if is_phase31 != has_phase31_identity:
        raise ValueError("cache experiment/3.1 valid-row identity fields disagree")
    count = len(result["valid_labels"])
    if result["raw_features"].shape != (count, 672):
        raise ValueError(f"Raw cache feature shape changed: {path}")
    if result["fmt_features"].shape != (count, 161):
        raise ValueError(f"FMT cache feature shape changed: {path}")
    if result["valid_seed_index"].shape != (count,) or result["valid_scale_id"].shape != (count,):
        raise ValueError(f"valid-row metadata shape changed: {path}")
    if is_phase31 and any(result[name].shape != (count,) for name in phase31_identity):
        raise ValueError(f"3.1 valid-row identity shape changed: {path}")
    if not np.isfinite(result["raw_features"]).all() or not np.isfinite(
        result["fmt_features"]
    ).all():
        raise ValueError(f"cache contains nonfinite features: {path}")
    return result


def recover_phase21_cache_summary(
    plan: Phase21Plan,
    *,
    cache_path: str | Path,
    dataset: str,
    source_ordinal: int,
    source_index: int,
    cache_builder_git_commit: str,
    strict_evidence: bool = True,
    expected_window_provenance: Mapping[str, Any] | None = None,
) -> CacheBuildSummary:
    """Validate an atomic shard and reconstruct a missing publish sidecar.

    This is the only supported Slurm-array resume path.  It never rewrites the
    cache and only publishes a sidecar after every stored array/metadata check
    succeeds.
    """

    path = Path(cache_path).resolve()
    cache = _load_cache(path)
    metadata = dict(cache["metadata"])
    split = plan.split_for(dataset)
    expected_metadata = {
        "schema": plan.cache_schema,
        "experiment": plan.experiment,
        "config_sha256": plan.config_sha256,
        "descriptor_id": plan.descriptor_config.descriptor_id,
        "dataset": dataset,
        "physical_family": plan.family_by_dataset[dataset],
        "split": split,
        "source_ordinal": int(source_ordinal),
        "source_index": int(source_index),
        "cache_builder_git_commit": cache_builder_git_commit,
    }
    drift = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if drift:
        raise ValueError(f"existing cache metadata cannot be resumed: {drift}")
    provenance = dict(metadata.get("window_provenance", {}))
    if strict_evidence and expected_window_provenance is None:
        raise ValueError(
            "strict cache recovery requires the currently validated portable-window provenance"
        )
    if expected_window_provenance is not None:
        expected_provenance = _json_safe(dict(expected_window_provenance))
        actual_provenance = _json_safe(provenance)
        if actual_provenance != expected_provenance:
            raise ValueError(
                "existing cache portable-window provenance differs from the current input"
            )
    if strict_evidence:
        for field, width in (
            ("dataset_registry_sha256", 64),
            ("builder_git_commit", 40),
        ):
            value = str(provenance.get(field, ""))
            if len(value) != width or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"existing cache has invalid portable provenance {field}")
    assigned_count = plan.assigned_primitive_count
    stored_array_names = [
        "raw_features",
        "fmt_features",
        "valid_labels",
        "valid_seed_index",
        "valid_scale_id",
        "center_sample_time",
        "seeds_xyz",
        "reference_labels_all",
        "ivd_values_all",
        "ivd_volume",
        "scale_assignment",
        "valid_mask",
        "line_steps",
        "line_travel",
        "line_end_time",
        "line_reached_target",
    ]
    if plan.experiment == EXPERIMENT31:
        stored_array_names.extend(
            (
                "valid_assigned_row_index",
                "valid_center_seed_index",
                "valid_scale_block_index",
            )
        )
    stored_arrays = {
        name: np.asarray(cache[name])
        for name in stored_array_names
    }
    valid_count = len(stored_arrays["valid_labels"])
    loaded_shape = metadata.get("loaded_shape_TZYXC")
    if not isinstance(loaded_shape, list) or len(loaded_shape) != 5:
        raise ValueError("existing cache loaded_shape_TZYXC metadata is invalid")
    ivd_shape = tuple(int(value) for value in loaded_shape[1:4])
    expected_contract = {
        "raw_features": (np.dtype(np.float32), (valid_count, 672)),
        "fmt_features": (np.dtype(np.float32), (valid_count, 161)),
        "valid_labels": (np.dtype(np.bool_), (valid_count,)),
        "valid_seed_index": (np.dtype(np.int64), (valid_count,)),
        "valid_scale_id": (np.dtype(np.int32), (valid_count,)),
        "center_sample_time": (np.dtype(np.float32), (valid_count, 32)),
        "seeds_xyz": (np.dtype(np.float64), (assigned_count, 3)),
        "reference_labels_all": (np.dtype(np.bool_), (assigned_count,)),
        "ivd_values_all": (np.dtype(np.float32), (assigned_count,)),
        "ivd_volume": (np.dtype(np.float32), ivd_shape),
        "scale_assignment": (np.dtype(np.int32), (assigned_count,)),
        "valid_mask": (np.dtype(np.bool_), (assigned_count,)),
        "line_steps": (np.dtype(np.int32), (assigned_count, 7)),
        "line_travel": (np.dtype(np.float32), (assigned_count, 7)),
        "line_end_time": (np.dtype(np.float32), (assigned_count, 7)),
        "line_reached_target": (np.dtype(np.bool_), (assigned_count, 7)),
    }
    if plan.experiment == EXPERIMENT31:
        expected_contract.update(
            {
                "valid_assigned_row_index": (
                    np.dtype(np.int64),
                    (valid_count,),
                ),
                "valid_center_seed_index": (
                    np.dtype(np.int64),
                    (valid_count,),
                ),
                "valid_scale_block_index": (
                    np.dtype(np.int8),
                    (valid_count,),
                ),
            }
        )
    contract_drift = {
        name: {
            "actual_dtype": str(stored_arrays[name].dtype),
            "expected_dtype": str(dtype),
            "actual_shape": list(stored_arrays[name].shape),
            "expected_shape": list(shape),
        }
        for name, (dtype, shape) in expected_contract.items()
        if stored_arrays[name].dtype != dtype or stored_arrays[name].shape != shape
    }
    if contract_drift:
        raise ValueError(f"existing cache stored-array contract changed: {contract_drift}")
    for name, values in stored_arrays.items():
        if values.dtype.kind in "f" and not np.isfinite(values).all():
            raise ValueError(f"existing cache {name} contains NaN or Inf")
    seeds = stored_arrays["seeds_xyz"]
    assignment = stored_arrays["scale_assignment"]
    labels = stored_arrays["reference_labels_all"]
    valid = stored_arrays["valid_mask"]
    valid_seed_index = stored_arrays["valid_seed_index"]
    valid_labels = stored_arrays["valid_labels"]
    valid_scale_id = stored_arrays["valid_scale_id"]
    line_steps = stored_arrays["line_steps"]
    line_travel = stored_arrays["line_travel"]
    line_end_time = stored_arrays["line_end_time"]
    line_reached = stored_arrays["line_reached_target"]
    center_time = stored_arrays["center_sample_time"]
    if not np.isfinite(seeds).all() or not np.isfinite(stored_arrays["ivd_values_all"]).all():
        raise ValueError("existing cache seed/IVD values contain NaN or Inf")
    if np.any(line_steps < 0) or np.any(line_travel < 0.0):
        raise ValueError("existing cache line diagnostics contain negative values")
    if not np.array_equal(valid, line_reached.all(axis=1)):
        raise ValueError("existing cache valid_mask disagrees with line_reached_target")
    if center_time.shape[0] and np.any(np.diff(center_time, axis=1) < -1e-7):
        raise ValueError("existing cache center_sample_time is not nondecreasing")
    expected_valid_indices = np.flatnonzero(valid)
    if not np.array_equal(valid_seed_index, expected_valid_indices):
        raise ValueError("existing cache valid_seed_index disagrees with valid_mask")
    if not np.array_equal(valid_labels, labels[valid]):
        raise ValueError("existing cache valid labels disagree with assigned labels")
    if not np.array_equal(valid_scale_id, assignment[valid]):
        raise ValueError("existing cache valid scale ids disagree with assignment")
    if assignment.size and (
        assignment.min() < 0 or assignment.max() >= len(plan.scale_table)
    ):
        raise ValueError("existing cache scale assignment is outside the frozen table")
    expected_assignment = plan.primitive_scale_assignment()
    if not np.array_equal(assignment, expected_assignment):
        raise ValueError(
            "existing cache scale assignment differs from the frozen block PCG64 assignment"
        )
    if plan.experiment == EXPERIMENT31:
        center_count = plan.assigned_seed_count
        valid_assigned_row_index = stored_arrays["valid_assigned_row_index"]
        valid_center_seed_index = stored_arrays["valid_center_seed_index"]
        valid_scale_block_index = stored_arrays["valid_scale_block_index"]
        if not np.array_equal(valid_assigned_row_index, valid_seed_index):
            raise ValueError(
                "existing cache valid_seed_index is not the documented assigned-row alias"
            )
        if not np.array_equal(
            valid_center_seed_index, valid_assigned_row_index % center_count
        ):
            raise ValueError(
                "existing cache center-seed identity disagrees with assigned rows"
            )
        if not np.array_equal(
            valid_scale_block_index,
            valid_assigned_row_index // center_count,
        ):
            raise ValueError(
                "existing cache scale-block identity disagrees with assigned rows"
            )
        if valid_scale_block_index.size and (
            valid_scale_block_index.min() < 0
            or valid_scale_block_index.max() >= len(plan.effective_scale_blocks)
        ):
            raise ValueError("existing cache scale-block index is outside the plan")
        for block_index, block in enumerate(plan.effective_scale_blocks):
            within = valid_scale_block_index == block_index
            if np.any(
                (valid_scale_id[within] < block.scale_id_start)
                | (valid_scale_id[within] >= block.scale_id_stop)
            ):
                raise ValueError(
                    "existing cache scale IDs disagree with valid scale-block identity"
                )
        center_seeds = np.ascontiguousarray(seeds[:center_count])
        center_labels = np.ascontiguousarray(labels[:center_count])
        center_ivd = np.ascontiguousarray(
            stored_arrays["ivd_values_all"][:center_count]
        )
        expected_block_rows = []
        for ordinal, block in enumerate(plan.effective_scale_blocks):
            start = ordinal * center_count
            stop = start + center_count
            if not np.array_equal(seeds[start:stop], center_seeds):
                raise ValueError("existing cache center seed rows differ between blocks")
            if not np.array_equal(labels[start:stop], center_labels):
                raise ValueError("existing cache reference labels differ between blocks")
            if not np.array_equal(
                stored_arrays["ivd_values_all"][start:stop], center_ivd
            ):
                raise ValueError("existing cache seed IVD values differ between blocks")
            expected_block_rows.append(
                {
                    "id": block.block_id,
                    "scale_id_start": int(block.scale_id_start),
                    "scale_id_stop_exclusive": int(block.scale_id_stop),
                    "assignment_seed": int(block.assignment_seed),
                    "primitive_row_start": int(start),
                    "primitive_row_stop_exclusive": int(stop),
                    "assignment_sha256": canonical_array_sha256(
                        np.ascontiguousarray(assignment[start:stop])
                    ),
                }
            )
        expected_phase31_identity = {
            "unique_center_seed_count": center_count,
            "assigned_primitive_count": assigned_count,
            "assignment_count_per_seed": 2,
            "center_seed_repetition_order": "block_major_then_center_seed_index",
            "valid_seed_index_semantics": (
                "legacy_alias_of_valid_assigned_row_index_not_center_seed_index"
            ),
            "valid_identity_fields": [
                "valid_assigned_row_index",
                "valid_center_seed_index",
                "valid_scale_block_index",
            ],
            "scale_block_ids_by_index": [
                block.block_id for block in plan.effective_scale_blocks
            ],
            "center_seed_xyz_sha256": canonical_array_sha256(center_seeds),
            "maximum_source_frame_intervals": 48.0,
            "scale_assignment_blocks": expected_block_rows,
        }
        phase31_drift = {
            key: (metadata.get(key), value)
            for key, value in expected_phase31_identity.items()
            if metadata.get(key) != value
        }
        if phase31_drift:
            raise ValueError(
                f"existing cache 3.1 block/horizon identity changed: {phase31_drift}"
            )
    if int(metadata.get("assigned_count", -1)) != assigned_count:
        raise ValueError("existing cache metadata assigned_count changed")
    if int(metadata.get("valid_count", -1)) != valid_count:
        raise ValueError("existing cache metadata valid_count changed")
    if int(metadata.get("invalid_count", -1)) != assigned_count - valid_count:
        raise ValueError("existing cache metadata invalid_count changed")

    stored_hashes = dict(metadata.get("array_sha256", {}))
    if set(stored_hashes) != set(stored_arrays):
        raise ValueError(
            "existing cache array_sha256 does not exactly cover every stored array"
        )
    actual_hashes = {
        name: canonical_array_sha256(values) for name, values in stored_arrays.items()
    }
    mismatches = {
        name: (stored_hashes.get(name), digest)
        for name, digest in actual_hashes.items()
        if stored_hashes.get(name) != digest
    }
    if mismatches:
        raise ValueError(f"existing cache canonical array SHA-256 mismatch: {mismatches}")
    combined_expected = canonical_json_sha256(stored_hashes)
    if metadata.get("combined_array_sha256") != combined_expected:
        raise ValueError("existing cache combined array SHA-256 mismatch")
    ivd_volume_hash = str(stored_hashes["ivd_volume"])
    cache_sha = sha256_file(path)
    if valid_count != int(valid.sum()):
        raise ValueError("existing cache valid row population changed")
    cache_row = {
        "dataset": dataset,
        "physical_family": plan.family_by_dataset[dataset],
        "split": split,
        "source_ordinal": int(source_ordinal),
        "source_index": int(source_index),
        "path": str(path),
        "file_size": int(path.stat().st_size),
        "file_sha256": cache_sha,
        "config_sha256": plan.config_sha256,
        "dataset_registry_sha256": provenance.get("dataset_registry_sha256"),
        "portable_builder_git_commit": provenance.get("builder_git_commit"),
        "cache_builder_git_commit": cache_builder_git_commit,
        "assigned_count": assigned_count,
        "valid_count": valid_count,
        "invalid_count": assigned_count - valid_count,
    }
    raw_input_row = {
        "dataset": dataset,
        "split": split,
        "registered_path": provenance.get("source_file", provenance.get("source_path")),
        "size_bytes": provenance.get("source_file_size"),
        "sha256": provenance.get("source_file_sha256"),
        "kind": provenance.get("source_kind", "flow_source"),
        "config_sha256": plan.config_sha256,
        "dataset_registry_sha256": provenance.get("dataset_registry_sha256"),
        "portable_builder_git_commit": provenance.get("builder_git_commit"),
        "cache_builder_git_commit": cache_builder_git_commit,
    }
    derived_window_row = {
        "dataset": dataset,
        "split": split,
        "source_index": int(source_index),
        "frame_index_range": [int(source_index), int(source_index + plan.window_frame_count - 1)],
        "source_time": float(metadata["source_time"]),
        "canonical_array_sha256s": provenance.get("array_sha256"),
        "combined_sha256": provenance.get("combined_array_sha256"),
        "portable_path": provenance.get("portable_path"),
        "portable_file_sha256": provenance.get("portable_file_sha256"),
        "coordinate_audit": provenance.get("coordinate_audit"),
        "config_sha256": plan.config_sha256,
        "dataset_registry_sha256": provenance.get("dataset_registry_sha256"),
        "portable_builder_git_commit": provenance.get("builder_git_commit"),
        "cache_builder_git_commit": cache_builder_git_commit,
    }
    counts = np.bincount(assignment, minlength=len(plan.scale_table))
    assignment_row = {
        "dataset": dataset,
        "split": split,
        "source_index": int(source_index),
        "seed_count": assigned_count,
        "seed_xyz_sha256": stored_hashes["seeds_xyz"],
        "assignment_sha256": stored_hashes["scale_assignment"],
        "assignment_seed": int(plan.assignment_seed),
        "scale_count": int(len(plan.scale_table)),
        "config_sha256": plan.config_sha256,
        "dataset_registry_sha256": provenance.get("dataset_registry_sha256"),
        "portable_builder_git_commit": provenance.get("builder_git_commit"),
        "cache_builder_git_commit": cache_builder_git_commit,
        "minimum_count_per_scale": int(counts.min()),
        "maximum_count_per_scale": int(counts.max()),
    }
    if plan.experiment == EXPERIMENT31:
        assignment_row.update(
            {
                "unique_center_seed_count": int(plan.assigned_seed_count),
                "assigned_primitive_count": assigned_count,
                "assignment_count_per_seed": int(plan.assignment_count_per_seed),
                "center_seed_repetition_order": "block_major_then_center_seed_index",
                "center_seed_xyz_sha256": metadata["center_seed_xyz_sha256"],
                "maximum_source_frame_intervals": float(
                    plan.maximum_source_frame_intervals
                ),
                "blocks": metadata["scale_assignment_blocks"],
            }
        )
    label_row = {
        "dataset": dataset,
        "split": split,
        "source_index": int(source_index),
        "ivd_threshold": float(metadata["ivd_threshold"]),
        "ivd_volume_sha256": ivd_volume_hash,
        "ivd_volume_storage": "stored_in_cache",
        "reference_labels_sha256": stored_hashes["reference_labels_all"],
        "assigned_positive_count": int(labels.sum()),
        "assigned_positive_fraction": float(labels.mean()),
        "valid_positive_count": int(valid_labels.sum()),
        "config_sha256": plan.config_sha256,
        "dataset_registry_sha256": provenance.get("dataset_registry_sha256"),
        "portable_builder_git_commit": provenance.get("builder_git_commit"),
        "cache_builder_git_commit": cache_builder_git_commit,
    }
    primitive_row = {
        "dataset": dataset,
        "split": split,
        "source_index": int(source_index),
        "assigned_count": assigned_count,
        "valid_count": valid_count,
        "invalid_count": assigned_count - valid_count,
        "valid_mask_sha256": stored_hashes["valid_mask"],
        "raw_features_sha256": stored_hashes["raw_features"],
        "fmt_features_sha256": stored_hashes["fmt_features"],
        "cache_file_sha256": cache_sha,
        "config_sha256": plan.config_sha256,
        "dataset_registry_sha256": provenance.get("dataset_registry_sha256"),
        "portable_builder_git_commit": provenance.get("builder_git_commit"),
        "cache_builder_git_commit": cache_builder_git_commit,
    }
    if plan.experiment == EXPERIMENT31:
        identity = {
            "experiment": plan.experiment,
            "maximum_source_frame_intervals": float(
                plan.maximum_source_frame_intervals
            ),
            "assignment_count_per_seed": int(plan.assignment_count_per_seed),
            "unique_center_seed_count": int(plan.assigned_seed_count),
        }
        cache_row.update(identity)
        label_row.update(identity)
        primitive_row.update(identity)
    codes = assignment.astype(np.int64) * 2 + labels.astype(np.int64)
    assigned_counts = np.bincount(codes, minlength=2 * len(plan.scale_table))
    valid_counts = np.bincount(codes[valid], minlength=2 * len(plan.scale_table))
    audit_rows = []
    for scale_id in range(len(plan.scale_table)):
        for class_id in (0, 1):
            code = scale_id * 2 + class_id
            audit_rows.append(
                {
                    "dataset": dataset,
                    "physical_family": plan.family_by_dataset[dataset],
                    "split": split,
                    "source_ordinal": int(source_ordinal),
                    "source_index": int(source_index),
                    "scale_id": scale_id,
                    "dx_grid_scale": float(plan.scale_table.dx_grid_scale[scale_id]),
                    "ds_frame_scale": float(plan.scale_table.ds_frame_scale[scale_id]),
                    "arc_length_grid_scale": float(
                        plan.scale_table.arc_length_grid_scale[scale_id]
                    ),
                    "reference_class": class_id,
                    "assigned_count": int(assigned_counts[code]),
                    "valid_count": int(valid_counts[code]),
                    "invalid_count": int(assigned_counts[code] - valid_counts[code]),
                }
            )
            if plan.experiment == EXPERIMENT31:
                block_index = next(
                    index
                    for index, block in enumerate(plan.effective_scale_blocks)
                    if block.scale_id_start <= scale_id < block.scale_id_stop
                )
                audit_rows[-1].update(
                    {
                        "scale_block_index": int(block_index),
                        "scale_block_id": plan.effective_scale_blocks[
                            block_index
                        ].block_id,
                    }
                )
    return CacheBuildSummary(
        cache_row=cache_row,
        raw_input_row=raw_input_row,
        derived_window_row=derived_window_row,
        assignment_row=assignment_row,
        label_row=label_row,
        primitive_row=primitive_row,
        audit_rows=tuple(audit_rows),
    )


def _read_manifest_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"manifest rows are invalid: {path}")
    if value.get("rows_content_sha256") != canonical_json_sha256(rows):
        raise ValueError(f"manifest row content SHA-256 mismatch: {path}")
    return [dict(row) for row in rows]


def _validate_cache_provenance(
    plan: Phase21Plan,
    cache: Mapping[str, Any],
    cache_row: Mapping[str, Any],
) -> None:
    metadata = cache["metadata"]
    expected = {
        "schema": plan.cache_schema,
        "experiment": plan.experiment,
        "config_sha256": plan.config_sha256,
        "dataset": str(cache_row["dataset"]),
        "physical_family": plan.family_by_dataset[str(cache_row["dataset"])],
        "split": str(cache_row["split"]),
        "source_ordinal": int(cache_row["source_ordinal"]),
        "source_index": int(cache_row["source_index"]),
        "dataset_registry_sha256": cache_row.get("dataset_registry_sha256"),
        "portable_builder_git_commit": cache_row.get("portable_builder_git_commit"),
        "cache_builder_git_commit": cache_row.get("cache_builder_git_commit"),
    }
    if plan.experiment == EXPERIMENT31:
        expected.update(
            {
                "maximum_source_frame_intervals": 48.0,
                "assignment_count_per_seed": 2,
                "assigned_primitive_count": plan.assigned_primitive_count,
            }
        )
    drift = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if drift:
        raise ValueError(f"cache metadata/sidecar provenance mismatch: {drift}")


def _choose_two_class_candidate_rows(
    generator: np.random.Generator,
    class_rows: Sequence[np.ndarray],
) -> tuple[int | None, int | None]:
    """Draw negative then positive only when both candidate classes are nonempty."""

    if len(class_rows) != 2:
        raise ValueError("exactly two class candidate arrays are required")
    negative = np.asarray(class_rows[0], dtype=np.int64)
    positive = np.asarray(class_rows[1], dtype=np.int64)
    if len(negative) == 0 or len(positive) == 0:
        return None, None
    negative_row = int(negative[int(generator.integers(0, len(negative)))])
    positive_row = int(positive[int(generator.integers(0, len(positive)))])
    return negative_row, positive_row


def _select_library_and_fit_pca(
    plan: Phase21Plan,
    root: Path,
    train_rows: Sequence[Mapping[str, Any]],
    *,
    verify_cache_hashes: bool,
) -> tuple[
    StreamingCovariancePCA,
    dict[str, np.ndarray],
    float,
    list[dict[str, Any]],
]:
    feature_sum = np.zeros(672, dtype=np.float64)
    total_positive = 0
    total_count = 0
    selected: dict[str, list[Any]] = {
        "raw_features": [],
        "fmt_features": [],
        "labels": [],
        "dataset": [],
        "physical_family": [],
        "source_ordinal": [],
        "source_index": [],
        "seed_index": [],
        "scale_id": [],
    }
    if plan.experiment == EXPERIMENT31:
        selected.update(
            {
                "assigned_row_index": [],
                "center_seed_index": [],
                "scale_block_index": [],
                "scale_block_id": [],
            }
        )
    library_audit: list[dict[str, Any]] = []
    generator = np.random.Generator(np.random.PCG64(plan.library_seed))

    ordered_rows = sorted(
        train_rows,
        key=lambda row: (
            plan.train_datasets.index(str(row["dataset"])),
            int(row["source_ordinal"]),
        ),
    )
    for row in ordered_rows:
        path = Path(str(row["path"]))
        cache = _load_cache(
            path,
            expected_sha256=str(row["file_sha256"]) if verify_cache_hashes else None,
        )
        _validate_cache_provenance(plan, cache, row)
        labels = np.asarray(cache["valid_labels"], dtype=bool)
        seed_index = np.asarray(cache["valid_seed_index"], dtype=np.int64)
        if plan.experiment == EXPERIMENT31:
            assigned_row_index = np.asarray(
                cache["valid_assigned_row_index"], dtype=np.int64
            )
            center_seed_index = np.asarray(
                cache["valid_center_seed_index"], dtype=np.int64
            )
            scale_block_index = np.asarray(
                cache["valid_scale_block_index"], dtype=np.int8
            )
        scale_id = np.asarray(cache["valid_scale_id"], dtype=np.int32)
        candidate_raw = np.asarray(cache["raw_features"], dtype=np.float32)
        for start in range(0, len(candidate_raw), 8192):
            feature_sum += candidate_raw[start : start + 8192].astype(np.float64).sum(
                axis=0
            )
        total_positive += int(labels.sum())
        total_count += len(labels)

        for current_scale in range(len(plan.scale_table)):
            class_rows: list[np.ndarray] = []
            counts: list[int] = []
            for class_id in (0, 1):
                candidates = np.flatnonzero(
                    (scale_id == current_scale) & (labels == bool(class_id))
                )
                if len(candidates):
                    order = np.argsort(seed_index[candidates], kind="stable")
                    candidates = candidates[order]
                class_rows.append(candidates)
                counts.append(int(len(candidates)))
            both_nonempty = counts[0] > 0 and counts[1] > 0
            chosen_rows = list(_choose_two_class_candidate_rows(generator, class_rows))
            if both_nonempty:
                for class_id in (0, 1):
                    candidates = class_rows[class_id]
                    chosen = chosen_rows[class_id]
                    if chosen is None:
                        raise RuntimeError("two-class candidate selection returned no row")
                    draw_row = int(chosen)
                    selected["raw_features"].append(cache["raw_features"][draw_row])
                    selected["fmt_features"].append(cache["fmt_features"][draw_row])
                    selected["labels"].append(bool(class_id))
                    selected["dataset"].append(str(row["dataset"]))
                    selected["physical_family"].append(
                        plan.family_by_dataset[str(row["dataset"])]
                    )
                    selected["source_ordinal"].append(int(row["source_ordinal"]))
                    selected["source_index"].append(int(row["source_index"]))
                    selected["seed_index"].append(int(seed_index[draw_row]))
                    selected["scale_id"].append(current_scale)
                    if plan.experiment == EXPERIMENT31:
                        block_index = int(scale_block_index[draw_row])
                        selected["assigned_row_index"].append(
                            int(assigned_row_index[draw_row])
                        )
                        selected["center_seed_index"].append(
                            int(center_seed_index[draw_row])
                        )
                        selected["scale_block_index"].append(block_index)
                        selected["scale_block_id"].append(
                            plan.effective_scale_blocks[block_index].block_id
                        )
            library_audit.append(
                {
                    "dataset": str(row["dataset"]),
                    "physical_family": plan.family_by_dataset[str(row["dataset"])],
                    "source_ordinal": int(row["source_ordinal"]),
                    "source_index": int(row["source_index"]),
                    "scale_id": current_scale,
                    "negative_candidate_count": counts[0],
                    "positive_candidate_count": counts[1],
                    "selected_per_class": int(both_nonempty),
                    "selected_negative_seed_index": (
                        None
                        if chosen_rows[0] is None
                        else int(seed_index[int(chosen_rows[0])])
                    ),
                    "selected_positive_seed_index": (
                        None
                        if chosen_rows[1] is None
                        else int(seed_index[int(chosen_rows[1])])
                    ),
                    "skip_reason": "" if both_nonempty else "one_or_both_classes_empty",
                }
            )
            if plan.experiment == EXPERIMENT31:
                block_index = next(
                    index
                    for index, block in enumerate(plan.effective_scale_blocks)
                    if block.scale_id_start <= current_scale < block.scale_id_stop
                )
                library_audit[-1].update(
                    {
                        "scale_block_index": block_index,
                        "scale_block_id": plan.effective_scale_blocks[
                            block_index
                        ].block_id,
                        "selected_negative_assigned_row_index": (
                            None
                            if chosen_rows[0] is None
                            else int(
                                assigned_row_index[int(chosen_rows[0])]
                            )
                        ),
                        "selected_positive_assigned_row_index": (
                            None
                            if chosen_rows[1] is None
                            else int(
                                assigned_row_index[int(chosen_rows[1])]
                            )
                        ),
                        "selected_negative_center_seed_index": (
                            None
                            if chosen_rows[0] is None
                            else int(center_seed_index[int(chosen_rows[0])])
                        ),
                        "selected_positive_center_seed_index": (
                            None
                            if chosen_rows[1] is None
                            else int(center_seed_index[int(chosen_rows[1])])
                        ),
                    }
                )

    if total_count < 2 or total_positive == 0 or total_positive == total_count:
        raise ValueError("valid train candidates must contain both reference classes")
    mean = feature_sum / total_count
    scatter = np.zeros((672, 672), dtype=np.float64)
    second_pass_count = 0
    for row in ordered_rows:
        cache = _load_cache(
            Path(str(row["path"])), expected_sha256=str(row["file_sha256"])
        )
        _validate_cache_provenance(plan, cache, row)
        candidate_raw = np.asarray(cache["raw_features"], dtype=np.float32)
        for start in range(0, len(candidate_raw), 8192):
            centered = candidate_raw[start : start + 8192].astype(np.float64) - mean
            scatter += centered.T @ centered
        second_pass_count += len(candidate_raw)
    if second_pass_count != total_count:
        raise RuntimeError("PCA train candidate population changed between its two passes")
    pca = _pca_from_statistics(
        sample_count=total_count,
        feature_sum=feature_sum,
        centered_scatter=scatter,
        components=plan.pca_components,
    )

    library: dict[str, np.ndarray] = {
        "raw_features": np.ascontiguousarray(np.asarray(selected["raw_features"]), dtype=np.float32),
        "fmt_features": np.ascontiguousarray(np.asarray(selected["fmt_features"]), dtype=np.float32),
        "labels": np.asarray(selected["labels"], dtype=np.bool_),
        "dataset": np.asarray(selected["dataset"], dtype=np.str_),
        "physical_family": np.asarray(selected["physical_family"], dtype=np.str_),
        "source_ordinal": np.asarray(selected["source_ordinal"], dtype=np.int16),
        "source_index": np.asarray(selected["source_index"], dtype=np.int32),
        "seed_index": np.asarray(selected["seed_index"], dtype=np.int64),
        "scale_id": np.asarray(selected["scale_id"], dtype=np.int32),
    }
    if plan.experiment == EXPERIMENT31:
        library.update(
            {
                "assigned_row_index": np.asarray(
                    selected["assigned_row_index"], dtype=np.int64
                ),
                "center_seed_index": np.asarray(
                    selected["center_seed_index"], dtype=np.int64
                ),
                "scale_block_index": np.asarray(
                    selected["scale_block_index"], dtype=np.int8
                ),
                "scale_block_id": np.asarray(
                    selected["scale_block_id"], dtype=np.str_
                ),
            }
        )
    if len(library["labels"]) < 2 or np.unique(library["labels"]).size != 2:
        raise ValueError("balanced library did not retain both classes")
    if len(library["labels"]) > plan.maximum_library_templates:
        raise ValueError(
            "selected library exceeds the frozen maximum of "
            f"{plan.maximum_library_templates}"
        )
    library["pca_features"] = pca.transform(library["raw_features"])
    return pca, library, float(total_positive / total_count), library_audit


def _matcher_artifacts(
    library: Mapping[str, np.ndarray],
    *,
    device: str,
) -> tuple[dict[str, ExhaustiveOneNearestNeighbor], dict[str, np.ndarray]]:
    labels = library["labels"]
    features = {
        METHOD_RAW: library["raw_features"],
        METHOD_PCA: library["pca_features"],
        METHOD_FMT: library["fmt_features"],
    }
    matchers = {
        method: ExhaustiveOneNearestNeighbor(values, labels, device=device)
        for method, values in features.items()
    }
    artifacts: dict[str, np.ndarray] = {}
    for method, matcher in matchers.items():
        stem = {METHOD_RAW: "raw", METHOD_PCA: "pca", METHOD_FMT: "fmt"}[method]
        values64 = np.asarray(features[method], dtype=np.float64)
        original_std = values64.std(axis=0)
        artifacts[f"{stem}_feature_mean"] = matcher.feature_mean
        artifacts[f"{stem}_feature_standard_deviation"] = matcher.feature_scale
        artifacts[f"{stem}_zero_variance_feature_mask"] = np.asarray(
            original_std < 1e-12, dtype=np.bool_
        )
    return matchers, artifacts


def _metric_values(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    targets = np.asarray(labels, dtype=bool)
    predicted = np.asarray(predictions, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    if targets.shape != predicted.shape or targets.shape != values.shape:
        raise ValueError("metric inputs must have identical one-dimensional shapes")
    if not np.isfinite(values).all():
        raise ValueError("metric scores contain NaN or Inf")
    if len(targets) == 0:
        return {
            "sample_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "true_positive": 0,
            "false_positive": 0,
            "true_negative": 0,
            "false_negative": 0,
            "single_class_group": True,
            **{name: float("nan") for name in METRIC_NAMES},
        }
    tp = int(np.sum(targets & predicted))
    fp = int(np.sum(~targets & predicted))
    tn = int(np.sum(~targets & ~predicted))
    fn = int(np.sum(targets & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    both_classes = bool(targets.any() and (~targets).any())
    balanced = (
        0.5 * (recall + specificity)
        if both_classes
        else float("nan")
    )
    return {
        "sample_count": int(len(targets)),
        "positive_count": int(targets.sum()),
        "negative_count": int((~targets).sum()),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": float((tp + tn) / len(targets)),
        "average_precision": (
            float(average_precision(targets, values)) if both_classes else float("nan")
        ),
        "auroc": float(auroc(targets, values)) if both_classes else float("nan"),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(balanced),
        "single_class_group": not both_classes,
    }


def _metric_row(
    method: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    *,
    assigned_count: int,
    **identity: Any,
) -> dict[str, Any]:
    metrics = _metric_values(labels, predictions, scores)
    valid_count = int(len(labels))
    return {
        **identity,
        "method": method,
        "assigned_count": int(assigned_count),
        "valid_count": valid_count,
        "invalid_count": int(assigned_count) - valid_count,
        "coverage": valid_count / int(assigned_count) if assigned_count else float("nan"),
        **metrics,
    }


def cache_summary_payload(summary: CacheBuildSummary) -> dict[str, Any]:
    """Return the canonical publish payload for one cache shard."""

    experiment = str(summary.cache_row.get("experiment", EXPERIMENT))
    if experiment not in SUPPORTED_EXPERIMENTS:
        raise ValueError(f"unsupported cache summary experiment {experiment}")
    artifact_tag = "phase31" if experiment == EXPERIMENT31 else "phase21"
    payload: dict[str, Any] = {
        "schema": f"pathline_template_matching.{artifact_tag}_cache_sidecar.v1",
        "experiment": experiment,
        "cache_row": summary.cache_row,
        "raw_input_row": summary.raw_input_row,
        "derived_window_row": summary.derived_window_row,
        "assignment_row": summary.assignment_row,
        "label_row": summary.label_row,
        "primitive_row": summary.primitive_row,
        "audit_rows": list(summary.audit_rows),
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    return payload


def write_cache_summary_sidecar(
    summary: CacheBuildSummary, path: str | Path
) -> str:
    """Atomically publish one independently-built shard's full evidence."""

    payload = cache_summary_payload(summary)
    return _atomic_json(Path(path), payload)


def load_cache_summary_sidecar(path: str | Path) -> CacheBuildSummary:
    """Read one shard sidecar and verify its self-contained digest."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    expected = payload.pop("content_sha256", None)
    if expected != canonical_json_sha256(payload):
        raise ValueError(f"cache sidecar content SHA-256 mismatch: {source}")
    experiment = payload.get("experiment")
    expected_schema = (
        "pathline_template_matching.phase31_cache_sidecar.v1"
        if experiment == EXPERIMENT31
        else "pathline_template_matching.phase21_cache_sidecar.v1"
    )
    if payload.get("schema") != expected_schema:
        raise ValueError(f"unsupported cache sidecar schema: {source}")
    if experiment not in SUPPORTED_EXPERIMENTS:
        raise ValueError(f"cache sidecar experiment mismatch: {source}")
    return CacheBuildSummary(
        cache_row=dict(payload["cache_row"]),
        raw_input_row=dict(payload["raw_input_row"]),
        derived_window_row=dict(payload["derived_window_row"]),
        assignment_row=dict(payload["assignment_row"]),
        label_row=dict(payload["label_row"]),
        primitive_row=dict(payload["primitive_row"]),
        audit_rows=tuple(dict(row) for row in payload["audit_rows"]),
    )


def discover_phase21_cache_sidecars(
    plan: Phase21Plan, cache_root: str | Path
) -> list[CacheBuildSummary]:
    """Discover and validate the exact frozen dataset/source-time cache set."""

    root = Path(cache_root).resolve()
    sources = sorted(root.rglob("*.summary.json"))
    summaries = [load_cache_summary_sidecar(path) for path in sources]
    expected_keys: list[tuple[str, int]] = []
    # Source indices are field-length dependent.  The sidecars preserve ordinal,
    # so completeness can be checked without reopening any query field.
    for dataset in plan.datasets:
        expected_keys.extend((dataset, ordinal) for ordinal in range(plan.source_count))
    found: dict[tuple[str, int], CacheBuildSummary] = {}
    for summary in summaries:
        row = summary.cache_row
        key = (str(row["dataset"]), int(row["source_ordinal"]))
        if key in found:
            raise ValueError(f"duplicate cache sidecar for {key}")
        if str(row.get("split")) != plan.split_for(key[0]):
            raise ValueError(f"cache sidecar split mismatch for {key}")
        if str(row.get("experiment", EXPERIMENT)) != plan.experiment:
            raise ValueError(f"cache sidecar experiment mismatch for {key}")
        found[key] = summary
    missing = sorted(set(expected_keys) - set(found))
    extra = sorted(set(found) - set(expected_keys))
    if missing or extra:
        raise ValueError(f"cache sidecar population mismatch: missing={missing}, extra={extra}")
    return [found[key] for key in expected_keys]


def authorize_phase31_portable_population_marker_path(
    plan: Phase21Plan, marker_path: str | Path, *, access_scope: str
) -> Path:
    """Authorize a marker lexically before any sidecar-supplied path is opened."""

    scope_directory = {"train-only": "train_only", "all": "all"}.get(
        access_scope
    )
    marker_name = {
        "train-only": "TRAIN_PORTABLES_PASS.json",
        "all": "ALL_PORTABLES_PASS.json",
    }.get(access_scope)
    if scope_directory is None or marker_name is None:
        raise ValueError("portable-population marker scope is invalid")
    trusted_root = Path(
        os.path.abspath(
            Path(plan.output_root)
            / "verification"
            / "portable_population"
            / scope_directory
        )
    )
    lexical_path = Path(os.path.abspath(Path(marker_path)))
    try:
        lexical_path.relative_to(trusted_root)
    except ValueError as error:
        raise ValueError(
            "portable-population marker path is outside the frozen output root"
        ) from error
    if lexical_path.name != marker_name:
        raise ValueError(f"portable-population marker must be named {marker_name}")
    return lexical_path


def validate_phase31_cache_portable_population_evidence(
    plan: Phase21Plan,
    cache_rows: Sequence[Mapping[str, Any]],
    *,
    usage: str,
    expected_git_commit: str,
    synthetic_pass_file_sha256: str,
    train_coverage_pass_file_sha256: str | None = None,
    authorized_marker_paths_by_scope: Mapping[str, str | Path] | None = None,
) -> list[dict[str, Any]]:
    """Authenticate portable-population markers without opening a cache NPZ."""

    if plan.experiment != EXPERIMENT31:
        raise ValueError("portable-population cache evidence is restricted to 3.1")
    if usage == "train-coverage":
        expected_scope_by_split = {"train": "train-only"}
        expected_datasets = plan.train_datasets
    elif usage == "evaluation":
        expected_scope_by_split = {"train": "train-only", "test": "all"}
        expected_datasets = plan.datasets
    else:
        raise ValueError("portable-population evidence usage is invalid")
    expected_keys = {
        (dataset, ordinal)
        for dataset in expected_datasets
        for ordinal in range(plan.source_count)
    }
    rows = [dict(row) for row in cache_rows]
    found_keys: set[tuple[str, int]] = set()
    identities_by_split: dict[str, set[tuple[str, int, str, str]]] = {
        split: set() for split in expected_scope_by_split
    }
    commits_by_split: dict[str, set[str]] = {
        split: set() for split in expected_scope_by_split
    }
    for row in rows:
        dataset = str(row.get("dataset", ""))
        split = str(row.get("split", ""))
        ordinal = int(row.get("source_ordinal", -1))
        key = (dataset, ordinal)
        if key not in expected_keys or key in found_keys:
            raise ValueError(
                "portable-population evidence cache population is incomplete or duplicated"
            )
        found_keys.add(key)
        if split not in expected_scope_by_split or plan.split_for(dataset) != split:
            raise ValueError("portable-population evidence cache split changed")
        row_provenance = {
            "config_sha256": plan.config_sha256,
            "dataset_registry_sha256": plan.dataset_registry_sha256,
            "portable_builder_git_commit": expected_git_commit,
            "cache_builder_git_commit": expected_git_commit,
        }
        drift = {
            name: (row.get(name), value)
            for name, value in row_provenance.items()
            if row.get(name) != value
        }
        if drift:
            raise ValueError(
                f"portable-population cache sidecar provenance changed: {drift}"
            )
        scope = str(row.get("portable_population_scope", ""))
        if scope != expected_scope_by_split[split]:
            raise ValueError(
                f"{split} caches must carry one {expected_scope_by_split[split]} "
                "portable-population marker"
            )
        path = str(row.get("portable_population_pass_path", ""))
        size = int(row.get("portable_population_pass_file_size", -1))
        file_sha = str(row.get("portable_population_pass_file_sha256", ""))
        rows_sha = str(row.get("portable_population_rows_content_sha256", ""))
        if (
            not path
            or size < 1
            or len(file_sha) != 64
            or any(character not in "0123456789abcdef" for character in file_sha)
            or len(rows_sha) != 64
            or any(character not in "0123456789abcdef" for character in rows_sha)
        ):
            raise ValueError("portable-population marker identity is invalid")
        identities_by_split[split].add((path, size, file_sha, rows_sha))
        commits_by_split[split].add(str(row.get("cache_builder_git_commit", "")))
    if found_keys != expected_keys or len(rows) != len(expected_keys):
        raise ValueError("portable-population evidence does not cover the exact cache set")
    if any(len(values) != 1 for values in identities_by_split.values()):
        raise ValueError(
            "portable-population marker path/size/file-SHA/rows-SHA must be "
            "singleton within each split"
        )
    if any(len(values) != 1 for values in commits_by_split.values()):
        raise ValueError("cache builder Git commit must be singleton within each split")

    evidence = []
    for split, scope in expected_scope_by_split.items():
        path_text, expected_size, expected_file_sha, expected_rows_sha = next(
            iter(identities_by_split[split])
        )
        lexical_path = Path(os.path.abspath(Path(path_text)))
        if authorized_marker_paths_by_scope is None:
            marker_path = authorize_phase31_portable_population_marker_path(
                plan, lexical_path, access_scope=scope
            )
        else:
            authorized_value = authorized_marker_paths_by_scope.get(scope)
            if authorized_value is None:
                raise ValueError(f"no authorized {scope} marker path was supplied")
            marker_path = Path(os.path.abspath(Path(authorized_value)))
            if lexical_path != marker_path:
                raise ValueError(
                    "cache sidecars point to a marker other than the authorized path"
                )
            expected_name = (
                "TRAIN_PORTABLES_PASS.json"
                if scope == "train-only"
                else "ALL_PORTABLES_PASS.json"
            )
            if marker_path.name != expected_name:
                raise ValueError(f"authorized marker must be named {expected_name}")
        if (
            not marker_path.is_file()
            or marker_path.stat().st_size != expected_size
            or sha256_file(marker_path) != expected_file_sha
        ):
            raise ValueError(
                f"{split} portable-population marker file size/hash changed"
            )
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        expected_dataset_count = (
            len(plan.train_datasets) if scope == "train-only" else len(plan.datasets)
        )
        marker_rows = marker.get("rows")
        if (
            not isinstance(marker_rows, list)
            or len(marker_rows) != expected_dataset_count * plan.source_count
            or canonical_json_sha256(marker_rows) != expected_rows_sha
        ):
            raise ValueError(
                f"{split} portable-population marker row evidence changed"
            )
        expected_marker = {
            "schema": "pathline_template_matching.phase31_portable_population_pass.v1",
            "experiment": plan.experiment,
            "status": "passed",
            "access_scope": scope,
            "git_commit": expected_git_commit,
            "worktree_clean": True,
            "config_sha256": plan.config_sha256,
            "dataset_registry_sha256": plan.dataset_registry_sha256,
            "dataset_count": expected_dataset_count,
            "window_count": expected_dataset_count * plan.source_count,
            "rows_content_sha256": expected_rows_sha,
            "synthetic_pass_file_sha256": synthetic_pass_file_sha256,
        }
        if scope == "all":
            expected_marker["train_coverage_pass_file_sha256"] = (
                train_coverage_pass_file_sha256
            )
        drift = {
            name: (marker.get(name), value)
            for name, value in expected_marker.items()
            if marker.get(name) != value
        }
        if drift:
            raise ValueError(
                f"{split} portable-population marker content changed: {drift}"
            )
        evidence.append(
            {
                "split": split,
                "access_scope": scope,
                "path": str(marker_path),
                "file_size": expected_size,
                "file_sha256": expected_file_sha,
                "rows_content_sha256": expected_rows_sha,
                "git_commit": expected_marker["git_commit"],
                "synthetic_pass_file_sha256": synthetic_pass_file_sha256,
                "train_coverage_pass_file_sha256": (
                    train_coverage_pass_file_sha256 if scope == "all" else None
                ),
                "dataset_count": expected_dataset_count,
                "window_count": expected_dataset_count * plan.source_count,
            }
        )
    if usage == "evaluation" and (
        len(evidence) != 2
        or {row["access_scope"] for row in evidence} != {"train-only", "all"}
        or len({row["file_sha256"] for row in evidence}) != 2
    ):
        raise ValueError(
            "strict 3.1 evaluation requires two distinct portable-population markers"
        )
    return evidence


def audit_phase31_train_coverage(
    plan: Phase21Plan,
    cache_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    verify_cache_hashes: bool = True,
    expected_git_commit: str,
    synthetic_pass_file_sha256: str,
    authorized_portable_population_marker_path: str | Path,
) -> dict[str, Any]:
    """Run the immutable, train-only feasibility gate for 3.1.

    The gate intentionally stops before Principal Component Analysis (PCA),
    template matching, query prediction, or test metrics.  It accepts exactly
    the frozen 8x4 train-cache population and refuses every row outside the
    train split before opening any cache path.
    """

    if plan.experiment != EXPERIMENT31:
        raise ValueError("train-only long-arc coverage is restricted to mainExp 3.1")
    plan.validate_production_contract()
    rows = [dict(row) for row in cache_rows]
    expected_keys = {
        (dataset, ordinal)
        for dataset in plan.train_datasets
        for ordinal in range(plan.source_count)
    }
    found: dict[tuple[str, int], dict[str, Any]] = {}
    forbidden_names = set(plan.test_datasets)
    for row in rows:
        dataset = str(row.get("dataset", ""))
        split = str(row.get("split", ""))
        ordinal = int(row.get("source_ordinal", -1))
        if dataset in forbidden_names or dataset not in plan.train_datasets:
            raise ValueError(
                f"train-only coverage refuses non-train dataset row: {dataset!r}"
            )
        if split != "train":
            raise ValueError("train-only coverage refuses a non-train cache row")
        path_text = str(row.get("path", ""))
        if not path_text:
            raise ValueError("train-only coverage cache row has no path")
        if any(name.lower() in path_text.lower() for name in forbidden_names):
            raise ValueError("train-only coverage refuses a test-named cache path")
        key = (dataset, ordinal)
        if key in found:
            raise ValueError(f"duplicate train coverage cache row: {key}")
        found[key] = row
    if set(found) != expected_keys or len(rows) != 32:
        raise ValueError(
            "train-only coverage requires exactly the frozen 8x4 train caches: "
            f"missing={sorted(expected_keys-set(found))}, "
            f"extra={sorted(set(found)-expected_keys)}, count={len(rows)}"
        )
    portable_population_evidence = (
        validate_phase31_cache_portable_population_evidence(
            plan,
            rows,
            usage="train-coverage",
            expected_git_commit=expected_git_commit,
            synthetic_pass_file_sha256=synthetic_pass_file_sha256,
            authorized_marker_paths_by_scope={
                "train-only": authorized_portable_population_marker_path
            },
        )[0]
    )

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    diagnostic_path = root / "train_only_coverage_diagnostics.csv"
    summary_path = root / "train_only_coverage_summary.json"
    if diagnostic_path.exists() or summary_path.exists():
        raise FileExistsError("train-only coverage outputs are immutable")

    diagnostic_rows: list[dict[str, Any]] = []
    selected_by_block = {
        block.block_id: {"negative": 0, "positive": 0}
        for block in plan.effective_scale_blocks
    }
    expanded_arc_valid = np.zeros(10, dtype=np.int64)
    cache_evidence: list[dict[str, Any]] = []
    for dataset in plan.train_datasets:
        for ordinal in range(plan.source_count):
            row = found[(dataset, ordinal)]
            path = Path(str(row["path"])).resolve()
            expected_sha = str(row.get("file_sha256", ""))
            if verify_cache_hashes and (
                len(expected_sha) != 64 or sha256_file(path) != expected_sha
            ):
                raise ValueError(f"train coverage cache SHA-256 mismatch: {path}")
            # Coverage consumes only the assigned scale, validity, and train
            # reference arrays.  Raw/FMT feature members are deliberately not
            # decompressed, because descriptor fitting/matching is forbidden
            # in this feasibility phase.  The complete immutable NPZ is still
            # authenticated by its file SHA-256 above.
            with np.load(path, allow_pickle=False) as archive:
                required = {
                    "metadata_json",
                    "scale_assignment",
                    "valid_mask",
                    "reference_labels_all",
                }
                if not required.issubset(archive.files):
                    raise ValueError(
                        f"train coverage cache misses required arrays: {path}"
                    )
                metadata_scalar = np.asarray(archive["metadata_json"])
                if metadata_scalar.ndim != 0:
                    raise ValueError("train coverage cache metadata is not scalar")
                metadata = json.loads(str(metadata_scalar.item()))
                assignment = np.asarray(archive["scale_assignment"])
                valid = np.asarray(archive["valid_mask"])
                labels = np.asarray(archive["reference_labels_all"])
            cache_identity = {"metadata": metadata}
            _validate_cache_provenance(plan, cache_identity, row)
            if metadata.get("split") != "train":
                raise ValueError("train-only coverage opened a non-train cache")
            expected_shape = (plan.assigned_primitive_count,)
            if (
                assignment.dtype != np.int32
                or valid.dtype != np.bool_
                or labels.dtype != np.bool_
                or assignment.shape != expected_shape
                or valid.shape != expected_shape
                or labels.shape != expected_shape
            ):
                raise ValueError("train coverage cache array contract changed")
            stored_hashes = metadata.get("array_sha256")
            if not isinstance(stored_hashes, Mapping):
                raise ValueError("train coverage cache has no canonical array hashes")
            for name, values in (
                ("scale_assignment", assignment),
                ("valid_mask", valid),
                ("reference_labels_all", labels),
            ):
                if stored_hashes.get(name) != canonical_array_sha256(values):
                    raise ValueError(f"train coverage cache {name} SHA-256 changed")
            if not np.array_equal(assignment, plan.primitive_scale_assignment()):
                raise ValueError("train coverage scale assignment differs from plan")
            block_index_all = np.arange(len(assignment), dtype=np.int64) // int(
                plan.assigned_seed_count
            )
            for block_index, block in enumerate(plan.effective_scale_blocks):
                block_mask = block_index_all == block_index
                if int(block_mask.sum()) != plan.assigned_seed_count:
                    raise ValueError("train cache assignment block population changed")
                for scale_id in range(block.scale_id_start, block.scale_id_stop):
                    assigned_mask = block_mask & (assignment == scale_id)
                    valid_mask = assigned_mask & valid
                    assigned_count = int(assigned_mask.sum())
                    valid_count = int(valid_mask.sum())
                    positive_count = int((valid_mask & labels).sum())
                    negative_count = valid_count - positive_count
                    if assigned_count != 64:
                        raise ValueError(
                            "train coverage assignment is not exactly 64 rows per tuple"
                        )
                    eligible = positive_count > 0 and negative_count > 0
                    if eligible:
                        selected_by_block[block.block_id]["negative"] += 1
                        selected_by_block[block.block_id]["positive"] += 1
                    if block.block_id == "expanded_3_1":
                        expanded_arc_valid[(scale_id - block.scale_id_start) % 10] += (
                            valid_count
                        )
                    diagnostic_rows.append(
                        {
                            "dataset": dataset,
                            "physical_family": plan.family_by_dataset[dataset],
                            "source_ordinal": ordinal,
                            "source_index": int(row["source_index"]),
                            "scale_block_index": block_index,
                            "scale_block_id": block.block_id,
                            "scale_id": scale_id,
                            "dx_grid_scale": float(
                                plan.scale_table.dx_grid_scale[scale_id]
                            ),
                            "ds_frame_scale": float(
                                plan.scale_table.ds_frame_scale[scale_id]
                            ),
                            "arc_length_grid_scale": float(
                                plan.scale_table.arc_length_grid_scale[scale_id]
                            ),
                            "assigned_count": assigned_count,
                            "valid_count": valid_count,
                            "invalid_count": assigned_count - valid_count,
                            "coverage": valid_count / assigned_count,
                            "valid_negative_candidate_count": negative_count,
                            "valid_positive_candidate_count": positive_count,
                            "frozen_library_stratum_two_class_nonempty": eligible,
                            "selected_negative_template_count": int(eligible),
                            "selected_positive_template_count": int(eligible),
                        }
                    )
            cache_evidence.append(
                {
                    "dataset": dataset,
                    "source_ordinal": ordinal,
                    "source_index": int(row["source_index"]),
                    "path": str(path),
                    "file_sha256": sha256_file(path),
                    "cache_builder_git_commit": row.get(
                        "cache_builder_git_commit"
                    ),
                }
            )

    expected_diagnostic_count = 8 * 4 * 2 * 1_000
    if len(diagnostic_rows) != expected_diagnostic_count:
        raise RuntimeError("train coverage did not preserve every required stratum")
    fields = (
        "dataset",
        "physical_family",
        "source_ordinal",
        "source_index",
        "scale_block_index",
        "scale_block_id",
        "scale_id",
        "dx_grid_scale",
        "ds_frame_scale",
        "arc_length_grid_scale",
        "assigned_count",
        "valid_count",
        "invalid_count",
        "coverage",
        "valid_negative_candidate_count",
        "valid_positive_candidate_count",
        "frozen_library_stratum_two_class_nonempty",
        "selected_negative_template_count",
        "selected_positive_template_count",
    )
    diagnostic_sha = _atomic_csv(diagnostic_path, diagnostic_rows, fields)
    expanded_block_counts = selected_by_block["expanded_3_1"]
    pass_conditions = {
        "every_expanded_arc_level_has_valid_train_primitive": bool(
            np.all(expanded_arc_valid > 0)
        ),
        "expanded_block_has_selected_positive_template": (
            expanded_block_counts["positive"] > 0
        ),
        "expanded_block_has_selected_negative_template": (
            expanded_block_counts["negative"] > 0
        ),
        "every_dataset_source_block_scale_stratum_reported": (
            len(diagnostic_rows) == expected_diagnostic_count
        ),
        "one_authenticated_train_portable_population_marker": True,
        "no_test_dataset_opened": True,
    }
    summary: dict[str, Any] = {
        "schema": "pathline_template_matching.phase31_train_only_coverage.v1",
        "experiment": "Verify_LongArcHorizon_1.1",
        "parent_experiment": plan.experiment,
        "status": "pass" if all(pass_conditions.values()) else "fail",
        "config_sha256": plan.config_sha256,
        "dataset_registry_sha256": plan.dataset_registry_sha256,
        "maximum_source_frame_intervals": 48.0,
        "opened_dataset_split": "train_only",
        "train_datasets": list(plan.train_datasets),
        "forbidden_test_datasets": list(plan.test_datasets),
        "cache_count": len(cache_evidence),
        "cache_evidence": cache_evidence,
        "cache_evidence_content_sha256": canonical_json_sha256(cache_evidence),
        "train_portable_population_pass": portable_population_evidence,
        "diagnostic_row_count": len(diagnostic_rows),
        "diagnostics_file": str(diagnostic_path),
        "diagnostics_file_sha256": diagnostic_sha,
        "expanded_arc_level_valid_counts": [
            {
                "arc_index": index,
                "arc_length_grid_scale": float(EXPANDED_ARC_VALUES[index]),
                "valid_train_count": int(count),
            }
            for index, count in enumerate(expanded_arc_valid)
        ],
        "globally_selected_template_counts_by_block": selected_by_block,
        "library_rule": (
            "one_negative_and_one_positive_only_for_each_two_class_nonempty_"
            "dataset_source_scale_stratum"
        ),
        "pass_conditions": pass_conditions,
        "prohibited_operations_performed": [],
    }
    summary["summary_content_sha256"] = canonical_json_sha256(summary)
    _atomic_json(summary_path, summary)
    return summary


def _phase31_evaluation_gate_evidence(
    plan: Phase21Plan,
    evidence: Mapping[str, Any] | None,
    *,
    evaluator_git_commit: str,
    strict_protocol: bool,
) -> dict[str, Any] | None:
    """Normalize the two authenticated Verify gates for result provenance."""

    if plan.experiment != EXPERIMENT31:
        if evidence is not None:
            raise ValueError("Verify gate evidence is defined only for mainExp 3.1")
        return None
    if evidence is None:
        if strict_protocol:
            raise ValueError("strict mainExp 3.1 evaluation requires Verify gate evidence")
        return None
    coverage = dict(evidence)
    synthetic_value = coverage.get("synthetic_pass")
    if not isinstance(synthetic_value, Mapping):
        raise ValueError("Phase B evidence does not retain authenticated Phase A evidence")
    synthetic = dict(synthetic_value)
    expected = {
        "git_commit": evaluator_git_commit,
        "main_config_sha256": plan.config_sha256,
        "dataset_registry_sha256": plan.dataset_registry_sha256,
    }
    for gate_name, gate in (("synthetic_pass", synthetic), ("train_coverage_pass", coverage)):
        drift = {
            name: (gate.get(name), value)
            for name, value in expected.items()
            if gate.get(name) != value
        }
        if drift:
            raise ValueError(f"{gate_name} evaluation evidence changed: {drift}")
        for name in ("path", "file_size", "file_sha256", "verify_config_sha256", "outputs"):
            if name not in gate:
                raise ValueError(f"{gate_name} evaluation evidence lacks {name}")
        if len(str(gate["file_sha256"])) != 64 or int(gate["file_size"]) <= 0:
            raise ValueError(f"{gate_name} marker file evidence is invalid")
        if not isinstance(gate["outputs"], list) or not gate["outputs"]:
            raise ValueError(f"{gate_name} output evidence is invalid")
    if coverage.get("synthetic_pass_file_sha256") != synthetic["file_sha256"]:
        raise ValueError("Phase B evidence points to a different Phase A marker")
    if coverage["verify_config_sha256"] != synthetic["verify_config_sha256"]:
        raise ValueError("Phase A and Phase B Verify config SHA-256 differ")
    return {
        "synthetic_pass": {
            name: synthetic[name]
            for name in (
                "path",
                "file_size",
                "file_sha256",
                "git_commit",
                "main_config_sha256",
                "verify_config_sha256",
                "dataset_registry_sha256",
                "outputs",
            )
        },
        "train_coverage_pass": {
            name: coverage[name]
            for name in (
                "path",
                "file_size",
                "file_sha256",
                "git_commit",
                "main_config_sha256",
                "verify_config_sha256",
                "dataset_registry_sha256",
                "synthetic_pass_file_sha256",
                "outputs",
            )
        },
    }


def _freeze_cache_evidence(
    plan: Phase21Plan,
    root: Path,
    summaries: Sequence[CacheBuildSummary],
    *,
    evaluator_git_commit: str,
    verification_gate_evidence: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Freeze manifests from completed array shards before query evaluation."""

    expected_count = len(plan.datasets) * plan.source_count
    if len(summaries) != expected_count:
        raise ValueError(f"received {len(summaries)} shard summaries, expected {expected_count}")
    cache_rows = [dict(summary.cache_row) for summary in summaries]
    raw_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for summary in summaries:
        row = dict(summary.raw_input_row)
        key = (str(row["dataset"]), str(row["registered_path"]))
        previous = raw_by_key.get(key)
        if previous is not None and previous != row:
            raise ValueError(f"raw provenance changed across sidecars: {key}")
        raw_by_key[key] = row
    manifests = {
        "cache_manifest.json": _manifest_payload(
            f"{plan.artifact_tag}_cache_manifest",
            cache_rows,
            experiment=plan.experiment,
        ),
        "raw_input_manifest.json": _manifest_payload(
            f"{plan.artifact_tag}_raw_input_manifest",
            list(raw_by_key.values()),
            experiment=plan.experiment,
        ),
        "derived_window_manifest.json": _manifest_payload(
            f"{plan.artifact_tag}_derived_window_manifest",
            [summary.derived_window_row for summary in summaries],
            experiment=plan.experiment,
        ),
        "seed_and_scale_assignment_manifest.json": _manifest_payload(
            f"{plan.artifact_tag}_seed_assignment_manifest",
            [summary.assignment_row for summary in summaries],
            experiment=plan.experiment,
        ),
        "label_manifest.json": _manifest_payload(
            f"{plan.artifact_tag}_label_manifest",
            [summary.label_row for summary in summaries],
            experiment=plan.experiment,
        ),
        "primitive_manifest.json": _manifest_payload(
            f"{plan.artifact_tag}_primitive_manifest",
            [summary.primitive_row for summary in summaries],
            experiment=plan.experiment,
        ),
    }
    manifest_hashes = {name: _atomic_json(root / name, value) for name, value in manifests.items()}
    audit_fields = (
        "dataset",
        "physical_family",
        "split",
        "source_ordinal",
        "source_index",
        "scale_id",
        "dx_grid_scale",
        "ds_frame_scale",
        "arc_length_grid_scale",
        "reference_class",
        "assigned_count",
        "valid_count",
        "invalid_count",
    )
    if plan.experiment == EXPERIMENT31:
        audit_fields = audit_fields[:5] + (
            "scale_block_index",
            "scale_block_id",
        ) + audit_fields[5:]
    audit_sha = _atomic_csv(
        root / "audit_counts.csv",
        (row for summary in summaries for row in summary.audit_rows),
        audit_fields,
    )
    input_manifest = {
        "schema": f"pathline_template_matching.{plan.artifact_tag}_input_manifest.v1",
        "experiment": plan.experiment,
        "config_sha256": plan.config_sha256,
        "dataset_registry_path": str(plan.dataset_registry_path),
        "dataset_registry_sha256": plan.dataset_registry_sha256,
        "dataset_order": list(plan.datasets),
        "train_datasets": list(plan.train_datasets),
        "test_datasets": list(plan.test_datasets),
        "cache_count": len(cache_rows),
        "dataset_registry_sha256s": sorted(
            {
                str(row["dataset_registry_sha256"])
                for row in cache_rows
                if row.get("dataset_registry_sha256") is not None
            }
        ),
        "portable_builder_git_commits": sorted(
            {
                str(row["portable_builder_git_commit"])
                for row in cache_rows
                if row.get("portable_builder_git_commit") is not None
            }
        ),
        "cache_builder_git_commits": sorted(
            {
                str(row["cache_builder_git_commit"])
                for row in cache_rows
                if row.get("cache_builder_git_commit") is not None
            }
        ),
        "evaluator_git_commit": evaluator_git_commit,
        "manifest_file_sha256": manifest_hashes,
        "audit_counts_file_sha256": audit_sha,
        "cache_file_sha256": {
            str(row["path"]): str(row["file_sha256"]) for row in cache_rows
        },
    }
    if plan.experiment == EXPERIMENT31:
        portable_population_passes = sorted(
            {
                (
                    str(row["portable_population_scope"]),
                    str(row["portable_population_pass_path"]),
                    int(row["portable_population_pass_file_size"]),
                    str(row["portable_population_pass_file_sha256"]),
                    str(row["portable_population_rows_content_sha256"]),
                )
                for row in cache_rows
                if "portable_population_scope" in row
            }
        )
        input_manifest.update(
            {
                "maximum_source_frame_intervals": float(
                    plan.maximum_source_frame_intervals
                ),
                "assignment_count_per_seed": int(plan.assignment_count_per_seed),
                "unique_center_seed_count_per_source_time": int(
                    plan.assigned_seed_count
                ),
                "assigned_primitive_count_per_source_time": int(
                    plan.assigned_primitive_count
                ),
                "verification_gates": verification_gate_evidence,
                "portable_population_passes": [
                    {
                        "access_scope": scope,
                        "path": path,
                        "file_size": size,
                        "file_sha256": file_sha,
                        "rows_content_sha256": rows_sha,
                    }
                    for scope, path, size, file_sha, rows_sha in portable_population_passes
                ],
            }
        )
    _atomic_json(root / "input_manifest.json", input_manifest)
    return cache_rows


def _initialize_evaluation_run(
    plan: Phase21Plan,
    run_dir: str | Path,
    *,
    git_commit: str,
    strict_protocol: bool,
) -> Path:
    root = Path(run_dir).resolve()
    if root.exists():
        raise FileExistsError(f"immutable run directory already exists: {root}")
    if strict_protocol:
        plan.validate_production_contract()
        if len(git_commit) != 40 or any(character not in "0123456789abcdef" for character in git_commit):
            raise ValueError("production evaluation requires a full lowercase Git commit SHA")
    root.mkdir(parents=True, exist_ok=False)
    config_payload = plan.config_path.read_bytes()
    if _sha256_bytes(config_payload) != plan.config_sha256:
        raise ValueError("config changed after Phase21Plan was loaded")
    _atomic_bytes(root / "frozen_config.yaml", config_payload)
    _atomic_json(root / "scale_manifest.json", _scale_manifest(plan))
    _atomic_json(
        root / "run_start.json",
        {
            "schema": (
                f"pathline_template_matching.{plan.artifact_tag}_run_state.v1"
            ),
            "experiment": plan.experiment,
            "status": "inputs_not_yet_frozen",
            "config_sha256": plan.config_sha256,
            "git_commit": git_commit,
            "started_unix_seconds": time.time(),
        },
    )
    return root


def _query_csv_header(plan: Phase21Plan | None = None) -> tuple[str, ...]:
    phase31 = plan is not None and plan.experiment == EXPERIMENT31
    identity = (
        (
            "query_assigned_row_index",
            "query_center_seed_index",
            "query_scale_block_index",
            "query_scale_block_id",
        )
        if phase31
        else ("query_seed_index",)
    )
    common = (
        "query_dataset",
        "query_physical_family",
        "query_source_ordinal",
        "query_source_index",
    ) + identity + (
        "query_seed_x",
        "query_seed_y",
        "query_seed_z",
        "query_scale_id",
        "query_dx_grid_scale",
        "query_ds_frame_scale",
        "query_arc_length_grid_scale",
        "reference_label",
        "prior_prediction",
        "prior_score",
    )
    method_columns: list[str] = []
    for stem in ("raw", "pca", "fmt"):
        method_identity = (
            (
                f"{stem}_match_assigned_row_index",
                f"{stem}_match_center_seed_index",
                f"{stem}_match_scale_block_index",
                f"{stem}_match_scale_block_id",
            )
            if phase31
            else (f"{stem}_match_seed_index",)
        )
        method_columns.extend(
            (
                f"{stem}_prediction",
                f"{stem}_score",
                f"{stem}_nearest_positive_distance",
                f"{stem}_nearest_negative_distance",
                f"{stem}_selected_template_distance",
                f"{stem}_match_dataset",
                f"{stem}_match_physical_family",
                f"{stem}_match_source_ordinal",
                f"{stem}_match_source_index",
            )
            + method_identity
            + (
                f"{stem}_match_scale_id",
                f"{stem}_match_label",
            )
        )
    return common + tuple(method_columns)


def _write_query_rows(
    writer: csv.DictWriter,
    *,
    plan: Phase21Plan,
    cache_row: Mapping[str, Any],
    cache: Mapping[str, Any],
    library: Mapping[str, np.ndarray],
    prior_prediction: bool,
    prior_score: float,
    matches: Mapping[str, ExhaustiveMatchResult],
) -> None:
    valid_seed_index = np.asarray(cache["valid_seed_index"], dtype=np.int64)
    valid_scale_id = np.asarray(cache["valid_scale_id"], dtype=np.int32)
    all_seeds = np.asarray(cache["seeds_xyz"], dtype=np.float32)
    labels = np.asarray(cache["valid_labels"], dtype=bool)
    if plan.experiment == EXPERIMENT31:
        assigned_row_index = np.asarray(
            cache["valid_assigned_row_index"], dtype=np.int64
        )
        center_seed_index = np.asarray(
            cache["valid_center_seed_index"], dtype=np.int64
        )
        scale_block_index = np.asarray(
            cache["valid_scale_block_index"], dtype=np.int8
        )
    stems = {METHOD_RAW: "raw", METHOD_PCA: "pca", METHOD_FMT: "fmt"}
    for query_index in range(len(labels)):
        seed_id = int(valid_seed_index[query_index])
        scale_id = int(valid_scale_id[query_index])
        row: dict[str, Any] = {
            "query_dataset": str(cache_row["dataset"]),
            "query_physical_family": plan.family_by_dataset[str(cache_row["dataset"])],
            "query_source_ordinal": int(cache_row["source_ordinal"]),
            "query_source_index": int(cache_row["source_index"]),
            "query_seed_x": float(all_seeds[seed_id, 0]),
            "query_seed_y": float(all_seeds[seed_id, 1]),
            "query_seed_z": float(all_seeds[seed_id, 2]),
            "query_scale_id": scale_id,
            "query_dx_grid_scale": float(plan.scale_table.dx_grid_scale[scale_id]),
            "query_ds_frame_scale": float(plan.scale_table.ds_frame_scale[scale_id]),
            "query_arc_length_grid_scale": float(
                plan.scale_table.arc_length_grid_scale[scale_id]
            ),
            "reference_label": bool(labels[query_index]),
            "prior_prediction": prior_prediction,
            "prior_score": prior_score,
        }
        if plan.experiment == EXPERIMENT31:
            block_index = int(scale_block_index[query_index])
            row.update(
                {
                    "query_assigned_row_index": int(
                        assigned_row_index[query_index]
                    ),
                    "query_center_seed_index": int(
                        center_seed_index[query_index]
                    ),
                    "query_scale_block_index": block_index,
                    "query_scale_block_id": plan.effective_scale_blocks[
                        block_index
                    ].block_id,
                }
            )
        else:
            row["query_seed_index"] = seed_id
        for method, stem in stems.items():
            result = matches[method]
            match_index = int(result.nearest_indices[query_index])
            row.update(
                {
                    f"{stem}_prediction": bool(result.labels[query_index]),
                    f"{stem}_score": float(result.scores[query_index]),
                    f"{stem}_nearest_positive_distance": float(
                        result.nearest_positive_distances[query_index]
                    ),
                    f"{stem}_nearest_negative_distance": float(
                        result.nearest_negative_distances[query_index]
                    ),
                    f"{stem}_selected_template_distance": float(
                        result.nearest_distances[query_index]
                    ),
                    f"{stem}_match_dataset": str(library["dataset"][match_index]),
                    f"{stem}_match_physical_family": str(
                        library["physical_family"][match_index]
                    ),
                    f"{stem}_match_source_ordinal": int(
                        library["source_ordinal"][match_index]
                    ),
                    f"{stem}_match_source_index": int(
                        library["source_index"][match_index]
                    ),
                    f"{stem}_match_scale_id": int(library["scale_id"][match_index]),
                    f"{stem}_match_label": bool(library["labels"][match_index]),
                }
            )
            if plan.experiment == EXPERIMENT31:
                row.update(
                    {
                        f"{stem}_match_assigned_row_index": int(
                            library["assigned_row_index"][match_index]
                        ),
                        f"{stem}_match_center_seed_index": int(
                            library["center_seed_index"][match_index]
                        ),
                        f"{stem}_match_scale_block_index": int(
                            library["scale_block_index"][match_index]
                        ),
                        f"{stem}_match_scale_block_id": str(
                            library["scale_block_id"][match_index]
                        ),
                    }
                )
            else:
                row[f"{stem}_match_seed_index"] = int(
                    library["seed_index"][match_index]
                )
        writer.writerow({name: _csv_value(row.get(name)) for name in writer.fieldnames})


def _metrics_fieldnames(identity: Sequence[str]) -> tuple[str, ...]:
    return tuple(identity) + (
        "method",
        "assigned_count",
        "valid_count",
        "invalid_count",
        "coverage",
        "sample_count",
        "positive_count",
        "negative_count",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "single_class_group",
    ) + METRIC_NAMES


def _bootstrap_difference_rows(
    plan: Phase21Plan,
    time_rows: Sequence[Mapping[str, Any]],
    main_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    families = tuple(
        dict.fromkeys(plan.family_by_dataset[dataset] for dataset in plan.test_datasets)
    )
    units: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for family in families:
        for method in METHODS:
            matching = [
                row
                for row in time_rows
                if row["physical_family"] == family and row["method"] == method
            ]
            matching.sort(key=lambda row: (str(row["dataset"]), int(row["source_ordinal"])))
            if not matching:
                raise ValueError(f"bootstrap has no source-time units for {family}/{method}")
            units[(family, method)] = matching
    generator = np.random.Generator(np.random.PCG64(plan.bootstrap_seed))
    draws = {
        family: generator.integers(
            0,
            len(units[(family, METHOD_FMT)]),
            size=(plan.bootstrap_replicates, len(units[(family, METHOD_FMT)])),
        )
        for family in families
    }
    macro_row = {
        str(row["method"]): row
        for row in main_rows
        if row["aggregation"] == "equal_weight_test_family_source_time_macro"
    }
    output: list[dict[str, Any]] = []
    for metric in METRIC_NAMES:
        method_distributions: dict[str, np.ndarray] = {}
        for method in METHODS:
            family_distributions = []
            for family in families:
                values = np.asarray(
                    [float(row[metric]) for row in units[(family, method)]],
                    dtype=np.float64,
                )
                if np.isfinite(values).all():
                    family_distributions.append(np.mean(values[draws[family]], axis=1))
                else:
                    family_distributions.append(
                        np.full(plan.bootstrap_replicates, np.nan, dtype=np.float64)
                    )
            method_distributions[method] = np.mean(
                np.stack(family_distributions, axis=1), axis=1
            )
        for comparator in (METHOD_PRIOR, METHOD_RAW, METHOD_PCA):
            difference = method_distributions[METHOD_FMT] - method_distributions[comparator]
            finite = difference[np.isfinite(difference)]
            if len(finite):
                lower, upper = np.percentile(finite, [2.5, 97.5], method="linear")
            else:
                lower = upper = float("nan")
            point = float(macro_row[METHOD_FMT][metric]) - float(
                macro_row[comparator][metric]
            )
            output.append(
                {
                    "metric": metric,
                    "method": METHOD_FMT,
                    "comparator": comparator,
                    "point_estimate": point,
                    "ci95_lower": float(lower),
                    "ci95_upper": float(upper),
                    "finite_replicate_count": int(len(finite)),
                    "replicate_count": int(plan.bootstrap_replicates),
                    "seed": int(plan.bootstrap_seed),
                    "paired_unit": "dataset_source_timeslice",
                }
            )
    return output


def _main_table_rows(
    plan: Phase21Plan,
    time_rows: Sequence[Mapping[str, Any]],
    pooled_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    families = tuple(
        dict.fromkeys(plan.family_by_dataset[dataset] for dataset in plan.test_datasets)
    )
    output: list[dict[str, Any]] = []

    def strict_mean(values: Sequence[float]) -> float:
        array = np.asarray(values, dtype=np.float64)
        return float(array.mean()) if len(array) and np.isfinite(array).all() else float("nan")

    for method in METHODS:
        family_values: dict[str, dict[str, float]] = {}
        for family in families:
            rows = [
                row
                for row in time_rows
                if row["method"] == method and row["physical_family"] == family
            ]
            family_values[family] = {
                name: strict_mean([float(row[name]) for row in rows])
                for name in ("coverage",) + METRIC_NAMES
            }
        macro = {
            name: strict_mean([family_values[family][name] for family in families])
            for name in ("coverage",) + METRIC_NAMES
        }
        output.append(
            {
                "aggregation": "equal_weight_test_family_source_time_macro",
                "method": method,
                "test_family_count": len(families),
                "sample_count": sum(
                    int(row["sample_count"])
                    for row in time_rows
                    if row["method"] == method
                ),
                **macro,
            }
        )
        pooled = next(row for row in pooled_rows if row["method"] == method)
        output.append(
            {
                "aggregation": "pooled_all_valid_test_queries_descriptive",
                "method": method,
                "test_family_count": len(families),
                "sample_count": int(pooled["sample_count"]),
                "coverage": float(pooled["coverage"]),
                **{name: float(pooled[name]) for name in METRIC_NAMES},
            }
        )
    return output


def _main_table_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "aggregation",
        "method",
        "sample_count",
        "coverage",
        "accuracy",
        "average_precision",
        "f1",
        "balanced_accuracy",
        "auroc",
        "precision",
        "recall",
    )
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for name in columns:
            value = row[name]
            if isinstance(value, (float, np.floating)):
                values.append("null" if not np.isfinite(value) else f"{float(value):.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _build_phase21_visualization_artifacts(
    plan: Phase21Plan,
    *,
    root: Path,
    test_rows: Sequence[Mapping[str, Any]],
    query: Mapping[str, np.ndarray],
    fmt_prediction: np.ndarray,
    git_commit: str,
    verify_cache_hashes: bool,
) -> dict[str, Any] | None:
    """Build fixed ordinal-2 triptychs after all numerical metrics exist."""

    visualization_config = plan.config.get("visualization")
    if not isinstance(visualization_config, Mapping):
        return None
    from .phase21_visualization import (
        FIXED_SOURCE_ORDINAL,
        build_phase21_visualization_scene,
        ordered_fmt_prediction,
        render_phase21_scene_artifact,
        write_phase21_scene_artifact,
    )

    configured_ordinal = int(visualization_config.get("source_ordinal", -1))
    if configured_ordinal != FIXED_SOURCE_ORDINAL:
        raise ValueError("template-matching visualization source ordinal drifted")
    if visualization_config.get("metric_based_or_prediction_based_scene_selection") != "forbidden":
        raise ValueError("visualization must forbid metric-selected scenes")

    entries: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(plan.test_datasets):
        candidates = [
            row
            for row in test_rows
            if str(row["dataset"]) == dataset
            and int(row["source_ordinal"]) == FIXED_SOURCE_ORDINAL
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"fixed visualization cache is not unique for {dataset}/ordinal2"
            )
        cache_row = candidates[0]
        cache = _load_cache(
            Path(str(cache_row["path"])),
            expected_sha256=(
                str(cache_row["file_sha256"]) if verify_cache_hashes else None
            ),
        )
        _validate_cache_provenance(plan, cache, cache_row)
        dataset_mask = (query["dataset_index"] == dataset_index) & (
            query["source_ordinal"] == FIXED_SOURCE_ORDINAL
        )
        cache_seed_index = np.asarray(cache["valid_seed_index"], dtype=np.int64)
        cache_scale_id = np.asarray(cache["valid_scale_id"], dtype=np.int32)
        cache_labels = np.asarray(cache["valid_labels"], dtype=np.bool_)
        if not np.array_equal(
            query["valid_seed_index"][dataset_mask], cache_seed_index
        ):
            raise RuntimeError("visualization query seed order differs from the cache")
        if not np.array_equal(query["scale_id"][dataset_mask], cache_scale_id):
            raise RuntimeError("visualization query scale order differs from the cache")
        if not np.array_equal(query["labels"][dataset_mask], cache_labels):
            raise RuntimeError("visualization query labels differ from the cache")
        work_items: list[tuple[int | None, ScaleAssignmentBlock]] = (
            list(enumerate(plan.effective_scale_blocks))
            if plan.experiment == EXPERIMENT31
            else [(None, ScaleAssignmentBlock("legacy_2_1", 0, 1_000, 15068))]
        )
        for block_index, block in work_items:
            cache_mask = (cache_scale_id >= block.scale_id_start) & (
                cache_scale_id < block.scale_id_stop
            )
            query_mask = dataset_mask & (
                query["scale_id"] >= block.scale_id_start
            ) & (query["scale_id"] < block.scale_id_stop)
            if plan.experiment == EXPERIMENT31:
                if not np.array_equal(
                    query["assigned_row_index"][query_mask],
                    np.asarray(cache["valid_assigned_row_index"])[cache_mask],
                ) or not np.array_equal(
                    query["center_seed_index"][query_mask],
                    np.asarray(cache["valid_center_seed_index"])[cache_mask],
                ):
                    raise RuntimeError(
                        "visualization query assigned/center identities differ from cache"
                    )
                if not np.all(
                    query["scale_block_index"][query_mask] == block_index
                ):
                    raise RuntimeError("visualization query mixes scale blocks")

            scene_cache: Mapping[str, Any] = cache
            suffix = ""
            if plan.experiment == EXPERIMENT31:
                valid_array_names = (
                    "raw_features",
                    "fmt_features",
                    "valid_labels",
                    "valid_seed_index",
                    "valid_scale_id",
                    "center_sample_time",
                    "valid_assigned_row_index",
                    "valid_center_seed_index",
                    "valid_scale_block_index",
                )
                scene_cache_dict = dict(cache)
                for name in valid_array_names:
                    scene_cache_dict[name] = np.ascontiguousarray(
                        np.asarray(cache[name])[cache_mask]
                    )
                scene_metadata = dict(cache["metadata"])
                scene_metadata.update(
                    {
                        "visualization_scale_block_index": int(block_index),
                        "visualization_scale_block_id": block.block_id,
                        "visualization_scale_id_start": block.scale_id_start,
                        "visualization_scale_id_stop_exclusive": block.scale_id_stop,
                    }
                )
                hashes = dict(scene_metadata.get("array_sha256", {}))
                for name in (
                    "raw_features",
                    "valid_labels",
                    "valid_seed_index",
                    "valid_scale_id",
                    "center_sample_time",
                    "valid_assigned_row_index",
                    "valid_center_seed_index",
                    "valid_scale_block_index",
                    "seeds_xyz",
                    "ivd_volume",
                ):
                    hashes[name] = canonical_array_sha256(scene_cache_dict[name])
                scene_metadata["array_sha256"] = hashes
                scene_cache_dict["metadata"] = scene_metadata
                scene_cache = scene_cache_dict
                suffix = f"_{block.block_id}"

            selected_seed_index = cache_seed_index[cache_mask]
            selected_scale_id = cache_scale_id[cache_mask]
            prediction_values = np.asarray(
                fmt_prediction[query_mask], dtype=np.bool_
            )
            prediction_contract = ordered_fmt_prediction(
                prediction_values,
                selected_seed_index,
                selected_scale_id,
            )
            scene, scientific_audit = build_phase21_visualization_scene(
                scene_cache, prediction_contract
            )
            if plan.experiment == EXPERIMENT31:
                scene["title"] = f"{scene['title']} | {block.block_id}"
                scientific_audit["scale_block"] = {
                    "scale_block_index": int(block_index),
                    "scale_block_id": block.block_id,
                    "scale_id_start": block.scale_id_start,
                    "scale_id_stop_exclusive": block.scale_id_stop,
                    "selection_normalization": (
                        "within_block_cartesian_indices_dx_ds_arc_divided_by_9"
                    ),
                    "dx_grid_scale_values": [
                        f"{value:.12f}"
                        for value in np.unique(
                            plan.scale_table.dx_grid_scale[
                                block.scale_id_start : block.scale_id_stop
                            ]
                        )
                    ],
                    "ds_frame_scale_values": [
                        f"{value:.12f}"
                        for value in np.unique(
                            plan.scale_table.ds_frame_scale[
                                block.scale_id_start : block.scale_id_stop
                            ]
                        )
                    ],
                    "arc_length_grid_scale_values": [
                        f"{value:.12f}"
                        for value in np.unique(
                            plan.scale_table.arc_length_grid_scale[
                                block.scale_id_start : block.scale_id_stop
                            ]
                        )
                    ],
                }

            scene_stem = root / "scenes" / f"{dataset}_source_ordinal_2{suffix}"
            scene_path = scene_stem.with_suffix(".scene.npz")
            scene_manifest_path = scene_stem.with_suffix(".scene.json")
            scene_manifest = write_phase21_scene_artifact(
                scene, scientific_audit, scene_path, scene_manifest_path
            )
            render_stem = (
                root
                / "figures"
                / f"{dataset}_source_ordinal_2{suffix}_triptych"
            )
            rendered = render_phase21_scene_artifact(
                scene_path,
                scene_manifest_path,
                render_stem,
                dpi=int(visualization_config.get("png_dpi", 360)),
            )
            relative = lambda path: str(Path(path).resolve().relative_to(root))
            entry = {
                "dataset": dataset,
                "physical_family": plan.family_by_dataset[dataset],
                "source_ordinal": FIXED_SOURCE_ORDINAL,
                "source_index": int(cache_row["source_index"]),
                "source_cache": str(cache_row["path"]),
                "source_cache_sha256": str(cache_row["file_sha256"]),
                "fmt_prediction_sha256": canonical_array_sha256(
                    prediction_values
                ),
                "scene_npz": relative(scene_path),
                "scene_npz_sha256": str(scene_manifest["scene_npz_sha256"]),
                "scene_manifest": relative(scene_manifest_path),
                "scene_manifest_file_sha256": str(
                    scene_manifest["scene_manifest_file_sha256"]
                ),
                "png": relative(rendered.png_path),
                "png_sha256": sha256_file(rendered.png_path),
                "pdf": relative(rendered.pdf_path),
                "pdf_sha256": sha256_file(rendered.pdf_path),
                "render_metadata": relative(rendered.metadata_path),
                "render_metadata_sha256": sha256_file(rendered.metadata_path),
                "panel_alignment": relative(rendered.alignment_path),
                "panel_alignment_sha256": sha256_file(rendered.alignment_path),
                "query_count": int(query_mask.sum()),
                "confusion_counts": dict(rendered.metadata["counts"]),
            }
            if plan.experiment == EXPERIMENT31:
                if rendered.svg_path is None:
                    raise RuntimeError("3.1 triptych did not export required SVG")
                required_exports = []
                additional_audit_files = []
                export_groups = (
                    ("required", "scene_npz", scene_path),
                    (
                        "required",
                        "svg_with_editable_text_and_rasterized_3d_marks",
                        rendered.svg_path,
                    ),
                    (
                        "required",
                        "pdf_with_editable_text_and_rasterized_3d_marks",
                        rendered.pdf_path,
                    ),
                    ("required", "png_360dpi", rendered.png_path),
                    ("required", "panel_alignment_json", rendered.alignment_path),
                    ("additional_audit", "scene_manifest_json", scene_manifest_path),
                    ("additional_audit", "render_metadata_json", rendered.metadata_path),
                )
                for group, export_kind, export_path in export_groups:
                    export_path = Path(export_path).resolve()
                    if not export_path.is_file() or export_path.stat().st_size <= 0:
                        raise RuntimeError(f"missing 3.1 visualization export: {export_path}")
                    export_row = {
                        "relative_path": str(export_path.relative_to(root)),
                        "export_kind": export_kind,
                        "size_bytes": int(export_path.stat().st_size),
                        "sha256": sha256_file(export_path),
                    }
                    if export_row["sha256"] != sha256_file(export_path):
                        raise RuntimeError(f"3.1 visualization export changed: {export_path}")
                    if group == "required":
                        required_exports.append(export_row)
                    else:
                        additional_audit_files.append(export_row)
                entry.update(
                    {
                        "scale_block_index": int(block_index),
                        "scale_block_id": block.block_id,
                        "svg": relative(rendered.svg_path),
                        "svg_sha256": sha256_file(rendered.svg_path),
                        "required_exports": required_exports,
                        "additional_audit_files": additional_audit_files,
                    }
                )
            entries.append(entry)

    if plan.experiment == EXPERIMENT31:
        keys = [(entry["dataset"], entry["scale_block_id"]) for entry in entries]
        if len(entries) != 4 or len(set(keys)) != 4:
            raise RuntimeError(
                "3.1 visualization must contain four unique dataset/block figures"
            )
        expected_export_kinds = {
            "scene_npz",
            "svg_with_editable_text_and_rasterized_3d_marks",
            "pdf_with_editable_text_and_rasterized_3d_marks",
            "png_360dpi",
            "panel_alignment_json",
        }
        for entry in entries:
            exports = entry["required_exports"]
            if (
                len(exports) != len(expected_export_kinds)
                or {row["export_kind"] for row in exports} != expected_export_kinds
            ):
                raise RuntimeError("3.1 visualization required-export set is incomplete")
            additional = entry["additional_audit_files"]
            if (
                len(additional) != 2
                or {row["export_kind"] for row in additional}
                != {"scene_manifest_json", "render_metadata_json"}
            ):
                raise RuntimeError("3.1 visualization additional audit set is incomplete")
            for row in exports + additional:
                if set(row) != {
                    "relative_path",
                    "export_kind",
                    "size_bytes",
                    "sha256",
                }:
                    raise RuntimeError("3.1 visualization export audit fields changed")
                path = (root / row["relative_path"]).resolve()
                if (
                    not path.is_file()
                    or int(row["size_bytes"]) != path.stat().st_size
                    or row["sha256"] != sha256_file(path)
                ):
                    raise RuntimeError("3.1 visualization export failed final hash audit")

    manifest: dict[str, Any] = {
        "schema": (
            "pathline_template_matching.phase31_visualization_manifest.v1"
            if plan.experiment == EXPERIMENT31
            else "pathline_template_matching.phase21_visualization_manifest.v1"
        ),
        "experiment": plan.experiment,
        "evidence_scope": "explanatory_exposed_development_only",
        "aggregate_performance_proof": False,
        "config_sha256": plan.config_sha256,
        "git_commit": git_commit,
        "source_selection": "fixed_test_datasets_source_ordinal_2_before_performance",
        "scene_selection_uses_predictions_or_metrics": False,
        "entry_count": len(entries),
        "entries": entries,
    }
    if plan.experiment == EXPERIMENT31:
        manifest.update(
            {
                "cross_block_aggregation_or_majority_vote": False,
                "unique_key": ["dataset", "scale_block_id"],
                "expected_figure_count": 4,
                "required_export_file_fields": [
                    "relative_path",
                    "export_kind",
                    "size_bytes",
                    "sha256",
                ],
                "required_export_count_per_figure": 5,
                "additional_audit_file_count_per_figure": 2,
                "visualization_manifest_json_hash_location": (
                    "result_manifest.json artifacts after this manifest is written"
                ),
            }
        )
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    _atomic_json(root / "visualization_manifest.json", manifest)
    return manifest


def evaluate_phase21_caches(
    plan: Phase21Plan,
    *,
    cache_summaries: Sequence[CacheBuildSummary],
    run_dir: str | Path,
    git_commit: str,
    device: str = "cpu",
    strict_protocol: bool = True,
    verify_cache_hashes: bool = True,
    query_chunk_size: int = 1024,
    library_chunk_size: int = 8192,
    phase31_verification_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze cache evidence, fit train-only components, and evaluate test rows."""

    verification_gate_evidence = _phase31_evaluation_gate_evidence(
        plan,
        phase31_verification_evidence,
        evaluator_git_commit=git_commit,
        strict_protocol=strict_protocol,
    )
    root = _initialize_evaluation_run(
        plan, run_dir, git_commit=git_commit, strict_protocol=strict_protocol
    )
    if strict_protocol:
        for summary in cache_summaries:
            row = summary.cache_row
            if row.get("config_sha256") != plan.config_sha256:
                raise ValueError("cache sidecar config SHA-256 mismatch")
            registry_sha = str(row.get("dataset_registry_sha256", ""))
            portable_commit = str(row.get("portable_builder_git_commit", ""))
            cache_commit = str(row.get("cache_builder_git_commit", ""))
            if len(registry_sha) != 64 or any(
                character not in "0123456789abcdef" for character in registry_sha
            ):
                raise ValueError("cache sidecar has no valid dataset-registry SHA-256")
            if registry_sha != plan.dataset_registry_sha256:
                raise ValueError(
                    "cache sidecar dataset-registry SHA-256 differs from committed config registry"
                )
            for name, commit in (
                ("portable builder", portable_commit),
                ("cache builder", cache_commit),
            ):
                if len(commit) != 40 or any(
                    character not in "0123456789abcdef" for character in commit
                ):
                    raise ValueError(f"cache sidecar has no valid {name} Git commit")
            if portable_commit != git_commit or cache_commit != git_commit:
                raise ValueError(
                    "portable staging, cache build, and evaluator must use one Git commit"
                )
        if plan.experiment == EXPERIMENT31:
            validate_phase31_cache_portable_population_evidence(
                plan,
                [summary.cache_row for summary in cache_summaries],
                usage="evaluation",
                expected_git_commit=git_commit,
                synthetic_pass_file_sha256=verification_gate_evidence[
                    "synthetic_pass"
                ]["file_sha256"],
                train_coverage_pass_file_sha256=verification_gate_evidence[
                    "train_coverage_pass"
                ]["file_sha256"],
            )
    cache_rows = _freeze_cache_evidence(
        plan,
        root,
        cache_summaries,
        evaluator_git_commit=git_commit,
        verification_gate_evidence=verification_gate_evidence,
    )
    # The input manifest must exist before this point: all subsequent cache
    # reads may expose test validity, labels, features, matches, or metrics.
    if not (root / "input_manifest.json").is_file():
        raise RuntimeError("input manifest was not frozen before evaluation")

    execution_contract = configure_deterministic_execution()
    selected_device = str(device)
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if selected_device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu, cuda, or auto")
    _validate_strict_cuda_workspace(
        plan, selected_device=selected_device, strict_protocol=strict_protocol
    )
    train_rows = [row for row in cache_rows if row["split"] == "train"]
    test_rows = [row for row in cache_rows if row["split"] == "test"]
    if len(train_rows) != len(plan.train_datasets) * plan.source_count:
        raise ValueError("train cache population is incomplete")
    if len(test_rows) != len(plan.test_datasets) * plan.source_count:
        raise ValueError("test cache population is incomplete")

    pca, library, prior_positive_fraction, library_audit = _select_library_and_fit_pca(
        plan,
        root,
        train_rows,
        verify_cache_hashes=verify_cache_hashes,
    )
    library_array_hashes = {
        name: canonical_array_sha256(values) for name, values in library.items()
    }
    library_metadata = {
        "schema": (
            f"pathline_template_matching.{plan.artifact_tag}_template_library.v1"
        ),
        "experiment": plan.experiment,
        "config_sha256": plan.config_sha256,
        "descriptor_id": plan.descriptor_config.descriptor_id,
        "candidate_positive_fraction": prior_positive_fraction,
        "template_count": int(len(library["labels"])),
        "negative_template_count": int((~library["labels"]).sum()),
        "positive_template_count": int(library["labels"].sum()),
        "array_sha256": library_array_hashes,
        "combined_array_sha256": canonical_json_sha256(library_array_hashes),
    }
    if plan.experiment == EXPERIMENT31:
        library_metadata.update(
            {
                "maximum_source_frame_intervals": 48.0,
                "scale_count": len(plan.scale_table),
                "assignment_count_per_seed": plan.assignment_count_per_seed,
                "fit_rebuilt_from_full_phase31_train_population": True,
                "phase21_selected_library_appended": False,
            }
        )
    library_path = root / "template_library.npz"
    library_file_sha = _atomic_npz(
        library_path,
        {
            **library,
            "metadata_json": np.asarray(
                json.dumps(library_metadata, sort_keys=True, separators=(",", ":"))
            ),
        },
    )
    library_audit_fields = (
        "dataset",
        "physical_family",
        "source_ordinal",
        "source_index",
        "scale_id",
        "negative_candidate_count",
        "positive_candidate_count",
        "selected_per_class",
        "selected_negative_seed_index",
        "selected_positive_seed_index",
        "skip_reason",
    )
    if plan.experiment == EXPERIMENT31:
        library_audit_fields = library_audit_fields[:4] + (
            "scale_block_index",
            "scale_block_id",
        ) + library_audit_fields[4:-1] + (
            "selected_negative_assigned_row_index",
            "selected_positive_assigned_row_index",
            "selected_negative_center_seed_index",
            "selected_positive_center_seed_index",
            "skip_reason",
        )
    library_audit_sha = _atomic_csv(
        root / "template_library_audit.csv", library_audit, library_audit_fields
    )
    template_manifest = {
        **library_metadata,
        "library_file": str(library_path),
        "library_file_sha256": library_file_sha,
        "library_audit_file_sha256": library_audit_sha,
        "selection_seed": plan.library_seed,
        "selection_rule": (
            "one global PCG64 generator draws negative then positive only for each "
            "two-class-nonempty dataset/source-time/scale stratum; skipped strata "
            "consume no random draws"
        ),
    }
    _atomic_json(root / "template_library_manifest.json", template_manifest)

    matchers, preprocessing_arrays = _matcher_artifacts(library, device=selected_device)
    preprocessing_arrays.update(
        {
            "pca_mean": pca.mean,
            "pca_components": pca.components,
            "pca_singular_values": pca.singular_values,
            "pca_explained_variance_ratio": pca.explained_variance_ratio,
        }
    )
    preprocessing_hashes = {
        name: canonical_array_sha256(values)
        for name, values in preprocessing_arrays.items()
    }
    preprocessing_path = root / "preprocessing_artifacts.npz"
    preprocessing_file_sha = _atomic_npz(preprocessing_path, preprocessing_arrays)
    _atomic_json(
        root / "preprocessing_manifest.json",
        {
            "schema": (
                f"pathline_template_matching.{plan.artifact_tag}_preprocessing.v1"
            ),
            "experiment": plan.experiment,
            "config_sha256": plan.config_sha256,
            "fit_population": "valid_train_candidates_for_pca_selected_library_for_scalers",
            "pca_sample_count": int(pca.sample_count),
            "pca_input_width": int(pca.input_width),
            "pca_output_width": int(pca.components.shape[0]),
            "pca_solver": pca.solver,
            "artifact_file": str(preprocessing_path),
            "artifact_file_sha256": preprocessing_file_sha,
            "array_sha256": preprocessing_hashes,
            "combined_array_sha256": canonical_json_sha256(preprocessing_hashes),
        },
    )

    prior_prediction = bool(prior_positive_fraction > 0.5)
    query_blocks: dict[str, list[np.ndarray]] = {
        "labels": [],
        "valid_seed_index": [],
        "dataset_index": [],
        "source_ordinal": [],
        "source_index": [],
        "scale_id": [],
    }
    if plan.experiment == EXPERIMENT31:
        query_blocks.update(
            {
                "assigned_row_index": [],
                "center_seed_index": [],
                "scale_block_index": [],
            }
        )
    prediction_blocks = {method: [] for method in METHODS}
    score_blocks = {method: [] for method in METHODS}
    assigned_time: dict[tuple[int, int], int] = {}
    source_index_by_time: dict[tuple[int, int], int] = {}
    assigned_scale = np.zeros(len(plan.scale_table), dtype=np.int64)

    query_path = root / "per_query_matches.csv"
    query_temporary = query_path.with_name(f".{query_path.name}.{os.getpid()}.partial")
    if query_path.exists() or query_temporary.exists():
        raise FileExistsError("per-query evidence already exists")
    header = _query_csv_header(plan)
    ordered_test_rows = sorted(
        test_rows,
        key=lambda row: (
            plan.test_datasets.index(str(row["dataset"])),
            int(row["source_ordinal"]),
        ),
    )
    try:
        with query_temporary.open("x", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(header), extrasaction="raise")
            writer.writeheader()
            for cache_row in ordered_test_rows:
                cache = _load_cache(
                    Path(str(cache_row["path"])),
                    expected_sha256=(
                        str(cache_row["file_sha256"]) if verify_cache_hashes else None
                    ),
                )
                _validate_cache_provenance(plan, cache, cache_row)
                labels = np.asarray(cache["valid_labels"], dtype=bool)
                raw = np.asarray(cache["raw_features"], dtype=np.float32)
                fmt = np.asarray(cache["fmt_features"], dtype=np.float32)
                query_features = {
                    METHOD_RAW: raw,
                    METHOD_PCA: pca.transform(raw),
                    METHOD_FMT: fmt,
                }
                matches = {
                    method: matchers[method].query(
                        query_features[method],
                        query_chunk_size=query_chunk_size,
                        library_chunk_size=library_chunk_size,
                    )
                    for method in ONE_NEAREST_NEIGHBOUR_METHODS
                }
                _write_query_rows(
                    writer,
                    plan=plan,
                    cache_row=cache_row,
                    cache=cache,
                    library=library,
                    prior_prediction=prior_prediction,
                    prior_score=prior_positive_fraction,
                    matches=matches,
                )

                dataset_index = plan.test_datasets.index(str(cache_row["dataset"]))
                ordinal = int(cache_row["source_ordinal"])
                query_blocks["labels"].append(labels)
                query_blocks["valid_seed_index"].append(
                    np.asarray(cache["valid_seed_index"], dtype=np.int64)
                )
                query_blocks["dataset_index"].append(
                    np.full(len(labels), dataset_index, dtype=np.int16)
                )
                query_blocks["source_ordinal"].append(
                    np.full(len(labels), ordinal, dtype=np.int16)
                )
                query_blocks["source_index"].append(
                    np.full(len(labels), int(cache_row["source_index"]), dtype=np.int32)
                )
                query_blocks["scale_id"].append(
                    np.asarray(cache["valid_scale_id"], dtype=np.int32)
                )
                if plan.experiment == EXPERIMENT31:
                    query_blocks["assigned_row_index"].append(
                        np.asarray(
                            cache["valid_assigned_row_index"], dtype=np.int64
                        )
                    )
                    query_blocks["center_seed_index"].append(
                        np.asarray(
                            cache["valid_center_seed_index"], dtype=np.int64
                        )
                    )
                    query_blocks["scale_block_index"].append(
                        np.asarray(
                            cache["valid_scale_block_index"], dtype=np.int8
                        )
                    )
                prediction_blocks[METHOD_PRIOR].append(
                    np.full(len(labels), prior_prediction, dtype=np.bool_)
                )
                score_blocks[METHOD_PRIOR].append(
                    np.full(len(labels), prior_positive_fraction, dtype=np.float32)
                )
                for method in ONE_NEAREST_NEIGHBOUR_METHODS:
                    prediction_blocks[method].append(matches[method].labels)
                    score_blocks[method].append(matches[method].scores)
                assignment = np.asarray(cache["scale_assignment"], dtype=np.int32)
                assigned_time[(dataset_index, ordinal)] = len(assignment)
                source_index_by_time[(dataset_index, ordinal)] = int(
                    cache_row["source_index"]
                )
                assigned_scale += np.bincount(
                    assignment, minlength=len(plan.scale_table)
                ).astype(np.int64)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(query_temporary, query_path)
    finally:
        if query_temporary.exists():
            query_temporary.unlink()

    query = {
        name: np.concatenate(blocks) if blocks else np.empty(0)
        for name, blocks in query_blocks.items()
    }
    predictions = {
        method: np.concatenate(blocks) if blocks else np.empty(0, dtype=bool)
        for method, blocks in prediction_blocks.items()
    }
    scores = {
        method: np.concatenate(blocks) if blocks else np.empty(0, dtype=np.float32)
        for method, blocks in score_blocks.items()
    }
    if len(query["labels"]) == 0:
        raise ValueError("no valid test primitive is available for evaluation")

    time_rows: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(plan.test_datasets):
        for ordinal in range(plan.source_count):
            mask = (query["dataset_index"] == dataset_index) & (
                query["source_ordinal"] == ordinal
            )
            source_index = source_index_by_time[(dataset_index, ordinal)]
            for method in METHODS:
                time_rows.append(
                    _metric_row(
                        method,
                        query["labels"][mask],
                        predictions[method][mask],
                        scores[method][mask],
                        assigned_count=assigned_time[(dataset_index, ordinal)],
                        dataset=dataset,
                        physical_family=plan.family_by_dataset[dataset],
                        source_ordinal=ordinal,
                        source_index=source_index,
                    )
                )
    _atomic_csv(
        root / "per_dataset_source_time_metrics.csv",
        time_rows,
        _metrics_fieldnames(
            ("dataset", "physical_family", "source_ordinal", "source_index")
        ),
    )

    dataset_rows: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(plan.test_datasets):
        mask = query["dataset_index"] == dataset_index
        assigned = sum(
            assigned_time[(dataset_index, ordinal)] for ordinal in range(plan.source_count)
        )
        for method in METHODS:
            dataset_rows.append(
                _metric_row(
                    method,
                    query["labels"][mask],
                    predictions[method][mask],
                    scores[method][mask],
                    assigned_count=assigned,
                    dataset=dataset,
                    physical_family=plan.family_by_dataset[dataset],
                )
            )
    _atomic_csv(
        root / "per_dataset_metrics.csv",
        dataset_rows,
        _metrics_fieldnames(("dataset", "physical_family")),
    )

    family_rows: list[dict[str, Any]] = []
    for family in dict.fromkeys(plan.family_by_dataset[name] for name in plan.test_datasets):
        indices = [
            index
            for index, dataset in enumerate(plan.test_datasets)
            if plan.family_by_dataset[dataset] == family
        ]
        mask = np.isin(query["dataset_index"], indices)
        assigned = sum(
            assigned_time[(index, ordinal)]
            for index in indices
            for ordinal in range(plan.source_count)
        )
        for method in METHODS:
            family_rows.append(
                _metric_row(
                    method,
                    query["labels"][mask],
                    predictions[method][mask],
                    scores[method][mask],
                    assigned_count=assigned,
                    physical_family=family,
                )
            )
    _atomic_csv(
        root / "per_test_family_metrics.csv",
        family_rows,
        _metrics_fieldnames(("physical_family",)),
    )

    scale_rows: list[dict[str, Any]] = []
    for scale_id in range(len(plan.scale_table)):
        mask = query["scale_id"] == scale_id
        for method in METHODS:
            scale_rows.append(
                _metric_row(
                    method,
                    query["labels"][mask],
                    predictions[method][mask],
                    scores[method][mask],
                    assigned_count=int(assigned_scale[scale_id]),
                    scale_id=scale_id,
                    dx_grid_scale=float(plan.scale_table.dx_grid_scale[scale_id]),
                    ds_frame_scale=float(plan.scale_table.ds_frame_scale[scale_id]),
                    arc_length_grid_scale=float(
                        plan.scale_table.arc_length_grid_scale[scale_id]
                    ),
                )
            )
    _atomic_csv(
        root / "per_scale_tuple_metrics.csv",
        scale_rows,
        _metrics_fieldnames(
            ("scale_id", "dx_grid_scale", "ds_frame_scale", "arc_length_grid_scale")
        ),
    )
    if plan.experiment == EXPERIMENT31:
        block_rows: list[dict[str, Any]] = []
        for block_index, block in enumerate(plan.effective_scale_blocks):
            mask = (query["scale_id"] >= block.scale_id_start) & (
                query["scale_id"] < block.scale_id_stop
            )
            if "scale_block_index" in query and not np.all(
                query["scale_block_index"][mask] == block_index
            ):
                raise RuntimeError(
                    "query scale IDs disagree with explicit scale-block identity"
                )
            assigned = int(
                assigned_scale[block.scale_id_start : block.scale_id_stop].sum()
            )
            for method in METHODS:
                block_rows.append(
                    _metric_row(
                        method,
                        query["labels"][mask],
                        predictions[method][mask],
                        scores[method][mask],
                        assigned_count=assigned,
                        scale_block_index=block_index,
                        scale_block_id=block.block_id,
                        scale_id_start=block.scale_id_start,
                        scale_id_stop_exclusive=block.scale_id_stop,
                    )
                )
        _atomic_csv(
            root / "per_scale_block_metrics.csv",
            block_rows,
            _metrics_fieldnames(
                (
                    "scale_block_index",
                    "scale_block_id",
                    "scale_id_start",
                    "scale_id_stop_exclusive",
                )
            ),
        )

    pooled_rows = [
        _metric_row(
            method,
            query["labels"],
            predictions[method],
            scores[method],
            assigned_count=int(sum(assigned_time.values())),
            aggregation="pooled_all_valid_test_queries_descriptive",
        )
        for method in METHODS
    ]
    main_rows = _main_table_rows(plan, time_rows, pooled_rows)
    main_fields = (
        "aggregation",
        "method",
        "test_family_count",
        "sample_count",
        "coverage",
    ) + METRIC_NAMES
    _atomic_csv(root / "main_table.csv", main_rows, main_fields)
    _atomic_bytes(root / "main_table.md", _main_table_markdown(main_rows).encode("utf-8"))

    bootstrap_rows = _bootstrap_difference_rows(plan, time_rows, main_rows)
    _atomic_csv(
        root / "bootstrap_differences.csv",
        bootstrap_rows,
        (
            "metric",
            "method",
            "comparator",
            "point_estimate",
            "ci95_lower",
            "ci95_upper",
            "finite_replicate_count",
            "replicate_count",
            "seed",
            "paired_unit",
        ),
    )

    visualization_manifest = _build_phase21_visualization_artifacts(
        plan,
        root=root,
        test_rows=test_rows,
        query=query,
        fmt_prediction=predictions[METHOD_FMT],
        git_commit=git_commit,
        verify_cache_hashes=verify_cache_hashes,
    )

    report_lines = [
        f"# {plan.experiment} development report",
        "",
        "This run uses the previously exposed ten-flow development resource. ",
        "It is not a sealed confirmation experiment and supports descriptive conclusions only.",
        "",
        f"Git commit: `{git_commit}`  ",
        f"Config SHA-256: `{plan.config_sha256}`  ",
        f"Device: `{selected_device}`  ",
        f"Train flows: {', '.join(plan.train_datasets)}  ",
        f"Test flows: {', '.join(plan.test_datasets)}  ",
        f"Valid test queries: {len(query['labels'])} / {sum(assigned_time.values())}",
        "",
        "## Main descriptive table",
        "",
        _main_table_markdown(main_rows).rstrip(),
        "",
    ]
    if visualization_manifest is not None:
        if plan.experiment == EXPERIMENT31:
            triptych_description = (
                "The four triptychs use the pre-frozen test source ordinal 2, "
                "with one independent figure for each test-dataset and scale-block pair. "
                "They provide spatial context and TP/FP/FN/TN error decomposition; "
                "they are not aggregate performance evidence."
            )
            evidence_description = (
                "See `visualization_manifest.json` for scene, PNG, PDF, SVG, "
                "panel-alignment, and SHA-256 evidence."
            )
        else:
            triptych_description = (
                "The two triptychs use the pre-frozen test source ordinal 2. "
                "They provide spatial context and TP/FP/FN/TN error decomposition; "
                "they are not aggregate performance evidence."
            )
            evidence_description = (
                "See `visualization_manifest.json` for scene, PNG, PDF, panel-alignment, "
                "and SHA-256 evidence."
            )
        report_lines.extend(
            (
                "## Fixed explanatory triptychs",
                "",
                triptych_description,
                "",
                evidence_description,
                "",
            )
        )
    _atomic_bytes(
        root / "development_report.md", ("\n".join(report_lines) + "\n").encode("utf-8")
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if selected_device == "cuda"
            else None
        ),
        "device": selected_device,
        "deterministic_execution": execution_contract,
    }
    if plan.experiment == EXPERIMENT31:
        environment["maximum_source_frame_intervals"] = 48.0
    _atomic_json(root / "environment_versions.json", environment)
    _atomic_bytes(
        root / "evaluation_summary.log",
        (
            f"{plan.experiment} evaluation completed\n"
            f"valid_test_queries={len(query['labels'])}\n"
            f"device={selected_device}\n"
        ).encode("utf-8"),
    )

    before_final = {path.name for path in root.iterdir() if path.is_file()}
    missing = [
        name
        for name in plan.required_outputs
        if name not in {"result_manifest.json", "RUN_COMPLETE.json"}
        and name not in before_final
    ]
    if strict_protocol and missing:
        raise RuntimeError(f"required outputs are missing before finalization: {missing}")
    artifact_rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in {"result_manifest.json", "RUN_COMPLETE.json"}:
            continue
        artifact_rows.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    result_manifest: dict[str, Any] = {
        "schema": (
            f"pathline_template_matching.{plan.artifact_tag}_result_manifest.v1"
        ),
        "experiment": plan.experiment,
        "status": "development_completed_confirmation_not_run",
        "conclusion_scope": "descriptive_exposed_development_only",
        "formal_confirmation": False,
        "config_sha256": plan.config_sha256,
        "dataset_registry_sha256": plan.dataset_registry_sha256,
        "git_commit": git_commit,
        "portable_builder_git_commits": sorted(
            {str(row["portable_builder_git_commit"]) for row in cache_rows}
        ),
        "cache_builder_git_commits": sorted(
            {str(row["cache_builder_git_commit"]) for row in cache_rows}
        ),
        "device": selected_device,
        "deterministic_execution": execution_contract,
        "valid_test_query_count": int(len(query["labels"])),
        "assigned_test_query_count": int(sum(assigned_time.values())),
        "template_count": int(len(library["labels"])),
        "artifact_count": len(artifact_rows),
        "artifacts": artifact_rows,
        "artifacts_content_sha256": canonical_json_sha256(artifact_rows),
    }
    if plan.experiment == EXPERIMENT31:
        portable_population_passes = sorted(
            {
                (
                    str(row["portable_population_scope"]),
                    str(row["portable_population_pass_path"]),
                    int(row["portable_population_pass_file_size"]),
                    str(row["portable_population_pass_file_sha256"]),
                    str(row["portable_population_rows_content_sha256"]),
                )
                for row in cache_rows
                if "portable_population_scope" in row
            }
        )
        result_manifest.update(
            {
                "maximum_source_frame_intervals": float(
                    plan.maximum_source_frame_intervals
                ),
                "portable_window_frame_count": plan.window_frame_count,
                "scale_count": len(plan.scale_table),
                "assignment_count_per_seed": plan.assignment_count_per_seed,
                "unique_center_seed_count_per_source_time": plan.assigned_seed_count,
                "assigned_primitive_count_per_source_time": (
                    plan.assigned_primitive_count
                ),
                "train_preprocessing_rebuilt": True,
                "phase21_selected_library_appended": False,
                "verification_gates": verification_gate_evidence,
                "portable_population_passes": [
                    {
                        "access_scope": scope,
                        "path": path,
                        "file_size": size,
                        "file_sha256": file_sha,
                        "rows_content_sha256": rows_sha,
                    }
                    for scope, path, size, file_sha, rows_sha in portable_population_passes
                ],
            }
        )
    result_manifest["manifest_content_sha256"] = canonical_json_sha256(result_manifest)
    _atomic_json(root / "result_manifest.json", result_manifest)
    result_file_sha = sha256_file(root / "result_manifest.json")
    _atomic_json(
        root / "RUN_COMPLETE.json",
        {
            "schema": (
                f"pathline_template_matching.{plan.artifact_tag}_completion.v1"
            ),
            "experiment": plan.experiment,
            "status": "development_completed_confirmation_not_run",
            "result_manifest_file_sha256": result_file_sha,
            "result_manifest_content_sha256": result_manifest[
                "manifest_content_sha256"
            ],
        },
    )
    return {
        **result_manifest,
        "run_dir": str(root),
        "result_manifest_file_sha256": result_file_sha,
    }


def run_phase21_from_resolvers(
    plan: Phase21Plan,
    *,
    run_dir: str | Path,
    frame_count_resolver: Callable[[str], int],
    window_resolver: Callable[[str, int, int], Any],
    git_commit: str,
    device: str = "cpu",
    strict_protocol: bool = True,
    integration_chunk_size: int = 2048,
    encoding_chunk_size: int = 4096,
    verify_cache_hashes: bool = True,
    phase31_verification_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Small-diagnostic convenience path; production should parallelize shards."""

    if plan.experiment == EXPERIMENT31 and strict_protocol:
        raise ValueError(
            "strict mainExp 3.1 production must use the marker-gated parallel "
            "build-slice/evaluate CLI; resolver execution is diagnostic-only"
        )
    temporary_cache = Path(run_dir).resolve().with_name(
        f".{Path(run_dir).name}.cache.{os.getpid()}"
    )
    if temporary_cache.exists():
        raise FileExistsError(f"temporary cache root exists: {temporary_cache}")
    temporary_cache.mkdir(parents=True)
    summaries: list[CacheBuildSummary] = []
    try:
        for dataset in plan.datasets:
            source_indices = plan.source_indices(int(frame_count_resolver(dataset)))
            for ordinal, source_index in enumerate(source_indices):
                cache_path = (
                    temporary_cache
                    / plan.split_for(dataset)
                    / dataset
                    / f"source_{source_index:06d}.npz"
                )
                summaries.append(
                    build_phase21_cache_slice(
                        plan,
                        dataset=dataset,
                        source_ordinal=ordinal,
                        source_index=source_index,
                        resolved_input=window_resolver(
                            dataset, source_index, plan.window_frame_count
                        ),
                        cache_path=cache_path,
                        integration_chunk_size=integration_chunk_size,
                        encoding_chunk_size=encoding_chunk_size,
                        strict_evidence=strict_protocol,
                        cache_builder_git_commit=git_commit,
                    )
                )
        return evaluate_phase21_caches(
            plan,
            cache_summaries=summaries,
            run_dir=run_dir,
            git_commit=git_commit,
            device=device,
            strict_protocol=strict_protocol,
            verify_cache_hashes=verify_cache_hashes,
            phase31_verification_evidence=phase31_verification_evidence,
        )
    finally:
        # Keep successful and failed scientific caches in production.  This
        # convenience path is explicitly limited to non-production diagnostics;
        # remove only its process-specific temporary tree after verifying scope.
        if temporary_cache.exists() and not strict_protocol:
            import shutil

            shutil.rmtree(temporary_cache)

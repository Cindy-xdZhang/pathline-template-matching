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
class Phase21Plan:
    """Validated numerical plan derived from the frozen experiment YAML."""

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

    @property
    def datasets(self) -> tuple[str, ...]:
        return self.train_datasets + self.test_datasets

    @property
    def assigned_seed_count(self) -> int:
        return int(np.prod(self.seed_shape_xyz, dtype=np.int64))

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
        """Reject any silent drift from the committed 2.1 protocol."""

        if self.experiment != EXPERIMENT:
            raise ValueError(f"expected experiment {EXPERIMENT}, got {self.experiment}")
        if len(self.train_datasets) != 8 or len(self.test_datasets) != 2:
            raise ValueError("production split must contain exactly eight train and two test flows")
        train_families = {self.family_by_dataset[name] for name in self.train_datasets}
        test_families = {self.family_by_dataset[name] for name in self.test_datasets}
        if train_families.intersection(test_families):
            raise ValueError("a physical family appears in both train and test")
        if self.source_count != 4 or self.window_frame_count != 13:
            raise ValueError("production requires four source times and thirteen-frame windows")
        if self.seed_shape_xyz != (40, 40, 40) or self.assigned_seed_count != 64_000:
            raise ValueError("production seed grid must be exactly 40x40x40")
        if len(self.scale_table) != 1_000:
            raise ValueError("production scale table must contain exactly 1000 tuples")
        scale_counts = np.bincount(
            balanced_scale_assignment(
                self.assigned_seed_count, len(self.scale_table), self.assignment_seed
            ),
            minlength=len(self.scale_table),
        )
        if not np.all(scale_counts == 64):
            raise ValueError("production assignment must give every scale exactly 64 seeds")
        if self.assignment_seed != 15068 or self.library_seed != 15068:
            raise ValueError("production assignment and library seeds must both equal 15068")
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


def load_phase21_plan(config_path: str | Path) -> Phase21Plan:
    """Strictly parse and validate ``mainExp_TemplateMatching_2.1``."""

    path = Path(config_path).resolve()
    payload = path.read_bytes()
    parsed = yaml.safe_load(payload)
    config = _as_mapping(parsed, name="config")
    _require(config.get("experiment") == EXPERIMENT, "wrong experiment config")
    _require(
        config.get("phase") == "development_raw_reintegration",
        "wrong mainExp_TemplateMatching_2.1 phase",
    )
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
    _require(
        int(seed_grid.get("assigned_seed_count_per_source_time", -1))
        == int(np.prod(shape, dtype=np.int64)),
        "assigned seed count disagrees with seed grid shape",
    )
    _require(
        seed_grid.get("seed_index_order") == "z_outer_y_middle_x_inner",
        "unsupported seed-index order",
    )

    dx = _as_mapping(scale_protocol.get("dx_grid_scale"), name="dx_grid_scale").get("values")
    ds = _as_mapping(scale_protocol.get("ds_frame_scale"), name="ds_frame_scale").get("values")
    arc = _as_mapping(
        scale_protocol.get("arc_length_grid_scale"), name="arc_length_grid_scale"
    ).get("values")
    scales = build_arc_length_scale_table(dx, ds, arc)
    _require(
        int(scale_protocol.get("expected_unique_tuple_count", -1)) == len(scales),
        "scale tuple count disagrees with explicit config values",
    )
    _require(
        scale_protocol.get("cartesian_order") == "dx_outer_ds_middle_arc_inner",
        "scale Cartesian order drifted",
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
        assignment_seed=int(scale_assignment.get("seed", -1)),
        library_seed=int(library.get("sampling_random_seed", -1)),
        pca_components=pca_width,
        bootstrap_seed=int(bootstrap.get("seed", -1)),
        bootstrap_replicates=int(bootstrap.get("replicates", -1)),
        descriptor_config=descriptor_config,
        method_ids=method_ids,
        required_outputs=tuple(str(value) for value in config.get("required_outputs", ())),
    )
    plan.validate_production_contract()
    _require(
        int(source_times.get("minimum_required_frame_count", -1)) >= 16,
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
    return plan


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
    seeds = generate_phase21_seeds(vector_field, plan.seed_shape_xyz, maximum_dx)
    if len(seeds) != plan.assigned_seed_count:
        raise RuntimeError("seed generator changed the assigned population")
    assignment = balanced_scale_assignment(
        len(seeds), len(plan.scale_table), plan.assignment_seed
    )

    reference_labels, ivd_values, ivd_threshold, ivd_mask = ivd_p95_reference_at_seeds(
        window.velocity[0],
        window.spacing_xyz,
        window.coordinates_xyz,
        seeds,
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
    array_hashes = {
        name: canonical_array_sha256(values) for name, values in cache_arrays.items()
    }
    metadata = {
        "schema": "pathline_template_matching.phase21_cache.v2",
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
    return CacheBuildSummary(
        cache_row=cache_row,
        raw_input_row=raw_input_row,
        derived_window_row=derived_window_row,
        assignment_row=assignment_row,
        label_row=label_row,
        primitive_row=primitive_row,
        audit_rows=tuple(audit_rows),
    )


def _manifest_payload(kind: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    safe_rows = [_json_safe(dict(row)) for row in rows]
    return {
        "schema": f"pathline_template_matching.{kind}.v1",
        "experiment": EXPERIMENT,
        "row_count": len(safe_rows),
        "rows": safe_rows,
        "rows_content_sha256": canonical_json_sha256(safe_rows),
    }


def _scale_manifest(plan: Phase21Plan) -> dict[str, Any]:
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
        "cache_manifest.json": _manifest_payload("phase21_cache_manifest", cache_rows),
        "raw_input_manifest.json": _manifest_payload(
            "phase21_raw_input_manifest", list(raw_by_key.values())
        ),
        "derived_window_manifest.json": _manifest_payload(
            "phase21_derived_window_manifest",
            [summary.derived_window_row for summary in summaries],
        ),
        "seed_and_scale_assignment_manifest.json": _manifest_payload(
            "phase21_seed_assignment_manifest",
            [summary.assignment_row for summary in summaries],
        ),
        "label_manifest.json": _manifest_payload(
            "phase21_label_manifest", [summary.label_row for summary in summaries]
        ),
        "primitive_manifest.json": _manifest_payload(
            "phase21_primitive_manifest",
            [summary.primitive_row for summary in summaries],
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
    audit_sha = _atomic_csv(root / "audit_counts.csv", audit_rows, audit_fields)
    input_manifest = {
        "schema": "pathline_template_matching.phase21_input_manifest.v1",
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
    _atomic_json(root / "input_manifest.json", input_manifest)
    return cache_rows


def _load_cache(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f"cache file SHA-256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {
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
        if set(archive.files) != required:
            raise ValueError(
                f"cache keys disagree for {path}: missing={sorted(required-set(archive.files))}, "
                f"extra={sorted(set(archive.files)-required)}"
            )
        result = {name: np.asarray(archive[name]) for name in required - {"metadata_json"}}
        metadata_scalar = np.asarray(archive["metadata_json"])
        if metadata_scalar.ndim != 0:
            raise ValueError(f"cache metadata_json is not scalar: {path}")
        result["metadata"] = json.loads(str(metadata_scalar.item()))
    count = len(result["valid_labels"])
    if result["raw_features"].shape != (count, 672):
        raise ValueError(f"Raw cache feature shape changed: {path}")
    if result["fmt_features"].shape != (count, 161):
        raise ValueError(f"FMT cache feature shape changed: {path}")
    if result["valid_seed_index"].shape != (count,) or result["valid_scale_id"].shape != (count,):
        raise ValueError(f"valid-row metadata shape changed: {path}")
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
        "schema": "pathline_template_matching.phase21_cache.v2",
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
    assigned_count = plan.assigned_seed_count
    stored_arrays = {
        name: np.asarray(cache[name])
        for name in (
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
        )
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
    if not np.array_equal(
        assignment,
        balanced_scale_assignment(assigned_count, len(plan.scale_table), plan.assignment_seed),
    ):
        raise ValueError("existing cache scale assignment differs from frozen PCG64 assignment")
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
    if len(library["labels"]) < 2 or np.unique(library["labels"]).size != 2:
        raise ValueError("balanced library did not retain both classes")
    if len(library["labels"]) > 64_000 and plan.experiment == EXPERIMENT:
        raise ValueError("selected library exceeds the frozen maximum of 64000")
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

    payload: dict[str, Any] = {
        "schema": "pathline_template_matching.phase21_cache_sidecar.v1",
        "experiment": EXPERIMENT,
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
    if payload.get("schema") != "pathline_template_matching.phase21_cache_sidecar.v1":
        raise ValueError(f"unsupported cache sidecar schema: {source}")
    if payload.get("experiment") != EXPERIMENT:
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
        found[key] = summary
    missing = sorted(set(expected_keys) - set(found))
    extra = sorted(set(found) - set(expected_keys))
    if missing or extra:
        raise ValueError(f"cache sidecar population mismatch: missing={missing}, extra={extra}")
    return [found[key] for key in expected_keys]


def _freeze_cache_evidence(
    plan: Phase21Plan,
    root: Path,
    summaries: Sequence[CacheBuildSummary],
    *,
    evaluator_git_commit: str,
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
        "cache_manifest.json": _manifest_payload("phase21_cache_manifest", cache_rows),
        "raw_input_manifest.json": _manifest_payload(
            "phase21_raw_input_manifest", list(raw_by_key.values())
        ),
        "derived_window_manifest.json": _manifest_payload(
            "phase21_derived_window_manifest",
            [summary.derived_window_row for summary in summaries],
        ),
        "seed_and_scale_assignment_manifest.json": _manifest_payload(
            "phase21_seed_assignment_manifest",
            [summary.assignment_row for summary in summaries],
        ),
        "label_manifest.json": _manifest_payload(
            "phase21_label_manifest", [summary.label_row for summary in summaries]
        ),
        "primitive_manifest.json": _manifest_payload(
            "phase21_primitive_manifest",
            [summary.primitive_row for summary in summaries],
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
    audit_sha = _atomic_csv(
        root / "audit_counts.csv",
        (row for summary in summaries for row in summary.audit_rows),
        audit_fields,
    )
    input_manifest = {
        "schema": "pathline_template_matching.phase21_input_manifest.v1",
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
            "schema": "pathline_template_matching.phase21_run_state.v1",
            "experiment": plan.experiment,
            "status": "inputs_not_yet_frozen",
            "config_sha256": plan.config_sha256,
            "git_commit": git_commit,
            "started_unix_seconds": time.time(),
        },
    )
    return root


def _query_csv_header() -> tuple[str, ...]:
    common = (
        "query_dataset",
        "query_physical_family",
        "query_source_ordinal",
        "query_source_index",
        "query_seed_index",
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
                f"{stem}_match_seed_index",
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
    stems = {METHOD_RAW: "raw", METHOD_PCA: "pca", METHOD_FMT: "fmt"}
    for query_index in range(len(labels)):
        seed_id = int(valid_seed_index[query_index])
        scale_id = int(valid_scale_id[query_index])
        row: dict[str, Any] = {
            "query_dataset": str(cache_row["dataset"]),
            "query_physical_family": plan.family_by_dataset[str(cache_row["dataset"])],
            "query_source_ordinal": int(cache_row["source_ordinal"]),
            "query_source_index": int(cache_row["source_index"]),
            "query_seed_index": seed_id,
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
                    f"{stem}_match_seed_index": int(library["seed_index"][match_index]),
                    f"{stem}_match_scale_id": int(library["scale_id"][match_index]),
                    f"{stem}_match_label": bool(library["labels"][match_index]),
                }
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
    """Build the two frozen source-ordinal-2 triptychs after all metrics exist."""

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
        raise ValueError("phase-2.1 visualization source ordinal drifted")
    if visualization_config.get("metric_based_or_prediction_based_scene_selection") != "forbidden":
        raise ValueError("phase-2.1 visualization must forbid metric-selected scenes")

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
        mask = (query["dataset_index"] == dataset_index) & (
            query["source_ordinal"] == FIXED_SOURCE_ORDINAL
        )
        cache_seed_index = np.asarray(cache["valid_seed_index"], dtype=np.int64)
        cache_scale_id = np.asarray(cache["valid_scale_id"], dtype=np.int32)
        cache_labels = np.asarray(cache["valid_labels"], dtype=np.bool_)
        if not np.array_equal(query["valid_seed_index"][mask], cache_seed_index):
            raise RuntimeError("visualization query seed order differs from the cache")
        if not np.array_equal(query["scale_id"][mask], cache_scale_id):
            raise RuntimeError("visualization query scale order differs from the cache")
        if not np.array_equal(query["labels"][mask], cache_labels):
            raise RuntimeError("visualization query labels differ from the cache")
        prediction_contract = ordered_fmt_prediction(
            np.asarray(fmt_prediction[mask], dtype=np.bool_),
            cache_seed_index,
            cache_scale_id,
        )
        scene, scientific_audit = build_phase21_visualization_scene(
            cache, prediction_contract
        )

        scene_stem = root / "scenes" / f"{dataset}_source_ordinal_2"
        scene_path = scene_stem.with_suffix(".scene.npz")
        scene_manifest_path = scene_stem.with_suffix(".scene.json")
        scene_manifest = write_phase21_scene_artifact(
            scene, scientific_audit, scene_path, scene_manifest_path
        )
        render_stem = root / "figures" / f"{dataset}_source_ordinal_2_triptych"
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
                np.asarray(fmt_prediction[mask], dtype=np.bool_)
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
            "query_count": int(mask.sum()),
            "confusion_counts": dict(rendered.metadata["counts"]),
        }
        entries.append(entry)

    manifest: dict[str, Any] = {
        "schema": "pathline_template_matching.phase21_visualization_manifest.v1",
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
) -> dict[str, Any]:
    """Freeze cache evidence, fit train-only components, and evaluate test rows."""

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
    cache_rows = _freeze_cache_evidence(
        plan, root, cache_summaries, evaluator_git_commit=git_commit
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
        "schema": "pathline_template_matching.phase21_template_library.v1",
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
            "schema": "pathline_template_matching.phase21_preprocessing.v1",
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
    prediction_blocks = {method: [] for method in METHODS}
    score_blocks = {method: [] for method in METHODS}
    assigned_time: dict[tuple[int, int], int] = {}
    source_index_by_time: dict[tuple[int, int], int] = {}
    assigned_scale = np.zeros(len(plan.scale_table), dtype=np.int64)

    query_path = root / "per_query_matches.csv"
    query_temporary = query_path.with_name(f".{query_path.name}.{os.getpid()}.partial")
    if query_path.exists() or query_temporary.exists():
        raise FileExistsError("per-query evidence already exists")
    header = _query_csv_header()
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
        report_lines.extend(
            (
                "## Fixed explanatory triptychs",
                "",
                "The two triptychs use the pre-frozen test source ordinal 2. "
                "They provide spatial context and TP/FP/FN/TN error decomposition; "
                "they are not aggregate performance evidence.",
                "",
                "See `visualization_manifest.json` for scene, PNG, PDF, panel-alignment, "
                "and SHA-256 evidence.",
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
        "schema": "pathline_template_matching.phase21_result_manifest.v1",
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
    result_manifest["manifest_content_sha256"] = canonical_json_sha256(result_manifest)
    _atomic_json(root / "result_manifest.json", result_manifest)
    result_file_sha = sha256_file(root / "result_manifest.json")
    _atomic_json(
        root / "RUN_COMPLETE.json",
        {
            "schema": "pathline_template_matching.phase21_completion.v1",
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
) -> dict[str, Any]:
    """Small-diagnostic convenience path; production should parallelize shards."""

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
        )
    finally:
        # Keep successful and failed scientific caches in production.  This
        # convenience path is explicitly limited to non-production diagnostics;
        # remove only its process-specific temporary tree after verifying scope.
        if temporary_cache.exists() and not strict_protocol:
            import shutil

            shutil.rmtree(temporary_cache)

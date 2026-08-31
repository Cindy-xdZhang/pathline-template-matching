"""Frozen train-only Raw672 to PCA161 representation.

This module implements the numerical Principal Component Analysis (PCA)
contract frozen by ``Verify_RawPCANegativeMetric_1.1``.  It computes the full
right-singular-vector solution through a two-pass float64 scatter matrix and
``numpy.linalg.eigh``; it is not randomized, incremental, whitened, or fitted
from query rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from typing import Callable, Iterable, Mapping

import numpy as np


RAW_PCA_SCHEMA = "pathline_template_matching.raw_pca161.v1"
RAW_INPUT_WIDTH = 672
RAW_OUTPUT_WIDTH = 161
RAW_PCA_ROW_CHUNK_SIZE = 8192
RAW_PCA_SOLVER = "deterministic_two_pass_streaming_covariance_eigendecomposition"

RAW_PCA_ARRAY_NAMES = (
    "mean_float32",
    "components_float32",
    "singular_values_float64",
    "explained_variance_ratio_float64",
    "sample_count_int64",
    "input_width_int32",
    "output_width_int32",
)

_ARRAY_SPECS = {
    "mean_float32": (np.dtype(np.float32), (RAW_INPUT_WIDTH,)),
    "components_float32": (
        np.dtype(np.float32),
        (RAW_OUTPUT_WIDTH, RAW_INPUT_WIDTH),
    ),
    "singular_values_float64": (np.dtype(np.float64), (RAW_OUTPUT_WIDTH,)),
    "explained_variance_ratio_float64": (
        np.dtype(np.float64),
        (RAW_OUTPUT_WIDTH,),
    ),
    "sample_count_int64": (np.dtype(np.int64), ()),
    "input_width_int32": (np.dtype(np.int32), ()),
    "output_width_int32": (np.dtype(np.int32), ()),
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return _sha256_bytes(contiguous.tobytes(order="C"))


def _copy_exact_array(
    values: object,
    *,
    name: str,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != dtype or array.shape != shape:
        raise ValueError(
            f"{name} must have dtype {dtype} and shape {shape}, "
            f"got {array.dtype} and {array.shape}"
        )
    if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    # Back the public view by immutable ``bytes`` rather than merely clearing
    # NumPy's writeable flag on an owning allocation.  An owning array can have
    # that flag re-enabled by a caller after authentication; a bytes-backed
    # view cannot.
    immutable = np.frombuffer(
        np.ascontiguousarray(array).tobytes(order="C"), dtype=dtype
    ).reshape(shape)
    immutable.setflags(write=False)
    return immutable


def _validate_sha256(value: object, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


@dataclass(frozen=True)
class RawPCAArrayAudit:
    """Authenticated identity of one serialized PCA array."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str

    def __post_init__(self) -> None:
        if self.name not in _ARRAY_SPECS:
            raise ValueError(f"unknown Raw-PCA array audit name: {self.name}")
        expected_dtype, expected_shape = _ARRAY_SPECS[self.name]
        if self.dtype != str(expected_dtype) or tuple(self.shape) != expected_shape:
            raise ValueError(f"invalid dtype or shape audit for {self.name}")
        _validate_sha256(self.sha256, name=f"{self.name} SHA-256")


@dataclass(frozen=True)
class SerializedRawPCA:
    """In-memory NPZ plus hashes that must be bound by an external manifest."""

    payload: bytes
    npz_sha256: str
    array_audits: tuple[RawPCAArrayAudit, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes):
            raise ValueError("serialized Raw-PCA payload must be immutable bytes")
        _validate_sha256(self.npz_sha256, name="Raw-PCA NPZ SHA-256")
        if _sha256_bytes(self.payload) != self.npz_sha256:
            raise ValueError("serialized Raw-PCA payload does not match its SHA-256")
        if not all(
            isinstance(audit, RawPCAArrayAudit) for audit in self.array_audits
        ):
            raise ValueError("Raw-PCA array audits have an invalid type")
        if tuple(audit.name for audit in self.array_audits) != RAW_PCA_ARRAY_NAMES:
            raise ValueError("Raw-PCA array audits are incomplete or out of order")


@dataclass(frozen=True)
class RawPCARepresentation:
    """Immutable fitted Raw672 PCA161 transform without whitening."""

    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    sample_count: int

    def __post_init__(self) -> None:
        mean = _copy_exact_array(
            self.mean,
            name="mean",
            dtype=np.dtype(np.float32),
            shape=(RAW_INPUT_WIDTH,),
        )
        components = _copy_exact_array(
            self.components,
            name="components",
            dtype=np.dtype(np.float32),
            shape=(RAW_OUTPUT_WIDTH, RAW_INPUT_WIDTH),
        )
        singular_values = _copy_exact_array(
            self.singular_values,
            name="singular_values",
            dtype=np.dtype(np.float64),
            shape=(RAW_OUTPUT_WIDTH,),
        )
        explained = _copy_exact_array(
            self.explained_variance_ratio,
            name="explained_variance_ratio",
            dtype=np.dtype(np.float64),
            shape=(RAW_OUTPUT_WIDTH,),
        )
        if isinstance(self.sample_count, (bool, np.bool_)) or not isinstance(
            self.sample_count, (int, np.integer)
        ):
            raise ValueError("Raw-PCA sample_count must be an integer")
        count = int(self.sample_count)
        if not 2 <= count <= np.iinfo(np.int64).max:
            raise ValueError("Raw-PCA sample_count must be at least 2")
        if np.any(singular_values < 0.0) or np.any(np.diff(singular_values) > 0.0):
            raise ValueError("Raw-PCA singular values must be nonnegative and descending")
        if np.any(explained < 0.0) or np.any(np.diff(explained) > 0.0):
            raise ValueError(
                "Raw-PCA explained variance ratios must be nonnegative and descending"
            )
        if float(explained.sum(dtype=np.float64)) > 1.0 + 1.0e-12:
            raise ValueError("Raw-PCA explained variance ratios sum above one")
        pivots = np.argmax(np.abs(components), axis=1)
        pivot_values = components[np.arange(RAW_OUTPUT_WIDTH), pivots]
        if np.any(pivot_values < 0.0):
            raise ValueError("Raw-PCA component signs do not follow the frozen pivot rule")
        gram = components.astype(np.float64) @ components.astype(np.float64).T
        if not np.allclose(
            gram,
            np.eye(RAW_OUTPUT_WIDTH, dtype=np.float64),
            rtol=0.0,
            atol=2.0e-6,
        ):
            raise ValueError("Raw-PCA components are not orthonormal")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "singular_values", singular_values)
        object.__setattr__(self, "explained_variance_ratio", explained)
        object.__setattr__(self, "sample_count", count)

    @property
    def input_width(self) -> int:
        return RAW_INPUT_WIDTH

    @property
    def output_width(self) -> int:
        return RAW_OUTPUT_WIDTH

    @property
    def whitening(self) -> bool:
        return False

    def transform(self, raw_features: np.ndarray) -> np.ndarray:
        """Project query rows without changing either the model or the input."""

        values = np.asarray(raw_features)
        if values.dtype != np.dtype(np.float32):
            raise ValueError(
                f"Raw-PCA input dtype must be float32, got {values.dtype}"
            )
        if values.ndim != 2 or values.shape[1] != RAW_INPUT_WIDTH:
            raise ValueError(
                f"Raw-PCA input must be [N,{RAW_INPUT_WIDTH}], got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("Raw-PCA input contains NaN or Inf")
        return np.ascontiguousarray(
            (values - self.mean) @ self.components.T,
            dtype=np.float32,
        )

    def export_arrays(self) -> dict[str, np.ndarray]:
        """Return independent arrays matching the frozen final-PCA NPZ schema."""

        return {
            "mean_float32": np.array(self.mean, copy=True, order="C"),
            "components_float32": np.array(self.components, copy=True, order="C"),
            "singular_values_float64": np.array(
                self.singular_values, copy=True, order="C"
            ),
            "explained_variance_ratio_float64": np.array(
                self.explained_variance_ratio, copy=True, order="C"
            ),
            "sample_count_int64": np.asarray(self.sample_count, dtype=np.int64),
            "input_width_int32": np.asarray(RAW_INPUT_WIDTH, dtype=np.int32),
            "output_width_int32": np.asarray(RAW_OUTPUT_WIDTH, dtype=np.int32),
        }

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, object]) -> "RawPCARepresentation":
        """Rebuild a model only from exact, finite, canonical schema arrays."""

        if tuple(arrays) != RAW_PCA_ARRAY_NAMES:
            raise ValueError("Raw-PCA arrays are incomplete, extra, or out of order")
        validated: dict[str, np.ndarray] = {}
        for name in RAW_PCA_ARRAY_NAMES:
            dtype, shape = _ARRAY_SPECS[name]
            validated[name] = _copy_exact_array(
                arrays[name], name=name, dtype=dtype, shape=shape
            )
        if int(validated["input_width_int32"]) != RAW_INPUT_WIDTH:
            raise ValueError("serialized Raw-PCA input width changed")
        if int(validated["output_width_int32"]) != RAW_OUTPUT_WIDTH:
            raise ValueError("serialized Raw-PCA output width changed")
        return cls(
            mean=validated["mean_float32"],
            components=validated["components_float32"],
            singular_values=validated["singular_values_float64"],
            explained_variance_ratio=validated[
                "explained_variance_ratio_float64"
            ],
            sample_count=int(validated["sample_count_int64"]),
        )


def _validated_fit_block(raw_block: object) -> np.ndarray:
    block = np.asarray(raw_block)
    if block.ndim != 2 or block.shape[1] != RAW_INPUT_WIDTH:
        raise ValueError(
            f"Raw-PCA fit block must be [N,{RAW_INPUT_WIDTH}], got {block.shape}"
        )
    if block.dtype != np.dtype(np.float32):
        raise ValueError("Raw-PCA fit block must have dtype float32")
    if not np.isfinite(block).all():
        raise ValueError("Raw-PCA fit block contains NaN or Inf")
    return block


def fit_raw_pca(
    train_block_factory: Callable[[], Iterable[np.ndarray]],
) -> RawPCARepresentation:
    """Fit the frozen PCA in two deterministic passes over train-only blocks.

    The callable must return the same ordered fit population on both calls.
    Query or held-out blocks must never be supplied to this function.
    """

    if not callable(train_block_factory):
        raise ValueError("train_block_factory must be callable")
    feature_sum = np.zeros(RAW_INPUT_WIDTH, dtype=np.float64)
    sample_count = 0
    for raw_block in train_block_factory():
        block = _validated_fit_block(raw_block)
        feature_sum += block.astype(np.float64, copy=True).sum(
            axis=0, dtype=np.float64
        )
        sample_count += len(block)
    if sample_count < 2:
        raise ValueError(f"Raw-PCA requires at least 2 train rows, got {sample_count}")
    if sample_count > np.iinfo(np.int64).max:
        raise ValueError("Raw-PCA sample count exceeds int64")

    mean64 = feature_sum / sample_count
    scatter = np.zeros((RAW_INPUT_WIDTH, RAW_INPUT_WIDTH), dtype=np.float64)
    second_pass_count = 0
    for raw_block in train_block_factory():
        block = _validated_fit_block(raw_block)
        for start in range(0, len(block), RAW_PCA_ROW_CHUNK_SIZE):
            centered = (
                block[start : start + RAW_PCA_ROW_CHUNK_SIZE].astype(
                    np.float64, copy=True
                )
                - mean64
            )
            scatter += centered.T @ centered
        second_pass_count += len(block)
    if second_pass_count != sample_count:
        raise ValueError("Raw-PCA block factory changed row count between passes")

    scatter = 0.5 * (scatter + scatter.T)
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    order = np.argsort(-eigenvalues, kind="stable")
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    tolerance = max(
        1.0e-12,
        float(max(0.0, eigenvalues[0])) * 1.0e-10,
    )
    if float(eigenvalues[-1]) < -tolerance:
        raise ValueError(
            "Raw-PCA scatter has a materially negative eigenvalue: "
            f"{eigenvalues[-1]}"
        )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    selected = np.ascontiguousarray(
        eigenvectors[:, :RAW_OUTPUT_WIDTH].T,
        dtype=np.float64,
    )
    pivots = np.argmax(np.abs(selected), axis=1)
    signs = np.sign(selected[np.arange(RAW_OUTPUT_WIDTH), pivots])
    signs[signs == 0.0] = 1.0
    selected *= signs[:, None]
    total_variance = float(eigenvalues.sum(dtype=np.float64))
    explained = (
        eigenvalues[:RAW_OUTPUT_WIDTH] / total_variance
        if total_variance > 0.0
        else np.zeros(RAW_OUTPUT_WIDTH, dtype=np.float64)
    )
    return RawPCARepresentation(
        mean=np.asarray(feature_sum / sample_count, dtype=np.float32),
        components=np.asarray(selected, dtype=np.float32),
        singular_values=np.sqrt(eigenvalues[:RAW_OUTPUT_WIDTH]).astype(np.float64),
        explained_variance_ratio=np.asarray(explained, dtype=np.float64),
        sample_count=sample_count,
    )


def serialize_raw_pca(model: RawPCARepresentation) -> SerializedRawPCA:
    """Serialize the exact seven-array NPZ and return its authentication data."""

    if not isinstance(model, RawPCARepresentation):
        raise ValueError("serialize_raw_pca requires RawPCARepresentation")
    arrays = model.export_arrays()
    audits = tuple(
        RawPCAArrayAudit(
            name=name,
            dtype=str(arrays[name].dtype),
            shape=tuple(arrays[name].shape),
            sha256=_array_sha256(arrays[name]),
        )
        for name in RAW_PCA_ARRAY_NAMES
    )
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    return SerializedRawPCA(
        payload=payload,
        npz_sha256=_sha256_bytes(payload),
        array_audits=audits,
    )


def deserialize_raw_pca(
    payload: bytes,
    *,
    expected_npz_sha256: str,
    expected_array_audits: tuple[RawPCAArrayAudit, ...],
) -> RawPCARepresentation:
    """Authenticate an NPZ before rebuilding an immutable PCA transform."""

    if not isinstance(payload, bytes):
        raise ValueError("serialized Raw-PCA payload must be immutable bytes")
    expected_file_hash = _validate_sha256(
        expected_npz_sha256, name="expected Raw-PCA NPZ SHA-256"
    )
    if _sha256_bytes(payload) != expected_file_hash:
        raise ValueError("Raw-PCA NPZ SHA-256 mismatch")
    audits = tuple(expected_array_audits)
    if not all(isinstance(audit, RawPCAArrayAudit) for audit in audits):
        raise ValueError("expected Raw-PCA array audits have an invalid type")
    if tuple(audit.name for audit in audits) != RAW_PCA_ARRAY_NAMES:
        raise ValueError("expected Raw-PCA array audits are incomplete or out of order")
    expected_by_name = {audit.name: audit for audit in audits}
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            if tuple(archive.files) != RAW_PCA_ARRAY_NAMES:
                raise ValueError("Raw-PCA NPZ member set or order changed")
            arrays = {
                name: np.array(archive[name], copy=True, order="C")
                for name in RAW_PCA_ARRAY_NAMES
            }
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("Raw-PCA NPZ cannot be decoded") from error
    for name, array in arrays.items():
        audit = expected_by_name[name]
        if (
            str(array.dtype) != audit.dtype
            or tuple(array.shape) != audit.shape
            or _array_sha256(array) != audit.sha256
        ):
            raise ValueError(f"Raw-PCA array authentication failed: {name}")
    return RawPCARepresentation.from_arrays(arrays)

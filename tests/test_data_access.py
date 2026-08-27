import json
from pathlib import Path
import tempfile

import numpy as np

from pathline_template_matching.data_access import _validate_cache_root


def _write_slice(path: Path, ordinal: int, *, complete: bool = True) -> None:
    metadata = {
        "experiment": "mainExp_Task5_3D_1.1",
        "dataset": "synthetic",
        "phase": "development",
        "ordinal": ordinal,
        "config_sha256": "a" * 64,
        "scale_table": [{"name": "only"}],
    }
    arrays = {
        "raw_features": np.zeros((2, 672), dtype=np.float32),
        "fmt_features": np.zeros((2, 161), dtype=np.float32),
        "reference": np.asarray([False, True]),
        "seeds": np.zeros((2, 3), dtype=np.float32),
        "scale_id": np.zeros(2, dtype=np.int16),
        "metadata_json": np.asarray(json.dumps(metadata)),
    }
    if not complete:
        arrays.pop("fmt_features")
    np.savez_compressed(path / f"slice_{ordinal:02d}.npz", **arrays)


def test_cache_validator_checks_every_slice_and_full_contract():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_slice(root, 0)
        _write_slice(root, 1)
        report = _validate_cache_root(
            root,
            2,
            expected_dataset="synthetic",
            expected_phase="development",
            expected_config_sha256="a" * 64,
        )
        assert report["valid"]
        assert report["checked_slices"] == 2
        assert report["sample_count_total"] == 4


def test_cache_validator_detects_damage_after_first_slice():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_slice(root, 0)
        _write_slice(root, 1, complete=False)
        report = _validate_cache_root(
            root,
            2,
            expected_dataset="synthetic",
            expected_phase="development",
            expected_config_sha256="a" * 64,
        )
        assert not report["valid"]
        assert any("slice_01" in error and "missing keys" in error for error in report["errors"])


def test_cache_validator_rejects_consistent_but_noncanonical_config_hash():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_slice(root, 0)
        report = _validate_cache_root(
            root,
            1,
            expected_dataset="synthetic",
            expected_phase="development",
            expected_config_sha256="b" * 64,
        )
        assert not report["valid"]
        assert any("does not equal canonical" in error for error in report["errors"])

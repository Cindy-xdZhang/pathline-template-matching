from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from pathline_template_matching.portable_flow import canonical_json_sha256


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mainExp_TemplateMatching_2.1"
CONFIG_SHA = "1" * 64
REGISTRY_SHA = "2" * 64
BUILDER_COMMIT = "3" * 40


def _expect_error(error_types, message: str, function) -> None:
    try:
        function()
    except error_types as error:
        assert message in str(error)
    else:
        raise AssertionError(f"expected {error_types} containing {message!r}")


def _load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(directory: Path) -> dict:
    source_file = str((directory / "source.nc").resolve())
    windows = []
    for ordinal in range(4):
        name = f"window_{ordinal}.npz"
        (directory / name).write_bytes(b"x")
        windows.append(
            {
                "relative_path": name,
                "file_size": 1,
                "file_sha256": "4" * 64,
                "dataset": "flow_a",
                "physical_family": "family_a",
                "split": "train",
                "experiment": EXPERIMENT,
                "config_sha256": CONFIG_SHA,
                "dataset_registry_sha256": REGISTRY_SHA,
                "builder_git_commit": BUILDER_COMMIT,
                "source_kind": "netcdf",
                "source_file": source_file,
                "source_file_size": 17,
                "source_file_sha256": "5" * 64,
                "source_total_frames": 16,
                "source_ordinal": ordinal,
                "source_start_index": ordinal,
                "frame_count": 13,
            }
        )
    payload = {
        "schema": "pathline_template_matching.portable_flow_dataset_manifest.v1",
        "experiment": EXPERIMENT,
        "config_sha256": CONFIG_SHA,
        "dataset_registry_sha256": REGISTRY_SHA,
        "builder_git_commit": BUILDER_COMMIT,
        "dataset": "flow_a",
        "physical_family": "family_a",
        "split": "train",
        "source_kind": "netcdf",
        "source_file": source_file,
        "source_file_size": 17,
        "source_file_sha256": "5" * 64,
        "source_total_frames": 16,
        "selected_source_indices": [0, 1, 2, 3],
        "window_count": 4,
        "windows": windows,
    }
    payload["manifest_content_sha256"] = canonical_json_sha256(payload)
    return payload


def _write_manifest(path: Path, payload: dict) -> None:
    digest_payload = dict(payload)
    digest_payload.pop("manifest_content_sha256", None)
    payload["manifest_content_sha256"] = canonical_json_sha256(digest_payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_index(module, manifest_path: Path):
    return module.PortableManifestIndex(
        manifest_path,
        expected_experiment=EXPERIMENT,
        expected_config_sha256=CONFIG_SHA,
        expected_dataset_registry_sha256=REGISTRY_SHA,
        expected_datasets=("flow_a",),
        expected_family_by_dataset={"flow_a": "family_a"},
        expected_split_by_dataset={"flow_a": "train"},
        expected_builder_git_commit=BUILDER_COMMIT,
        expected_source_kind_by_dataset={"flow_a": "netcdf"},
    )


def test_production_manifest_requires_v1_self_hash_file_hash_and_exact_provenance():
    module = _load_script(
        "phase21_runner_manifest_test",
        "scripts/run_mainexp_template_matching_2_1.py",
    )
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        manifest_path = directory / "manifest.json"
        base = _manifest(directory)
        _write_manifest(manifest_path, base)
        index = _build_index(module, manifest_path)
        assert index.native_frame_count("flow_a") == 16
        assert index.resolve_path(index.dataset_rows("flow_a")[0]) == (
            directory / "window_0.npz"
        ).resolve()

        for field in (
            "source_total_frames",
            "source_file",
            "source_file_size",
            "source_file_sha256",
            "source_kind",
            "dataset",
            "physical_family",
            "split",
            "experiment",
            "config_sha256",
            "dataset_registry_sha256",
            "builder_git_commit",
        ):
            drifted = _manifest(directory)
            drifted["windows"][0][field] = "drift"
            _write_manifest(manifest_path, drifted)
            _expect_error(
                (TypeError, ValueError),
                "provenance mismatch",
                lambda: _build_index(module, manifest_path),
            )

        missing_hash = _manifest(directory)
        missing_hash["windows"][0].pop("file_sha256")
        _write_manifest(manifest_path, missing_hash)
        _expect_error(
            ValueError,
            "file SHA-256",
            lambda: _build_index(module, manifest_path),
        )

        bad_self_hash = _manifest(directory)
        _write_manifest(manifest_path, bad_self_hash)
        bad_self_hash["manifest_content_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(bad_self_hash), encoding="utf-8")
        _expect_error(
            ValueError,
            "content SHA-256 mismatch",
            lambda: _build_index(module, manifest_path),
        )

        manifest_path.write_text(json.dumps({"rows": []}), encoding="utf-8")
        _expect_error(
            ValueError,
            "production accepts only",
            lambda: _build_index(module, manifest_path),
        )


def test_runner_clean_gate_treats_untracked_files_as_dirty():
    module = _load_script(
        "phase21_runner_clean_test",
        "scripts/run_mainexp_template_matching_2_1.py",
    )
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=BUILDER_COMMIT + "\n")
        if command[:3] == ["git", "status", "--porcelain"]:
            return SimpleNamespace(stdout="?? untracked.txt\n")
        return SimpleNamespace(stdout="")

    original = module.subprocess.run
    module.subprocess.run = fake_run
    try:
        _expect_error(
            RuntimeError,
            "tracked or untracked",
            module._git_commit_and_clean,
        )
    finally:
        module.subprocess.run = original
    assert ["git", "status", "--porcelain"] in calls


def test_staging_source_identity_rejects_mutation_during_hash():
    module = _load_script(
        "phase21_staging_toctou_test",
        "scripts/stage_mainexp_template_matching_2_1_windows.py",
    )
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "source.nc"
        source.write_bytes(b"before")

        def mutating_hash(path):
            Path(path).write_bytes(b"after-and-different-size")
            return "6" * 64

        original = module.sha256_file
        module.sha256_file = mutating_hash
        try:
            _expect_error(
                RuntimeError,
                "changed while",
                lambda: module._stable_source_identity(source),
            )
        finally:
            module.sha256_file = original


def test_ibex_staging_uses_strict_resume_mode():
    script = (ROOT / "ibex/mainexp_template_matching_2.1_stage_windows.sh").read_text(
        encoding="utf-8"
    )
    assert "git status --porcelain" in script
    assert "--resume" in script

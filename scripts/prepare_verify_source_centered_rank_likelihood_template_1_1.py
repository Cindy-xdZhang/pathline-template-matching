#!/usr/bin/env python3
"""Build or freshly authenticate the immutable parent-sidecar binding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT / "src", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts import (  # noqa: E402
    run_verify_source_centered_rank_likelihood_template_1_1 as runner,
)


_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _identity(expected_commit: str) -> str:
    runner._require(_HEX40.fullmatch(expected_commit) is not None, "expected commit must be 40-hex")
    commit, dirty = runner._git_identity()
    runner._require(not dirty and commit == expected_commit, "clean checkout commit differs from expected")
    return commit


def build(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    expected_git_commit: str,
) -> dict[str, Any]:
    commit = _identity(expected_git_commit)
    plan = runner.load_plan(config_path)
    destination = Path(output_dir).resolve()
    runner._require(not destination.exists(), f"immutable binding directory exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    replayed = runner.authenticate_historical_sidecar_population(plan)
    binding = runner._manifest(
        {
            "schema": runner.PARENT_BINDING_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": "passed",
            "created_utc": runner._utc_now(),
            "git_commit": commit,
            "config_sha256": plan.sha256,
            "historical_source_centered_evidence": runner._json_safe(
                replayed.source_evidence
            ),
            "sidecar_npz_members_opened": [],
            "labels_or_references_opened": [],
        }
    )
    binding_path = destination / "parent_sidecar_binding.json"
    binding_sha = runner.source_runner.early._atomic_json(binding_path, binding)
    completion = runner._manifest(
        {
            "schema": runner.BINDING_COMPLETE_SCHEMA,
            "experiment": runner.EXPERIMENT,
            "status": "passed",
            "completed_utc": runner._utc_now(),
            "git_commit": commit,
            "config_sha256": plan.sha256,
            "parent_binding_file": binding_path.name,
            "parent_binding_file_sha256": binding_sha,
            "parent_binding_content_sha256": binding["content_sha256"],
        }
    )
    completion_path = destination / "BINDING_COMPLETE.json"
    completion_sha = runner.source_runner.early._atomic_json(
        completion_path, completion
    )
    runner._require(
        {path.name for path in destination.iterdir()}
        == {binding_path.name, completion_path.name},
        "binding file set drifted",
    )
    return {
        "status": "passed",
        "directory": str(destination),
        "parent_binding": str(binding_path),
        "parent_binding_file_sha256": binding_sha,
        "binding_completion": str(completion_path),
        "binding_completion_file_sha256": completion_sha,
        "config_sha256": plan.sha256,
        "git_commit": commit,
    }


def authenticate(
    config_path: str | Path,
    binding_path: str | Path,
    binding_sha256: str,
    completion_path: str | Path,
    completion_sha256: str,
    *,
    expected_git_commit: str,
) -> dict[str, Any]:
    commit = _identity(expected_git_commit)
    runner._require(
        _HEX64.fullmatch(binding_sha256) is not None
        and _HEX64.fullmatch(completion_sha256) is not None,
        "binding hashes must be lowercase SHA-256",
    )
    plan = runner.bind_parent_sidecar_release(
        runner.load_plan(config_path),
        parent_binding_path=binding_path,
        parent_binding_file_sha256=binding_sha256,
        binding_completion_path=completion_path,
        binding_completion_file_sha256=completion_sha256,
    )
    return {
        "status": "authenticated",
        "parent_binding": str(plan.parent_binding_path),
        "parent_binding_file_sha256": plan.parent_binding_file_sha256,
        "binding_completion": str(plan.binding_completion_path),
        "binding_completion_file_sha256": plan.binding_completion_file_sha256,
        "config_sha256": plan.sha256,
        "git_commit": commit,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            ROOT
            / "config"
            / "Verify_SourceCenteredRankLikelihoodTemplate_1.1.yaml"
        ),
    )
    parser.add_argument("--expected-git-commit", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--output-dir", required=True)
    auth = subparsers.add_parser("authenticate")
    auth.add_argument("--parent-binding", required=True)
    auth.add_argument("--parent-binding-sha256", required=True)
    auth.add_argument("--binding-completion", required=True)
    auth.add_argument("--binding-completion-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    if arguments.command == "build":
        result = build(
            arguments.config,
            arguments.output_dir,
            expected_git_commit=arguments.expected_git_commit,
        )
    else:
        result = authenticate(
            arguments.config,
            arguments.parent_binding,
            arguments.parent_binding_sha256,
            arguments.binding_completion,
            arguments.binding_completion_sha256,
            expected_git_commit=arguments.expected_git_commit,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

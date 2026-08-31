#!/usr/bin/env python3
"""Immutable preparation stages for Verify_EarlyOppositePairKinematics_1.1.

The synthetic stage executes the production oracle implementation itself.  No
caller-provided PASS booleans or caller-authored oracle evidence are accepted.
Every command captures the exact clean Git/source identity before work and
recaptures it before returning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching.early_kinematic_preparation import (  # noqa: E402
    EXPERIMENT,
    PRODUCTION_CONTRACT,
    authenticate_kinematic_input_manifest,
    authenticate_row_completion,
    authenticate_sidecar_population_manifest,
    authenticate_synthetic_pass_marker,
    build_kinematic_input_manifest,
    build_one_sidecar_and_completion,
    capture_clean_source_identity,
    sidecar_row_relative_directory,
    write_sidecar_population_manifest,
    write_synthetic_pass_marker,
)
from pathline_template_matching.portable_flow import sha256_file  # noqa: E402


_LOWER_HEX_40 = re.compile(r"[0-9a-f]{40}\Z")
_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_confirmation_tokens(value: object, *, name: str) -> None:
    text = "".join(character for character in str(value).lower() if character.isalnum())
    if "tangaroa" in text or "smokebuoyancy" in text:
        raise ValueError(f"{name} contains a forbidden confirmation dataset token")


def _sha256(value: str, *, name: str) -> str:
    _require(_LOWER_HEX_64.fullmatch(value) is not None, f"{name} must be lowercase SHA-256")
    return value


def _row_key(row_index: int) -> tuple[str, int]:
    contract = PRODUCTION_CONTRACT
    _require(
        isinstance(row_index, int)
        and not isinstance(row_index, bool)
        and 0 <= row_index < contract.row_count,
        f"row index must be in 0..{contract.row_count - 1}",
    )
    dataset_index, source_ordinal = divmod(row_index, contract.source_count)
    return str(contract.datasets[dataset_index]), int(source_ordinal)


def _capture_identity(project_root: Path, expected_git_commit: str):
    _require(
        _LOWER_HEX_40.fullmatch(expected_git_commit) is not None,
        "expected Git commit must be lowercase 40-hex",
    )
    identity = capture_clean_source_identity(project_root)
    _require(
        identity.git_commit == expected_git_commit,
        "clean checkout commit differs from --expected-git-commit",
    )
    return identity


def _manifest_row(manifest: Mapping[str, Any], row_index: int) -> Mapping[str, Any]:
    dataset, source_ordinal = _row_key(row_index)
    rows = manifest.get("rows")
    _require(isinstance(rows, tuple), "authenticated input manifest rows are not immutable")
    matches = [
        row
        for row in rows
        if row.get("dataset") == dataset
        and int(row.get("source_ordinal", -1)) == source_ordinal
    ]
    _require(len(matches) == 1, "profile row does not resolve uniquely")
    return matches[0]


def _summary(
    value: Mapping[str, Any],
    *,
    path: Path,
    identity: Any,
    stage: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "stage": stage,
        "status": str(value.get("status", "authenticated")),
        "schema": str(value.get("schema", "")),
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "file_sha256": sha256_file(path),
        "content_sha256": str(value.get("content_sha256", "")),
        "git_commit": identity.git_commit,
        "source_content_sha256": identity.source_content_sha256,
    }
    for name in ("row_count", "sidecar_count", "sidecar_row_count_total"):
        if name in value:
            result[name] = int(value[name])
    for name in ("dataset", "source_ordinal", "source_index"):
        if name in value:
            result[name] = value[name]
    return result


def _recapture(project_root: Path, identity: Any) -> None:
    observed = capture_clean_source_identity(project_root)
    _require(observed == identity, "clean commit or frozen source hashes changed during stage")


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(arguments.project_root).resolve()
    for name, value in vars(arguments).items():
        if isinstance(value, (str, Path)):
            _reject_confirmation_tokens(value, name=name)
    identity = _capture_identity(project_root, arguments.expected_git_commit)
    contract = PRODUCTION_CONTRACT

    if arguments.command == "synthetic":
        run_dir = Path(arguments.run_dir).resolve()
        value = write_synthetic_pass_marker(
            run_dir,
            identity=identity,
            contract=contract,
        )
        output = run_dir / "SYNTHETIC_PASS.json"
        stage = "synthetic_production_oracles"
    elif arguments.command == "freeze-input":
        output = Path(arguments.output_path).resolve()
        value = build_kinematic_input_manifest(
            output,
            parent_input_manifest_path=arguments.parent_input_manifest,
            train_portable_marker_path=arguments.train_portable_marker,
            synthetic_pass_path=arguments.synthetic_pass,
            synthetic_pass_file_sha256=_sha256(
                arguments.synthetic_pass_sha256,
                name="synthetic PASS SHA-256",
            ),
            identity=identity,
            contract=contract,
        )
        stage = "freeze_exact_32_input_rows"
    elif arguments.command == "build-sidecar":
        dataset, source_ordinal = _row_key(arguments.row_index)
        value = build_one_sidecar_and_completion(
            arguments.sidecar_root,
            dataset=dataset,
            source_ordinal=source_ordinal,
            input_manifest_path=arguments.input_manifest,
            input_manifest_file_sha256=_sha256(
                arguments.input_manifest_sha256,
                name="input manifest SHA-256",
            ),
            synthetic_pass_path=arguments.synthetic_pass,
            synthetic_pass_file_sha256=_sha256(
                arguments.synthetic_pass_sha256,
                name="synthetic PASS SHA-256",
            ),
            identity=identity,
            contract=contract,
        )
        input_manifest = authenticate_kinematic_input_manifest(
            arguments.input_manifest,
            expected_file_sha256=arguments.input_manifest_sha256,
            identity=identity,
            contract=contract,
            authenticate_all_referenced_rows=False,
        )
        row = _manifest_row(input_manifest, arguments.row_index)
        output = (
            Path(arguments.sidecar_root).resolve()
            / sidecar_row_relative_directory(
                dataset,
                source_ordinal,
                int(row["source_index"]),
            )
            / "SIDECAR_COMPLETE.json"
        )
        stage = "build_one_sidecar"
    elif arguments.command == "authenticate-profile":
        authenticate_synthetic_pass_marker(
            arguments.synthetic_pass,
            expected_file_sha256=_sha256(
                arguments.synthetic_pass_sha256,
                name="synthetic PASS SHA-256",
            ),
            identity=identity,
            contract=contract,
        )
        input_manifest = authenticate_kinematic_input_manifest(
            arguments.input_manifest,
            expected_file_sha256=_sha256(
                arguments.input_manifest_sha256,
                name="input manifest SHA-256",
            ),
            identity=identity,
            contract=contract,
            authenticate_all_referenced_rows=False,
        )
        row = _manifest_row(input_manifest, arguments.row_index)
        dataset, source_ordinal = _row_key(arguments.row_index)
        output = (
            Path(arguments.sidecar_root).resolve()
            / sidecar_row_relative_directory(
                dataset,
                source_ordinal,
                int(row["source_index"]),
            )
            / "SIDECAR_COMPLETE.json"
        )
        value = authenticate_row_completion(
            output,
            sidecar_root=arguments.sidecar_root,
            expected_file_sha256=_sha256(
                arguments.completion_sha256,
                name="profile completion SHA-256",
            ),
            input_manifest=input_manifest,
            input_manifest_file_sha256=arguments.input_manifest_sha256,
            synthetic_pass_file_sha256=arguments.synthetic_pass_sha256,
            identity=identity,
            contract=contract,
            authenticate_sidecar=True,
        )
        stage = "authenticate_single_row_resource_profile"
    elif arguments.command == "seal-population":
        value = write_sidecar_population_manifest(
            arguments.sidecar_root,
            input_manifest_path=arguments.input_manifest,
            input_manifest_file_sha256=_sha256(
                arguments.input_manifest_sha256,
                name="input manifest SHA-256",
            ),
            synthetic_pass_path=arguments.synthetic_pass,
            synthetic_pass_file_sha256=_sha256(
                arguments.synthetic_pass_sha256,
                name="synthetic PASS SHA-256",
            ),
            identity=identity,
            contract=contract,
        )
        output = Path(arguments.sidecar_root).resolve() / "SIDECAR_POPULATION.json"
        stage = "authenticate_and_seal_exact_32_sidecars"
    elif arguments.command == "authenticate-population":
        output = Path(arguments.population_manifest).resolve()
        value = authenticate_sidecar_population_manifest(
            output,
            sidecar_root=arguments.sidecar_root,
            expected_file_sha256=_sha256(
                arguments.population_manifest_sha256,
                name="population manifest SHA-256",
            ),
            input_manifest_path=arguments.input_manifest,
            input_manifest_file_sha256=_sha256(
                arguments.input_manifest_sha256,
                name="input manifest SHA-256",
            ),
            synthetic_pass_file_sha256=_sha256(
                arguments.synthetic_pass_sha256,
                name="synthetic PASS SHA-256",
            ),
            identity=identity,
            contract=contract,
        )
        stage = "fresh_authenticate_exact_32_sidecar_population"
    else:  # pragma: no cover - argparse makes this unreachable.
        raise RuntimeError(f"unsupported command: {arguments.command}")

    result = _summary(value, path=output, identity=identity, stage=stage)
    _recapture(project_root, identity)
    return result


def _evidence_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--synthetic-pass", required=True)
    parser.add_argument("--synthetic-pass-sha256", required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--expected-git-commit", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic")
    synthetic.add_argument("--run-dir", required=True)

    freeze = subparsers.add_parser("freeze-input")
    freeze.add_argument("--output-path", required=True)
    freeze.add_argument("--parent-input-manifest", required=True)
    freeze.add_argument("--train-portable-marker", required=True)
    freeze.add_argument("--synthetic-pass", required=True)
    freeze.add_argument("--synthetic-pass-sha256", required=True)

    sidecar = subparsers.add_parser("build-sidecar")
    sidecar.add_argument("--sidecar-root", required=True)
    sidecar.add_argument("--row-index", type=int, required=True)
    _evidence_arguments(sidecar)

    profile = subparsers.add_parser("authenticate-profile")
    profile.add_argument("--sidecar-root", required=True)
    profile.add_argument("--row-index", type=int, required=True)
    profile.add_argument("--completion-sha256", required=True)
    _evidence_arguments(profile)

    population = subparsers.add_parser("seal-population")
    population.add_argument("--sidecar-root", required=True)
    _evidence_arguments(population)

    authenticate = subparsers.add_parser("authenticate-population")
    authenticate.add_argument("--sidecar-root", required=True)
    authenticate.add_argument("--population-manifest", required=True)
    authenticate.add_argument("--population-manifest-sha256", required=True)
    _evidence_arguments(authenticate)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(_run(_parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

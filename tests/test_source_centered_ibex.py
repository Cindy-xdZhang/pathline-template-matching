from __future__ import annotations

import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IBEX = ROOT / "ibex"
PREFIX = "verify_source_centered_paired_scale_template_1.1_"
COMMON = IBEX / f"{PREFIX}common.sh"
STAGES = {
    "prepare": IBEX / f"{PREFIX}prepare.sh",
    "sidecar_profile": IBEX / f"{PREFIX}sidecar_profile.sh",
    "sidecars": IBEX / f"{PREFIX}sidecars.sh",
    "population": IBEX / f"{PREFIX}population.sh",
    "first_fold": IBEX / f"{PREFIX}first_fold.sh",
    "first_fold_auth": IBEX / f"{PREFIX}first_fold_auth.sh",
    "all_folds": IBEX / f"{PREFIX}all_folds.sh",
    "aggregate_five": IBEX / f"{PREFIX}aggregate_five.sh",
}
CONFIG_SHA256 = "15ac5b0e82b30cbaf952475a7fbb6d19dc070c1121bc9aa8db980d75600260cc"


def _read(path: Path) -> str:
    payload = path.read_bytes()
    assert payload.startswith(b"#!/bin/bash\n")
    assert b"\r\n" not in payload
    return payload.decode("utf-8")


def _directive(text: str, name: str) -> str:
    match = re.search(rf"^#SBATCH --{re.escape(name)}=(.+)$", text, re.MULTILINE)
    assert match is not None, name
    return match.group(1).strip()


def test_exact_ibex_wrapper_population_and_shared_identity_gate() -> None:
    expected = {COMMON, *STAGES.values()}
    actual = set(IBEX.glob(f"{PREFIX}*.sh"))
    assert actual == expected
    common = _read(COMMON)
    assert hashlib.sha256(
        (ROOT / "config" / "Verify_SourceCenteredPairedScaleTemplate_1.1.yaml").read_bytes()
    ).hexdigest() == CONFIG_SHA256
    assert f"SOURCE_CENTERED_CONFIG_SHA256={CONFIG_SHA256}" in common
    assert (
        "SOURCE_CENTERED_REMOTE_URL=git@github.com:Cindy-xdZhang/"
        "pathline-template-matching.git"
    ) in common
    for required in (
        "git status --porcelain=v1 --untracked-files=all",
        "git rev-parse --verify HEAD^{commit}",
        "git remote get-url origin",
        "git show-ref --verify --quiet refs/remotes/origin/main",
        'git merge-base --is-ancestor "$expected_commit" refs/remotes/origin/main',
        'git ls-files --error-unmatch "$source"',
        'git show "${expected_commit}:${source}"',
        "ptm_require_file_sha256",
        "tests/test_source_centered_aggregate.py",
        "tests/test_source_centered_runner_contract.py",
        "tests/test_source_centered_ibex.py",
    ):
        assert required in common
    assert "source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh" in common
    assert "conda activate deepvortex" in common
    assert "unset PYTHONOPTIMIZE" in common


def test_every_slurm_stage_is_cpu_rome_and_uses_fail_closed_shared_gates() -> None:
    for name, path in STAGES.items():
        text = _read(path)
        assert _directive(text, "nodes") == "1"
        assert _directive(text, "account") == "pi-hadwigm"
        assert _directive(text, "partition") == "batch"
        assert _directive(text, "constraint") == "rome"
        assert _directive(text, "chdir") == (
            "/home/zhanx0o/pathline-template-matching-source-centered"
        )
        assert "--gres" not in text
        assert "--gpus" not in text
        assert "cuda" not in text.lower()
        assert "set -euo pipefail" in text
        assert (
            "source ibex/verify_source_centered_paired_scale_template_1.1_common.sh"
            in text
        )
        assert 'ptm_stage_gate "$WRAPPER"' in text or "ptm_stage_gate \\\n" in text
        assert 'ptm_stage_unchanged "$WRAPPER"' in text
        assert "--device cpu" in text or name in {
            "prepare",
            "sidecar_profile",
            "sidecars",
            "population",
        }


def test_preparation_profile_array_and_population_form_a_closed_release_chain() -> None:
    prepare = _read(STAGES["prepare"])
    assert "freeze-input" in prepare
    assert "--early-input-manifest \"$SOURCE_CENTERED_EARLY_INPUT_MANIFEST\"" in prepare
    assert "authenticate-input" in prepare
    assert "--authenticate-all-rows" in prepare
    assert "source_centered_input_manifest.json" in prepare
    assert "ptm_full_preflight" in prepare

    profile = _read(STAGES["sidecar_profile"])
    assert "profile_row_index=0" in profile
    assert "build-sidecar" in profile
    assert "--row-index 0" in profile
    assert "authenticate-row" in profile
    assert "source_centered_seed_time_kinematics.npz" in profile

    sidecars = _read(STAGES["sidecars"])
    assert _directive(sidecars, "array") == "0-31%2"
    assert "sidecar row index must be 0..31" in sidecars
    assert "profile completion is not the authenticated row-0 artifact" in sidecars
    assert sidecars.count("authenticate-row") == 2
    assert sidecars.count("build-sidecar") == 1
    assert '--row-index "$ROW_INDEX"' in sidecars
    assert "ROW_COMPLETION_SHA256" in sidecars

    population = _read(STAGES["population"])
    assert "SOURCE_CENTERED_SIDECAR_ARRAY_JOB_ID" in population
    assert "seal-population" in population
    assert "authenticate-population" in population
    assert "SIDECAR_POPULATION.json" in population
    assert population.index("seal-population") < population.index(
        "authenticate-population"
    )


def test_first_fold_authentication_remaining_folds_and_five_fold_aggregate_are_bound() -> None:
    first = _read(STAGES["first_fold"])
    assert "ptm_full_preflight" in first
    assert "outer-family half_cylinder" in first
    assert "--device cpu" in first
    for argument in (
        "--sidecar-input-manifest",
        "--sidecar-input-manifest-sha256",
        "--sidecar-root",
        "--sidecar-population-manifest",
        "--sidecar-population-manifest-sha256",
    ):
        assert argument in first
    assert "RUN_COMPLETE.json" in first

    # Complete population authentication opens/recomputes all 32 NPZ files.
    # A fold may only hash-check the manifest envelope before the runner has
    # sealed the nonouter model and candidate; the runner owns the outer-open
    # gate after that point.
    for name in ("first_fold", "first_fold_auth", "all_folds", "aggregate_five"):
        stage = _read(STAGES[name])
        assert "authenticate-population" not in stage
        assert 'python "$SOURCE_CENTERED_PREPARER"' not in stage

    authentication = _read(STAGES["first_fold_auth"])
    assert 'python "$SOURCE_CENTERED_AGGREGATOR"' in authentication
    assert authentication.count("--run-dir") == 1
    assert "single_fold_authentication_report.json" in authentication
    assert 'report["outer_families"] == ["half_cylinder"]' in authentication
    assert 'isinstance(report["stop_version"], bool)' in authentication
    assert 'completion["source_centered_evidence"]' in authentication
    assert 'sidecars["sidecar_count"] == 32' in authentication
    for argument in (
        "--sidecar-input-manifest",
        "--sidecar-input-manifest-sha256",
        "--sidecar-root",
        "--sidecar-population-manifest",
        "--sidecar-population-manifest-sha256",
    ):
        assert argument in authentication

    remaining = _read(STAGES["all_folds"])
    assert _directive(remaining, "array") == "1-4%2"
    assert "FIRST_AUTH_COMPLETE_SHA256" in remaining
    assert 'report["stop_version"] is False' in remaining
    assert 'manifest["source_folds"]' in remaining
    assert 'completion["source_centered_evidence"]' in remaining
    assert 'python "$SOURCE_CENTERED_RUNNER"' in remaining
    assert '--outer-family "$OUTER_FAMILY"' in remaining

    aggregate = _read(STAGES["aggregate_five"])
    assert "ptm_full_preflight" in aggregate
    assert aggregate.count("--run-dir") == 5
    assert 'python "$SOURCE_CENTERED_AGGREGATOR"' in aggregate
    assert "aggregate_summary.json" in aggregate
    assert 'report["paired_bootstrap"]["replicates"] == 5000' in aggregate
    assert 'report["paired_bootstrap"]["seed"] == 17068' in aggregate
    assert 'len(manifest["source_folds"]) == 5' in aggregate
    assert 'completion["source_centered_evidence"]' in aggregate
    for argument in (
        "--sidecar-input-manifest",
        "--sidecar-input-manifest-sha256",
        "--sidecar-root",
        "--sidecar-population-manifest",
        "--sidecar-population-manifest-sha256",
    ):
        assert argument in aggregate


def test_resource_envelopes_cover_the_1800_candidate_cpu_workload() -> None:
    assert _directive(_read(STAGES["sidecars"]), "cpus-per-task") == "8"
    assert _directive(_read(STAGES["sidecars"]), "mem") == "96G"
    assert _directive(_read(STAGES["population"]), "cpus-per-task") == "16"
    for name in ("first_fold", "first_fold_auth", "all_folds", "aggregate_five"):
        text = _read(STAGES[name])
        assert _directive(text, "cpus-per-task") == "32"
        assert _directive(text, "mem") == "128G"
        hours = int(_directive(text, "time").split(":", 1)[0])
        assert hours >= 12


def test_wrappers_preserve_immutable_outputs_and_contain_no_destructive_git_or_file_action() -> None:
    forbidden = (
        "rm -rf",
        "git reset",
        "git checkout",
        "git clean",
        "git pull",
        "git fetch",
    )
    for name, path in STAGES.items():
        text = _read(path)
        assert not any(value in text for value in forbidden), name
        if name in {
            "prepare",
            "sidecar_profile",
            "population",
            "first_fold",
            "first_fold_auth",
            "all_folds",
            "aggregate_five",
        }:
            assert "[[ ! -e" in text

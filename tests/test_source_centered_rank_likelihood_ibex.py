from __future__ import annotations

import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IBEX = ROOT / "ibex"
PREFIX = "verify_source_centered_rank_likelihood_template_1.1_"
COMMON = IBEX / f"{PREFIX}common.sh"
STAGES = {
    "prepare": IBEX / f"{PREFIX}prepare.sh",
    "first_fold": IBEX / f"{PREFIX}first_fold.sh",
    "first_fold_auth": IBEX / f"{PREFIX}first_fold_auth.sh",
    "all_folds": IBEX / f"{PREFIX}all_folds.sh",
    "aggregate_five": IBEX / f"{PREFIX}aggregate_five.sh",
}
CONFIG_SHA256 = (
    "41d6e7be70b898715c6df6f92cfb17176d2f1bb6153fa37b09dd4da9a6059ffa"
)


def _read(path: Path) -> str:
    payload = path.read_bytes()
    assert payload.startswith(b"#!/bin/bash\n")
    assert b"\r\n" not in payload
    return payload.decode("utf-8")


def _directive(text: str, name: str) -> str:
    match = re.search(rf"^#SBATCH --{re.escape(name)}=(.+)$", text, re.MULTILINE)
    assert match is not None, name
    return match.group(1).strip()


def test_exact_rank_likelihood_wrapper_set_and_shared_commit_gate() -> None:
    assert set(IBEX.glob(f"{PREFIX}*.sh")) == {COMMON, *STAGES.values()}
    common = _read(COMMON)
    config = ROOT / "config" / "Verify_SourceCenteredRankLikelihoodTemplate_1.1.yaml"
    assert hashlib.sha256(config.read_bytes()).hexdigest() == CONFIG_SHA256
    assert f"RANK_CONFIG_SHA256={CONFIG_SHA256}" in common
    assert (
        "RANK_PROJECT_ROOT:-/home/zhanx0o/"
        "pathline-template-matching-rank-likelihood"
    ) in common
    assert (
        "RANK_REMOTE_URL=git@github.com:Cindy-xdZhang/"
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
        "rank_require_file_sha256",
        "source_centered_rank_likelihood.py",
        "tests/test_source_centered_rank_likelihood_runner.py",
        "tests/test_source_centered_rank_likelihood_aggregate.py",
        "tests/test_source_centered_rank_likelihood_ibex.py",
    ):
        assert required in common
    assert "source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh" in common
    assert "conda activate deepvortex" in common
    assert "unset PYTHONOPTIMIZE" in common


def test_all_stages_are_cpu_rome_and_use_fail_closed_shared_gates() -> None:
    for name, path in STAGES.items():
        text = _read(path)
        assert _directive(text, "nodes") == "1"
        assert _directive(text, "account") == "pi-hadwigm"
        assert _directive(text, "partition") == "batch"
        assert _directive(text, "constraint") == "rome"
        assert _directive(text, "chdir") == (
            "/home/zhanx0o/pathline-template-matching-rank-likelihood"
        )
        assert "--gres" not in text
        assert "--gpus" not in text
        assert "cuda" not in text.lower()
        assert "set -euo pipefail" in text
        assert (
            "source ibex/verify_source_centered_rank_likelihood_template_1.1_common.sh"
            in text
        )
        assert 'rank_stage_gate "$WRAPPER"' in text
        assert 'rank_stage_unchanged "$WRAPPER"' in text
        assert "rank_targeted_preflight" in text
        assert "--device cpu" in text or name in {"prepare", "first_fold_auth", "aggregate_five"}


def test_prepare_only_binds_existing_parent_sidecars_without_rebuild() -> None:
    prepare = _read(STAGES["prepare"])
    assert 'python "$RANK_PREPARER"' in prepare
    assert "  build \\\n" in prepare
    assert "parent_sidecar_binding.json" in prepare
    assert "BINDING_COMPLETE.json" in prepare
    assert 'binding["sidecar_npz_members_opened"] == []' in prepare
    assert 'binding["labels_or_references_opened"] == []' in prepare
    assert "rank_authenticate_parent_binding" in prepare
    for forbidden in (
        "build-sidecar",
        "seal-population",
        "authenticate-population",
        "SIDECAR_POPULATION.json",
    ):
        assert forbidden not in prepare


def test_first_auth_remaining_and_five_fold_release_are_bound() -> None:
    first = _read(STAGES["first_fold"])
    assert "rank_full_preflight" in first
    assert "--outer-family half_cylinder" in first
    assert "--device cpu" in first
    assert "set(runner.REQUIRED_FOLD_FILES)" in first
    assert "first_fold_exact_18_file_completion=passed" in first
    for argument in (
        "--parent-binding",
        "--parent-binding-sha256",
        "--binding-completion",
        "--binding-completion-sha256",
    ):
        assert argument in first

    authentication = _read(STAGES["first_fold_auth"])
    assert 'python "$RANK_AGGREGATOR"' in authentication
    assert authentication.count("--run-dir") == 1
    assert "single_fold_authentication_report.json" in authentication
    assert 'report["outer_families"] == ["half_cylinder"]' in authentication
    assert 'isinstance(report["stop_version"], bool)' in authentication
    assert 'aggregate.sha256_file(Path(expected_fold) / "RUN_COMPLETE.json")' in authentication
    assert 'aggregate.sha256_file(Path(expected_fold) / "result_manifest.json")' in authentication
    assert 'evidence["parent_binding_file_sha256"] == binding_sha' in authentication

    remaining = _read(STAGES["all_folds"])
    assert _directive(remaining, "array") == "1-4%2"
    assert "RANK_FIRST_AUTH_COMPLETE_SHA256" in remaining
    assert 'report["stop_version"] is False' in remaining
    assert 'manifest["source_folds"]' in remaining
    assert 'python "$RANK_RUNNER"' in remaining
    assert '--outer-family "$OUTER_FAMILY"' in remaining
    assert "remaining_fold_exact_18_file_completion=passed" in remaining

    aggregate = _read(STAGES["aggregate_five"])
    assert "rank_full_preflight" in aggregate
    assert aggregate.count("--run-dir") == 5
    assert 'python "$RANK_AGGREGATOR"' in aggregate
    assert "aggregate_summary.json" in aggregate
    assert 'report["paired_bootstrap"]["replicates"] == 5000' in aggregate
    assert 'report["paired_bootstrap"]["seed"] == 17068' in aggregate
    assert 'len(manifest["source_folds"]) == 5' in aggregate
    assert 'fold / "RUN_COMPLETE.json"' in aggregate
    assert 'fold / "result_manifest.json"' in aggregate
    assert 'report["controls_can_satisfy_primary_success"] is False' in aggregate


def test_resource_envelopes_cover_rank_likelihood_cpu_workload() -> None:
    prepare = _read(STAGES["prepare"])
    assert _directive(prepare, "cpus-per-task") == "16"
    assert _directive(prepare, "mem") == "128G"
    for name in ("first_fold", "first_fold_auth", "all_folds", "aggregate_five"):
        text = _read(STAGES[name])
        assert _directive(text, "cpus-per-task") == "32"
        assert _directive(text, "mem") == "128G"
        assert int(_directive(text, "time").split(":", 1)[0]) >= 12


def test_wrappers_keep_immutable_outputs_and_have_no_destructive_action() -> None:
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
        assert "[[ ! -e" in text

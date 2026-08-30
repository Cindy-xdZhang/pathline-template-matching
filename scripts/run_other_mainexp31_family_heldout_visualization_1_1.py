#!/usr/bin/env python3
"""Run the frozen family-held-out 3.1 classification visualization experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching.family_heldout_visualization import (  # noqa: E402
    EXPERIMENT,
    load_other_visualization_plan,
    run_family_heldout_visualization,
)


DEFAULT_CONFIG = ROOT / "config/Other_MainExp31FamilyHeldOutVisualization_1.1.yaml"


def _git_commit_and_clean() -> str:
    critical_paths = (
        "config/Other_MainExp31FamilyHeldOutVisualization_1.1.yaml",
        "config/mainExp_TemplateMatching_3.1.yaml",
        "scripts/run_other_mainexp31_family_heldout_visualization_1_1.py",
        "src/pathline_template_matching/family_heldout_visualization.py",
        "src/pathline_template_matching/phase21_pipeline.py",
        "src/pathline_template_matching/phase21_visualization.py",
        "src/pathline_template_matching/visualization.py",
        "src/pathline_template_matching/matcher.py",
    )
    for relative in critical_paths:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(
            "Ibex experiment requires a clean committed worktree; "
            f"uncommitted entries:\n{dirty}"
        )
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    plan = load_other_visualization_plan(args.config)
    cache_root = (
        args.cache_root
        if args.cache_root is not None
        else Path(str(plan.config["parent"]["cache_root"]))
    )
    run_dir = args.run_dir.resolve()
    expected_parent = (Path(str(plan.output_root)) / "runs").resolve()
    if run_dir.parent != expected_parent:
        raise ValueError(
            f"run directory must be a direct child of {expected_parent}, got {run_dir}"
        )
    commit = _git_commit_and_clean()
    result = run_family_heldout_visualization(
        plan,
        cache_root=cache_root,
        run_dir=run_dir,
        git_commit=commit,
        device=args.device,
    )
    print(json.dumps({"experiment": EXPERIMENT, **result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

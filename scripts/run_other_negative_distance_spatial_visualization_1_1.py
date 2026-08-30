#!/usr/bin/env python3
"""Render the frozen negative-distance spatial candidate on four 3D flows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathline_template_matching.negative_distance_spatial_visualization import (  # noqa: E402
    EXPERIMENT,
    load_visualization_plan,
    run_negative_distance_spatial_visualization,
)


DEFAULT_CONFIG = ROOT / "config/Other_NegativeDistanceSpatialVisualization_1.1.yaml"


def _git_commit_and_clean() -> str:
    critical_paths = (
        "config/Other_NegativeDistanceSpatialVisualization_1.1.yaml",
        "scripts/run_other_negative_distance_spatial_visualization_1_1.py",
        "src/pathline_template_matching/negative_distance_spatial_visualization.py",
        "src/pathline_template_matching/phase21_visualization.py",
        "src/pathline_template_matching/visualization.py",
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
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
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
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = load_visualization_plan(args.config)
    run_dir = args.run_dir.resolve()
    expected_parent = (plan.output_root / "runs").resolve()
    if run_dir.parent != expected_parent:
        raise ValueError(
            f"run directory must be a direct child of {expected_parent}, got {run_dir}"
        )
    commit = _git_commit_and_clean()
    result = run_negative_distance_spatial_visualization(
        plan, run_dir=run_dir, git_commit=commit
    )
    print(json.dumps({"experiment": EXPERIMENT, **result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

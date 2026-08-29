#!/usr/bin/env python3
"""Build parallel cache shards or evaluate mainExp_TemplateMatching_3.1."""

from __future__ import annotations

from pathlib import Path

from run_mainexp_template_matching_2_1 import ROOT, main


if __name__ == "__main__":
    main(
        default_config=ROOT / "config/mainExp_TemplateMatching_3.1.yaml",
        expected_experiment="mainExp_TemplateMatching_3.1",
    )

#!/usr/bin/env python3
"""Build immutable 49-frame portable windows for mainExp_TemplateMatching_3.1."""

from __future__ import annotations

from stage_mainexp_template_matching_2_1_windows import ROOT, main


if __name__ == "__main__":
    main(
        default_config=ROOT / "config/mainExp_TemplateMatching_3.1.yaml",
        expected_experiment="mainExp_TemplateMatching_3.1",
    )

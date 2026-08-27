# Pathline Template Matching project instructions

## Before changing research code

1. Read `docs/research_tasks_and_protocol.md`, `docs/experiment_log.md`, `docs/ibex_run_registry.md`, and the active experiment document/config.
2. Treat `docs/research_tasks_and_protocol.md` as the unique task definition.
3. Check `git status`; preserve unrelated user changes and never reset or clean them.
4. Do not modify the copied FMT baseline silently. Any descriptor change requires a new experiment version, config, tests, and provenance update.

## Experiment discipline

- Core experiments use `mainExp_[name]_x.y`; component checks use `Verify_[name]_x.y`; explorations use `Other_[name]_x.y`; ablations use `Ablation_[name]_x.y`.
- Freeze the complete method and confirmation manifest before first reading confirmation raw fields, features, valid rates, labels, or metrics.
- Never split random spatial seeds across library/query. Split by physical family and account for the complete pathline source window.
- Fit normalization, distance weights, thresholds, template sampling, and model selection only on library/development data.
- Preserve failed, cancelled, invalid, timed-out, negative, and superseded results. Never overwrite an old experiment output directory.
- Every conclusion must identify experiment version, config, Git commit, data/cache manifest, per-run results, summary, and device.
- When a conclusion changes, record the previous conclusion, current conclusion, reason, and the previous error or narrower scope.

## Ibex-first execution

- Except for unit tests, smoke tests, and genuinely small diagnostics, run experiments on Ibex from a committed Git revision.
- Sync through `git push` and an Ibex clone/pull; Slurm jobs must not depend on uncommitted local files.
- Register every Slurm process in `docs/ibex_run_registry.md` immediately after submission and update the same row with node/GPU, result, logs, and output.
- Use `glogin.ibex.kaust.edu.sa`; there is no local `ibex` SSH alias.
- Keep raw-flow availability distinct from derived-cache availability. Use `config/datasets.yaml` as the machine-path registry.

## Fixed terminology

- FMT is the inherited project/descriptor name; do not expand it because the source project uses more than one historical full name.
- IVD means Instantaneous Vorticity Deviation: `||curl(v)-spatial_mean(curl(v))||`.
- The default 3D primitive order is `center, x+, x-, y+, y-, z+, z-`.
- Do not call the old Task5 268D recipe an independent template descriptor; its 44D kinematic block depends on batch composition.

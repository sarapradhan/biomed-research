# Environment Baseline

Frozen snapshot of the project state that the reproducibility analyses
were developed against. Any environment change is compared against this
baseline so result differences can be attributed to code, not drift.

## Git

See `git_head.txt` for the full record.

- Commit: `24140c9a6f26026f2eafd9cc6eb347cc3b1c7f30`
- Title:  `Add DeepWiki badge and documentation link to README`
- Author: Sara Pradhan <pradhan.sara@gmail.com>
- Date:   2026-04-18 23:27:08 -0700

## Declared Environment (from the repo)

- `environment.yml.snapshot` — conda environment file as of the baseline commit
- `requirements.txt.snapshot` — pinned pip requirements as of the baseline commit
- `pyproject.toml.snapshot` — package metadata and build config as of the baseline commit

Reconstruct the baseline environment with:

```bash
conda env create -f docs/baseline/environment.yml.snapshot
conda activate fmri-unified-pipeline
```

The pinned pip layer is already pointed at `requirements.txt` via the `-r` directive in `environment.yml`.

## Captured Environment (from the machine running the analyses)

Before running a reproducibility analysis on a given machine, capture the
concrete installed versions with:

```bash
pip freeze > docs/baseline/pip_freeze.<host>.txt
conda list --explicit > docs/baseline/conda_explicit.<host>.txt
python --version > docs/baseline/python_version.<host>.txt
```

Commit those files alongside any reproducibility results so the exact
versions that produced each figure/number are recoverable.

## Why this baseline matters

The reproducibility analyses (FC, ReHo, ICA, graph) must be runnable on
the same inputs and environment so that any change in outputs is
attributable to code changes, not dependency drift. If you bump a
dependency version, re-snapshot the environment and record the change in
`docs/baseline/CHANGES.md`.

## Data inputs

The data inventory that describes which subjects, runs, and tasks the
baseline analyses used lives in `data_inventory_template.csv` (blank
template) and `data_inventory.<dataset>.csv` (filled per dataset). See
also `scripts/data_inventory.py`.

#!/usr/bin/env python3
"""Run the ICA component stability analysis (JEI revision Step 1.3).

Loads ICA decompositions saved as ``{tag}_ica_components.npy`` from
two sub-directories under ``paths.ica_input_dir``:

    {ica_input_dir}/seeds/    - one .npy per random_state seed
    {ica_input_dir}/lorocv/   - one .npy per leave-one-run-out subset

Either sub-sweep may be absent; only the present one(s) will be
reported. For each sweep the script writes a summary CSV, per-component
mean/std spatial r, the full pairwise table, and a JSON snapshot under
``paths.output_root``.

Example:

    python scripts/run_ica_stability.py --config config/reproducibility.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fmri_pipeline.reproducibility import ica_stability  # noqa: E402


def _summary_for_print(result) -> dict:
    summary = asdict(result)
    # Drop large arrays from stdout; they're persisted in the JSON / CSV.
    summary.pop("per_component_mean_r", None)
    summary.pop("per_component_std_r", None)
    summary.pop("pairwise_records", None)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to reproducibility YAML config (e.g. config/reproducibility.yaml).",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Echo each sweep's compact summary JSON to stdout.",
    )
    args = parser.parse_args()

    results = ica_stability.run(args.config)

    for sweep, res in results.items():
        print(
            f"ICA stability ({sweep}): K={res.k_components}, "
            f"n_runs={res.n_runs}, mean |r|={res.mean_matched_correlation:.3f} "
            f"(SD {res.std_matched_correlation:.3f}), "
            f"n_robust(>{res.robust_threshold:.2f})={res.n_robust_components}"
        )
        if args.print_summary:
            print(json.dumps(_summary_for_print(res), indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())

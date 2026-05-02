#!/usr/bin/env python3
"""Run the ReHo run-to-run stability analysis (JEI revision Step 1.2).

Reads per-subject, per-run ReHo ROI vectors saved as
``sub-<id>_run-<id>_reho.npy`` under ``paths.reho_input_dir`` from the
reproducibility config, computes within-subject vs between-subject
similarity, and writes the similarity matrix + summary CSV/JSON to
``paths.output_root``.

Example:

    python scripts/run_reho_stability.py --config config/reproducibility.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fmri_pipeline.reproducibility import reho_stability  # noqa: E402


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
        help="Echo the JSON summary to stdout after writing it to disk.",
    )
    args = parser.parse_args()

    result = reho_stability.run(args.config)

    print(
        f"ReHo stability: within={result.within_mean:.3f} "
        f"(SD {result.within_std:.3f}), between={result.between_mean:.3f} "
        f"(SD {result.between_std:.3f}), gap={result.gap_mean:.3f} "
        f"[{result.gap_ci_low:.3f}, {result.gap_ci_high:.3f}], "
        f"p={result.p_value:.4g}, d={result.effect_size:.3f}, "
        f"n_subjects={result.n_subjects}, n_runs={result.n_runs}"
    )

    if args.print_summary:
        # Produce a JSON-friendly dict (drop the full similarity matrix —
        # it lives in the .npy already).
        from dataclasses import asdict

        summary = asdict(result)
        summary.pop("similarity_matrix", None)
        print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())

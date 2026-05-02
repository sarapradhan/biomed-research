#!/usr/bin/env python3
"""Run dynamic FC window-size sensitivity (JEI revision Phase 2).

Sweeps dynamic FC over a list of window sizes (default 20, 30, 40 TR),
clusters each window's FC into ``k`` states with k-means, and computes
the Adjusted Rand Index between state-assignment label sequences from
different window sizes (after projecting them onto a common per-TR
timeline).

Reads ROI timeseries saved as ``sub-<id>_run-<id>_roi.npy`` (T x n_rois)
under ``paths.dfc_input_dir`` from the reproducibility config.

Example:

    python scripts/run_dfc_sensitivity.py --config config/reproducibility.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fmri_pipeline.reproducibility import dfc_sensitivity  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Echo group summary as JSON to stdout.",
    )
    args = parser.parse_args()

    group = dfc_sensitivity.run(args.config)

    print(f"dFC sensitivity: n_runs={group.n_runs}, window_sizes={group.window_sizes}")
    for w in group.window_sizes:
        m = group.per_window_variability_mean.get(w, float("nan"))
        s = group.per_window_variability_std.get(w, float("nan"))
        print(f"  window={w} TR  mean FC variability = {m:.4f} (SD {s:.4f})")
    for key, m in group.pairwise_ari_mean.items():
        s = group.pairwise_ari_std.get(key, float("nan"))
        print(f"  ARI({key}) = {m:.3f} (SD {s:.3f})")

    if args.print_summary:
        snapshot = {
            "window_sizes": group.window_sizes,
            "n_runs": group.n_runs,
            "per_window_variability_mean": group.per_window_variability_mean,
            "per_window_variability_std": group.per_window_variability_std,
            "pairwise_ari_mean": group.pairwise_ari_mean,
            "pairwise_ari_std": group.pairwise_ari_std,
        }
        print(json.dumps(snapshot, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

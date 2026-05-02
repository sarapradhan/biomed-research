#!/usr/bin/env python3
"""Run the canonical-network biological anchor (JEI revision Phase 3).

Reads a group-mean FC matrix (or computes it from a directory of per-run
FC matrices) and a ROI-labels file, assigns each ROI to a canonical
network using Schaefer/Yeo conventions, and reports:

* mean within-network FC vs mean between-network FC
* a one-sided permutation test for within > between (default 1000 perms)
* Newman modularity Q of the network partition on the absolute FC
* per-block mean FC for every (network_a, network_b) pair
* the network-reordered FC matrix (.npy) for figure 3

Example:

    python scripts/run_network_anchor.py --config config/reproducibility.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fmri_pipeline.reproducibility import network_anchor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Echo the summary JSON to stdout.",
    )
    args = parser.parse_args()

    result = network_anchor.run(args.config)

    print(
        f"Network anchor: n_rois={result.n_rois}, "
        f"n_networks={result.n_networks}, "
        f"within={result.within_mean:.4f}, between={result.between_mean:.4f}, "
        f"gap={result.gap_mean:.4f}, p={result.p_value:.4g}, "
        f"Q={result.modularity_q:.4f}"
    )

    if args.print_summary:
        d = asdict(result)
        # Drop the matrix from stdout (it's persisted to .npy).
        d.pop("reordered_matrix", None)
        print(json.dumps(d, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

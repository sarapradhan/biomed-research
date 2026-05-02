#!/usr/bin/env python3
"""Build the JEI revision validation scorecard from existing module outputs.

Useful as a standalone command when the per-module CSVs are already on
disk and you only want to (re)render the scorecard. The
``run_reproducibility.py`` harness invokes the same code path as the
final step.

Example:

    python scripts/build_scorecard.py --config config/reproducibility.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fmri_pipeline.reproducibility import scorecard  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    paths = scorecard.run(args.config)
    print(f"Scorecard markdown: {paths['markdown']}")
    print(f"Scorecard CSV:      {paths['csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

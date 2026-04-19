#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fmri_pipeline.pipeline import run_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified fMRI pipeline end-to-end")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    args = parser.parse_args()
    run_all(args.config)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fmri_pipeline.config import load_config
from fmri_pipeline.pipeline import (
    run_group_stats,
    run_ica_step,
    run_ingest,
    run_isc_step,
    run_pca_step,
    run_preprocess_qc,
    run_reho_step,
    run_roi_step,
    run_static_dynamic_fc,
)
from fmri_pipeline.roi import get_schaefer_atlas
from fmri_pipeline.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one pipeline milestone")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--step",
        required=True,
        choices=[
            "ingest",
            "preprocess_qc",
            "roi_timeseries",
            "reho",
            "static_dynamic_fc",
            "ica",
            "pca",
            "isc",
            "group_stats",
        ],
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg["paths"]["logs_dir"], name=f"step_{args.step}")

    participants, runs = run_ingest(cfg, logger)
    if args.step == "ingest":
        return

    preproc = run_preprocess_qc(cfg, runs, logger)
    if args.step == "preprocess_qc":
        return

    roi = run_roi_step(cfg, preproc, logger)
    if args.step == "roi_timeseries":
        return

    if args.step == "reho":
        run_reho_step(cfg, preproc, logger)
        return

    if args.step == "static_dynamic_fc":
        run_static_dynamic_fc(cfg, roi, logger)
        return

    if args.step == "ica":
        run_ica_step(cfg, preproc[~preproc["exclude"]], logger)
        return

    if args.step == "pca":
        run_pca_step(cfg, roi, logger)
        return

    if args.step == "isc":
        run_isc_step(cfg, preproc, logger)
        return

    if args.step == "group_stats":
        run_reho_step(cfg, preproc, logger)
        run_static_dynamic_fc(cfg, roi, logger)
        run_ica_step(cfg, preproc[~preproc["exclude"]], logger)
        run_pca_step(cfg, roi, logger)
        _, labels = get_schaefer_atlas(cfg)
        labels = [l.decode("utf-8") if isinstance(l, bytes) else str(l) for l in labels]
        run_group_stats(cfg, preproc, labels, logger)


if __name__ == "__main__":
    main()

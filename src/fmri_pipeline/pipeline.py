from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import concat_imgs

from .bids_ingest import build_participants_table, collect_runs, fmriprep_command_template
from .connectivity import (
    dynamic_fc_summary,
    exploratory_window_clustering,
    plot_matrix,
    save_matrix,
    static_fc,
)
from .config import load_config
from .ica import (
    compute_subject_network_loadings,
    match_components_across_subjects,
    run_subject_spatial_ica,
    save_subject_ica,
)
from .isc import compute_leave_one_out_isc, permutation_pvalues, save_isc_maps
from .pca_metrics import append_pca_row, run_subject_pca, save_pca_table
from .preprocessing import load_confound_matrix, preprocess_bold, save_preprocessed_images
from .qc import plot_fd_trace, plot_qc_distributions, plot_scrub_mask, save_qc_summary, summarize_qc_row
from .reho import compute_reho_map, save_reho_map
from .roi import extract_roi_timeseries, get_schaefer_atlas, save_roi_timeseries
from .stats import (
    build_design_matrix,
    edge_results_table,
    mass_univariate_ols,
    network_summary,
    save_stats_tables,
    voxelwise_group_stats,
)
from .utils import metric_path, rebuild_symmetric, run_basename, save_json, set_global_seed, setup_logging, upper_triangle_vector
from .viz import plot_box_by_group, plot_voxel_map, save_thresholded_diff_matrix, save_voxel_from_vector


MANIFEST_DIR = "manifests"


def _limit_debug(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    if not cfg["project"].get("debug_mode", False):
        return df
    lim = cfg["project"].get("debug_subject_limit")
    if not lim:
        return df
    subjects = sorted(df["subject"].unique())[: int(lim)]
    return df[df["subject"].isin(subjects)].copy()


def run_ingest(cfg: Dict, logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ingest BIDS entities and write participants/run manifests."""
    participants = build_participants_table(cfg, logger)
    runs = collect_runs(cfg, participants, logger)
    runs = _limit_debug(runs, cfg)

    man_dir = Path(cfg["paths"]["output_root"]) / MANIFEST_DIR
    man_dir.mkdir(parents=True, exist_ok=True)
    participants.to_csv(man_dir / "participants_merged.csv", index=False)
    runs.to_csv(man_dir / "run_manifest_raw.csv", index=False)
    logger.info("Ingested %d runs across %d subjects", len(runs), runs["subject"].nunique())
    return participants, runs


def run_preprocess_qc(cfg: Dict, runs_df: pd.DataFrame, logger) -> pd.DataFrame:
    """Preprocess runs and generate QC summaries/plots."""
    output_root = cfg["paths"]["output_root"]
    qc_rows = []
    run_rows = []

    for _, row in runs_df.iterrows():
        run_key = run_basename(row.to_dict())
        out_dir = metric_path(output_root, "preprocessed", row["subject"], run_key)

        confounds, keep_mask, fd, motion = load_confound_matrix(row["confounds_file"], cfg)

        try:
            unsm, sm = preprocess_bold(
                bold_file=row["bold_file"],
                mask_file=row["brain_mask_file"],
                confound_matrix=confounds,
                keep_mask=keep_mask,
                tr=float(row["tr"]),
                cfg=cfg,
            )
            saved = save_preprocessed_images(unsm, sm, out_dir)
            exclude = bool(motion["exclude"])
        except Exception as e:
            logger.warning("Preprocess failed for %s: %s", run_key, str(e))
            saved = {"unsmoothed": "", "smoothed": ""}
            exclude = True

        plot_fd_trace(fd.to_numpy(), keep_mask, str(out_dir / "qc_fd_trace.png"))
        plot_scrub_mask(keep_mask, str(out_dir / "qc_scrub_mask.png"))

        qc_row = summarize_qc_row(
            subject=row["subject"],
            dataset=row["dataset"],
            site=row.get("site", row["dataset"]),
            run_key=run_key,
            diagnosis=row.get("diagnosis", "NA"),
            fd=fd.to_numpy(),
            keep_mask=keep_mask,
            motion_metrics={**motion, "exclude": exclude},
            cleaned_img_file=saved["unsmoothed"] if saved["unsmoothed"] else row["bold_file"],
            mask_file=row["brain_mask_file"],
        )
        qc_rows.append(qc_row)

        row_dict = row.to_dict()
        row_dict.update(
            {
                "run_key": run_key,
                "clean_unsmoothed_file": saved["unsmoothed"],
                "clean_smoothed_file": saved["smoothed"],
                "exclude": exclude,
                "mean_fd": float(np.mean(fd)),
            }
        )
        run_rows.append(row_dict)

    qc_df = save_qc_summary(qc_rows, output_root)
    plot_qc_distributions(qc_df, output_root)

    out_runs = pd.DataFrame(run_rows)
    man_dir = Path(output_root) / MANIFEST_DIR
    out_runs.to_csv(man_dir / "run_manifest_preprocessed.csv", index=False)
    return out_runs


def run_roi_step(cfg: Dict, runs_df: pd.DataFrame, logger) -> pd.DataFrame:
    """Extract Schaefer-200 ROI time series per run."""
    atlas_img, labels = get_schaefer_atlas(cfg)
    rows = []
    for _, row in runs_df.iterrows():
        if row.get("exclude", False):
            continue
        if not row.get("clean_unsmoothed_file"):
            continue

        out_dir = metric_path(cfg["paths"]["output_root"], "roi_timeseries", row["subject"], row["run_key"])
        ts = extract_roi_timeseries(row["clean_unsmoothed_file"], atlas_img, float(row["tr"]), cfg)
        npy, csv = save_roi_timeseries(ts, labels, out_dir)
        row_dict = row.to_dict()
        row_dict.update({"roi_npy": npy, "roi_csv": csv})
        rows.append(row_dict)

    roi_df = pd.DataFrame(rows)
    roi_df.to_csv(Path(cfg["paths"]["output_root"]) / MANIFEST_DIR / "run_manifest_roi.csv", index=False)
    logger.info("Extracted ROI timeseries for %d runs", len(roi_df))
    return roi_df


def run_reho_step(cfg: Dict, runs_df: pd.DataFrame, logger) -> pd.DataFrame:
    """Compute ReHo map per run and subject-average map."""
    rows = []
    for _, row in runs_df.iterrows():
        if row.get("exclude", False) or not row.get("clean_unsmoothed_file"):
            continue

        out_dir = metric_path(cfg["paths"]["output_root"], "reho", row["subject"], row["run_key"])
        reho_img = compute_reho_map(
            clean_bold_file=row["clean_unsmoothed_file"],
            gm_mask_file=row["brain_mask_file"],
            cfg=cfg,
            n_jobs=int(cfg["project"].get("n_jobs", 1)),
        )
        reho_path = save_reho_map(reho_img, out_dir)
        row_dict = row.to_dict()
        row_dict["reho_file"] = reho_path
        rows.append(row_dict)

    reho_df = pd.DataFrame(rows)
    reho_df.to_csv(Path(cfg["paths"]["output_root"]) / MANIFEST_DIR / "run_manifest_reho.csv", index=False)
    return reho_df


def _subject_average_matrix(run_df: pd.DataFrame, matrix_col: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for sub, g in run_df.groupby("subject"):
        mats = [np.load(p) for p in g[matrix_col].tolist() if isinstance(p, str) and p]
        if mats:
            out[sub] = np.mean(np.stack(mats, axis=0), axis=0)
    return out


def run_static_dynamic_fc(cfg: Dict, roi_df: pd.DataFrame, logger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute static FC and dFC summary matrices."""
    static_rows = []
    dyn_rows = []

    for _, row in roi_df.iterrows():
        ts = np.load(row["roi_npy"])

        s_fc = static_fc(ts)
        s_dir = metric_path(cfg["paths"]["output_root"], "static_fc", row["subject"], row["run_key"])
        s_path = save_matrix(s_fc, s_dir, "static_fc_fisherz")
        plot_matrix(s_fc, str(s_dir / "static_fc_matrix.png"), "Static FC (Fisher-z)")

        d_mean, d_std, windows = dynamic_fc_summary(ts, cfg)
        d_dir = metric_path(cfg["paths"]["output_root"], "dynamic_fc", row["subject"], row["run_key"])
        dm_path = save_matrix(d_mean, d_dir, "dfc_mean")
        dv_path = save_matrix(d_std, d_dir, "dfc_variability_std")
        plot_matrix(d_std, str(d_dir / "dfc_variability_matrix.png"), "dFC Variability (STD)")

        if cfg["dynamic_fc"].get("exploratory_clustering", False):
            exp = exploratory_window_clustering(windows, cfg)
            np.save(d_dir / "exploratory_state_labels.npy", exp["labels"])
            np.save(d_dir / "exploratory_state_centroids.npy", exp["centroids"])

        static_rows.append({**row.to_dict(), "static_fc_file": s_path})
        dyn_rows.append({**row.to_dict(), "dfc_mean_file": dm_path, "dfc_var_file": dv_path})

    static_df = pd.DataFrame(static_rows)
    dyn_df = pd.DataFrame(dyn_rows)

    static_df.to_csv(Path(cfg["paths"]["output_root"]) / MANIFEST_DIR / "run_manifest_static_fc.csv", index=False)
    dyn_df.to_csv(Path(cfg["paths"]["output_root"]) / MANIFEST_DIR / "run_manifest_dynamic_fc.csv", index=False)

    sub_static = []
    sub_dyn = []
    for sub, mat in _subject_average_matrix(static_df, "static_fc_file").items():
        sdir = metric_path(cfg["paths"]["output_root"], "static_fc", sub)
        spath = save_matrix(mat, sdir, "subject_mean_static_fc")
        sub_static.append({"subject": sub, "static_fc_subject_file": spath})

    for sub, mat in _subject_average_matrix(dyn_df, "dfc_var_file").items():
        ddir = metric_path(cfg["paths"]["output_root"], "dynamic_fc", sub)
        dpath = save_matrix(mat, ddir, "subject_mean_dfc_var")
        sub_dyn.append({"subject": sub, "dfc_var_subject_file": dpath})

    return static_df, dyn_df, pd.DataFrame(sub_static).merge(pd.DataFrame(sub_dyn), on="subject", how="outer")


def run_ica_step(cfg: Dict, runs_df: pd.DataFrame, logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run subject ICA then cross-subject component matching."""
    subject_maps: Dict[str, np.ndarray] = {}
    subject_rows = []
    common_mask_file = runs_df.iloc[0]["brain_mask_file"]

    for sub, g in runs_df.groupby("subject"):
        imgs = [p for p in g["clean_unsmoothed_file"].tolist() if isinstance(p, str) and p]
        if not imgs:
            continue
        concat = concat_imgs(imgs)
        tmp_dir = metric_path(cfg["paths"]["output_root"], "ica", sub)
        concat_path = tmp_dir / "concat_bold.nii.gz"
        nib.save(concat, str(concat_path))

        maps, tc = run_subject_spatial_ica(str(concat_path), common_mask_file, cfg)
        saved = save_subject_ica(maps, tc, tmp_dir)
        subject_maps[sub] = maps
        subject_rows.append({"subject": sub, "ica_maps": saved["maps"], "ica_timecourses": saved["timecourses"]})

    match_df, centroids = match_components_across_subjects(subject_maps, cfg)
    loadings_df = compute_subject_network_loadings(subject_maps, centroids)

    ica_dir = Path(cfg["paths"]["output_root"]) / "ica"
    ica_dir.mkdir(parents=True, exist_ok=True)
    match_df.to_csv(ica_dir / "ica_component_matching.csv", index=False)
    np.save(ica_dir / "ica_cluster_centroids.npy", centroids)
    loadings_df.to_csv(ica_dir / "ica_subject_loadings.csv", index=False)

    return pd.DataFrame(subject_rows), loadings_df


def run_pca_step(cfg: Dict, roi_df: pd.DataFrame, logger) -> pd.DataFrame:
    """Run subject-level PCA on concatenated ROI series."""
    rows = []
    merged_rows = []
    for sub, g in roi_df.groupby("subject"):
        ts = [np.load(p) for p in g["roi_npy"].tolist() if isinstance(p, str) and p]
        if not ts:
            continue
        concat_ts = np.concatenate(ts, axis=0)
        evr = run_subject_pca(concat_ts, cfg)
        diagnosis = str(g["diagnosis"].iloc[0]) if "diagnosis" in g.columns else "NA"
        append_pca_row(rows, sub, str(g["dataset"].iloc[0]), diagnosis, evr)
        merged_rows.append({"subject": sub, **{f"pca_evr_{i+1}": float(v) for i, v in enumerate(evr)}})

    pca_dir = Path(cfg["paths"]["output_root"]) / "pca"
    save_pca_table(rows, pca_dir)
    return pd.DataFrame(merged_rows)


def run_isc_step(cfg: Dict, runs_df: pd.DataFrame, logger) -> Dict[str, str]:
    """Compute leave-one-out ISC for Algonauts controls with permutation significance."""
    movie = runs_df[(runs_df["dataset"] == "algonauts") & (~runs_df["exclude"])].copy()
    if movie["subject"].nunique() < int(cfg["isc"].get("min_subjects", 4)):
        logger.warning("Skipping ISC: insufficient movie subjects.")
        return {}

    files = []
    for sub, g in movie.groupby("subject"):
        imgs = [p for p in g["clean_smoothed_file"].tolist() if isinstance(p, str) and p]
        if not imgs:
            continue
        concat = concat_imgs(imgs)
        out_dir = metric_path(cfg["paths"]["output_root"], "isc", sub)
        cpath = out_dir / "concat_movie_smoothed6mm.nii.gz"
        nib.save(concat, str(cpath))
        files.append(str(cpath))

    if len(files) < int(cfg["isc"].get("min_subjects", 4)):
        logger.warning("Skipping ISC: could not build enough concatenated files.")
        return {}

    mask_file = movie["brain_mask_file"].iloc[0]
    loo, mean_map = compute_leave_one_out_isc(files, mask_file)
    pvals, qvals = permutation_pvalues(files, mask_file, mean_map, cfg)

    out_dir = Path(cfg["paths"]["output_root"]) / "isc"
    paths = save_isc_maps(mean_map, pvals, qvals, mask_file, out_dir)

    np.save(out_dir / "isc_leave_one_out_by_subject.npy", loo)
    save_json(
        {
            "assumption": "N=4 controls; circular-shift permutation null used for voxelwise ISC significance.",
            "n_subjects": int(loo.shape[0]),
            "n_voxels": int(loo.shape[1]),
            "permutations": int(cfg["isc"].get("permutations", 1000)),
        },
        str(out_dir / "isc_assumptions.json"),
    )
    plot_voxel_map(paths["isc_mean"], str(Path(cfg["paths"]["output_root"]) / "figures" / "isc_mean_map.png"), "ISC Mean")
    plot_voxel_map(paths["isc_sig"], str(Path(cfg["paths"]["output_root"]) / "figures" / "isc_sig_fdrq05.png"), "ISC Significant (FDR q<0.05)")
    return paths


def run_group_stats(cfg: Dict, runs_df: pd.DataFrame, roi_labels: List[str], logger) -> None:
    """Run group statistics for FC, dFC variability, ReHo, ICA loadings, and PCA metrics."""
    out_root = Path(cfg["paths"]["output_root"])
    stats_dir = out_root / "group_stats"
    figs_dir = out_root / "figures"
    tabs_dir = out_root / "tables"
    diag_col = cfg["stats"]["diagnosis_column"]
    stats_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    tabs_dir.mkdir(parents=True, exist_ok=True)

    qc = pd.read_csv(out_root / "qc" / "qc_summary.csv")
    subj_cov = (
        qc.groupby("subject", as_index=False)
        .agg(mean_fd=("mean_fd", "mean"), diagnosis=("diagnosis", "first"), dataset=("dataset", "first"), site=("site", "first"))
    )

    # Harmonize subject IDs to prevent int/string merge mismatches (e.g., 1 vs 01).
    subj_cov["subject"] = (
        subj_cov["subject"].astype(str).str.replace("sub-", "", regex=False).str.lstrip("0").replace("", "0")
    )

    participants = pd.read_csv(out_root / MANIFEST_DIR / "participants_merged.csv")
    participants["subject"] = (
        participants["subject_id"].astype(str).str.replace("sub-", "", regex=False).str.lstrip("0").replace("", "0")
    )
    cov = subj_cov.merge(participants, on=[c for c in ["subject", "dataset"] if c in participants.columns], how="left")

    static_manifest = pd.read_csv(out_root / MANIFEST_DIR / "run_manifest_static_fc.csv")
    dyn_manifest = pd.read_csv(out_root / MANIFEST_DIR / "run_manifest_dynamic_fc.csv")
    reho_manifest = pd.read_csv(out_root / MANIFEST_DIR / "run_manifest_reho.csv")

    for mdf in (static_manifest, dyn_manifest, reho_manifest):
        mdf["subject"] = mdf["subject"].astype(str).str.replace("sub-", "", regex=False).str.lstrip("0").replace("", "0")

    static_sub = _subject_average_matrix(static_manifest, "static_fc_file")
    dyn_sub = _subject_average_matrix(dyn_manifest, "dfc_var_file")

    common_subs = sorted(set(static_sub.keys()) & set(cov["subject"].astype(str)))
    cov_fc = cov[cov["subject"].astype(str).isin(common_subs)].copy().sort_values("subject")
    x_fc, x_cols, x_df = build_design_matrix(cov_fc, cfg)

    ordered_subs = x_df["subject"].astype(str).tolist()
    mats = [static_sub[s] for s in ordered_subs if s in static_sub]
    if mats:
        y_mat = np.stack(mats, axis=0)
        y_edge, edge_idx = upper_triangle_vector(y_mat[0])
        y = np.stack([m[edge_idx] for m in y_mat], axis=0)
        res = mass_univariate_ols(y, x_fc, contrast_idx=1)

        edge_df = edge_results_table(res["beta"], res["t"], res["p"], res["q"], edge_idx, roi_labels, "static_fc")
        net_df = network_summary(edge_df)
        save_stats_tables(edge_df, net_df, tabs_dir, "static_fc")
        save_thresholded_diff_matrix(
            res["beta"], res["q"], edge_idx, n_rois=len(roi_labels),
            out_file=str(figs_dir / "static_fc_diff_sig_matrix.png"),
            title="Static FC Diagnosis Effect (SZ-HC, FDR q<0.05)",
        )

    common_subs_dyn = sorted(set(dyn_sub.keys()) & set(cov["subject"].astype(str)))
    cov_dyn = cov[cov["subject"].astype(str).isin(common_subs_dyn)].copy().sort_values("subject")
    x_dyn, _, x_dyn_df = build_design_matrix(cov_dyn, cfg)
    ordered_dyn = x_dyn_df["subject"].astype(str).tolist()
    dyn_mats = [dyn_sub[s] for s in ordered_dyn if s in dyn_sub]
    if dyn_mats:
        edge_idx_dyn = np.triu_indices(dyn_mats[0].shape[0], k=1)
        y_dyn = np.stack([m[edge_idx_dyn] for m in dyn_mats], axis=0)
        res_dyn = mass_univariate_ols(y_dyn, x_dyn, contrast_idx=1)
        edge_df_dyn = edge_results_table(
            res_dyn["beta"], res_dyn["t"], res_dyn["p"], res_dyn["q"], edge_idx_dyn, roi_labels, "dynamic_fc_variability"
        )
        net_df_dyn = network_summary(edge_df_dyn)
        save_stats_tables(edge_df_dyn, net_df_dyn, tabs_dir, "dynamic_fc_variability")
        save_thresholded_diff_matrix(
            res_dyn["beta"], res_dyn["q"], edge_idx_dyn, n_rois=len(roi_labels),
            out_file=str(figs_dir / "dynamic_fc_variability_diff_sig_matrix.png"),
            title="dFC Variability Diagnosis Effect (SZ-HC, FDR q<0.05)",
        )

    reho_sub_rows = []
    for sub, g in reho_manifest.groupby("subject"):
        imgs = [nib.load(p).get_fdata(dtype=np.float32) for p in g["reho_file"].tolist() if isinstance(p, str) and p]
        if imgs:
            reho_sub_rows.append((sub, np.mean(np.stack(imgs, axis=0), axis=0)))

    if reho_sub_rows:
        reho_subjects = [s for s, _ in reho_sub_rows]
        cov_reho = cov[cov["subject"].astype(str).isin(reho_subjects)].copy().sort_values("subject")
        x_reho, _, x_reho_df = build_design_matrix(cov_reho, cfg)
        ordered = x_reho_df["subject"].astype(str).tolist()
        ref_mask = nib.load(runs_df.iloc[0]["brain_mask_file"]).get_fdata() > 0

        y_list = []
        for s in ordered:
            for sub, arr in reho_sub_rows:
                if sub == s:
                    y_list.append(arr[ref_mask])
                    break

        if y_list:
            y_reho = np.stack(y_list, axis=0)
            reho_res = voxelwise_group_stats(y_reho, x_reho, cfg)
            beta_file = str(stats_dir / "reho_beta_diagnosis.nii.gz")
            q_file = str(stats_dir / "reho_qvals_diagnosis.nii.gz")
            save_voxel_from_vector(reho_res["beta"], runs_df.iloc[0]["brain_mask_file"], beta_file)
            save_voxel_from_vector(reho_res["q"], runs_df.iloc[0]["brain_mask_file"], q_file)
            plot_voxel_map(beta_file, str(figs_dir / "reho_group_diff_beta.png"), "ReHo Group Difference (SZ-HC)", threshold=0.0)

    ica_load = out_root / "ica" / "ica_subject_loadings.csv"
    if ica_load.exists():
        load_df = pd.read_csv(ica_load)
        load_df["subject"] = load_df["subject"].astype(str).str.replace("sub-", "", regex=False).str.lstrip("0").replace("", "0")
        cov_plot = cov[["subject", diag_col]].copy()
        load_df = load_df.merge(cov_plot, on="subject", how="left")
        plot_box_by_group(
            load_df,
            x=diag_col,
            y="loading",
            out_file=str(figs_dir / "ica_loadings_by_group.png"),
            title="ICA Network Loadings by Diagnosis",
        )
        load_df.to_csv(tabs_dir / "ica_loadings_with_groups.csv", index=False)

    pca_path = out_root / "pca" / "pca_explained_variance.csv"
    if pca_path.exists():
        pca_df = pd.read_csv(pca_path)
        pca_df["subject"] = pca_df["subject"].astype(str).str.replace("sub-", "", regex=False).str.lstrip("0").replace("", "0")
        cov_plot = cov[["subject", diag_col]].copy()
        pca_df = pca_df.drop(columns=[diag_col], errors="ignore").merge(cov_plot, on="subject", how="left")
        plot_box_by_group(
            pca_df,
            x=diag_col,
            y="explained_variance_ratio",
            out_file=str(figs_dir / "pca_variance_by_group.png"),
            title="PCA Explained Variance by Diagnosis",
        )
        pca_df.to_csv(tabs_dir / "pca_explained_variance_with_groups.csv", index=False)


def run_all(config_file: str) -> None:
    """Run full pipeline through group-level outputs."""
    cfg = load_config(config_file)
    logger = setup_logging(cfg["paths"]["logs_dir"], name="pipeline")
    set_global_seed(int(cfg["project"].get("random_seed", 42)))

    logger.info("Reliability layer: preprocessing + QC")
    participants, runs = run_ingest(cfg, logger)
    preproc_runs = run_preprocess_qc(cfg, runs, logger)

    logger.info("Feature modules: ROI, ReHo, FC, dFC, ICA, PCA, ISC")
    roi_runs = run_roi_step(cfg, preproc_runs, logger)
    reho_runs = run_reho_step(cfg, preproc_runs, logger)
    run_static_dynamic_fc(cfg, roi_runs, logger)
    run_ica_step(cfg, preproc_runs[~preproc_runs["exclude"]], logger)
    run_pca_step(cfg, roi_runs, logger)
    run_isc_step(cfg, preproc_runs, logger)

    logger.info("Decision layer: group statistics")
    _, roi_labels = get_schaefer_atlas(cfg)
    roi_labels = [l.decode("utf-8") if isinstance(l, bytes) else str(l) for l in roi_labels]
    run_group_stats(cfg, preproc_runs, roi_labels, logger)

    logger.info("Pipeline complete")
    template = fmriprep_command_template(cfg)
    save_json({"fmriprep_docker_template": template}, str(Path(cfg["paths"]["output_root"]) / "repro" / "fmriprep_command_template.json"))

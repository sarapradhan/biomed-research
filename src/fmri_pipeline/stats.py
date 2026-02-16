from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy.stats import t as t_dist


def build_design_matrix(subject_df: pd.DataFrame, cfg: Dict) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """Build design matrix with intercept + diagnosis + covariates + optional site dummies."""
    y_col = cfg["stats"]["diagnosis_column"]
    patient = cfg["stats"]["patient_label"]
    control = cfg["stats"]["control_label"]

    df = subject_df.copy()
    df = df[df[y_col].isin([patient, control])].copy()
    df["diagnosis_bin"] = (df[y_col] == patient).astype(int)

    covars = list(cfg["stats"].get("covariates", []))
    for c in cfg["stats"].get("optional_covariates_if_available", []):
        if c in df.columns:
            covars.append(c)

    x_parts = [pd.Series(np.ones(len(df)), name="intercept"), df["diagnosis_bin"]]
    colnames = ["intercept", "diagnosis_bin"]

    for c in covars:
        if c not in df.columns:
            continue
        if df[c].dtype == object:
            dummies = pd.get_dummies(df[c], prefix=c, drop_first=True)
            for dc in dummies.columns:
                x_parts.append(dummies[dc].astype(float))
                colnames.append(dc)
        else:
            vals = df[c].astype(float).fillna(df[c].astype(float).mean())
            vals = (vals - vals.mean()) / (vals.std() if vals.std() > 0 else 1.0)
            x_parts.append(vals.rename(c))
            colnames.append(c)

    x_df = pd.concat(x_parts, axis=1)
    return x_df.to_numpy(dtype=float), colnames, df


def mass_univariate_ols(y: np.ndarray, x: np.ndarray, contrast_idx: int) -> Dict[str, np.ndarray]:
    """Vectorized OLS for Y [subjects, features] with contrast index for diagnosis."""
    n, p = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    y_hat = x @ beta
    resid = y - y_hat

    dof = max(n - p, 1)
    sigma2 = np.sum(resid**2, axis=0) / dof
    c = np.zeros(p)
    c[contrast_idx] = 1.0

    cvc = float(c.T @ xtx_inv @ c)
    se = np.sqrt(np.maximum(sigma2 * cvc, 1e-12))
    tvals = beta[contrast_idx] / se
    pvals = 2.0 * (1.0 - t_dist.cdf(np.abs(tvals), dof))

    _, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    return {
        "beta": beta[contrast_idx],
        "t": tvals,
        "p": pvals,
        "q": qvals,
        "dof": np.array([dof], dtype=float),
    }


def edge_results_table(
    beta: np.ndarray,
    tvals: np.ndarray,
    pvals: np.ndarray,
    qvals: np.ndarray,
    edge_idx: Tuple[np.ndarray, np.ndarray],
    roi_labels: List[str],
    metric_name: str,
) -> pd.DataFrame:
    """Create tidy edge-level results table."""
    i, j = edge_idx
    df = pd.DataFrame(
        {
            "metric": metric_name,
            "roi_i": i,
            "roi_j": j,
            "label_i": [roi_labels[k] for k in i],
            "label_j": [roi_labels[k] for k in j],
            "beta_diagnosis": beta,
            "t": tvals,
            "p": pvals,
            "q": qvals,
            "significant_fdr": qvals < 0.05,
        }
    )
    return df.sort_values("q")


def network_summary(edge_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize significant edges by Schaefer network labels."""
    sig = edge_df[edge_df["significant_fdr"]].copy()
    if sig.empty:
        return pd.DataFrame(columns=["network_pair", "n_edges", "mean_beta"])

    def net(lbl: str) -> str:
        parts = lbl.split("_")
        return parts[1] if len(parts) > 2 else lbl

    sig["net_i"] = sig["label_i"].map(net)
    sig["net_j"] = sig["label_j"].map(net)
    sig["network_pair"] = np.where(sig["net_i"] <= sig["net_j"], sig["net_i"] + "__" + sig["net_j"], sig["net_j"] + "__" + sig["net_i"])

    out = (
        sig.groupby("network_pair", as_index=False)
        .agg(n_edges=("network_pair", "size"), mean_beta=("beta_diagnosis", "mean"))
        .sort_values("n_edges", ascending=False)
    )
    return out


def save_stats_tables(edge_df: pd.DataFrame, network_df: pd.DataFrame, out_dir: Path, stem: str) -> Dict[str, str]:
    """Save group-level stats tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    edge_path = out_dir / f"{stem}_edge_results.csv"
    net_path = out_dir / f"{stem}_network_summary.csv"
    edge_df.to_csv(edge_path, index=False)
    network_df.to_csv(net_path, index=False)
    return {"edges": str(edge_path), "networks": str(net_path)}


def voxelwise_group_stats(y: np.ndarray, design: np.ndarray, cfg: Dict) -> Dict[str, np.ndarray]:
    """Second-level voxelwise OLS + FDR for diagnosis effect."""
    return mass_univariate_ols(y=y, x=design, contrast_idx=1)

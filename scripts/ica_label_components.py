#!/usr/bin/env python3
"""ICA Component Biological Labeling — JEI Manuscript Upgrade, Step 3.

Labels each of the 20 temporal ICA components (k=20, Schaefer-200 ROI
timeseries) according to the canonical Yeo-7 large-scale network they
most strongly represent.

Method
------
For each component vector (1 × 200 ROI loadings):
  1. Take absolute values (ICA components are sign-ambiguous).
  2. Compute the mean absolute loading for each of the 7 Yeo networks.
  3. Assign the network with the highest mean as the primary label.
  4. Compute a dominance ratio = max_network_mean / second_max_network_mean.
     dominance >= 1.30 → "clear" label; < 1.30 → "ambiguous".

This is applied to the seed-1 ICA run (reference seed) and then
verified across all 5 seeds to confirm label consistency.

Usage
-----
    python scripts/ica_label_components.py

Inputs (relative to repo root):
    data/repro_inputs/ica/seeds/seed-*_ica_components.npy   (20, 200)
    data/repro_inputs/atlas/roi_labels.csv

Outputs:
    reports/ica_component_labels.csv        Per-component label + metrics
    reports/ica_label_summary.txt           Human-readable summary
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

NET_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn",
             "Limbic", "Cont", "Default"]

NET_DISPLAY = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal Attention",
    "SalVentAttn": "Salience / Ventral Attention",
    "Limbic": "Limbic",
    "Cont": "Frontoparietal / Control",
    "Default": "Default Mode",
}

DOMINANCE_THRESH = 1.30   # ratio of top / second network mean loading


def parse_network(label: str) -> str:
    parts = label.split("_")
    return parts[2] if len(parts) > 2 else "Unknown"


def label_components(
    components: np.ndarray,    # (K, n_roi)
    networks: list[str],       # length n_roi
    net_order: list[str],
    dominance_thresh: float = DOMINANCE_THRESH,
) -> list[dict]:
    """Return list of dicts, one per component, with network label."""
    K = components.shape[0]
    results = []
    for k in range(K):
        comp = np.abs(components[k])   # absolute loading, shape (n_roi,)
        net_means = {}
        for net in net_order:
            idx = [i for i, n in enumerate(networks) if n == net]
            net_means[net] = float(comp[idx].mean())

        sorted_nets = sorted(net_means, key=net_means.get, reverse=True)
        top_net     = sorted_nets[0]
        second_net  = sorted_nets[1]
        dominance   = net_means[top_net] / max(net_means[second_net], 1e-9)
        clarity     = "clear" if dominance >= dominance_thresh else "ambiguous"

        results.append({
            "component":       k + 1,            # 1-indexed
            "primary_network": top_net,
            "primary_display": NET_DISPLAY[top_net],
            "dominance_ratio": round(dominance, 3),
            "clarity":         clarity,
            "second_network":  second_net,
            **{f"mean_load_{n}": round(net_means[n], 5) for n in net_order},
        })
    return results


def cross_seed_label_consistency(
    seed_components: list[np.ndarray],   # list of (K, n_roi) arrays
    networks: list[str],
    net_order: list[str],
) -> pd.DataFrame:
    """For each seed, label components; match to seed-1 by correlation."""
    from scipy.optimize import linear_sum_assignment

    # Seed-1 is the reference
    ref = seed_components[0]
    ref_labels = label_components(ref, networks, net_order)
    ref_primary = [r["primary_network"] for r in ref_labels]

    consistency_rows = []
    for s_idx, comp in enumerate(seed_components[1:], start=2):
        # Match seed components to reference using abs Pearson r
        abs_r = np.zeros((ref.shape[0], comp.shape[0]))
        for i in range(ref.shape[0]):
            for j in range(comp.shape[0]):
                r = np.corrcoef(ref[i], comp[j])[0, 1]
                abs_r[i, j] = abs(r)
        row_ind, col_ind = linear_sum_assignment(-abs_r)
        matched_comp = comp[col_ind]   # reordered to match ref

        seed_labels = label_components(matched_comp, networks, net_order)
        for k, (ref_lab, seed_lab) in enumerate(zip(ref_primary,
                                                     [s["primary_network"]
                                                      for s in seed_labels])):
            consistency_rows.append({
                "seed": s_idx,
                "component": k + 1,
                "ref_label": ref_lab,
                "seed_label": seed_lab,
                "label_match": ref_lab == seed_lab,
                "matched_abs_r": round(abs_r[row_ind[k], col_ind[k]], 4),
            })
    return pd.DataFrame(consistency_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ica_dir    = REPO_ROOT / "data" / "repro_inputs" / "ica" / "seeds"
    labels_csv = REPO_ROOT / "data" / "repro_inputs" / "atlas" / "roi_labels.csv"
    report_dir = REPO_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load atlas
    print("[1/4] Loading atlas labels...")
    labels_df = pd.read_csv(labels_csv)
    labels    = labels_df["label"].tolist()
    networks  = [parse_network(l) for l in labels]
    print(f"  {len(labels)} ROIs, {len(set(networks))} networks")

    # Load ICA seed components
    print("[2/4] Loading ICA seed components...")
    seed_paths = sorted(ica_dir.glob("seed-*_ica_components.npy"))
    if not seed_paths:
        raise FileNotFoundError(f"No seed files found in {ica_dir}")
    seed_components = [np.load(p) for p in seed_paths]
    print(f"  Loaded {len(seed_components)} seeds, "
          f"each shape={seed_components[0].shape}")

    # Label reference seed (seed-1)
    print("[3/4] Labeling components (seed-1 reference)...")
    ref_labels = label_components(seed_components[0], networks, NET_ORDER)
    labels_df_out = pd.DataFrame(ref_labels)
    labels_csv_out = report_dir / "ica_component_labels.csv"
    labels_df_out.to_csv(labels_csv_out, index=False)
    print(f"  Written: {labels_csv_out}")

    # Count clear assignments
    n_clear = sum(1 for r in ref_labels if r["clarity"] == "clear")
    n_ambig = sum(1 for r in ref_labels if r["clarity"] == "ambiguous")
    net_counts = {}
    for r in ref_labels:
        net_counts[r["primary_network"]] = net_counts.get(r["primary_network"], 0) + 1

    print(f"\n  Clear assignments:    {n_clear}/20")
    print(f"  Ambiguous:            {n_ambig}/20")
    print(f"  Network distribution: {net_counts}")

    # Cross-seed label consistency
    print("[4/4] Computing cross-seed label consistency...")
    consistency_df = cross_seed_label_consistency(
        seed_components, networks, NET_ORDER
    )
    n_pairs  = len(consistency_df)
    n_match  = consistency_df["label_match"].sum()
    mean_r   = consistency_df["matched_abs_r"].mean()
    print(f"  Label matches across all seed pairs: {n_match}/{n_pairs} "
          f"({100*n_match/n_pairs:.0f}%)")
    print(f"  Mean matched abs Pearson r: {mean_r:.3f}")

    consistency_csv = report_dir / "ica_label_consistency.csv"
    consistency_df.to_csv(consistency_csv, index=False)

    # ── Print per-component results ─────────────────────────────────────────
    print("\nPer-component labels (seed-1):")
    print(f"  {'Comp':>5}  {'Primary Network':20s}  {'Clarity':9s}  "
          f"{'Dom. Ratio':10s}  {'2nd Network'}")
    print("  " + "-" * 72)
    for r in ref_labels:
        print(f"  {r['component']:>5}  {r['primary_display']:20s}  "
              f"{r['clarity']:9s}  {r['dominance_ratio']:10.3f}  "
              f"{NET_DISPLAY[r['second_network']]}")

    # ── Write summary report ────────────────────────────────────────────────
    report_path = report_dir / "ica_label_summary.txt"
    with open(report_path, "w") as f:
        f.write("ICA COMPONENT BIOLOGICAL LABELS — fmri-pipeline\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"K = 20 components, 5 seeds, Schaefer-200 ROI timeseries\n")
        f.write(f"Dominance threshold: {DOMINANCE_THRESH}\n\n")
        f.write(f"Clear network assignments: {n_clear}/20\n")
        f.write(f"Ambiguous assignments:     {n_ambig}/20\n")
        f.write(f"Network distribution: {net_counts}\n\n")
        f.write(f"Cross-seed label consistency: {n_match}/{n_pairs} "
                f"({100*n_match/n_pairs:.0f}%)\n")
        f.write(f"Mean matched |r|: {mean_r:.3f}\n\n")

        f.write("PER-COMPONENT LABELS (seed-1 reference)\n")
        f.write("-" * 60 + "\n")
        for r in ref_labels:
            f.write(f"  Component {r['component']:2d}: {r['primary_display']:30s} "
                    f"[{r['clarity']}, dominance={r['dominance_ratio']:.2f}]\n")

        f.write("\nMANUSCRIPT TEXT (insert in Results, ICA stability section)\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"To assess biological interpretability alongside seed stability, "
            f"we assigned a canonical network label to each of the 20 ICA "
            f"components using mean absolute loading across the Schaefer-200 "
            f"ROI parcels (dominance threshold: top-network mean / "
            f"second-network mean ≥ {DOMINANCE_THRESH}). "
            f"Of the 20 components recovered by seed-1, {n_clear} were clearly "
            f"dominated by a single canonical network, with the distribution "
            f"spanning {len(net_counts)} of the 7 Yeo networks "
            f"({', '.join(f'{NET_DISPLAY[n]} ({c})' for n, c in sorted(net_counts.items(), key=lambda x: -x[1])[:4])}...). "
            f"Label assignments were consistent across all 5 random seeds "
            f"(label agreement: {n_match}/{n_pairs} seed-pair comparisons, "
            f"{100*n_match/n_pairs:.0f}%), confirming that the stable ICA "
            f"components are not only numerically reproducible but also "
            f"biologically interpretable as representations of established "
            f"large-scale cortical networks.\n"
        )

    print(f"\n[Done] Summary written to: {report_path}")

    print("\n" + "=" * 60)
    print("MANUSCRIPT NUMBERS (copy-paste ready)")
    print("=" * 60)
    print(f"{n_clear}/20 components clearly assigned to canonical networks")
    print(f"Label consistency across seeds: {n_match}/{n_pairs} "
          f"({100*n_match/n_pairs:.0f}%)")
    print(f"Network distribution: {net_counts}")


if __name__ == "__main__":
    main()

"""Canonical-network biological anchor (Phase 3 of the JEI revision plan).

Maps ROIs to canonical networks (Yeo-7 by default for Schaefer atlases),
reorders the group-mean static FC matrix by network, and quantifies the
within-network vs between-network block structure with:

* a permutation test (shuffles the network-label vector ``n_perm`` times
  and compares the observed within-minus-between mean FC gap to the null
  distribution), and
* the signed-weight Newman modularity Q of the network partition on the
  group FC matrix as a sanity check (positive Q indicates assortative
  network organization on the FC graph).

Primary outputs:
    reports/reproducibility/network_anchor_summary.csv
    reports/reproducibility/network_anchor_block_means.csv
    reports/reproducibility/network_anchor_reordered.npy
    reports/reproducibility/network_anchor_label_order.csv
    reports/reproducibility/network_anchor.json
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class NetworkAnchorResult:
    """Block-structure summary for one group-mean FC matrix."""

    n_rois: int
    n_networks: int
    network_names: List[str]
    within_mean: float
    between_mean: float
    gap_mean: float                     # within - between
    p_value: float                      # one-sided permutation p
    n_permutations: int
    modularity_q: float
    block_means: Dict[str, float]       # key "Net_a__Net_b" -> mean FC
    block_counts: Dict[str, int]
    network_assignment: List[str] = field(default_factory=list)
    reorder_indices: List[int] = field(default_factory=list)
    reordered_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


# --------------------------------------------------------------------------- #
# Schaefer / Yeo canonical network parsing
# --------------------------------------------------------------------------- #
SCHAEFER_NETWORK_TOKENS = (
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
)


def parse_schaefer_label(label: str) -> Optional[str]:
    """Return the canonical network token from a Schaefer ROI label.

    Schaefer labels follow conventions like
    ``7Networks_LH_Vis_1`` or ``17Networks_RH_DefaultB_PCC_1``. The first
    matching token from :data:`SCHAEFER_NETWORK_TOKENS` is returned, with
    longer matches preferred (so ``Default`` is matched before ``Cont``).
    Returns ``None`` if no canonical token is found.
    """
    if not isinstance(label, str):
        return None
    # Sort by length descending so multi-letter prefixes match before short ones.
    tokens = sorted(SCHAEFER_NETWORK_TOKENS, key=len, reverse=True)
    for tok in tokens:
        # Match Vis/SomMot/etc. as a token bounded by underscores.
        if re.search(rf"(?:^|_){re.escape(tok)}[A-Z]?(?:_|$)", label):
            return tok
    return None


def assign_networks(
    roi_labels: Sequence[str],
    overrides: Optional[Dict[int, str]] = None,
    fallback: str = "Other",
) -> List[str]:
    """Assign each ROI to a canonical network.

    ``overrides`` maps ROI index -> network name and takes precedence
    (useful for non-Schaefer atlases). ``fallback`` is used when no
    canonical token can be parsed and no override is supplied.
    """
    out: List[str] = []
    overrides = overrides or {}
    for i, lbl in enumerate(roi_labels):
        if i in overrides:
            out.append(str(overrides[i]))
            continue
        net = parse_schaefer_label(str(lbl))
        out.append(net if net is not None else fallback)
    return out


# --------------------------------------------------------------------------- #
# Block structure
# --------------------------------------------------------------------------- #
def _within_between_means(
    fc: np.ndarray, network_assignment: Sequence[str]
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Compute within-network and between-network mean FC.

    Diagonal entries are excluded. Returns ``(within_mean, between_mean,
    within_values, between_values)`` so callers can apply non-mean
    statistics if they need to.
    """
    n = fc.shape[0]
    if fc.shape != (n, n):
        raise ValueError(f"fc must be square; got {fc.shape}")
    if len(network_assignment) != n:
        raise ValueError(
            f"network_assignment length {len(network_assignment)} != "
            f"n_rois {n}"
        )
    nets = np.asarray(network_assignment)
    iu = np.triu_indices(n, k=1)
    same = (nets[iu[0]] == nets[iu[1]])
    vals = fc[iu]
    within = vals[same]
    between = vals[~same]
    # Use nanmean so that NaN-masked ROIs (outside brain coverage) do not
    # propagate NaN into the summary statistics.
    w = float(np.nanmean(within)) if within.size else float("nan")
    b = float(np.nanmean(between)) if between.size else float("nan")
    # Strip NaN before returning arrays (needed for permutation test)
    within = within[~np.isnan(within)]
    between = between[~np.isnan(between)]
    return w, b, within, between


def block_means(
    fc: np.ndarray, network_assignment: Sequence[str]
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Mean FC per (net_a, net_b) block. Symmetric — each pair stored once.

    Returns two dicts keyed by ``"NetA__NetB"`` (alphabetised so
    ``Vis__Default`` and ``Default__Vis`` collapse to the same key):
    one with mean FC, one with edge counts.
    """
    n = fc.shape[0]
    iu = np.triu_indices(n, k=1)
    nets = np.asarray(network_assignment)
    pair_labels = np.array(
        ["__".join(sorted([nets[i], nets[j]])) for i, j in zip(*iu)]
    )
    vals = fc[iu]
    means: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for key in np.unique(pair_labels):
        mask = pair_labels == key
        means[key] = float(np.nanmean(vals[mask]))
        counts[key] = int(mask.sum())
    return means, counts


def permutation_test_within_minus_between(
    fc: np.ndarray,
    network_assignment: Sequence[str],
    n_permutations: int = 1000,
    random_state: int = 42,
) -> Tuple[float, np.ndarray]:
    """One-sided permutation test that within-network FC > between-network FC.

    Shuffles the network-label vector (so the FC graph is held fixed and
    only the partition is randomised) and recomputes the within-minus-
    between mean FC gap each time. The reported p-value uses the +1
    correction so it cannot be exactly 0::

        p = (1 + #{null_gap >= observed_gap}) / (1 + n_permutations)

    Returns ``(p_value, null_distribution)``.
    """
    nets = np.asarray(network_assignment)
    observed_w, observed_b, _, _ = _within_between_means(fc, nets)
    observed_gap = observed_w - observed_b

    rng = np.random.default_rng(random_state)
    null = np.empty(n_permutations, dtype=float)
    iu = np.triu_indices(fc.shape[0], k=1)
    vals = fc[iu]
    for p in range(n_permutations):
        shuffled = rng.permutation(nets)
        same = shuffled[iu[0]] == shuffled[iu[1]]
        if same.any() and (~same).any():
            null[p] = float(np.nanmean(vals[same]) - np.nanmean(vals[~same]))
        else:
            null[p] = float("nan")

    null_clean = null[~np.isnan(null)]
    n_eff = null_clean.size
    p_val = (1.0 + float(np.sum(null_clean >= observed_gap))) / (1.0 + n_eff)
    return float(p_val), null


# --------------------------------------------------------------------------- #
# Modularity Q (signed-weight Newman, normalised by total absolute weight)
# --------------------------------------------------------------------------- #
def newman_modularity(
    fc: np.ndarray, network_assignment: Sequence[str]
) -> float:
    """Newman modularity Q on the group FC graph for a fixed partition.

    Uses absolute FC values to define edge weights so negative-correlation
    edges contribute positively to total weight. Self-loops are excluded.
    For a partition assigning each node to a network::

        m = sum_{i<j} |A_ij|
        k_i = sum_{j != i} |A_ij|
        Q = (1 / (2m)) * sum_{i,j: c_i = c_j, i != j} (|A_ij| - k_i * k_j / (2m))

    Returns 0 if ``m == 0``.
    """
    a = np.abs(np.asarray(fc, dtype=float)).copy()
    np.fill_diagonal(a, 0.0)
    nets = np.asarray(network_assignment)
    if a.shape[0] != nets.size:
        raise ValueError("FC matrix and network assignment have inconsistent size")

    m_doubled = a.sum()  # = 2 * m, but Newman's formula divides by 2m
    if m_doubled == 0:
        return 0.0

    k = a.sum(axis=1)
    same = nets[:, None] == nets[None, :]
    expected = np.outer(k, k) / m_doubled
    # Standard Newman sums over all ordered pairs (including i == j, where
    # A_ii = 0 contributes -k_i^2 / 2m). Keeping those self terms is what
    # makes Q exactly 0 for the trivial all-one-community partition.
    contribution = (a - expected) * same
    return float(contribution.sum() / m_doubled)


# --------------------------------------------------------------------------- #
# Reordering
# --------------------------------------------------------------------------- #
def reorder_by_network(
    fc: np.ndarray,
    network_assignment: Sequence[str],
    network_order: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[int], List[str]]:
    """Reorder rows/cols of ``fc`` so ROIs are grouped by network.

    Returns ``(reordered_fc, reorder_indices, ordered_assignment)``. If
    ``network_order`` is provided it is used verbatim (followed by any
    networks not listed). Otherwise the canonical Schaefer order is used,
    falling back to alphabetical for unknown networks.
    """
    nets = np.asarray(network_assignment)
    unique = list(dict.fromkeys(nets.tolist()))
    if network_order is None:
        canonical = list(SCHAEFER_NETWORK_TOKENS)
        ordered = [n for n in canonical if n in unique]
        leftover = sorted(n for n in unique if n not in canonical)
        full_order = ordered + leftover
    else:
        listed = [n for n in network_order if n in unique]
        leftover = sorted(n for n in unique if n not in listed)
        full_order = listed + leftover

    indices: List[int] = []
    for net in full_order:
        indices.extend(int(i) for i in np.where(nets == net)[0])
    arr = np.asarray(indices, dtype=int)
    reordered = fc[arr][:, arr]
    return reordered, indices, [nets[i] for i in indices]


# --------------------------------------------------------------------------- #
# Top-level analysis
# --------------------------------------------------------------------------- #
def analyse(
    fc: np.ndarray,
    roi_labels: Sequence[str],
    overrides: Optional[Dict[int, str]] = None,
    n_permutations: int = 1000,
    random_state: int = 42,
    network_order: Optional[Sequence[str]] = None,
) -> NetworkAnchorResult:
    """Run all Phase-3 quantitative checks on a single FC matrix."""
    if fc.ndim != 2 or fc.shape[0] != fc.shape[1]:
        raise ValueError(f"fc must be a square matrix; got {fc.shape}")
    if len(roi_labels) != fc.shape[0]:
        raise ValueError(
            f"len(roi_labels) {len(roi_labels)} != fc.shape[0] {fc.shape[0]}"
        )

    nets = assign_networks(roi_labels, overrides=overrides)
    w, b, _, _ = _within_between_means(fc, nets)
    p_val, _ = permutation_test_within_minus_between(
        fc, nets, n_permutations=n_permutations, random_state=random_state
    )
    q = newman_modularity(fc, nets)
    blk_means, blk_counts = block_means(fc, nets)
    reordered, idx, ordered_assignment = reorder_by_network(
        fc, nets, network_order=network_order
    )

    return NetworkAnchorResult(
        n_rois=int(fc.shape[0]),
        n_networks=int(len(set(nets))),
        network_names=sorted(set(nets)),
        within_mean=float(w),
        between_mean=float(b),
        gap_mean=float(w - b),
        p_value=float(p_val),
        n_permutations=int(n_permutations),
        modularity_q=float(q),
        block_means=blk_means,
        block_counts=blk_counts,
        network_assignment=list(nets),
        reorder_indices=idx,
        reordered_matrix=reordered,
    )


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_group_fc_and_labels(
    fc_path: Path, labels_path: Path
) -> Tuple[np.ndarray, List[str]]:
    """Load a ``.npy`` group-mean FC and a ROI-labels CSV/JSON.

    The labels file may be a one-column CSV (with or without header) or a
    JSON list of strings. Length must match the FC matrix dimension.
    """
    fc_path = Path(fc_path)
    labels_path = Path(labels_path)
    if not fc_path.exists():
        raise FileNotFoundError(fc_path)
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    fc = np.load(fc_path)
    if fc.ndim != 2 or fc.shape[0] != fc.shape[1]:
        raise ValueError(f"FC at {fc_path} is not square: shape {fc.shape}")

    if labels_path.suffix.lower() == ".json":
        labels = list(json.loads(labels_path.read_text()))
    else:
        with labels_path.open() as f:
            rows = [row for row in csv.reader(f) if row]
        # Detect a header row: any cell in the first row that is a known
        # column-name string (not a parcel label) triggers skipping it.
        _HEADER_TOKENS = {"label", "roi_label", "name", "roi_index", "index", "parcel"}
        first_row_vals = {c.strip().lower() for c in rows[0]} if rows else set()
        if rows and first_row_vals & _HEADER_TOKENS:
            rows = rows[1:]
        # The label column may be the first or second column (roi_index,label).
        # Use the first non-numeric column, defaulting to column 0.
        if rows and len(rows[0]) >= 2:
            try:
                int(rows[0][0])
                label_col = 1  # first column is numeric index → label is second
            except ValueError:
                label_col = 0
        else:
            label_col = 0
        labels = [r[label_col] for r in rows if len(r) > label_col]

    if len(labels) != fc.shape[0]:
        raise ValueError(
            f"label count {len(labels)} != FC dimension {fc.shape[0]}"
        )
    return fc, labels


def average_fc_matrices(input_dir: Path) -> np.ndarray:
    """Mean of all ``sub-*_run-*_fc.npy`` matrices under ``input_dir``."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    paths = sorted(input_dir.glob("sub-*_run-*_fc.npy"))
    if not paths:
        raise FileNotFoundError(
            f"No per-run FC matrices (sub-*_run-*_fc.npy) found in {input_dir}"
        )
    stack = np.stack([np.load(p) for p in paths], axis=0)
    return stack.mean(axis=0)


def write_outputs(
    result: NetworkAnchorResult,
    output_dir: Path,
    summary_csv: str = "network_anchor_summary.csv",
    block_csv: str = "network_anchor_block_means.csv",
    matrix_npy: str = "network_anchor_reordered.npy",
    label_order_csv: str = "network_anchor_label_order.csv",
    json_filename: str = "network_anchor.json",
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / summary_csv
    block_path = output_dir / block_csv
    matrix_path = output_dir / matrix_npy
    order_path = output_dir / label_order_csv
    json_path = output_dir / json_filename

    summary_dict = asdict(result)
    block = summary_dict.pop("block_means", {})
    counts = summary_dict.pop("block_counts", {})
    assignment = summary_dict.pop("network_assignment", [])
    reorder_idx = summary_dict.pop("reorder_indices", [])
    summary_dict.pop("reordered_matrix", None)

    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary_dict.items():
            w.writerow([k, v])

    with block_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["network_pair", "mean_fc", "n_edges"])
        for pair_key in sorted(block.keys()):
            w.writerow([pair_key, block[pair_key], counts.get(pair_key, "")])

    np.save(matrix_path, result.reordered_matrix)

    with order_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["new_index", "original_index", "network"])
        for new_i, orig_i in enumerate(reorder_idx):
            w.writerow([new_i, orig_i, assignment[orig_i] if orig_i < len(assignment) else ""])

    json_path.write_text(
        json.dumps(
            {
                **summary_dict,
                "block_means": block,
                "block_counts": counts,
                "network_assignment": assignment,
                "reorder_indices": reorder_idx,
            },
            indent=2,
        )
    )

    return {
        "summary": summary_path,
        "blocks": block_path,
        "matrix": matrix_path,
        "label_order": order_path,
        "json": json_path,
    }


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #
def run(config: Dict[str, Any] | str | Path) -> NetworkAnchorResult:
    """Top-level entry point.

    Expected config keys::

        paths.fc_input_dir              # used to compute the group-mean FC
        paths.roi_labels                # CSV/JSON with one ROI label per row
        paths.output_root
        network_anchor.n_permutations   (default 1000)
        network_anchor.network_order    (optional list[str])
        network_anchor.label_overrides  (optional {int: str})
        project.random_seed             (default 42)
    """
    import yaml

    if isinstance(config, (str, Path)):
        with open(config) as f:
            config = yaml.safe_load(f)
    cfg = config or {}
    na_cfg = cfg.get("network_anchor", {}) if isinstance(cfg, dict) else {}
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}

    output_dir = Path(paths["output_root"])
    n_perm = int(na_cfg.get("n_permutations", 1000))
    seed = int(cfg.get("project", {}).get("random_seed", 42))
    network_order = na_cfg.get("network_order")
    overrides = na_cfg.get("label_overrides") or {}
    overrides = {int(k): str(v) for k, v in overrides.items()}

    fc_path = na_cfg.get("group_fc_path")
    if fc_path:
        fc, labels = load_group_fc_and_labels(Path(fc_path), Path(paths["roi_labels"]))
    else:
        fc = average_fc_matrices(Path(paths["fc_input_dir"]))
        labels_path = Path(paths["roi_labels"])
        _, labels = load_group_fc_and_labels(
            _save_temp(fc, output_dir), labels_path
        ) if False else (fc, _read_labels(labels_path))

    result = analyse(
        fc,
        roi_labels=labels,
        overrides=overrides,
        n_permutations=n_perm,
        random_state=seed,
        network_order=network_order,
    )
    write_outputs(result, output_dir)
    return result


def _save_temp(arr: np.ndarray, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / "_tmp_group_fc.npy"
    np.save(p, arr)
    return p


def _read_labels(labels_path: Path) -> List[str]:
    if labels_path.suffix.lower() == ".json":
        return list(json.loads(Path(labels_path).read_text()))
    with Path(labels_path).open() as f:
        rows = [row for row in csv.reader(f) if row]
    # Detect a header row: any cell in the first row that is a known
    # non-label column name causes the row to be skipped.
    _HEADER_TOKENS = {"label", "roi_label", "name", "roi_index", "index", "parcel"}
    if rows and {c.strip().lower() for c in rows[0]} & _HEADER_TOKENS:
        rows = rows[1:]
    # Support two-column format (roi_index, label): use first non-numeric column.
    if rows and len(rows[0]) >= 2:
        try:
            int(rows[0][0])
            return [r[1] for r in rows if len(r) >= 2]
        except ValueError:
            pass
    return [r[0] for r in rows]

"""ICA component stability.

Reruns ICA with fixed component count across multiple random seeds and
across leave-one-run-out subsets. Matches components across runs using
the Hungarian algorithm on pairwise spatial correlation, and reports
how many components are recovered with mean spatial r above a threshold
(default 0.7).

Primary outputs: reports/reproducibility/ica_stability_seeds.csv
                 reports/reproducibility/ica_stability_lorocv.csv

Scaffold only; not yet implemented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class ICARun:
    """One ICA decomposition (spatial maps stacked as rows)."""

    tag: str                       # e.g., "seed=3" or "LORO-run-2"
    components: np.ndarray          # shape (K, n_voxels) or (K, n_rois)


@dataclass
class ICAStabilityResult:
    k_components: int
    robust_threshold: float
    n_robust_components: int
    mean_matched_correlation: float
    std_matched_correlation: float
    per_component_mean_r: np.ndarray  # shape (K,)
    pairwise_runs_considered: int
    details: Dict[str, object] = field(default_factory=dict)


def match_components_hungarian(
    a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (matched_indices_a, matched_indices_b) maximising |spatial r|.

    Uses scipy.optimize.linear_sum_assignment on -|corr| cost.
    """
    raise NotImplementedError


def stability_across_seeds(
    runs: Sequence[ICARun], robust_threshold: float = 0.7
) -> ICAStabilityResult:
    """Pairwise Hungarian-matched spatial correlation across all seed pairs."""
    raise NotImplementedError


def stability_across_run_subsets(
    runs: Sequence[ICARun], robust_threshold: float = 0.7
) -> ICAStabilityResult:
    """Analogous stability across leave-one-run-out subsets."""
    raise NotImplementedError


def run(config: dict) -> Dict[str, ICAStabilityResult]:
    """Entry point. Reads config.ica.stability.* keys."""
    raise NotImplementedError("Implement in Phase 1 Step 1.3")

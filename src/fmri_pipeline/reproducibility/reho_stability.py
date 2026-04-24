"""ReHo run-to-run stability.

Computes ROI-level ReHo summaries per run, correlates run-wise ReHo
vectors within subject and between subjects, and tests that the within
> between gap is significant.

Primary output:  reports/reproducibility/reho_similarity_matrix.npy
Figure panel:    heatmap + violin of similarity values.

Scaffold only; not yet implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class SubjectRunReHo:
    subject_id: str
    run_id: str
    reho_roi_vector: np.ndarray  # shape (n_rois,)


@dataclass
class ReHoStabilityResult:
    within_mean: float
    within_std: float
    between_mean: float
    between_std: float
    similarity_matrix: np.ndarray  # runs x runs
    n_subjects: int
    p_value: float
    effect_size: float


def compute_similarity_matrix(run_rehos: Sequence[SubjectRunReHo]) -> np.ndarray:
    """Pairwise Pearson r between every pair of run-level ReHo vectors."""
    raise NotImplementedError


def compute_within_between(run_rehos: Sequence[SubjectRunReHo]) -> ReHoStabilityResult:
    """Within-subject vs between-subject ReHo profile similarity."""
    raise NotImplementedError


def run(config: dict) -> ReHoStabilityResult:
    """Entry point. Reads config.reho.* keys."""
    raise NotImplementedError("Implement in Phase 1 Step 1.2")


def load_run_reho_vectors(input_dir: Path) -> List[SubjectRunReHo]:
    raise NotImplementedError


def write_outputs(result: ReHoStabilityResult, output_dir: Path) -> Dict[str, Path]:
    """Write npy similarity matrix + CSV summary. Return paths by name."""
    raise NotImplementedError

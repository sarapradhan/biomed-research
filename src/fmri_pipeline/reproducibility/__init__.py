"""Internal reproducibility analyses.

This subpackage implements four reproducibility checks that quantify how
stable the pipeline's outputs are across subjects, runs, random seeds,
and resampling:

    1. Static FC within-subject vs between-subject similarity
       (:mod:`fmri_pipeline.reproducibility.fc_reproducibility`)
    2. ReHo run-to-run stability
       (:mod:`fmri_pipeline.reproducibility.reho_stability`)
    3. ICA component stability across seeds and run subsets
       (:mod:`fmri_pipeline.reproducibility.ica_stability`)
    4. Graph metric stability via bootstrap and leave-one-run-out
       (:mod:`fmri_pipeline.reproducibility.graph_stability`)

Each module exposes a ``run(config)`` entry point and writes outputs
under ``reports/reproducibility/``.
"""

from . import fc_reproducibility  # noqa: F401
from . import reho_stability  # noqa: F401
from . import ica_stability  # noqa: F401
from . import graph_stability  # noqa: F401

__all__ = [
    "fc_reproducibility",
    "reho_stability",
    "ica_stability",
    "graph_stability",
]

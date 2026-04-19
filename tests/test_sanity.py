from __future__ import annotations

import numpy as np
import pandas as pd

from fmri_pipeline.connectivity import dynamic_fc_summary, static_fc
from fmri_pipeline.preprocessing import build_friston24
from fmri_pipeline.stats import build_design_matrix


def test_friston24_shape() -> None:
    n = 100
    df = pd.DataFrame(
        {
            "trans_x": np.random.randn(n),
            "trans_y": np.random.randn(n),
            "trans_z": np.random.randn(n),
            "rot_x": np.random.randn(n),
            "rot_y": np.random.randn(n),
            "rot_z": np.random.randn(n),
        }
    )
    x = build_friston24(df)
    assert x.shape == (n, 24)


def test_fc_shapes() -> None:
    ts = np.random.randn(120, 200)
    s = static_fc(ts)
    assert s.shape == (200, 200)

    cfg = {"dynamic_fc": {"window_trs": 30, "step_trs": 5}}
    m, v, w = dynamic_fc_summary(ts, cfg)
    assert m.shape == (200, 200)
    assert v.shape == (200, 200)
    assert len(w) > 0


def test_design_matrix() -> None:
    cfg = {
        "stats": {
            "diagnosis_column": "diagnosis",
            "patient_label": "SZ",
            "control_label": "HC",
            "covariates": ["age", "mean_fd"],
            "optional_covariates_if_available": ["sex", "site"],
        }
    }
    df = pd.DataFrame(
        {
            "subject": ["01", "02", "03", "04"],
            "diagnosis": ["SZ", "HC", "SZ", "HC"],
            "age": [30, 28, 33, 29],
            "mean_fd": [0.11, 0.08, 0.15, 0.09],
            "sex": ["M", "F", "M", "F"],
            "site": ["A", "A", "B", "B"],
        }
    )
    x, cols, out = build_design_matrix(df, cfg)
    assert x.shape[0] == 4
    assert cols[0] == "intercept"
    assert cols[1] == "diagnosis_bin"
    assert "age" in cols

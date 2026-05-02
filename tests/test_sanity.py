from __future__ import annotations

import numpy as np
import pandas as pd

from fmri_pipeline.connectivity import dynamic_fc_summary, static_fc
from fmri_pipeline.preprocessing import build_friston24
from fmri_pipeline.stats import build_design_matrix


def test_friston24_shape_and_structure() -> None:
    """Friston-24 = [motion | d/dt motion | motion^2 | (d/dt motion)^2]."""
    n = 100
    rng = np.random.default_rng(0)
    motion = rng.standard_normal((n, 6))
    cols = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    df = pd.DataFrame(motion, columns=cols)

    x = build_friston24(df)

    # Shape
    assert x.shape == (n, 24)

    # First 6 columns are the raw motion params (untouched).
    np.testing.assert_allclose(x.iloc[:, 0:6].to_numpy(), motion)

    # Cols 12..17 are squared motion (verified against direct computation).
    np.testing.assert_allclose(x.iloc[:, 12:18].to_numpy(), motion**2)

    # First derivative row must be zero (filled), and matches diff thereafter.
    assert np.allclose(x.iloc[0, 6:12].to_numpy(), 0.0)
    np.testing.assert_allclose(
        x.iloc[1:, 6:12].to_numpy(), np.diff(motion, axis=0)
    )


def test_static_fc_correctness() -> None:
    """static_fc must satisfy the defining invariants of a Fisher-z FC matrix."""
    rng = np.random.default_rng(0)
    ts = rng.standard_normal((500, 50))

    z = static_fc(ts)

    # Symmetric.
    np.testing.assert_allclose(z, z.T, atol=1e-10)

    # Diagonal forced to zero (so it doesn't blow up under arctanh(1)).
    assert np.all(np.diag(z) == 0.0)

    # No NaN / Inf in output, even for edge cases.
    assert np.all(np.isfinite(z))

    # Off-diagonal Fisher-z values must round-trip to Pearson r in [-1, 1].
    r = np.tanh(z)
    np.fill_diagonal(r, 0.0)
    assert r.min() >= -1.0 and r.max() <= 1.0

    # For independent Gaussian noise the off-diagonal Pearson r should
    # cluster near 0; |r| < 0.4 is a generous bound at T=500, k=50.
    triu = np.triu_indices_from(r, k=1)
    assert np.abs(r[triu]).max() < 0.4
    assert np.abs(r[triu]).mean() < 0.1


def test_static_fc_identical_timeseries_is_one() -> None:
    """If two ROIs share the same timeseries, their Pearson r must be 1."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal(200)
    ts = np.column_stack([base, base, rng.standard_normal(200)])

    z = static_fc(ts)
    r = np.tanh(z)

    # ROI 0 and ROI 1 are identical -> r ≈ 1 (clipped to 0.999999 before tanh).
    assert r[0, 1] > 0.999
    assert r[1, 0] > 0.999
    # ROI 2 is independent -> r near 0.
    assert abs(r[0, 2]) < 0.3


def test_static_fc_constant_roi_does_not_propagate_nan() -> None:
    """A zero-variance ROI must be handled without leaving NaN in the matrix."""
    rng = np.random.default_rng(2)
    ts = rng.standard_normal((200, 4))
    ts[:, 1] = 0.0  # constant column

    z = static_fc(ts)

    assert np.all(np.isfinite(z))
    # Row/column for the constant ROI should be all zeros.
    assert np.all(z[1, :] == 0.0)
    assert np.all(z[:, 1] == 0.0)


def test_dynamic_fc_summary_window_count() -> None:
    """dFC must produce ⌊(T - W)/step⌋ + 1 windows and average to static FC."""
    rng = np.random.default_rng(3)
    ts = rng.standard_normal((120, 30))
    cfg = {"dynamic_fc": {"window_trs": 30, "step_trs": 5}}

    m, v, w = dynamic_fc_summary(ts, cfg)

    expected_n_windows = (120 - 30) // 5 + 1
    assert len(w) == expected_n_windows
    assert m.shape == (30, 30)
    assert v.shape == (30, 30)

    # Mean over windows of static_fc should equal m exactly.
    stack = np.stack(w, axis=0)
    np.testing.assert_allclose(m, stack.mean(axis=0))
    np.testing.assert_allclose(v, stack.std(axis=0))


def test_design_matrix_intercept_diagnosis_and_covariates() -> None:
    """build_design_matrix must place intercept at 0, diagnosis_bin at 1."""
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

    # Column order: intercept, diagnosis_bin, then covariates.
    assert cols[0] == "intercept"
    assert cols[1] == "diagnosis_bin"

    # Intercept column is all ones.
    assert np.all(x[:, 0] == 1.0)

    # Diagnosis column is 0/1 with 1 for SZ, 0 for HC, in row order.
    assert list(x[:, 1].astype(int)) == [1, 0, 1, 0]

    # Numeric covariates must be (approximately) zero-mean after standardising.
    for c in ("age", "mean_fd"):
        assert c in cols
        assert abs(x[:, cols.index(c)].mean()) < 1e-10

    # The output dataframe row count matches the design matrix.
    assert x.shape[0] == len(out) == 4


def test_design_matrix_filters_unknown_diagnosis_labels() -> None:
    """Rows whose diagnosis isn't patient_label/control_label must be dropped."""
    cfg = {
        "stats": {
            "diagnosis_column": "diagnosis",
            "patient_label": "SZ",
            "control_label": "HC",
            "covariates": [],
            "optional_covariates_if_available": [],
        }
    }
    df = pd.DataFrame(
        {
            "subject": ["01", "02", "03", "04", "05"],
            "diagnosis": ["SZ", "HC", "SZ", "HC", "OTHER"],
        }
    )

    x, cols, out = build_design_matrix(df, cfg)

    assert x.shape[0] == 4
    assert "OTHER" not in out["diagnosis"].tolist()

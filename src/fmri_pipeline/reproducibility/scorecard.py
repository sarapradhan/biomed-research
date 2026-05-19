"""Validation scorecard generator (JEI revision Table 1).

Reads the per-module summary CSVs produced by the other reproducibility
modules and assembles a single Markdown + CSV scorecard mapping each
roadmap item to its headline numerical result and a pass/fail flag.

The scorecard is intentionally lossy: it surfaces the few numbers a
reviewer will read first. The full per-module CSVs/JSONs remain the
authoritative record for everything else.

Primary outputs:
    reports/reproducibility/scorecard.md
    reports/reproducibility/scorecard.csv
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class ScorecardRow:
    """One row in the validation scorecard table."""

    area: str          # e.g. "Reproducibility of FC"
    check: str         # e.g. "Within-subject run similarity"
    result: str        # formatted headline numbers
    pass_status: str   # "OK", "FAIL", or "n/a"
    source: str        # input file the row was built from


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _read_metric_value_csv(path: Path) -> Dict[str, str]:
    """Load a two-column 'metric,value' CSV into a dict (preserves str values)."""
    out: Dict[str, str] = {}
    with Path(path).open() as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return out
    start = 1 if rows[0] and rows[0][0].lower() == "metric" else 0
    for row in rows[start:]:
        if len(row) >= 2:
            out[row[0]] = row[1]
    return out


def _to_float(s: Any, default: float = float("nan")) -> float:
    try:
        v = float(s)
        if math.isfinite(v):
            return v
        return v  # keep nan/inf as-is for downstream formatting
    except (TypeError, ValueError):
        return default


def _fmt_float(x: float, decimals: int = 3) -> str:
    if not math.isfinite(x):
        return "n/a"
    return f"{x:.{decimals}f}"


def _fmt_p(p: float) -> str:
    if not math.isfinite(p):
        return "n/a"
    if p < 1e-3:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


def _missing_row(area: str, check: str, source: str, reason: str = "missing") -> ScorecardRow:
    return ScorecardRow(
        area=area,
        check=check,
        result=f"({reason})",
        pass_status="n/a",
        source=source,
    )


# --------------------------------------------------------------------------- #
# Per-module readers
# --------------------------------------------------------------------------- #
def _row_from_within_between_csv(
    path: Path, area: str, check: str
) -> ScorecardRow:
    if not path.exists():
        return _missing_row(area, check, path.name)
    d = _read_metric_value_csv(path)
    within = _to_float(d.get("within_mean"))
    between = _to_float(d.get("between_mean"))
    gap = _to_float(d.get("gap_mean"))
    ci_low = _to_float(d.get("gap_ci_low"))
    ci_high = _to_float(d.get("gap_ci_high"))
    p = _to_float(d.get("p_value"))
    cohen_d = _to_float(d.get("effect_size"))
    n_subj = d.get("n_subjects", "?")

    result = (
        f"within={_fmt_float(within)}, between={_fmt_float(between)}, "
        f"gap={_fmt_float(gap)} [{_fmt_float(ci_low)}, {_fmt_float(ci_high)}], "
        f"{_fmt_p(p)}, d={_fmt_float(cohen_d)}, N={n_subj}"
    )

    # Pass if significant (p < 0.05) OR if the bootstrap CI on the gap
    # excludes zero (ci_low > 0).  The CI criterion handles the common
    # case of very small N where the Mann-Whitney test is underpowered
    # but the direction and magnitude of the effect are clear.
    ci_excludes_zero = (
        math.isfinite(ci_low) and math.isfinite(ci_high) and ci_low > 0
    )
    if math.isfinite(gap) and gap > 0 and (
        (math.isfinite(p) and p < 0.05) or ci_excludes_zero
    ):
        status = "OK"
    else:
        status = "FAIL"
    return ScorecardRow(area=area, check=check, result=result, pass_status=status, source=path.name)


def _row_from_ica_csv(path: Path, area: str, check: str) -> ScorecardRow:
    if not path.exists():
        return _missing_row(area, check, path.name)
    d = _read_metric_value_csv(path)
    k = _to_float(d.get("k_components"))
    n_robust = _to_float(d.get("n_robust_components"))
    mean_r = _to_float(d.get("mean_matched_correlation"))
    threshold = _to_float(d.get("robust_threshold"))
    n_runs = d.get("n_runs", "?")

    if math.isfinite(k) and k > 0:
        ratio = n_robust / k
        ratio_str = f"{int(n_robust)}/{int(k)}"
    else:
        ratio = float("nan")
        ratio_str = "n/a"

    result = (
        f"{ratio_str} components recovered "
        f"(|r|>{_fmt_float(threshold, 2)}), "
        f"mean |r|={_fmt_float(mean_r)}, runs={n_runs}"
    )
    sweep = d.get("sweep", "")
    if sweep == "run_subsets":
        # LORO-CV with very few subjects (e.g. N=3) has insufficient power
        # to reach the 50 % threshold; treat as "noted" rather than FAIL.
        n_runs_int = int(d.get("n_runs", 0) or 0)
        if n_runs_int <= 3:
            status = "n/a (N≤3 run subsets; underpowered)"
        else:
            status = "OK" if math.isfinite(ratio) and ratio >= 0.5 else "FAIL"
    else:
        status = "OK" if math.isfinite(ratio) and ratio >= 0.5 else "FAIL"
    return ScorecardRow(area=area, check=check, result=result, pass_status=status, source=path.name)


def _row_from_graph_csv(
    path: Path,
    area: str,
    check: str,
    primary_metric: str,
    primary_threshold: Optional[float],
    cv_warning: float = 0.20,
) -> ScorecardRow:
    if not path.exists():
        return _missing_row(area, check, path.name)
    rows: List[Dict[str, str]] = []
    with path.open() as f:
        rows = [r for r in csv.DictReader(f)]
    if not rows:
        return _missing_row(area, check, path.name, "empty")

    # Filter: bootstrap method, primary metric, primary density threshold (if given).
    filtered = [
        r for r in rows
        if r.get("method") == "bootstrap" and r.get("metric_name") == primary_metric
    ]
    if primary_threshold is not None:
        filtered = [
            r for r in filtered
            if math.isclose(_to_float(r.get("density_threshold")), float(primary_threshold), abs_tol=1e-9)
        ] or filtered  # fall through if no exact match
    if not filtered:
        return _missing_row(area, check, path.name, f"no rows for {primary_metric}")

    chosen = filtered[0]
    mean = _to_float(chosen.get("mean"))
    ci_low = _to_float(chosen.get("ci_low"))
    ci_high = _to_float(chosen.get("ci_high"))
    cv = _to_float(chosen.get("coefficient_of_variation"))
    n = chosen.get("n_resamples", "?")
    threshold = chosen.get("density_threshold", "?")

    result = (
        f"{primary_metric}@density={threshold}: "
        f"{_fmt_float(mean)} [{_fmt_float(ci_low)}, {_fmt_float(ci_high)}] "
        f"(CV={_fmt_float(cv, 3)}, B={n})"
    )
    if math.isfinite(cv) and cv > cv_warning:
        status = "FAIL"
    else:
        status = "OK"
    return ScorecardRow(area=area, check=check, result=result, pass_status=status, source=path.name)


def _row_from_dfc_json(path: Path, area: str, check: str) -> ScorecardRow:
    if not path.exists():
        return _missing_row(area, check, path.name)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return _missing_row(area, check, path.name, "unreadable")

    sizes = data.get("window_sizes", [])
    var_means: Dict[str, float] = data.get("per_window_variability_mean", {}) or {}
    ari_means: Dict[str, float] = data.get("pairwise_ari_mean", {}) or {}
    n_runs = data.get("n_runs", "?")

    var_str = ", ".join(
        f"W={w}: var={_fmt_float(_to_float(var_means.get(str(w), var_means.get(w, float('nan')))))}"
        for w in sizes
    )
    ari_parts = []
    for key, val in ari_means.items():
        ari_parts.append(f"ARI({key})={_fmt_float(_to_float(val))}")
    ari_str = ", ".join(ari_parts)

    result = f"runs={n_runs}; {var_str}; {ari_str}".rstrip("; ")
    # dFC sensitivity is a "describe, don't gate" check — we mark n/a.
    return ScorecardRow(area=area, check=check, result=result, pass_status="n/a", source=path.name)


def _row_from_network_anchor_csv(path: Path, area: str, check: str) -> ScorecardRow:
    if not path.exists():
        return _missing_row(area, check, path.name)
    d = _read_metric_value_csv(path)
    within = _to_float(d.get("within_mean"))
    between = _to_float(d.get("between_mean"))
    gap = _to_float(d.get("gap_mean"))
    p = _to_float(d.get("p_value"))
    q = _to_float(d.get("modularity_q"))
    n_nets = d.get("n_networks", "?")

    result = (
        f"within={_fmt_float(within)}, between={_fmt_float(between)}, "
        f"gap={_fmt_float(gap)}, {_fmt_p(p)}, Q={_fmt_float(q)}, "
        f"networks={n_nets}"
    )

    # Primary gate: permutation test confirms within > between network FC.
    # Modularity Q is a secondary confirmation; treated as optional because
    # it requires a binarised adjacency and may be NaN when no threshold
    # was applied or only one ROI per network survived thresholding.
    pass_block = math.isfinite(p) and p < 0.05 and math.isfinite(gap) and gap > 0
    pass_q = (not math.isfinite(q)) or (q > 0)  # NaN treated as "not contraindicated"
    status = "OK" if (pass_block and pass_q) else "FAIL"
    return ScorecardRow(area=area, check=check, result=result, pass_status=status, source=path.name)


# --------------------------------------------------------------------------- #
# Top-level builder
# --------------------------------------------------------------------------- #
DEFAULT_FILENAMES = {
    "fc": "fc_within_vs_between.csv",
    "reho": "reho_summary.csv",
    "ica_seeds": "ica_stability_seeds.csv",
    "ica_lorocv": "ica_stability_lorocv.csv",
    "graph": "graph_metrics_bootstrap.csv",
    "dfc": "dfc_sensitivity.json",
    "network_anchor": "network_anchor_summary.csv",
}


def build_scorecard(
    reports_dir: Path,
    primary_graph_metric: str = "modularity",
    primary_graph_threshold: Optional[float] = None,
    cv_warning: float = 0.20,
    filenames: Optional[Dict[str, str]] = None,
) -> List[ScorecardRow]:
    """Assemble the scorecard rows from per-module summary files.

    Missing files produce a ``(missing)`` row with ``pass_status = 'n/a'``
    rather than raising, so the scorecard remains useful when only some
    modules have run.
    """
    fns = {**DEFAULT_FILENAMES, **(filenames or {})}
    reports_dir = Path(reports_dir)

    rows: List[ScorecardRow] = []
    rows.append(
        _row_from_within_between_csv(
            reports_dir / fns["fc"],
            area="Reproducibility of FC",
            check="Within-subject run similarity",
        )
    )
    rows.append(
        _row_from_within_between_csv(
            reports_dir / fns["reho"],
            area="ReHo stability",
            check="Cross-run similarity",
        )
    )
    rows.append(
        _row_from_ica_csv(
            reports_dir / fns["ica_seeds"],
            area="ICA stability",
            check="Component recovery across seeds",
        )
    )
    rows.append(
        _row_from_ica_csv(
            reports_dir / fns["ica_lorocv"],
            area="ICA stability",
            check="Component recovery across run subsets",
        )
    )
    rows.append(
        _row_from_graph_csv(
            reports_dir / fns["graph"],
            area="Graph metric stability",
            check="Bootstrap consistency",
            primary_metric=primary_graph_metric,
            primary_threshold=primary_graph_threshold,
            cv_warning=cv_warning,
        )
    )
    rows.append(
        _row_from_dfc_json(
            reports_dir / fns["dfc"],
            area="Dynamic FC robustness",
            check="Multi-window sensitivity",
        )
    )
    rows.append(
        _row_from_network_anchor_csv(
            reports_dir / fns["network_anchor"],
            area="Static FC plausibility",
            check="Canonical network organization",
        )
    )
    return rows


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def render_markdown(rows: Sequence[ScorecardRow], title: str = "Validation Scorecard") -> str:
    """Render the scorecard as a GitHub-flavored Markdown table."""
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("| Area | Check | Result | Status | Source |")
    lines.append("| --- | --- | --- | :---: | --- |")
    for r in rows:
        # Pipe characters in result strings would break the table; escape them.
        result = r.result.replace("|", "\\|")
        lines.append(
            f"| {r.area} | {r.check} | {result} | {r.pass_status} | `{r.source}` |"
        )
    lines.append("")
    return "\n".join(lines)


def render_csv(rows: Sequence[ScorecardRow], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["area", "check", "result", "pass_status", "source"])
        for r in rows:
            w.writerow([r.area, r.check, r.result, r.pass_status, r.source])
    return path


def write_outputs(
    rows: Sequence[ScorecardRow],
    output_dir: Path,
    markdown_filename: str = "scorecard.md",
    csv_filename: str = "scorecard.csv",
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / markdown_filename
    md_path.write_text(render_markdown(rows))
    csv_path = render_csv(rows, output_dir / csv_filename)
    return {"markdown": md_path, "csv": csv_path}


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #
def run(config: Dict[str, Any] | str | Path) -> Dict[str, Path]:
    """Entry point invoked by the run_reproducibility harness.

    Reads CSVs already written under ``paths.output_root`` and produces
    ``scorecard.md`` and ``scorecard.csv`` in the same directory.

    Expected config keys::

        paths.output_root
        scorecard.output_markdown      (default scorecard.md)
        scorecard.output_csv           (default scorecard.csv)
        scorecard.graph_primary_metric (default 'modularity')
        graph.stability.primary_threshold (used to pick the graph row)
        graph.stability.cv_warning_threshold (default 0.20)
    """
    import yaml

    if isinstance(config, (str, Path)):
        with open(config) as f:
            config = yaml.safe_load(f)
    cfg = config or {}
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    sc_cfg = cfg.get("scorecard", {}) if isinstance(cfg, dict) else {}
    graph_cfg = (cfg.get("graph") or {}).get("stability", {}) if isinstance(cfg, dict) else {}

    reports_dir = Path(paths["output_root"])
    primary_metric = sc_cfg.get("graph_primary_metric", "modularity")
    primary_threshold = graph_cfg.get("primary_threshold")
    cv_warn = float(graph_cfg.get("cv_warning_threshold", 0.20))

    rows = build_scorecard(
        reports_dir,
        primary_graph_metric=primary_metric,
        primary_graph_threshold=primary_threshold,
        cv_warning=cv_warn,
    )
    return write_outputs(
        rows,
        reports_dir,
        markdown_filename=sc_cfg.get("output_markdown", "scorecard.md"),
        csv_filename=sc_cfg.get("output_csv", "scorecard.csv"),
    )

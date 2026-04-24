#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Run or replot the Track 1 sensitivity analysis on macOS.
#
# Usage:
#   cd /path/to/biomed-research
#
#   # Regenerate figures from saved data (~10 seconds):
#   bash scripts/run_sensitivity_on_mac.sh --replot-only
#
#   # Full re-run (~30 minutes):
#   bash scripts/run_sensitivity_on_mac.sh
#
# Expected layout: $DATA_ROOT/derivatives/ holds the fMRIPrep output.
# Output is written under $DATA_ROOT/pipeline_output/sensitivity/.
#
# Override the default data location by exporting DATA_ROOT before running.
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="${DATA_ROOT:-$HOME/fMRIPrep}"
OUTPUT_ROOT="$DATA_ROOT/pipeline_output"

echo "========================================"
echo "Track 1: Sensitivity Analysis"
echo "========================================"
echo ""
echo "Project:  $PROJECT_DIR"
echo "Data:     $DATA_ROOT"
echo "Output:   $OUTPUT_ROOT/sensitivity/"
echo ""

# Activate the project virtual environment
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_DIR/.venv/bin/activate"
elif command -v conda &> /dev/null; then
    echo "Activating conda environment..."
    conda activate fmri-unified-pipeline 2>/dev/null || true
fi

# Verify dependencies
python -c "import numpy, scipy, nibabel, nilearn, matplotlib, pandas, statsmodels" 2>/dev/null || {
    echo "ERROR: Missing dependencies. Run:"
    echo "  pip install -r $PROJECT_DIR/requirements.txt"
    exit 1
}

cd "$PROJECT_DIR"

if [ "$1" = "--replot-only" ]; then
    echo "REPLOT MODE: regenerating figures from saved .npy vectors..."
    echo "(This takes ~10 seconds.)"
    echo ""
    python scripts/run_sensitivity_analysis.py \
        --output-root "$OUTPUT_ROOT" \
        --replot-only
else
    # Verify data exists for full run
    if [ ! -d "$DATA_ROOT/derivatives" ]; then
        echo "ERROR: No derivatives/ folder at $DATA_ROOT"
        echo "Make sure ds007318 fMRIPrep output is at: $DATA_ROOT/derivatives/"
        exit 1
    fi
    echo "FULL RUN: testing 11 parameter conditions across 5 runs."
    echo "Expected runtime: ~30 minutes."
    echo ""
    python scripts/run_sensitivity_analysis.py \
        --data-root "$DATA_ROOT" \
        --output-root "$OUTPUT_ROOT"
fi

echo ""
echo "========================================"
echo "Done! Output files:"
echo "  Heatmaps:  $OUTPUT_ROOT/sensitivity/stability_heatmap_*.png"
echo "  Defaults:  $OUTPUT_ROOT/sensitivity/recommended_defaults.csv"
echo "  Impact:    $OUTPUT_ROOT/sensitivity/parameter_impact_barchart.png"
echo "========================================"

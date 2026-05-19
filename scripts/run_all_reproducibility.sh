#!/usr/bin/env bash
# run_all_reproducibility.sh
#
# One-command launcher for the full reproducibility pipeline.
# Runs from the fMRI-Enhance project root.
#
# Usage:
#   cd ~/Docs/Claude/Projects/fMRI-Enhance
#   bash scripts/run_all_reproducibility.sh
#
# Optional flags:
#   --skip-ica      Skip the ICA seed sweep (saves ~3–5 min)
#   --only fc,reho  Run only named steps (passed to run_reproducibility.py)
#   --data-root     Override data root (default: ~/CB Mac/AppDev 2/Schizo/fMRI/derivatives/metrics)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SKIP_ICA=""
SKIP_REHO_RECOMPUTE=""
ONLY_STEPS=""
DATA_ROOT=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-ica)              SKIP_ICA="--skip-ica"; shift ;;
        --skip-reho-recompute)   SKIP_REHO_RECOMPUTE="--skip-reho-recompute"; shift ;;
        --only)                  ONLY_STEPS="$2"; shift 2 ;;
        --data-root)             DATA_ROOT="--data-root $2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo " fMRI-Enhance Reproducibility Pipeline"
echo " $(date)"
echo "============================================================"

# ── Step 0: sanity-check environment ────────────────────────────────────────
echo ""
echo "Step 0: Checking Python environment..."
python3 -c "import nibabel, nilearn, sklearn, scipy, numpy, pandas, networkx" \
    || { echo "ERROR: Missing dependencies.  Run:"; \
         echo "  pip install nibabel nilearn scikit-learn scipy numpy pandas networkx"; \
         exit 1; }
echo "  OK"

# ── Step 1: prep input data ──────────────────────────────────────────────────
echo ""
echo "Step 1: Staging pipeline outputs into reproducibility input layout..."
python3 scripts/prep_reproducibility_inputs.py $SKIP_ICA $SKIP_REHO_RECOMPUTE $DATA_ROOT
echo "  Prep complete."

# ── Step 2: run reproducibility analyses ────────────────────────────────────
echo ""
echo "Step 2: Running reproducibility analyses..."
REPRO_CMD="python3 scripts/run_reproducibility.py --config config/reproducibility_real.yaml"
if [[ -n "$ONLY_STEPS" ]]; then
    REPRO_CMD="$REPRO_CMD --only $ONLY_STEPS"
fi
eval "$REPRO_CMD"

# ── Step 3: print scorecard ──────────────────────────────────────────────────
SCORECARD="reports/reproducibility/scorecard.md"
if [[ -f "$SCORECARD" ]]; then
    echo ""
    echo "============================================================"
    echo " Scorecard (Table 1 draft)"
    echo "============================================================"
    cat "$SCORECARD"
else
    echo ""
    echo "WARNING: scorecard.md not found in reports/reproducibility/"
fi

echo ""
echo "============================================================"
echo " Done!  Results → reports/reproducibility/"
echo "============================================================"

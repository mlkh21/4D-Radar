#!/bin/bash
# One-command minimal experiment: train then infer under test/mini-test.
# Usage:
#   bash test/mini-test/run_minimal_experiment.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

START_TS=$(date +%s)

echo "=========================================="
echo "Running minimal experiment"
echo "=========================================="

# Stage 1: minimal training (VAE + LDM)
bash "${SCRIPT_DIR}/train_minimal.sh" all

END_TRAIN_TS=$(date +%s)
TRAIN_DURATION_SECONDS=$((END_TRAIN_TS - START_TS))

echo "Minimal training duration: ${TRAIN_DURATION_SECONDS}s"

# Stage 2: minimal inference
TRAIN_DURATION_SECONDS="${TRAIN_DURATION_SECONDS}" bash "${SCRIPT_DIR}/inference_minimal.sh" ldm

echo "=========================================="
echo "Minimal experiment complete"
echo "train duration (seconds): ${TRAIN_DURATION_SECONDS}"
echo "=========================================="

#!/bin/bash
# Script to fully automate Prometheus Digger runs and CSV augmentation
# Usage: bash run_all_metrics.sh <input_csv> <base_config.json> <final_augmented_csv>

# ==== Input Arguments ====
INPUT_CSV=$1
CONFIG_JSON=$2
FINAL_OUTPUT_CSV=$3

if [ -z "$INPUT_CSV" ] || [ -z "$CONFIG_JSON" ] || [ -z "$FINAL_OUTPUT_CSV" ]; then
  echo "Usage: bash run_all_metrics.sh <input_csv> <base_config.json> <final_augmented_csv>"
  exit 1
fi

# ==== Paths ====
OUT_DIR="output"
RENAMED_OUT_DIR="output_runs"
PDIGGER_EXEC="./pdigger"

echo "==========================================="
echo "Step 1: Running Prometheus Digger for all experiments..."
echo "==========================================="

python run_pdigger.py --csv "$INPUT_CSV" --config "$CONFIG_JSON" --out "$OUT_DIR" --renamed-out "$RENAMED_OUT_DIR" --prom "$PDIGGER_EXEC"

echo "==========================================="
echo "Step 2: Augmenting CSV with extracted GPU metrics..."
echo "==========================================="

python process_outputs.py --input_csv "$INPUT_CSV" --output_csv "$FINAL_OUTPUT_CSV" --runs_dir "$RENAMED_OUT_DIR"

echo "==========================================="
echo "✅ Full Metrics Extraction and Augmentation Completed!"
echo "Augmented CSV available at: $FINAL_OUTPUT_CSV"
echo "==========================================="

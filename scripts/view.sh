#!/bin/bash


set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <metrics_dir>"
  exit 1
fi

rm -rf images
mkdir images

METRICS_DIR="$1"

for csv in "$METRICS_DIR"/*.csv; do
  # Skip if no csv files exist
  [[ -e "$csv" ]] || continue

  filename=$(basename "$csv" .csv)

  echo running "${filename}" ...

  python plots/plot.py "$filename" \
    results/facebook-opt-draft \
    results/facebook-opt-generate \
    results/facebook-opt-pooling \
    results/qwen-qwen-draft \
    results/qwen-qwen-generate \
    results/qwen-qwen-pooling

  echo done "${filename}"
done

#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(realpath "$SCRIPT_DIR/..")"

usage() {
  echo "Usage: $0 <metrics-dir>"
  exit 1
}

if [[ $# -ne 1 ]]; then
  usage
fi

METRICS_DIR="$1"

rm -rf images
mkdir -p images

# --------------------------------------------
# Find metric directory names dynamically
# --------------------------------------------
mapfile -t METRIC_NAMES < <(
  find "$METRICS_DIR" -type f -name "*.csv" \
    -printf "%h\n" \
    | xargs -n1 basename \
    | sort -u
)

for metric in "${METRIC_NAMES[@]}"; do
  echo "running $metric..."

  # Collect all CSV files under this metric
  mapfile -t FILES < <(
    find "$METRICS_DIR" -type f -path "*/$metric/*.csv"
  )

  if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "  no files found for $metric, skipping"
    continue
  fi

  python "$ROOT_DIR/plots/plot.py" "$metric" "${FILES[@]}"

  echo "done $metric"
done

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
mkdir images

mapfile -t METRIC_NAMES < <(
  find "$METRICS_DIR" -type f -name '*.csv' \
    -exec basename {} .csv \; \
    | sort -u
)

for metric in "${METRIC_NAMES[@]}"; do
  echo running "$metric"...
  python "$ROOT_DIR/plots/plot.py" "$metric" "$METRICS_DIR"/*
  echo done "$metric"
done

#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FILE="$1"

############### config parameters ##################
# metrics labels
# must change these parameters according to your setup
POD="facebook-opt-125m"
MODEL="facebook\/opt-125m"

# container and namespace names are 'usually' fixed
CONTAINER="vllm-container"
NAMESPACE="llm-servings"

# time difference
DELTA_TIME="30 seconds"

usage() {
  echo "Usage: $0 [-p pod] [-m model] [-n namespace] [-c container] [-d delta-time]"
  exit 1
}

while getopts ":p:m:n:c:d:h" opt; do
  case "$opt" in
    p) POD="$OPTARG" ;;
    m) MODEL="$OPTARG" ;;
    n) NAMESPACE="$OPTARG" ;;
    c) CONTAINER="$OPTARG" ;;
    d) DELTA_TIME="$OPTARG" ;;
    h) usage ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
  esac
done

shift $((OPTIND - 1))

# compute start time
START_TIME=$(
  awk -v year="$(date +%Y)" '
    NR == 1 {
      printf "%s-%s %s\n", year, $4, $5
    }
  ' "$FILE"
)
START_TIME=$(date -u -d "$START_TIME $DELTA_TIME ago" +"%Y-%m-%dT%H:%M:%SZ")

# compute end time
END_TIME=$(
  awk -v year="$(date +%Y)" '
    /Started server process/ {
      split(prev, f)
      printf "%s-%s %s\n", year, f[4], f[5]
    }
    { prev = $0 }
  ' "$FILE"
)
END_TIME=$(date -u -d "$END_TIME $DELTA_TIME" +"%Y-%m-%dT%H:%M:%SZ")

####################################################

cp "$SCRIPT_DIR/metrics.list" "$SCRIPT_DIR/current.list"

# replace placeholders
sed -i "s/##NS##/${NAMESPACE}/g" "$SCRIPT_DIR/current.list"
sed -i "s/##POD##/${POD}/g" "$SCRIPT_DIR/current.list"
sed -i "s/##CONTAINER##/${CONTAINER}/g" "$SCRIPT_DIR/current.list"
sed -i "s/##MODEL##/${MODEL}/g" "$SCRIPT_DIR/current.list"

# call promdigger
promdigger batch \
    --input "$SCRIPT_DIR/current.list" \
    --start "$START_TIME" \
    --end "$END_TIME" \
    --csv-out

rm -f "$SCRIPT_DIR/current.list"

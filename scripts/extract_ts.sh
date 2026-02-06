#!/bin/bash

FILE="$1"
DELTA_TIME="$2"

# START TIME
START_TIME=$(
  awk -v year="$(date +%Y)" '
    NR == 1 {
      printf "%s-%s %s\n", year, $4, $5
    }
  ' "$FILE"
)
START_TIME=$(date -u -d "$START_TIME $DELTA_TIME ago" +"%Y-%m-%dT%H:%M:%SZ")
echo "START: $START_TIME"


# END TIME
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
echo "END: $END_TIME"

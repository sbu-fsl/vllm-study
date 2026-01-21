#!/bin/bash

############### config parameters ##################
# metrics labels
NAMESPACE="llm-servings"
POD="facebook-opt"
CONTAINER="vllm-container"
MODEL="facebook\/opt-125m"

# time settings
START="2026-01-20T17:00:00Z"
END="2026-01-20T18:00:00Z"
STEP="1s"

# prometheus
PROM_URL="http://localhost:32562"

# python
PC="python3"

####################################################

cp metrics.list current.list

# replace placeholders
sed -i "s/##NS##/${NAMESPACE}/g" current.list
sed -i "s/##POD##/${POD}/g" current.list
sed -i "s/##CONTAINER##/${CONTAINER}/g" current.list
sed -i "s/##MODEL##/${MODEL}/g" current.list

INDEX=0

while IFS= read -r line; do
    # skip empty lines
    [[ -z "$line" ]] && continue

    # extract metric name (before '{')
    metric_name=$(echo "$line" | cut -d'{' -f1)

    # extract labels (inside {...})
    labels_raw=$(echo "$line" | sed -n 's/^[^{]*{\(.*\)}/\1/p')

    # convert labels to Prometheus CLI format:
    # namespace="x",pod="y" → namespace=x,pod=y
    labels=$(echo "$labels_raw" \
        | sed 's/"//g' \
        | tr ',' ',')

    ((INDEX++))
    output="${INDEX}.${metric_name}.csv"

    echo "fetching $metric_name ($labels) ..."

    $PC metric_to_csv.py \
        --url "$PROM_URL" \
        --metric "$metric_name" \
        --labels "$labels" \
        --start "$START" \
        --end "$END" \
        --step "$STEP" \
        --output "$output"

done < current.list

rm -f current.list

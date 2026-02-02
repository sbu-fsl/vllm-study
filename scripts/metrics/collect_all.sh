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

# python
PC="python3"

####################################################

cp metrics.list current.list

# replace placeholders
sed -i "s/##NS##/${NAMESPACE}/g" current.list
sed -i "s/##POD##/${POD}/g" current.list
sed -i "s/##CONTAINER##/${CONTAINER}/g" current.list
sed -i "s/##MODEL##/${MODEL}/g" current.list

# call promdigger
./promdigger batch --input current.list --start "${START}" --end "${END}"

rm -f current.list

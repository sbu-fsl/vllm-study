#!/bin/bash

############### config parameters ##################
# metrics labels
# must change these parameters according to your setup
POD="facebook-opt"
MODEL="facebook\/opt-125m"

# container and namespace names are 'usually' fixed
CONTAINER="vllm-container"
NAMESPACE="llm-servings"


# time settings
START="2026-01-20T17:00:00Z" # enter the start time you got from the extract_ts.sh, minus some buffer time (e.g., 30 seconds)
END="2026-01-20T18:00:00Z" # enter the end time you got from the extract_ts.sh, plus some buffer time (e.g., 30 seconds)

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
promdigger batch --input current.list --start "${START}" --end "${END}"

rm -f current.list

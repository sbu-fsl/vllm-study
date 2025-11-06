#!/bin/bash
# file: logs_pipeline.sh

# global vars
DIR="logs"
INP="new_logs.txt"

# remove the old directory
rm -rf $DIR

# create a new logs directory
mkdir $DIR

# take the logs and export it into 4 files
# read_times, read_bytes, write_times, write_bytes
grep -e "read_time" "${INP}" > "${DIR}/read_times.txt"
grep -e "write_time" "${INP}" > "${DIR}/write_times.txt"
grep -e "read_bytes" "${INP}" > "${DIR}/read_bytes.txt"
grep -e "write_bytes" "${INP}" > "${DIR}/write_bytes.txt"
grep -e "read_count" "${INP}" > "${DIR}/read_counts.txt"
grep -e "write_count" "${INP}" > "${DIR}/write_counts.txt"

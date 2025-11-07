#!/bin/bash
# file: script.sh

python analysis/create_datasets.py
python analysis/split_traces.py logs/traces.txt traces
python analysis/plot_traces_prefix_timeline.py traces/vllm.log

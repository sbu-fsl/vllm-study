#!/bin/bash
# Wrapper script to activate venv and run Phase 2 analysis

cd /home/sai/timeseries/SUNY-ibm-multicloud-gpus/timeseries_benchmarks_v2
source .venv/bin/activate
python3 analyze_phase2_results.py

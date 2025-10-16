#!/bin/bash
# Script to run Imagenet Distributed Benchmark Experiments
# Usage: bash run_experiments.sh

echo "==========================================="
echo "Running Imagenet Distributed Experiments..."
echo "==========================================="

python imagenet_distributed_benchmark.py --logfile ImagenetNet_Experiments.csv

echo "==========================================="
echo "Experiments Completed!"
echo "Logs and results saved in ImagenetNet_Experiments_2.csv and logs/ folder."
echo "==========================================="

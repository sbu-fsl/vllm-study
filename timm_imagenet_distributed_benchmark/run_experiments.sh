#!/bin/bash
# Script to run Timm-based Imagenet Distributed Benchmark Experiments
# Usage: bash run_experiments.sh

echo "==========================================="
echo "Running Timm-Imagenet Distributed Experiments..."
echo "==========================================="

python timm_imagenet_distributed_benchmark.py --csv_file timm_imagenet_experiments.csv

echo "==========================================="
echo "Experiments Completed!"
echo "Logs and results saved in timm_imagenet_experiments.csv and logs/ folder."
echo "==========================================="

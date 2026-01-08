#!/bin/bash
# Phase 2: Full NAB realKnownCause benchmark (15 datasets)
# Expected runtime: ~10-12 hours (135 runs: 15 datasets × 3 models × 3 detectors)
#
# Purpose: Publication-quality robust statistics
# - All NAB realKnownCause datasets
# - Comprehensive evaluation across diverse patterns
# - Final validation before production deployment

cd /home/sai/timeseries/SUNY-ibm-multicloud-gpus/timeseries_benchmarks_v2

# Activate virtual environment
source .venv/bin/activate

echo "========================================="
echo "Phase 2: Full 15-Dataset Benchmark"
echo "========================================="
echo "Detectors: SeasonalQuantile, VolatilityNormalized, AdaptiveQuantile"
echo "Datasets: 15 (ALL NAB realKnownCause)"
echo "Models: ARIMA, Toto (optimized), Granite"
echo "Total runs: 135 (15 × 3 × 3)"
echo "Expected: ~10-12 hours"
echo ""
echo "Purpose: Publication-quality comprehensive evaluation"
echo ""
echo "Starting run at $(date)"
echo "========================================="

# Set limit to 15 datasets (all available)
export TS_LIMIT=15

# Run benchmark with fixed detectors
python main_timeseries_benchmark.py

echo ""
echo "========================================="
echo "Phase 2 complete at $(date)"
echo "========================================="
echo ""
echo "Results ready for:"
echo "1. Publication/reporting"
echo "2. Production detector selection"
echo "3. Final model + detector combination choice"

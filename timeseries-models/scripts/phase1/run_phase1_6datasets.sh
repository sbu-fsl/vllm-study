#!/bin/bash
# Phase 1: Full validation with 6 datasets
# Expected runtime: ~4-5 hours (54 runs: 6 datasets × 3 models × 3 detectors)
#
# Purpose: Get robust statistics across multiple datasets
# - Compare: SeasonalQuantile (baseline) vs VolatilityNormalized vs AdaptiveQuantile
# - Models: ARIMA, Toto (optimized), Granite
# - Each run: 3-fold CV for robust evaluation

cd /home/sai/timeseries/SUNY-ibm-multicloud-gpus/timeseries_benchmarks_v2

# Activate virtual environment
source .venv/bin/activate

echo "========================================="
echo "Phase 1: Full 6-Dataset Benchmark"
echo "========================================="
echo "Detectors: SeasonalQuantile, VolatilityNormalized, AdaptiveQuantile"
echo "Datasets: 6 (all NAB realKnownCause)"
echo "Models: ARIMA, Toto (optimized), Granite"
echo "Total runs: 54 (6 × 3 × 3)"
echo "Expected: ~4-5 hours"
echo ""
echo "Success criteria:"
echo "  - Consistent detector rankings across datasets"
echo "  - VolatilityNormalized best for Granite"
echo "  - AdaptiveQuantile competitive or better"
echo ""
echo "Starting run at $(date)"
echo "========================================="

# Set limit to 6 datasets
export TS_LIMIT=6

# Run benchmark with fixed detectors
python main_timeseries_benchmark.py

echo ""
echo "========================================="
echo "Phase 1 complete at $(date)"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Check results in runs/ directory"
echo "2. Analyze exec_summary.csv across all 6 datasets"
echo "3. Compare detector performance consistency"
echo "4. Optional: Phase 2 (15 datasets) for publication-quality results"

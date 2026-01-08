#!/bin/bash
# Phase 0 validation: Test fixed detectors on 1 dataset
# Expected runtime: ~30-45 min (3 models × 3 detectors × 5 min each)
#
# Purpose: Validate that fixes work before expensive Phase 1
# - VolatilityNormalized: Should handle heteroscedastic residuals better
# - AdaptiveQuantile: Should properly tighten thresholds in high-volatility regions
# - SeasonalQuantile: Baseline for comparison

cd /home/sai/timeseries/SUNY-ibm-multicloud-gpus/timeseries_benchmarks_v2

# Activate virtual environment
source .venv/bin/activate

echo "========================================="
echo "Phase 0: Fixed Detectors Validation"
echo "========================================="
echo "Testing: SeasonalQuantile, VolatilityNormalized, AdaptiveQuantile"
echo "Dataset: 1 (nyc_taxi)"
echo "Models: ARIMA, Toto (optimized), Granite"
echo "Expected: ~30-45 minutes"
echo ""
echo "Success criteria:"
echo "  - VolatilityNormalized F1 >= SeasonalQuantile F1"
echo "  - AdaptiveQuantile F1 >= SeasonalQuantile F1 * 0.95"
echo "  - No crashes or errors"
echo ""
echo "Starting run at $(date)"
echo "========================================="

# Set limit to 1 dataset for quick validation
export TS_LIMIT=1

# Run benchmark with fixed detectors
python main_timeseries_benchmark.py

echo ""
echo "========================================="
echo "Phase 0 complete at $(date)"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Check results in runs/ directory"
echo "2. Compare detector_improvements.csv"
echo "3. If F1 improvements confirmed, proceed to Phase 1"
echo "4. Phase 1: 6 datasets × 3 models × 3 detectors = 54 runs (~4-5 hours)"

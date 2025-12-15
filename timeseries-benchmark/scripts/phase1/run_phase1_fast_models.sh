#!/bin/bash
# Phase 1: Full test with fast models only (ARIMA + Granite)
# Expected runtime: 30-45 minutes

set -e  # Exit on error

echo "======================================================================="
echo "PHASE 1: FULL TEST - FAST MODELS ONLY"
echo "Testing new detectors with ARIMA + Granite on all 6 datasets"
echo "Expected runtime: 30-45 minutes"
echo "======================================================================="

# Configuration
export SKIP_TOTO=1           # Skip Toto (optimize separately)
export TS_LIMIT=6            # All 6 datasets
export TS_WARMUP=200         # Full warmup
export TOTO_SAMPLES=128      # N/A (Toto skipped)
export TOTO_MAX_POINTS=5000  # N/A (Toto skipped)

echo ""
echo "Configuration:"
echo "  SKIP_TOTO: ${SKIP_TOTO}"
echo "  TS_LIMIT: ${TS_LIMIT} datasets"
echo "  TS_WARMUP: ${TS_WARMUP} points"
echo "  Detectors: SeasonalQuantile, AdaptiveQuantile, IsolationForest, Multivariate"
echo ""

# Run benchmark
python main_timeseries_benchmark.py

echo ""
echo "======================================================================="
echo "PHASE 1 COMPLETE!"
echo "======================================================================="
echo ""
echo "Analysis:"
find runs/TSB_* -name "exec_summary.csv" -exec tail -1 {} \; | head -20

echo ""
echo "Check for rogue_agent improvements:"
echo "  grep 'rogue_agent' runs/TSB_*/exec_summary.csv"
echo ""
echo "Next steps:"
echo "  1. Analyze results - did new detectors improve rogue_agent?"
echo "  2. If yes, proceed to Phase 2 (optimize Toto)"
echo "  3. If no, debug detector implementations"
echo ""

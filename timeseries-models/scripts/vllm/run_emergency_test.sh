#!/bin/bash
# Emergency test - Phase 0: Validate new detectors WITHOUT Toto
# Expected runtime: 5-10 minutes

set -e  # Exit on error

echo "======================================================================="
echo "PHASE 0: EMERGENCY TEST"
echo "Testing new detectors with ARIMA + Granite only (fast models)"
echo "Expected runtime: 5-10 minutes"
echo "======================================================================="

# Configuration
export SKIP_TOTO=1           # Skip Toto (too slow)
export TS_LIMIT=2            # Test on 2 smallest datasets
export TS_WARMUP=100         # Smaller warmup for speed
export TOTO_SAMPLES=128      # N/A (Toto skipped)
export TOTO_MAX_POINTS=5000  # N/A (Toto skipped)

echo ""
echo "Configuration:"
echo "  SKIP_TOTO: ${SKIP_TOTO}"
echo "  TS_LIMIT: ${TS_LIMIT} datasets"
echo "  TS_WARMUP: ${TS_WARMUP} points"
echo ""

# Run benchmark
python main_timeseries_benchmark.py

echo ""
echo "======================================================================="
echo "PHASE 0 COMPLETE!"
echo "======================================================================="
echo ""
echo "Next steps:"
echo "  1. Check results in runs/TSB_* directory"
echo "  2. Look for improvement on rogue_agent dataset"
echo "  3. If successful, run Phase 1 (all 6 datasets, no Toto)"
echo "  4. Then optimize and add Toto back"
echo ""

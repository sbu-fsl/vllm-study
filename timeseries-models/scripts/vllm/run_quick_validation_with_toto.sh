#!/bin/bash
# REVISED PLAN: Test new detectors on smallest dataset first (with ALL 3 models including Toto)
# This validates everything works before committing to full 6-dataset run
# Expected runtime: 30-45 minutes for 1 dataset × 3 models × 4 detectors = 12 runs

set -e  # Exit on error

echo "======================================================================="
echo "REVISED PHASE 0: QUICK VALIDATION WITH ALL 3 MODELS"
echo "Testing new detectors on 1 SMALL dataset with ARIMA + Toto + Granite"
echo "Expected runtime: 30-45 minutes"
echo "======================================================================="

# Configuration
export SKIP_TOTO=0              # ✅ INCLUDE Toto (we need it!)
export TS_LIMIT=1               # Just 1 dataset (smallest: rogue_agent has 1,881 points)
export TS_WARMUP=100            # Smaller warmup for speed
export TOTO_SAMPLES=128         # Optimized (was 512)
export TOTO_MAX_POINTS=2000     # Limit to 2000 points for quick test

echo ""
echo "Configuration:"
echo "  SKIP_TOTO: ${SKIP_TOTO} (Toto ENABLED with optimizations)"
echo "  TS_LIMIT: ${TS_LIMIT} dataset"
echo "  TS_WARMUP: ${TS_WARMUP} points"
echo "  TOTO_SAMPLES: ${TOTO_SAMPLES} (optimized from 512)"
echo "  TOTO_MAX_POINTS: ${TOTO_MAX_POINTS} (limit dataset size)"
echo ""
echo "This will test:"
echo "  - 3 models: ARIMA, Toto, Granite"
echo "  - 4 detectors: SeasonalQuantile, AdaptiveQuantile, IsolationForest, Multivariate"
echo "  - 1 dataset: smallest available"
echo "  = 12 total runs"
echo ""

# Run benchmark
echo "Starting benchmark..."
python main_timeseries_benchmark.py

echo ""
echo "======================================================================="
echo "PHASE 0 COMPLETE!"
echo "======================================================================="
echo ""

# Quick analysis
echo "Quick Results Check:"
runs_dir=$(ls -td runs/TSB_* | head -1)
if [ -f "${runs_dir}/exec_summary.csv" ]; then
    echo ""
    echo "Toto runtime check (if Toto ran successfully):"
    echo "  Expected: 10-15 minutes for 1 dataset with optimizations"
    echo "  If it took longer, we need more optimization"
    echo ""
    
    echo "Results preview:"
    python3 << 'EOF'
import pandas as pd
import glob

runs = glob.glob('runs/TSB_*')
latest = max(runs, key=lambda x: x.split('_')[-1])
df = pd.read_csv(f'{latest}/exec_summary.csv')

print("\nF1 Scores by Model and Detector:")
pivot = df.pivot_table(index='Detector', columns='Model', values='F1', aggfunc='mean')
print(pivot.round(3))

print("\nDid Toto run successfully?")
toto_results = df[df['Model'] == 'Toto']
if len(toto_results) > 0:
    print(f"✅ YES - Toto completed {len(toto_results)} runs")
    print(f"   Average F1: {toto_results['F1'].mean():.3f}")
else:
    print("❌ NO - Toto did not run")
EOF
fi

echo ""
echo "Next steps:"
echo "  1. If Toto completed in 10-15 min: ✅ Proceed to Phase 1 (all 6 datasets)"
echo "  2. If Toto took >30 min: ⚠️  Need more optimization"
echo "  3. Check: Did new detectors improve F1 scores?"
echo ""
echo "To run full benchmark (6 datasets × 3 models × 4 detectors):"
echo "  ./run_full_benchmark_with_toto.sh"
echo ""

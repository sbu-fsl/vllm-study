#!/bin/bash
# FULL BENCHMARK: All 6 datasets × 3 models × 4 detectors = 72 runs
# Expected runtime: 8-10 hours (with Toto optimizations)

set -e  # Exit on error

echo "======================================================================="
echo "FULL BENCHMARK WITH ALL 3 MODELS (INCLUDING TOTO)"
echo "Testing new detectors on all 6 NAB datasets"
echo "Expected runtime: 8-10 hours"
echo "======================================================================="

# Configuration
export SKIP_TOTO=0              # ✅ INCLUDE Toto
export TS_LIMIT=6               # All 6 datasets
export TS_WARMUP=200            # Full warmup
export TOTO_SAMPLES=128         # Optimized (was 512 in unfixed version)
export TOTO_MAX_POINTS=5000     # Limit to 5000 points per dataset

echo ""
echo "Configuration:"
echo "  SKIP_TOTO: ${SKIP_TOTO} (Toto ENABLED)"
echo "  TS_LIMIT: ${TS_LIMIT} datasets"
echo "  TS_WARMUP: ${TS_WARMUP} points"
echo "  TOTO_SAMPLES: ${TOTO_SAMPLES}"
echo "  TOTO_MAX_POINTS: ${TOTO_MAX_POINTS}"
echo ""
echo "This will run:"
echo "  - 3 models: ARIMA, Toto, Granite"
echo "  - 4 detectors: SeasonalQuantile, AdaptiveQuantile, IsolationForest, Multivariate"
echo "  - 6 datasets: All NAB realKnownCause"
echo "  = 72 total runs"
echo ""
echo "Expected time breakdown:"
echo "  - ARIMA: ~3 minutes total (very fast)"
echo "  - Granite: ~60-90 minutes total (fast GPU)"
echo "  - Toto: ~6-8 hours total (optimized but still slowest)"
echo "  - Total: ~8-10 hours"
echo ""

# Confirm before starting long run
read -p "This will take 8-10 hours. Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Run benchmark
echo "Starting full benchmark at $(date)..."
start_time=$(date +%s)

python main_timeseries_benchmark.py

end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))

echo ""
echo "======================================================================="
echo "FULL BENCHMARK COMPLETE!"
echo "======================================================================="
echo "Total runtime: ${hours}h ${minutes}m"
echo "Completed at: $(date)"
echo ""

# Analysis
echo "Generating analysis..."
runs_dir=$(ls -td runs/TSB_* | head -1)

if [ -f "${runs_dir}/exec_summary.csv" ]; then
    python3 << 'EOF'
import pandas as pd
import glob

runs = glob.glob('runs/TSB_*')
latest = max(runs, key=lambda x: x.split('_')[-1])
df = pd.read_csv(f'{latest}/exec_summary.csv')

print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)

print("\n1. Average F1 by Model (across all detectors and datasets):")
model_avg = df.groupby('Model')['F1'].agg(['mean', 'std'])
print(model_avg.round(3))

print("\n2. Average F1 by Detector (across all models and datasets):")
det_avg = df.groupby('Detector')['F1'].agg(['mean', 'std'])
print(det_avg.round(3))

print("\n3. CRITICAL: Rogue Agent Results")
print("   (Testing if new detectors solve the paradox)")
print("-"*70)
rogue = df[df['Dataset'] == 'rogue_agent_key_hold']
if len(rogue) > 0:
    print(rogue[['Model', 'Detector', 'F1', 'sMAPE']].to_string(index=False))
    
    best = rogue.loc[rogue['F1'].idxmax()]
    baseline_f1 = rogue[rogue['Detector'] == 'SeasonalQuantile']['F1'].mean()
    improvement = (best['F1'] / baseline_f1 - 1) * 100
    
    print(f"\nBest detector: {best['Detector']}")
    print(f"Best model: {best['Model']}")
    print(f"F1: {best['F1']:.4f} (improvement: {improvement:+.1f}%)")
    
    if best['F1'] > 0.05:
        print("\n✅ SUCCESS! New detector solved rogue_agent paradox!")
    else:
        print("\n⚠️  Still challenging. May need further improvements.")

print("\n4. Detector Improvements over Baseline:")
print("-"*70)
baseline = df[df['Detector'] == 'SeasonalQuantile'].groupby('Model')['F1'].mean()
for detector in df['Detector'].unique():
    if detector != 'SeasonalQuantile':
        det_means = df[df['Detector'] == detector].groupby('Model')['F1'].mean()
        print(f"\n{detector}:")
        for model in det_means.index:
            if model in baseline.index:
                pct_imp = (det_means[model] / baseline[model] - 1) * 100
                print(f"  {model}: {det_means[model]:.3f} ({pct_imp:+.1f}%)")

print("\n5. Model Performance on Different Patterns:")
print("-"*70)
for dataset in df['Dataset'].unique():
    ds_data = df[df['Dataset'] == dataset]
    best_combo = ds_data.loc[ds_data['F1'].idxmax()]
    print(f"{dataset[:20]:20s}: {best_combo['Model']:10s} + {best_combo['Detector']:20s} (F1={best_combo['F1']:.3f})")

print("\n" + "="*70)
print(f"Full results saved in: {latest}")
print("="*70)
EOF
fi

echo ""
echo "Key files:"
echo "  - ${runs_dir}/exec_summary.csv"
echo "  - ${runs_dir}/detector_improvements.csv"
echo "  - ${runs_dir}/plots/smape_vs_f1_scatter_*.png"
echo "  - ${runs_dir}/plots/detector_comparison_*.png"
echo ""

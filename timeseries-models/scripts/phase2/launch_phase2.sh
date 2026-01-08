#!/bin/bash
# Launch Phase 2: Full 15-dataset benchmark
# Run this AFTER Phase 0 v3 completes successfully

cd /home/sai/timeseries/SUNY-ibm-multicloud-gpus/timeseries_benchmarks_v2

echo "========================================="
echo "Phase 2: Full 15-Dataset Benchmark Launch"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - 15 datasets (all NAB realKnownCause)"
echo "  - 3 models: ARIMA, Toto (optimized), Granite"
echo "  - 3 detectors: SeasonalQuantile, VolatilityNormalized, AdaptiveQuantile"
echo "  - Total: 135 runs (15 × 3 × 3)"
echo "  - Estimated time: 10-12 hours"
echo ""
echo "========================================="
read -p "Ready to launch Phase 2? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Launching Phase 2 in tmux session 'phase2_benchmark'..."
    echo ""
    
    tmux new-session -d -s phase2_benchmark "./run_phase2_15datasets.sh 2>&1 | tee phase2.log"
    
    echo "✅ Phase 2 launched!"
    echo ""
    echo "Monitor progress:"
    echo "  - Watch log: tail -f phase2.log"
    echo "  - Attach session: tmux attach -t phase2_benchmark"
    echo "  - Detach: Ctrl+b then d"
    echo ""
    echo "Started at: $(date)"
    echo "Expected completion: ~10-12 hours from now"
    echo "========================================="
else
    echo "Launch cancelled."
fi

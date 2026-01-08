#!/bin/bash
# Quick check script for Phase 0 v3 completion
# Run this to see if Phase 0 v3 finished and get quick summary

echo "========================================="
echo "Phase 0 v3 Status Check"
echo "========================================="

# Check if still running
if tmux has-session -t phase0_v3 2>/dev/null; then
    echo "Status: RUNNING ⏳"
    echo ""
    echo "Last 20 lines of output:"
    tail -20 /home/sai/timeseries/SUNY-ibm-multicloud-gpus/timeseries_benchmarks_v2/phase0_v3.log
else
    echo "Status: COMPLETED ✅"
fi

echo ""
echo "========================================="

# Find latest results
LATEST_RUN=$(find /home/sai/timeseries/SUNY-ibm-multicloud-gpus/timeseries_benchmarks_v2/runs -name "detector_improvements.csv" -mmin -120 -exec dirname {} \; | tail -1)

if [ -n "$LATEST_RUN" ]; then
    echo "Latest Results: $(basename $LATEST_RUN)"
    echo "========================================="
    echo ""
    echo "Detector Improvements:"
    cat "$LATEST_RUN/detector_improvements.csv"
    echo ""
    echo "========================================="
    echo "Key Metrics:"
    echo ""
    
    # Show just Toto + VolatilityNormalized (the one we fixed)
    grep "Toto.*VolatilityNormalized" "$LATEST_RUN/exec_summary.csv" | awk -F',' '{print "Toto+VolatilityNormalized F1: " $4}'
    grep "Toto.*SeasonalQuantile" "$LATEST_RUN/exec_summary.csv" | awk -F',' '{print "Toto+SeasonalQuantile F1: " $4 " (baseline)"}'
    
    echo ""
    echo "Full results in: $LATEST_RUN"
else
    echo "No recent results found (check if still running)"
fi

echo ""
echo "========================================="
echo ""
echo "Next steps:"
echo "1. If Phase 0 v3 successful → Launch Phase 2 (15 datasets)"
echo "2. Command: ./launch_phase2.sh"
echo "========================================="

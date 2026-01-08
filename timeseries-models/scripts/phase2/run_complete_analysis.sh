#!/bin/bash
# ============================================================
# Complete Analysis Pipeline
# Runs benchmark + all enhanced analyses + publication materials
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "TIME-SERIES ANOMALY DETECTION - COMPLETE ANALYSIS PIPELINE"
echo "============================================================"
echo ""
echo "This script will:"
echo "  1. Run the main benchmark (if needed)"
echo "  2. Enhanced statistical analysis"
echo "  3. Pattern characterization"
echo "  4. Publication report generation"
echo ""

# Configuration
RUN_BENCHMARK=false
BENCHMARK_DATASETS=6
RUN_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark)
            RUN_BENCHMARK=true
            shift
            ;;
        --datasets)
            BENCHMARK_DATASETS="$2"
            shift 2
            ;;
        --run-dir)
            RUN_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--benchmark] [--datasets N] [--run-dir PATH]"
            exit 1
            ;;
    esac
done

# Step 1: Run benchmark if requested or if no run directory specified
if [ "$RUN_BENCHMARK" = true ] || [ -z "$RUN_DIR" ]; then
    echo ""
    echo "============================================================"
    echo "STEP 1: RUNNING BENCHMARK"
    echo "============================================================"
    echo ""
    
    export TS_LIMIT=$BENCHMARK_DATASETS
    export TS_WARMUP=200
    export TS_ALPHA=0.10
    export TOTO_SAMPLES=128
    export TS_SEED=0
    
    echo "Configuration:"
    echo "  Datasets: $TS_LIMIT"
    echo "  Warmup: $TS_WARMUP"
    echo "  Alpha: $TS_ALPHA"
    echo "  Toto samples: $TOTO_SAMPLES"
    echo ""
    
    python main_timeseries_benchmark.py
    
    # Find the most recent run directory
    RUN_DIR=$(ls -td runs/TSB_* | head -1)
    echo ""
    echo "✅ Benchmark complete: $RUN_DIR"
else
    echo ""
    echo "Skipping benchmark, using existing run: $RUN_DIR"
fi

# Verify run directory exists
if [ ! -d "$RUN_DIR" ]; then
    echo "❌ Error: Run directory not found: $RUN_DIR"
    exit 1
fi

echo ""
echo "Working with run directory: $RUN_DIR"

# Step 2: Enhanced analysis
echo ""
echo "============================================================"
echo "STEP 2: ENHANCED STATISTICAL ANALYSIS"
echo "============================================================"
echo ""

python enhanced_analysis.py "$RUN_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Enhanced analysis complete"
else
    echo "⚠️  Enhanced analysis failed (may be missing dependencies)"
fi

# Step 3: Pattern characterization
echo ""
echo "============================================================"
echo "STEP 3: PATTERN CHARACTERIZATION"
echo "============================================================"
echo ""

python pattern_characterization.py "$RUN_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Pattern analysis complete"
else
    echo "⚠️  Pattern analysis failed (may be missing dependencies)"
fi

# Step 4: Publication report
echo ""
echo "============================================================"
echo "STEP 4: PUBLICATION REPORT GENERATION"
echo "============================================================"
echo ""

python publication_report_generator.py "$RUN_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Publication materials generated"
else
    echo "⚠️  Report generation failed"
fi

# Step 5: Summary
echo ""
echo "============================================================"
echo "✅ COMPLETE ANALYSIS PIPELINE FINISHED"
echo "============================================================"
echo ""
echo "📁 All outputs saved to:"
echo "   $RUN_DIR/"
echo ""
echo "📊 Key files:"
echo "   Executive Summary:"
echo "     - exec_summary.csv"
echo "     - exec_summary.md"
echo "     - enhanced_analysis/PROGRESS_REPORT.md"
echo "     - enhanced_analysis/ONE_PAGER.md"
echo ""
echo "   Visualizations:"
echo "     - plots/smape_vs_f1_scatter.png (CRITICAL PLOT)"
echo "     - enhanced_analysis/separability_vs_f1_comprehensive.png"
echo "     - enhanced_analysis/performance_by_pattern_type.png"
echo "     - enhanced_analysis/feature_importance.png"
echo ""
echo "   Publication Materials:"
echo "     - enhanced_analysis/PUBLICATION_REPORT.md"
echo "     - enhanced_analysis/paper_outline.tex"
echo "     - enhanced_analysis/presentation_outline.md"
echo "     - enhanced_analysis/EMAIL_TEMPLATE.txt"
echo ""
echo "   Data & Traces:"
echo "     - datasets/*.csv"
echo "     - traces/*.csv"
echo "     - pr_traces/*.csv"
echo ""
echo "🚀 Next Steps:"
echo "   1. Review PROGRESS_REPORT.md for key insights"
echo "   2. Check critical plots in enhanced_analysis/"
echo "   3. Use PUBLICATION_REPORT.md as paper draft"
echo "   4. Share ONE_PAGER.md with collaborators"
echo ""
echo "📧 To share results, use:"
echo "   enhanced_analysis/EMAIL_TEMPLATE.txt"
echo ""
echo "============================================================"

# Generate a quick summary table
echo ""
echo "📈 QUICK RESULTS SUMMARY"
echo "============================================================"
echo ""

if [ -f "$RUN_DIR/exec_summary.csv" ]; then
    python3 << EOF
import pandas as pd
df = pd.read_csv('$RUN_DIR/exec_summary.csv')
print("Model Performance (mean across datasets):")
print("="*60)
summary = df.groupby('Model').agg({
    'F1': 'mean',
    'sMAPE': 'mean',
    'Coverage': 'mean'
}).round(3)
print(summary.to_string())
print("")
print("="*60)
print(f"Total experiments: {len(df)} (models × datasets)")
print(f"Datasets: {df['Dataset'].nunique()}")
print(f"Models: {df['Model'].nunique()}")
print("")
corr = df[['sMAPE', 'F1']].corr().iloc[0, 1]
print(f"🔍 Key Finding: Correlation (sMAPE vs F1) = {corr:.3f}")
if abs(corr) > 0.5:
    print("   → STRONG correlation: Forecast quality matters!")
elif abs(corr) > 0.3:
    print("   → MODERATE correlation: Forecast quality has impact")
else:
    print("   → WEAK correlation: Detector may be the bottleneck")
EOF
fi

echo ""
echo "============================================================"
echo "🎉 ANALYSIS COMPLETE - READY TO SHOW PROGRESS!"
echo "============================================================"

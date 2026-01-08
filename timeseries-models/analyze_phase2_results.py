#!/usr/bin/env python3
"""
Phase 2 Results Analysis
Compares optimized Toto + new detectors against baseline
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Phase 2 results path
PHASE2_DIR = Path("runs/TSB_2025-10-26_22-20-54_cu124_toto128_lim15_sai-workspace-7698d6c79d-lv28t")

def main():
    # Load Phase 2 results
    df = pd.read_csv(PHASE2_DIR / "exec_summary.csv")
    
    print("="*80)
    print("PHASE 2 RESULTS ANALYSIS: 15 Datasets (135 runs)")
    print("="*80)
    print(f"\nTotal runs: {len(df)}")
    print(f"Datasets: {df['Dataset'].nunique()}")
    print(f"Models: {df['Model'].unique()}")
    print(f"Detectors: {df['Detector'].unique()}")
    
    # ============================================================================
    # 1. OVERALL MODEL PERFORMANCE
    # ============================================================================
    print("\n" + "="*80)
    print("1. OVERALL MODEL PERFORMANCE (Averaged Across All Datasets & Detectors)")
    print("="*80)
    
    model_stats = df.groupby('Model').agg({
        'F1': ['mean', 'std', 'min', 'max'],
        'sMAPE': ['mean', 'std'],
        'Coverage': ['mean', 'std']
    }).round(3)
    print(model_stats)
    
    # ============================================================================
    # 2. DETECTOR PERFORMANCE BY MODEL
    # ============================================================================
    print("\n" + "="*80)
    print("2. DETECTOR PERFORMANCE BY MODEL")
    print("="*80)
    
    for model in ['ARIMA', 'Toto', 'GraniteTTM']:
        model_df = df[df['Model'] == model]
        print(f"\n{model}:")
        print("-" * 40)
        
        det_stats = model_df.groupby('Detector').agg({
            'F1': ['mean', 'std', 'count']
        }).round(3)
        print(det_stats)
        
        # Calculate improvement vs baseline
        baseline_f1 = model_df[model_df['Detector'] == 'SeasonalQuantile']['F1'].mean()
        print(f"\nBaseline (SeasonalQuantile) F1: {baseline_f1:.3f}")
        
        for det in ['VolatilityNormalized', 'AdaptiveQuantile']:
            det_df = model_df[model_df['Detector'] == det]
            if len(det_df) > 0:
                det_f1 = det_df['F1'].mean()
                improvement = ((det_f1 / baseline_f1) - 1) * 100
                print(f"  {det}: F1={det_f1:.3f} ({improvement:+.1f}% vs baseline)")
    
    # ============================================================================
    # 3. TOTO OPTIMIZATION CHECK
    # ============================================================================
    print("\n" + "="*80)
    print("3. TOTO OPTIMIZATION CHECK: Did 128 samples hurt accuracy?")
    print("="*80)
    
    toto_df = df[df['Model'] == 'Toto']
    print(f"\nToto runs completed: {len(toto_df)}/45 expected (15 datasets × 3 detectors)")
    
    # Check for failures (F1=0)
    zero_f1 = toto_df[toto_df['F1'] == 0.0]
    print(f"Toto failures (F1=0): {len(zero_f1)}/{len(toto_df)} ({len(zero_f1)/len(toto_df)*100:.1f}%)")
    
    if len(zero_f1) > 0:
        print("\nFailed datasets:")
        print(zero_f1[['Dataset', 'Detector', 'sMAPE', 'Coverage']])
    
    # Compare Toto sMAPE by detector (forecast quality check)
    print("\nToto forecast quality (sMAPE) by detector:")
    toto_smape = toto_df.groupby('Detector')['sMAPE'].agg(['mean', 'std', 'count']).round(2)
    print(toto_smape)
    
    print("\nToto F1 scores by detector:")
    toto_f1 = toto_df.groupby('Detector')['F1'].agg(['mean', 'std', 'min', 'max']).round(3)
    print(toto_f1)
    
    # ============================================================================
    # 4. ROGUE_AGENT BREAKTHROUGH CHECK
    # ============================================================================
    print("\n" + "="*80)
    print("4. ROGUE_AGENT BREAKTHROUGH: Did new detectors solve the paradox?")
    print("="*80)
    
    rogue = df[df['Dataset'].str.contains('rogue_agent')]
    print(f"\nRogue agent datasets found: {rogue['Dataset'].unique()}")
    
    # Focus on rogue_agent_key_hold (the main one from Phase 1)
    rogue_hold = df[df['Dataset'] == 'rogue_agent_key_hold']
    
    if len(rogue_hold) > 0:
        print("\n" + "-"*60)
        print("ROGUE_AGENT_KEY_HOLD Results (Phase 1 baseline: F1≈0.006-0.011)")
        print("-"*60)
        
        # Create pivot table
        pivot = rogue_hold.pivot_table(
            index='Model',
            columns='Detector',
            values=['F1', 'sMAPE'],
            aggfunc='mean'
        )
        print("\nF1 Scores:")
        print(pivot['F1'].round(3))
        print("\nsMAPE (forecast quality):")
        print(pivot['sMAPE'].round(2))
        
        # Find best combination
        best_idx = rogue_hold['F1'].idxmax()
        best = rogue_hold.loc[best_idx]
        
        print(f"\n🏆 BEST RESULT:")
        print(f"   Model: {best['Model']}")
        print(f"   Detector: {best['Detector']}")
        print(f"   F1 Score: {best['F1']:.3f}")
        print(f"   sMAPE: {best['sMAPE']:.2f}")
        
        # Compare to Phase 1 baseline
        baseline_granite = 0.006  # From Phase 1 report
        if best['F1'] > baseline_granite * 2:
            improvement_pct = (best['F1'] / baseline_granite - 1) * 100
            print(f"   ✅ BREAKTHROUGH: {improvement_pct:.0f}% improvement vs Phase 1!")
        else:
            print(f"   ⚠️  Still poor detection (only {best['F1']/baseline_granite:.1f}× baseline)")
    
    # ============================================================================
    # 5. DATASET-SPECIFIC WINNERS
    # ============================================================================
    print("\n" + "="*80)
    print("5. DATASET-SPECIFIC WINNERS (Best Model+Detector per Dataset)")
    print("="*80)
    
    print(f"\n{'Dataset':<35} | {'Winner':<30} | {'F1':>6} | {'sMAPE':>7}")
    print("-"*80)
    
    for dataset in sorted(df['Dataset'].unique()):
        ds_df = df[df['Dataset'] == dataset]
        best = ds_df.loc[ds_df['F1'].idxmax()]
        winner = f"{best['Model']} + {best['Detector']}"
        print(f"{dataset:<35} | {winner:<30} | {best['F1']:6.3f} | {best['sMAPE']:7.2f}")
    
    # ============================================================================
    # 6. DETECTOR EFFECTIVENESS SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("6. DETECTOR EFFECTIVENESS SUMMARY")
    print("="*80)
    
    # Load detector improvements file
    if (PHASE2_DIR / "detector_improvements.csv").exists():
        improvements = pd.read_csv(PHASE2_DIR / "detector_improvements.csv")
        print("\nImprovement vs SeasonalQuantile Baseline:")
        print(improvements.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    else:
        # Calculate manually
        print("\nAverage F1 improvement by Model+Detector:")
        baseline = df[df['Detector'] == 'SeasonalQuantile'].groupby('Model')['F1'].mean()
        
        for model in ['ARIMA', 'Toto', 'GraniteTTM']:
            print(f"\n{model}:")
            baseline_f1 = baseline[model]
            
            for det in ['VolatilityNormalized', 'AdaptiveQuantile']:
                det_df = df[(df['Model'] == model) & (df['Detector'] == det)]
                if len(det_df) > 0:
                    det_f1 = det_df['F1'].mean()
                    improvement = ((det_f1 / baseline_f1) - 1) * 100
                    print(f"  {det}: {improvement:+.1f}%")
    
    # ============================================================================
    # 7. KEY FINDINGS SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("7. KEY FINDINGS & RECOMMENDATIONS")
    print("="*80)
    
    # Calculate key metrics
    toto_failure_rate = len(toto_df[toto_df['F1'] == 0]) / len(toto_df) * 100
    
    # Best detector per model
    best_detectors = {}
    for model in ['ARIMA', 'Toto', 'GraniteTTM']:
        model_df = df[df['Model'] == model]
        det_means = model_df.groupby('Detector')['F1'].mean()
        best_det = det_means.idxmax()
        best_f1 = det_means.max()
        baseline_f1 = det_means.get('SeasonalQuantile', 0)
        improvement = ((best_f1 / baseline_f1) - 1) * 100 if baseline_f1 > 0 else 0
        best_detectors[model] = (best_det, best_f1, improvement)
    
    print("\n✅ SUCCESSES:")
    print("-" * 40)
    
    if toto_failure_rate < 10:
        print(f"1. Toto optimization successful: {100-toto_failure_rate:.0f}% success rate")
        print(f"   → 128 samples works well (down from 512)")
    
    for model, (det, f1, imp) in best_detectors.items():
        if imp > 5:
            print(f"2. {model} + {det}: +{imp:.1f}% improvement")
            print(f"   → Mean F1 = {f1:.3f}")
    
    print("\n⚠️  CONCERNS:")
    print("-" * 40)
    
    if toto_failure_rate > 5:
        print(f"1. Toto has {toto_failure_rate:.1f}% failure rate (F1=0)")
        print(f"   → Investigate failed datasets")
    
    # Check if rogue_agent improved
    if len(rogue_hold) > 0:
        best_rogue_f1 = rogue_hold['F1'].max()
        if best_rogue_f1 < 0.05:
            print(f"2. Rogue_agent still fails: Best F1 = {best_rogue_f1:.3f}")
            print(f"   → New detectors didn't solve subtle anomaly detection")
    
    print("\n📋 RECOMMENDATIONS:")
    print("-" * 40)
    
    print("1. Use model-specific detectors:")
    for model, (det, f1, imp) in best_detectors.items():
        print(f"   • {model}: {det} (F1={f1:.3f})")
    
    if toto_failure_rate > 5:
        print("\n2. Investigate Toto failures:")
        print(f"   • Check {len(zero_f1)} failed datasets")
        print(f"   • Consider fallback to ARIMA/Granite on failure")
    
    if len(rogue_hold) > 0 and best_rogue_f1 < 0.05:
        print("\n3. Rogue_agent requires different approach:")
        print(f"   • Current detectors don't handle subtle anomalies")
        print(f"   • Consider: LSTM autoencoder, Transformer attention, or supervised learning")
    
    print("\n" + "="*80)
    print("Analysis complete. Check plots in:")
    print(f"  {PHASE2_DIR / 'plots'}")
    print("="*80)

if __name__ == "__main__":
    main()

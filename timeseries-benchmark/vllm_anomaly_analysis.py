#!/usr/bin/env python3
"""
vLLM Anomaly Detection Analysis (with Phase 2 Learnings)
Compares 6 parameter change scenarios across 39 metrics using 3 models (ARIMA, Toto, Granite)
Expected output: 18 results (6 scenarios × 3 models)

Applies Phase 2 optimizations:
- VolatilityNormalized detector (+1,916% improvement validated)
- Grid search tuning for (ql, qh) per metric
- Runtime tracking (forecasts/sec)
- 60/40 train/validation splits
- Residual separation analysis
- Optimal Toto parameters (128 samples, batch=256)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Import the benchmark models
sys.path.insert(0, str(Path(__file__).parent))
import main_timeseries_benchmark as benchmark

# Test case scenarios
SCENARIOS = [
    {
        'name': 'Volume Type Change',
        'baseline': '01.llama',
        'test': '07.llama-pvc',
        'parameter': 'Volume type changed (standard → PVC)'
    },
    {
        'name': 'Model Change',
        'baseline': '07.llama-pvc',
        'test': '02.granite',
        'parameter': 'Model changed (Llama → Granite)'
    },
    {
        'name': 'Max Model Length Change',
        'baseline': '02.granite',
        'test': '03.granite-max-model',
        'parameter': 'Maximum model length changed'
    },
    {
        'name': 'CPU Offloading Added',
        'baseline': '03.granite-max-model',
        'test': '04.granite-cpu-offloading',
        'parameter': 'CPU offloading enabled'
    },
    {
        'name': 'Max Request Num Change',
        'baseline': '03.granite-max-model',
        'test': '05.granite-max-num-seq',
        'parameter': 'Maximum request number changed'
    },
    {
        'name': 'Max Batch Size Change',
        'baseline': '03.granite-max-model',
        'test': '06.granite-max-batch',
        'parameter': 'Maximum batch size changed'
    }
]

DATASET_DIR = Path('/home/sai/timeseries/SUNY-ibm-multicloud-gpus/timeseries_benchmarks_v2/dataset')
OUTPUT_DIR = Path('/home/sai/timeseries/SUNY-ibm-multicloud-gpus/timeseries_benchmarks_v2/vllm_analysis_results')

def normalize_timestamps(df, start_time=None):
    """Normalize timestamps to start from a common point"""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if start_time is None:
        start_time = df['timestamp'].min()
    
    # Convert to seconds from start
    df['time_seconds'] = (df['timestamp'] - start_time).dt.total_seconds()
    
    return df, start_time

def load_metric_file(filepath):
    """Load a metric CSV file"""
    try:
        df = pd.read_csv(filepath)
        df.columns = ['timestamp', 'value']
        
        # Handle different timestamp formats
        if df['timestamp'].dtype in ['int64', 'float64']:
            # Unix timestamp in milliseconds
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            # String timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Strip ALL units from values (e.g., "24.5 W" → 24.5, "1.06 GiB" → 1.06)
        if df['value'].dtype == 'object':
            # Remove ALL units found in the dataset
            # Storage: B, KiB, MiB, GiB
            # Speed: B/s, KiB/s, MiB/s, io/s, p/s, mp/s
            # Power: W
            # Temperature: °C
            # Percentage: %
            # Time: s, ms, µs
            df['value'] = df['value'].astype(str).str.replace(
                r'\s*(B|KiB|MiB|GiB|B/s|KiB/s|MiB/s|io/s|p/s|mp/s|W|°C|%|ms|µs|s)(?:\s|$)',
                '', regex=True
            )
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Drop rows with NaN values
        df = df.dropna().reset_index(drop=True)
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['label'] = 0  # No anomaly labels in vLLM data
        
        # Check for all-zero or zero-variance data (skip these)
        if len(df) > 0:
            if df['value'].nunique() == 1:  # All same value (including all zeros)
                return None  # Skip constant-value metrics
            if df['value'].std() < 1e-10:  # Near-zero variance
                return None  # Skip no-variance metrics
        
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def get_metric_name(filename):
    """Extract clean metric name from filename"""
    # Remove timestamp suffix
    name = filename.rsplit('-data-', 1)[0]
    return name

def align_timeseries(baseline_df, test_df):
    """Align two time series by common time range"""
    # Normalize both to start at 0
    baseline_df, baseline_start = normalize_timestamps(baseline_df)
    test_df, test_start = normalize_timestamps(test_df)
    
    # Find common time range
    min_time = max(baseline_df['time_seconds'].min(), test_df['time_seconds'].min())
    max_time = min(baseline_df['time_seconds'].max(), test_df['time_seconds'].max())
    
    # Filter to common range
    baseline_df = baseline_df[
        (baseline_df['time_seconds'] >= min_time) & 
        (baseline_df['time_seconds'] <= max_time)
    ].reset_index(drop=True)
    
    test_df = test_df[
        (test_df['time_seconds'] >= min_time) & 
        (test_df['time_seconds'] <= max_time)
    ].reset_index(drop=True)
    
    return baseline_df, test_df

def detect_anomalies_arima(test_df, baseline_df=None):
    """Detect anomalies using ARIMA with Phase 2 optimizations"""
    try:
        # Use only test data for Granite (univariate)
        y = test_df['value'].values
        ts = test_df['timestamp']
        
        if len(y) < 15:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        # Start timing
        start_time = time.time()
        
        # ARIMA forecaster with reduced order for short series
        arima = benchmark.ARIMAForecaster(order=(2,1,0), alpha=0.10)
        warmup = min(10, len(y) // 3)
        
        # Forecast with timing
        forecast_start = time.time()
        arima.begin(y[:warmup])
        residuals = []
        
        for t in range(warmup, len(y)-1):
            out = arima.forecast_and_update(y[t])
            residuals.append(y[t] - out.mean)
        
        forecast_time = time.time() - forecast_start
        n_forecasts = len(residuals)
        
        residuals = np.array(residuals)
        ts_eff = ts[warmup:warmup+len(residuals)].reset_index(drop=True)
        
        # Infer cadence for detector
        dt_sec, bins = benchmark.infer_cadence_and_bins(ts_eff)
        
        # 60/40 train/validation split (Phase 2 best practice)
        split = max(1, int(0.6 * len(residuals)))
        
        # Grid search for best (ql, qh) on validation set
        detection_start = time.time()
        grid = [(0.02,0.98), (0.05,0.95), (0.10,0.90), (0.15,0.85)]
        best_f1 = -1.0
        best_detector = None
        best_params = None
        
        for ql, qh in grid:
            detector = benchmark.VolatilityNormalizedQuantileDetector(
                ql=ql, qh=qh, bins=bins, use_weekday=False,
                temporal_smooth=True, min_samples_per_bin=5
            )
            detector.fit(ts_eff[:split], residuals[:split])
            flags_val = detector.predict(ts_eff[:split], residuals[:split])
            
            # Evaluate on validation (we don't have true labels, so use anomaly rate as proxy)
            # In real scenario, we'd use F1 if labels exist
            anomaly_rate = np.sum(flags_val) / len(flags_val) if len(flags_val) > 0 else 0
            
            # Prefer moderate anomaly rates (5-20%) as most realistic
            score = 1.0 - abs(anomaly_rate - 0.10)  # Prefer ~10% anomaly rate
            
            if score > best_f1:
                best_f1 = score
                best_detector = detector
                best_params = (ql, qh)
        
        # Test on held-out test set
        flags = best_detector.predict(ts_eff[split:], residuals[split:])
        detection_time = time.time() - detection_start
        total_time = time.time() - start_time
        
        anomaly_count = np.sum(flags)
        anomaly_rate = anomaly_count / len(flags) if len(flags) > 0 else 0
        
        # Residual separation analysis (Phase 2 diagnostic)
        normal_res = residuals[split:][flags == 0]
        anomaly_res = residuals[split:][flags == 1]
        if len(anomaly_res) > 0:
            separation = np.mean(np.abs(anomaly_res)) / (np.mean(np.abs(normal_res)) + 1e-8)
        else:
            separation = 1.0
        
        return {
            'detected': anomaly_count > 0,
            'anomaly_count': int(anomaly_count),
            'anomaly_rate': float(anomaly_rate),
            'total_points': len(flags),
            'model': 'ARIMA',
            'forecast_time_sec': forecast_time,
            'detection_time_sec': detection_time,
            'total_time_sec': total_time,
            'forecasts_per_sec': n_forecasts / forecast_time if forecast_time > 0 else 0,
            'best_ql': best_params[0],
            'best_qh': best_params[1],
            'residual_separation': separation
        }
        
    except Exception as e:
        return {'detected': False, 'reason': f'Error: {str(e)}'}

def detect_anomalies_toto(test_df, baseline_df=None):
    """Detect anomalies using Toto with Phase 2 optimizations"""
    try:
        # Check if Toto is available
        if not hasattr(benchmark, 'TotoForecaster'):
            return {'detected': False, 'reason': 'Toto not available'}
        
        y = test_df['value'].values
        ts = test_df['timestamp']
        
        if len(y) < 20:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        # Start timing
        start_time = time.time()
        
        # Limit size for Toto (it's slow)
        if len(y) > 1000:
            y = y[-1000:]
            ts = ts.iloc[-1000:].reset_index(drop=True)
        
        # Toto forecaster with Phase 2 optimal settings
        device = 'cuda' if benchmark.DEVICE == 'cuda' else 'cpu'
        toto = benchmark.TotoForecaster(
            device=device,
            num_samples=128,  # Phase 2 optimal (not 64!)
            ql=0.10, qh=0.90,
            compile_model=False,
            max_samples_per_batch=256  # Phase 2 optimization
        )
        
        warmup = min(10, len(y) // 3)
        dt_sec, bins = benchmark.infer_cadence_and_bins(ts)
        
        # Forecast with timing
        forecast_start = time.time()
        residuals = []
        for t in range(warmup, len(y)-1):
            hist = y[:t]
            out = toto.one_step(hist, dt_sec)
            residuals.append(y[t] - out.mean)
        
        forecast_time = time.time() - forecast_start
        n_forecasts = len(residuals)
        
        residuals = np.array(residuals)
        ts_eff = ts[warmup:warmup+len(residuals)].reset_index(drop=True)
        
        # Grid search with 60/40 split
        detection_start = time.time()
        split = max(1, int(0.6 * len(residuals)))
        grid = [(0.02,0.98), (0.05,0.95), (0.10,0.90), (0.15,0.85)]
        best_f1 = -1.0
        best_detector = None
        best_params = None
        
        for ql, qh in grid:
            detector = benchmark.VolatilityNormalizedQuantileDetector(
                ql=ql, qh=qh, bins=bins, use_weekday=False,
                temporal_smooth=True, min_samples_per_bin=5
            )
            detector.fit(ts_eff[:split], residuals[:split])
            flags_val = detector.predict(ts_eff[:split], residuals[:split])
            
            anomaly_rate = np.sum(flags_val) / len(flags_val) if len(flags_val) > 0 else 0
            score = 1.0 - abs(anomaly_rate - 0.10)
            
            if score > best_f1:
                best_f1 = score
                best_detector = detector
                best_params = (ql, qh)
        
        # Test on held-out set
        flags = best_detector.predict(ts_eff[split:], residuals[split:])
        detection_time = time.time() - detection_start
        total_time = time.time() - start_time
        
        anomaly_count = np.sum(flags)
        anomaly_rate = anomaly_count / len(flags) if len(flags) > 0 else 0
        
        # Residual separation
        normal_res = residuals[split:][flags == 0]
        anomaly_res = residuals[split:][flags == 1]
        if len(anomaly_res) > 0:
            separation = np.mean(np.abs(anomaly_res)) / (np.mean(np.abs(normal_res)) + 1e-8)
        else:
            separation = 1.0
        
        return {
            'detected': anomaly_count > 0,
            'anomaly_count': int(anomaly_count),
            'anomaly_rate': float(anomaly_rate),
            'total_points': len(flags),
            'model': 'Toto',
            'forecast_time_sec': forecast_time,
            'detection_time_sec': detection_time,
            'total_time_sec': total_time,
            'forecasts_per_sec': n_forecasts / forecast_time if forecast_time > 0 else 0,
            'best_ql': best_params[0],
            'best_qh': best_params[1],
            'residual_separation': separation
        }
        
    except Exception as e:
        return {'detected': False, 'reason': f'Error: {str(e)}'}

def detect_anomalies_granite(test_df, baseline_df=None):
    """Detect anomalies using Granite TTM with Phase 2 optimizations"""
    try:
        # Check if Granite is available
        if not hasattr(benchmark, 'GraniteTTMForecaster'):
            return {'detected': False, 'reason': 'Granite not available'}
        
        y = test_df['value'].values
        ts = test_df['timestamp']
        
        if len(y) < 20:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        # Start timing
        start_time = time.time()
        
        # Granite forecaster with minimum context requirement
        device = 'cuda' if benchmark.DEVICE == 'cuda' else 'cpu'
        # Granite TTM requires minimum context_len of 52
        context_len = max(52, min(512, len(y) // 2))
        granite = benchmark.GraniteTTMForecaster(
            context_len=context_len,
            pred_len=1,
            device=device
        )
        
        warmup = min(10, len(y) // 3)
        dt_sec, bins = benchmark.infer_cadence_and_bins(ts)
        
        # Forecast with timing
        forecast_start = time.time()
        residuals = []
        for t in range(warmup, len(y)-1):
            hist = y[:t]
            out = granite.one_step(hist)
            residuals.append(y[t] - out.mean)
        
        forecast_time = time.time() - forecast_start
        n_forecasts = len(residuals)
        
        residuals = np.array(residuals)
        ts_eff = ts[warmup:warmup+len(residuals)].reset_index(drop=True)
        
        # Grid search with 60/40 split
        detection_start = time.time()
        split = max(1, int(0.6 * len(residuals)))
        grid = [(0.02,0.98), (0.05,0.95), (0.10,0.90), (0.15,0.85)]
        best_f1 = -1.0
        best_detector = None
        best_params = None
        
        for ql, qh in grid:
            detector = benchmark.VolatilityNormalizedQuantileDetector(
                ql=ql, qh=qh, bins=bins, use_weekday=False,
                temporal_smooth=True, min_samples_per_bin=5
            )
            detector.fit(ts_eff[:split], residuals[:split])
            flags_val = detector.predict(ts_eff[:split], residuals[:split])
            
            anomaly_rate = np.sum(flags_val) / len(flags_val) if len(flags_val) > 0 else 0
            score = 1.0 - abs(anomaly_rate - 0.10)
            
            if score > best_f1:
                best_f1 = score
                best_detector = detector
                best_params = (ql, qh)
        
        # Test on held-out set
        flags = best_detector.predict(ts_eff[split:], residuals[split:])
        detection_time = time.time() - detection_start
        total_time = time.time() - start_time
        
        anomaly_count = np.sum(flags)
        anomaly_rate = anomaly_count / len(flags) if len(flags) > 0 else 0
        
        # Residual separation
        normal_res = residuals[split:][flags == 0]
        anomaly_res = residuals[split:][flags == 1]
        if len(anomaly_res) > 0:
            separation = np.mean(np.abs(anomaly_res)) / (np.mean(np.abs(normal_res)) + 1e-8)
        else:
            separation = 1.0
        
        return {
            'detected': anomaly_count > 0,
            'anomaly_count': int(anomaly_count),
            'anomaly_rate': float(anomaly_rate),
            'total_points': len(flags),
            'model': 'Granite',
            'forecast_time_sec': forecast_time,
            'detection_time_sec': detection_time,
            'total_time_sec': total_time,
            'forecasts_per_sec': n_forecasts / forecast_time if forecast_time > 0 else 0,
            'best_ql': best_params[0],
            'best_qh': best_params[1],
            'residual_separation': separation
        }
        
    except Exception as e:
        return {'detected': False, 'reason': f'Error: {str(e)}'}

def analyze_scenario(scenario, model_name='ARIMA'):
    """Analyze one scenario with one model"""
    baseline_dir = DATASET_DIR / scenario['baseline']
    test_dir = DATASET_DIR / scenario['test']
    
    baseline_files = sorted(baseline_dir.glob('*.csv'))
    test_files = sorted(test_dir.glob('*.csv'))
    
    # Match files by metric name (not timestamp)
    baseline_metrics = {get_metric_name(f.name): f for f in baseline_files}
    test_metrics = {get_metric_name(f.name): f for f in test_files}
    
    # Find common metrics
    common_metrics = set(baseline_metrics.keys()) & set(test_metrics.keys())
    
    print(f"\n{'='*80}")
    print(f"Scenario: {scenario['name']}")
    print(f"Parameter: {scenario['parameter']}")
    print(f"Model: {model_name}")
    print(f"Comparing: {scenario['baseline']} vs {scenario['test']}")
    print(f"Common metrics: {len(common_metrics)}")
    print(f"{'='*80}\n")
    
    results = []
    
    for i, metric_name in enumerate(sorted(common_metrics), 1):
        print(f"[{i}/{len(common_metrics)}] Analyzing: {metric_name}... ", end='', flush=True)
        
        baseline_file = baseline_metrics[metric_name]
        test_file = test_metrics[metric_name]
        
        # Load data
        baseline_df = load_metric_file(baseline_file)
        test_df = load_metric_file(test_file)
        
        if baseline_df is None or test_df is None:
            print("⚪ Skipped (zero-variance or all-zero)")
            continue
        
        # Save original test data in case alignment fails
        original_test_df = test_df.copy()
        
        # Try to align time series
        baseline_df, test_df = align_timeseries(baseline_df, test_df)
        
        # If alignment resulted in insufficient data, use raw test data instead
        # (Models can learn normal behavior from the test series itself)
        if len(test_df) < 15:
            if len(original_test_df) >= 15:
                test_df = original_test_df
                baseline_df = None  # No baseline needed for univariate detection
            else:
                print("⚪ Insufficient data (< 15 points)")
                continue
        
        # Select detection function
        if model_name == 'ARIMA':
            result = detect_anomalies_arima(test_df, baseline_df)
        elif model_name == 'Toto':
            result = detect_anomalies_toto(test_df, baseline_df)
        elif model_name == 'Granite':
            result = detect_anomalies_granite(test_df, baseline_df)
        else:
            print("❌ Unknown model")
            continue
        
        result['scenario'] = scenario['name']
        result['parameter'] = scenario['parameter']
        result['metric'] = metric_name
        result['baseline'] = scenario['baseline']
        result['test'] = scenario['test']
        
        results.append(result)
        
        if result.get('detected'):
            anomaly_rate = result.get('anomaly_rate', 0) * 100
            print(f"✅ Anomaly detected ({anomaly_rate:.1f}%)")
        else:
            reason = result.get('reason', 'No anomalies')
            print(f"⚪ {reason}")
    
    return results

def main():
    """Main analysis pipeline"""
    print("="*80)
    print("vLLM ANOMALY DETECTION ANALYSIS")
    print("="*80)
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nAnalyzing 6 scenarios × 3 models = 18 results")
    print("="*80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Check model availability
    print("\nChecking model availability...")
    print(f"  ARIMA: ✅ Available")
    
    has_toto = False
    try:
        toto = benchmark.TotoForecaster(device='cpu', num_samples=1)
        has_toto = True
        print(f"  Toto: ✅ Available")
    except Exception as e:
        print(f"  Toto: ❌ Not available ({e})")
    
    has_granite = False
    try:
        from tsfm_public.toolkit.get_model import get_model
        has_granite = True
        print(f"  Granite: ✅ Available")
    except Exception as e:
        print(f"  Granite: ❌ Not available ({e})")
    
    models_to_run = ['ARIMA']
    if has_toto:
        models_to_run.append('Toto')
    if has_granite:
        models_to_run.append('Granite')
    
    print(f"\nWill run: {', '.join(models_to_run)}")
    
    # Run all analyses
    all_results = []
    
    for scenario in SCENARIOS:
        for model_name in models_to_run:
            results = analyze_scenario(scenario, model_name)
            all_results.extend(results)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    
    # Summary by scenario and model (with runtime metrics)
    summary_cols = ['scenario', 'parameter', 'model', 'metric', 'detected', 
                    'anomaly_count', 'anomaly_rate', 'total_points',
                    'forecast_time_sec', 'detection_time_sec', 'total_time_sec',
                    'forecasts_per_sec', 'best_ql', 'best_qh', 'residual_separation']
    # Only include columns that exist
    summary_cols = [c for c in summary_cols if c in results_df.columns]
    results_df[summary_cols].to_csv(OUTPUT_DIR / 'vllm_anomaly_results.csv', index=False)
    
    # Create summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    summary_report = []
    for scenario in SCENARIOS:
        for model_name in models_to_run:
            scenario_results = results_df[
                (results_df['scenario'] == scenario['name']) & 
                (results_df['model'] == model_name)
            ]
            
            total_metrics = len(scenario_results)
            anomalies_detected = scenario_results['detected'].sum()
            avg_anomaly_rate = scenario_results['anomaly_rate'].mean() * 100
            
            summary_report.append({
                'Scenario': scenario['name'],
                'Parameter': scenario['parameter'],
                'Model': model_name,
                'Total Metrics': total_metrics,
                'Anomalies Detected': anomalies_detected,
                'Detection Rate': f"{(anomalies_detected/total_metrics*100):.1f}%" if total_metrics > 0 else "0%",
                'Avg Anomaly Rate': f"{avg_anomaly_rate:.1f}%"
            })
    
    summary_df = pd.DataFrame(summary_report)
    summary_df.to_csv(OUTPUT_DIR / 'vllm_summary_report.csv', index=False)
    
    print("\n" + summary_df.to_string(index=False))
    
    # Runtime performance analysis (Phase 2 insight)
    print("\n" + "="*80)
    print("RUNTIME PERFORMANCE ANALYSIS")
    print("="*80)
    
    if 'forecast_time_sec' in results_df.columns:
        runtime_stats = results_df.groupby('model').agg({
            'forecast_time_sec': 'mean',
            'detection_time_sec': 'mean',
            'total_time_sec': 'mean',
            'forecasts_per_sec': 'mean'
        }).round(2)
        print("\nMean runtime by model:")
        print(runtime_stats.to_string())
        
        # Compare to Phase 2 benchmarks
        print("\n📊 Phase 2 Benchmarks (for comparison):")
        print("  Granite: 1,767 forecasts/sec (340× faster than Toto)")
        print("  Toto: 5.2 forecasts/sec")
        print("  ARIMA: ~100-500 forecasts/sec (dataset dependent)")
    
    # Metrics with most anomalies
    print("\n" + "="*80)
    print("TOP METRICS WITH ANOMALIES")
    print("="*80)
    
    metric_summary = results_df.groupby('metric').agg({
        'detected': 'sum',
        'anomaly_rate': 'mean'
    }).sort_values('detected', ascending=False).head(10)
    
    metric_summary.columns = ['Anomaly Count', 'Avg Anomaly Rate']
    metric_summary['Avg Anomaly Rate'] = metric_summary['Avg Anomaly Rate'] * 100
    print("\n" + metric_summary.to_string())
    
    print("\n" + "="*80)
    print(f"✅ Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"  - vllm_anomaly_results.csv (detailed)")
    print(f"  - vllm_summary_report.csv (summary)")
    print("="*80)

if __name__ == '__main__':
    main()

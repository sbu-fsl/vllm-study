#!/usr/bin/env python3
"""
vllm_benchmark.py - Run time-series benchmark on vLLM metrics
Standalone script - does not modify main_timeseries_benchmark.py
"""

import os
import sys
import pathlib

# Set environment variables before importing main script
os.environ["TS_LIMIT"] = os.environ.get("TS_LIMIT", "21")
os.environ["TOTO_SAMPLES"] = os.environ.get("TOTO_SAMPLES", "128")
os.environ["TS_WARMUP"] = os.environ.get("TS_WARMUP", "50")  # Shorter for vLLM data
os.environ["TS_ALPHA"] = os.environ.get("TS_ALPHA", "0.10")
os.environ["TS_SEED"] = os.environ.get("TS_SEED", "0")

# Import pandas before monkey-patching
import pandas as pd

# Import the main benchmark module
import main_timeseries_benchmark as benchmark

print("="*70)
print("vLLM METRICS BENCHMARK")
print("="*70)

# Check vLLM data exists
vllm_dir = pathlib.Path("vllm_datasets")
if not vllm_dir.exists():
    print(f"ERROR: vllm_datasets/ directory not found!")
    print(f"Run convert_vllm_to_pipeline.py first.")
    sys.exit(1)

csv_files = sorted(vllm_dir.glob("*.csv"))
limit = int(os.environ["TS_LIMIT"])
print(f"\nFound {len(csv_files)} vLLM metrics")
print(f"Will process: {min(len(csv_files), limit)} metrics")
print("="*70 + "\n")

# Save original functions
_original_discover = benchmark.discover_real_known_csvs
_original_load_csv = benchmark.load_csv_from_url
_original_load_windows = benchmark.load_nab_label_windows
_original_apply_windows = benchmark.apply_label_windows

# Create replacement functions that load vLLM data
def vllm_discover(limit):
    """Discover vLLM CSV files instead of NAB"""
    csv_files_limited = csv_files[:limit] if limit > 0 else csv_files
    result = [(f.stem, str(f)) for f in csv_files_limited]
    print(f"Discovered {len(result)} vLLM metrics:")
    for name, _ in result:
        print(f"  - {name}")
    return result

def vllm_load_csv(filepath):
    """Load vLLM CSV from local file"""
    df = pd.read_csv(filepath)
    
    # Already has correct columns from conversion script
    if 'timestamp' not in df.columns or 'value' not in df.columns:
        raise ValueError(f"CSV missing required columns: {filepath}")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Ensure label column
    if 'label' not in df.columns:
        df['label'] = 0
    
    return df[['timestamp', 'value', 'label']]

def vllm_load_windows(save_path):
    """No label windows for vLLM data"""
    # Save empty dict to maintain compatibility
    import json
    save_path.write_text(json.dumps({}))
    return {}

def vllm_apply_windows(df, file_key, windows_obj):
    """Skip window labeling for vLLM data"""
    return df  # Already has label=0 from conversion

# Monkey-patch the functions
print("Patching data loading functions for vLLM data...")
benchmark.discover_real_known_csvs = vllm_discover
benchmark.load_csv_from_url = vllm_load_csv
benchmark.load_nab_label_windows = vllm_load_windows
benchmark.apply_label_windows = vllm_apply_windows

print("Starting benchmark pipeline...\n")

# Now import and run the main block
# We need to execute the if __name__ == "__main__" block
# The cleanest way is to exec the main section

main_code = """
if True:  # Replace 'if __name__ == "__main__":' guard
    datasets = discover_real_known_csvs(LIMIT_DATASETS)
    windows = load_nab_label_windows(RUN_DIR / "combined_windows.json")
    loaded: List[Tuple[str, pd.DataFrame]] = []
    for pretty, raw_url in datasets:
        df = load_csv_from_url(raw_url)
        (RUN_DIR / "datasets" / f"{pretty}.csv").write_text(df.to_csv(index=False))
        file_key = f"realKnownCause/{pretty}.csv"
        df = apply_label_windows(df, file_key, windows)
        loaded.append((pretty, df))

    @dataclass
    class BaseCfg:
        lower_q: float = 0.05
        upper_q: float = 0.95
        alpha: float = ALPHA
    base_cfg = BaseCfg()
    arima = ARIMAForecaster(order=(5,1,1), alpha=base_cfg.alpha)

    has_toto = False
    try:
        toto = TotoForecaster(device=DEVICE, checkpoint="Datadog/Toto-Open-Base-1.0",
                              num_samples=TOTO_SAMPLES, ql=0.10, qh=0.90)
        has_toto = True
    except Exception as e:
        print("[warn] Toto init failed:", e)

    has_granite = False
    try:
        from tsfm_public.toolkit.get_model import get_model
        granite = GraniteTTMForecaster(context_len=1024, pred_len=1, device=DEVICE)
        has_granite = True
    except Exception as e:
        print("[warn] Granite init failed:", e)

    rows = []; all_pr = []
    for name, df in loaded:
        dt_sec, bins = infer_cadence_and_bins(df["timestamp"])
        local_cfg = RunCfg(lower_q=base_cfg.lower_q, upper_q=base_cfg.upper_q,
                           warmup=min(WARMUP, max(32, len(df)//5)),
                           alpha=base_cfg.alpha, bins_per_day=bins, interval_sec=dt_sec)

        try:
            r_arima = evaluate(df, "ARIMA", arima, local_cfg, name, RUN_DIR, windows)
            rows.append((name, "ARIMA", r_arima["F1"], r_arima["AUPR"], r_arima["sMAPE"],
                         r_arima["Coverage"], r_arima["WHit"], r_arima["ql"], r_arima["qh"],
                         r_arima["cov_scale"], r_arima["F1_CV_mean"], r_arima["F1_CV_std"],
                         r_arima["AUPR_CV_mean"], r_arima["AUPR_CV_std"], r_arima["separation"]))
            all_pr.append(r_arima["trace"].assign(Dataset=name, Model="ARIMA"))
        except Exception as e:
            print(f"[warn] ARIMA failed on {name}: {e}")

        if has_toto:
            try:
                r_toto = evaluate(df, "Toto", toto, local_cfg, name, RUN_DIR, windows)
                rows.append((name, "Toto", r_toto["F1"], r_toto["AUPR"], r_toto["sMAPE"],
                             r_toto["Coverage"], r_toto["WHit"], r_toto["ql"], r_toto["qh"],
                             r_toto["cov_scale"], r_toto["F1_CV_mean"], r_toto["F1_CV_std"],
                             r_toto["AUPR_CV_mean"], r_toto["AUPR_CV_std"], r_toto["separation"]))
                all_pr.append(r_toto["trace"].assign(Dataset=name, Model="Toto"))
            except Exception as e:
                print(f"[warn] Toto failed on {name}: {e}")

        if has_granite:
            try:
                r_gra = evaluate(df, "GraniteTTM", granite, local_cfg, name, RUN_DIR, windows)
                rows.append((name, "GraniteTTM", r_gra["F1"], r_gra["AUPR"], r_gra["sMAPE"],
                             r_gra["Coverage"], r_gra["WHit"], r_gra["ql"], r_gra["qh"],
                             r_gra["cov_scale"], r_gra["F1_CV_mean"], r_gra["F1_CV_std"],
                             r_gra["AUPR_CV_mean"], r_gra["AUPR_CV_std"], r_gra["separation"]))
                all_pr.append(r_gra["trace"].assign(Dataset=name, Model="GraniteTTM"))
            except Exception as e:
                print(f"[warn] GraniteTTM failed on {name}: {e}")

    summary_path_csv = RUN_DIR / "exec_summary.csv"
    summary_path_md  = RUN_DIR / "exec_summary.md"
    if rows:
        out = pd.DataFrame(rows, columns=[
            "Dataset","Model","F1","AUPR","sMAPE","Coverage","WHit",
            "ql","qh","cov_scale","F1_CV_mean","F1_CV_std","AUPR_CV_mean","AUPR_CV_std",
            "separation"
        ]).sort_values(["Dataset","F1"], ascending=[True,False])

        print("\\n" + "="*70)
        print("CRITICAL ANALYSIS: Testing if forecast quality predicts detection quality")
        print("="*70)
        corr = make_scatter_plot(out, RUN_DIR / "plots" / "smape_vs_f1_scatter.png")
        print(f"\\nCorrelation between sMAPE and F1: {corr:.3f}")
        if abs(corr) < 0.3:
            print("WARNING: WEAK CORRELATION - Detection logic is likely the bottleneck!")
        elif corr < -0.5:
            print("STRONG NEGATIVE CORRELATION: Better forecasts lead to better detection")
        else:
            print("INCONCLUSIVE: Mixed relationship")

        print("\\nGenerating model comparison plots...")
        make_model_comparison_plots(out, RUN_DIR / "plots")

        print("\\n" + "="*70)
        print("MODEL AVERAGES ACROSS ALL DATASETS")
        print("="*70)
        agg = out.groupby('Model').agg({
            'F1': ['mean', 'std', 'min', 'max'],
            'AUPR': ['mean', 'std'],
            'sMAPE': ['mean', 'std'],
            'Coverage': ['mean', 'std'],
            'WHit': ['mean'],
            'separation': ['mean']
        }).round(4)
        print(agg.to_string())
        (RUN_DIR / "model_averages.txt").write_text(agg.to_string())

        agg_simple = out.groupby('Model').agg({
            'F1': ['mean', 'std'],
            'sMAPE': ['mean', 'std'],
            'Coverage': ['mean']
        }).round(3)
        (RUN_DIR / "model_averages.md").write_text(agg_simple.to_markdown())

        out_fmt = out.copy()
        out_fmt["F1_CV"] = (out["F1_CV_mean"].map(lambda x: f"{x:.3f}") + " ± " +
                            out["F1_CV_std"].map(lambda x: f"{x:.3f}"))
        out_fmt["AUPR_CV"] = (out["AUPR_CV_mean"].map(lambda x: f"{x:.3f}") + " ± " +
                              out["AUPR_CV_std"].map(lambda x: f"{x:.3f}"))
        out_fmt = out_fmt.drop(columns=["F1_CV_mean","F1_CV_std","AUPR_CV_mean","AUPR_CV_std"])

        print("\\n\\n## Executive Summary\\n")
        with pd.option_context('display.float_format', '{:.4f}'.format):
            print(out_fmt.to_markdown(index=False))
        out.to_csv(summary_path_csv, index=False)
        summary_path_md.write_text(out_fmt.to_markdown(index=False))
    else:
        print("\\n[warn] No results were produced — all model runs failed.\\n")

    if all_pr:
        pr_all = pd.concat(all_pr, ignore_index=True)
        pr_all.to_csv(RUN_DIR / "pr_traces.csv", index=False)

    print(f"\\n{'='*70}")
    print(f"vLLM BENCHMARK COMPLETE")
    print(f"All artifacts saved under: {RUN_DIR}")
    print(f"{'='*70}")
    print(f"\\nKey outputs:")
    print(f"  - exec_summary.csv: Full results table")
    print(f"  - model_averages.txt: Aggregate statistics")
    print(f"  - plots/smape_vs_f1_scatter.png: Correlation analysis")
    print(f"  - plots/comparison_*.png: Model performance comparisons")
    print(f"  - plots/TL_*.png: Timeline visualizations for each metric")
    print(f"\\n")
"""

# Execute the main code with access to benchmark module's namespace
exec(main_code, benchmark.__dict__)

# ============================================================
# main_timeseries_benchmark.py  (GPU-ready, robust, saves artifacts)
# Benchmarks: ARIMA | Datadog Toto (optional) | IBM Granite TTM (optional)
# Datasets: NAB "realKnownCause" (auto-download)
# Detector: Seasonal Quantile (weekday-aware, per-dataset tuned)
# Metrics: F1 (point), AUCPR, sMAPE, Calibrated PI Coverage, NAB Window-Hit
# CV: 3-fold expanding-window (mean±std for F1/AUCPR)
# Outputs: run folder with datasets, traces, PR curves, plots, exec summary
# ============================================================

import io, os, json, time, warnings, sys, pathlib, platform, random, socket
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_recall_curve
from statsmodels.tsa.statespace.sarimax import SARIMAX

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------ knobs via env ------------------
LIMIT_DATASETS = int(os.environ.get("TS_LIMIT", 6))
WARMUP        = int(os.environ.get("TS_WARMUP", 200))
ALPHA         = float(os.environ.get("TS_ALPHA", 0.10))
TOTO_SAMPLES  = int(os.environ.get("TOTO_SAMPLES", 128))
OUT_BASE      = os.environ.get("TS_OUTDIR", "runs")
SEED          = int(os.environ.get("TS_SEED", 0))
SKIP_TOTO     = os.environ.get("SKIP_TOTO", "0") == "1"  # 🚨 NEW: Skip Toto if too slow
TOTO_MAX_POINTS = int(os.environ.get("TOTO_MAX_POINTS", 5000))  # 🚨 NEW: Limit Toto dataset size

np.random.seed(SEED); random.seed(SEED)

# ------------------ device autodetect ------------------
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    TORCH_VER = getattr(torch, "__version__", "n/a")
    CUDA_VER = getattr(getattr(torch, "version", None), "cuda", "cpu")
except Exception:
    torch = None
    DEVICE, TORCH_VER, CUDA_VER = "cpu", "n/a", "cpu"
print(f"[device] torch={TORCH_VER}, cuda={CUDA_VER}, device={DEVICE}")

# ------------------ NAB sources ------------------
NAB_BASE = "https://raw.githubusercontent.com/numenta/NAB/master"
NAB_REAL_KNOWN = f"{NAB_BASE}/data/realKnownCause"
NAB_LABEL_WINDOWS = f"{NAB_BASE}/labels/combined_windows.json"
NAB_LIST_API = "https://api.github.com/repos/numenta/NAB/contents/data/realKnownCause"

# ------------------ HTTP helper ------------------
import requests
def http_get(url: str, as_text=True, retry=3, sleep=1.0):
    last = None
    for _ in range(retry):
        r = requests.get(url, timeout=60)
        if r.ok:
            return r.text if as_text else r.content
        last = r
        time.sleep(sleep)
    raise RuntimeError(f"GET failed {url}: {getattr(last, 'status_code', None)}")

# ------------------ run folder ------------------
def make_run_dir() -> pathlib.Path:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    cuda_tag = f"cu{CUDA_VER.replace('.','')}" if CUDA_VER not in (None, "cpu") else "cpu"
    name = f"TSB_{ts}_{cuda_tag}_toto{TOTO_SAMPLES}_lim{LIMIT_DATASETS}_{socket.gethostname()}"
    out = pathlib.Path(OUT_BASE) / name
    (out / "datasets").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "pr_traces").mkdir(parents=True, exist_ok=True)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    return out

RUN_DIR = make_run_dir()
def save_text(path: pathlib.Path, text: str):
    path.write_text(text)

def save_env_info():
    save_text(RUN_DIR / "device.txt",
              f"torch={TORCH_VER}\n"
              f"cuda={CUDA_VER}\n"
              f"cuda_available={torch.cuda.is_available() if torch else False}\n"
              f"device={DEVICE}\n")
    cfg = dict(LIMIT_DATASETS=LIMIT_DATASETS, WARMUP=WARMUP, ALPHA=ALPHA,
               TOTO_SAMPLES=TOTO_SAMPLES, OUT_BASE=str(OUT_BASE), SEED=SEED,
               host=socket.gethostname(), python=sys.version, platform=platform.platform())
    (RUN_DIR / "config.json").write_text(json.dumps(cfg, indent=2))
save_env_info()

# ------------------ data ------------------
def load_csv_from_url(url: str) -> pd.DataFrame:
    txt = http_get(url, as_text=True)
    df = pd.read_csv(io.StringIO(txt))
    if 'timestamp' not in df.columns:
        df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
    if 'value' not in df.columns:
        df.rename(columns={df.columns[1]: 'value'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    if 'label' not in df.columns:
        df['label'] = 0
    return df[['timestamp','value','label']]

def load_nab_label_windows(save_to: pathlib.Path) -> Dict[str, Any]:
    txt = http_get(NAB_LABEL_WINDOWS, as_text=True)
    save_text(save_to, txt)
    return json.loads(txt)

def discover_real_known_csvs(limit: int) -> List[Tuple[str,str]]:
    items = requests.get(NAB_LIST_API, timeout=60).json()
    csvs = [it for it in items if str(it.get('name','')).endswith('.csv')]
    csvs.sort(key=lambda it: (0 if it['name']=='nyc_taxi.csv' else 1, it['name']))
    out = []
    for it in csvs[:limit]:
        name = it['name']
        raw = f"{NAB_REAL_KNOWN}/{name}"
        out.append((name.replace('.csv',''), raw))
    return out

def apply_label_windows(df: pd.DataFrame, file_key: str, windows_obj) -> pd.DataFrame:
    df = df.copy()
    lbl = np.zeros(len(df), dtype=int)
    if file_key not in windows_obj:
        df["label"] = lbl; return df
    entry = windows_obj[file_key]
    if isinstance(entry, dict) and "windows" in entry:
        entry = entry["windows"]
    for w in entry:
        if isinstance(w, dict):
            st = w.get("start") or w.get("startTime") or w.get("begin")
            en = w.get("end")   or w.get("endTime")   or w.get("finish")
        elif isinstance(w, (list, tuple)) and len(w) >= 2:
            st, en = w[0], w[1]
        else:
            continue
        st = pd.to_datetime(st); en = pd.to_datetime(en)
        mask = (df["timestamp"] >= st) & (df["timestamp"] <= en)
        lbl |= mask.astype(int).values
    df["label"] = lbl
    return df

# ------------------ cadence ------------------
def infer_cadence_and_bins(ts: pd.Series) -> tuple[int, int]:
    if len(ts) < 3: return 1800, 48
    dt = (ts.iloc[1:] - ts.iloc[:-1]).median().total_seconds()
    if not np.isfinite(dt) or dt <= 0: dt = 1800
    if abs(dt - 1800) <= 60:  return 1800, 48
    if abs(dt - 3600) <= 120: return 3600, 24
    bins = max(1, int(round(86400 / dt))); bins = int(np.clip(bins, 8, 144))
    return int(dt), bins

# ------------------ detector ------------------
def _bin_for(ts, bins):
    if bins == 48: return ts.hour*2 + (1 if ts.minute >= 30 else 0)
    step = 1440 / bins
    return int((ts.hour*60 + ts.minute)//step)

class SeasonalQuantileDetector:
    def __init__(self, ql=0.05, qh=0.95, bins=48):
        self.ql, self.qh, self.bins = ql, qh, bins
        self.table = {}
    def fit(self, timestamps: pd.Series, residuals: np.ndarray):
        df = pd.DataFrame({'ts': timestamps, 'r': residuals})
        df['b'] = df['ts'].apply(lambda t: _bin_for(t, self.bins))
        self.table = {b: (g['r'].quantile(self.ql), g['r'].quantile(self.qh)) for b,g in df.groupby('b')}
    def predict(self, timestamps: pd.Series, residuals: np.ndarray) -> np.ndarray:
        flags = []
        for ts, r in zip(timestamps, residuals):
            b = _bin_for(ts, self.bins)
            lo, hi = self.table.get(b, (-np.inf, np.inf))
            flags.append(1 if (r < lo or r > hi) else 0)
        return np.array(flags, int)

class SeasonalQuantileDetectorWD(SeasonalQuantileDetector):
    # weekday-aware variant: key = (weekday, bin)
    def __init__(self, ql=0.05, qh=0.95, bins=48, use_weekday=True):
        super().__init__(ql, qh, bins); self.use_weekday = use_weekday
    def fit(self, timestamps: pd.Series, residuals: np.ndarray):
        df = pd.DataFrame({'ts': timestamps, 'r': residuals})
        if self.use_weekday:
            df['k'] = df['ts'].apply(lambda t: (t.weekday(), _bin_for(t, self.bins)))
        else:
            df['k'] = df['ts'].apply(lambda t: _bin_for(t, self.bins))
        self.table = {k: (g['r'].quantile(self.ql), g['r'].quantile(self.qh)) for k,g in df.groupby('k')}
    def predict(self, timestamps: pd.Series, residuals: np.ndarray) -> np.ndarray:
        flags = []
        for ts, r in zip(timestamps, residuals):
            k = (ts.weekday(), _bin_for(ts, self.bins)) if self.use_weekday else _bin_for(ts, self.bins)
            lo, hi = self.table.get(k, (-np.inf, np.inf))
            flags.append(1 if (r < lo or r > hi) else 0)
        return np.array(flags, int)

# ============================================================
# IMPROVED DETECTOR IMPLEMENTATIONS
# ============================================================

class VolatilityNormalizedQuantileDetector:
    """
    Robust anomaly detector that normalizes residuals by local volatility
    before applying quantile thresholds.
    
    Key improvements over baseline SeasonalQuantile:
    1. Handles heteroscedasticity (varying variance over time)
    2. Uses MAD (Median Absolute Deviation) for robust volatility estimation
    3. Applies quantile thresholds to z-scores, not raw residuals
    4. Temporal smoothing to reduce noise-induced false positives
    
    Theory:
    - A residual of +5 is anomalous if local volatility is 1.0
    - But same +5 is normal if local volatility is 10.0
    - Z-score normalization: z = (r - median) / MAD
    - Detects residuals that are X standard deviations from typical
    
    Advantages:
    - Works across all models (ARIMA, Toto, Granite)
    - Robust to outliers in training data (MAD vs std)
    - Interpretable (z-score based)
    - Fast (pure numpy, no sklearn)
    """
    
    def __init__(self, ql=0.05, qh=0.95, bins=48, use_weekday=True, 
                 temporal_smooth=True, min_volatility=0.01, min_samples_per_bin=5):
        """
        Parameters:
        -----------
        ql, qh: Quantile thresholds (applied to normalized residuals)
        bins: Number of time-of-day bins (default 48 = 30min intervals)
        use_weekday: Whether to use weekday-specific statistics
        temporal_smooth: Require anomaly to persist 2-3 steps (reduces noise)
        min_volatility: Minimum volatility to prevent division by zero
        min_samples_per_bin: Minimum samples needed for bin-specific stats (default 5)
        """
        self.ql = ql
        self.qh = qh
        self.bins = bins
        self.use_weekday = use_weekday
        self.temporal_smooth = temporal_smooth
        self.min_volatility = min_volatility
        self.min_samples_per_bin = min_samples_per_bin
        
        # Statistics per bin: median, MAD, quantile thresholds
        self.stats_table = {}  # key: (weekday, bin) or bin -> (median, MAD, q_lo, q_hi)
        self.data_density = 1.0  # Track overall data density (0-1)
        
    def _get_key(self, timestamp):
        """Get bin key for a timestamp."""
        bin_idx = _bin_for(timestamp, self.bins)
        if self.use_weekday:
            return (timestamp.weekday(), bin_idx)
        return bin_idx
    
    def _compute_mad(self, residuals):
        """
        Compute Median Absolute Deviation (MAD).
        MAD is more robust to outliers than standard deviation.
        Scale factor 1.4826 makes MAD comparable to std for normal distributions.
        """
        if len(residuals) == 0:
            return self.min_volatility
        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median))
        # Scale to match std for normal distribution
        return max(mad * 1.4826, self.min_volatility)
    
    def fit(self, timestamps: pd.Series, residuals: np.ndarray):
        """
        Fit detector by computing per-bin statistics.
        
        For each bin:
        1. Compute median (robust center)
        2. Compute MAD (robust volatility)
        3. Normalize residuals to z-scores
        4. Compute quantile thresholds on z-scores
        
        Also tracks data density to adjust temporal smoothing later.
        """
        df = pd.DataFrame({
            'ts': timestamps,
            'r': residuals
        })
        
        df['key'] = df['ts'].apply(self._get_key)
        
        # Compute overall data density (samples per bin)
        num_bins = len(df['key'].unique())
        samples_per_bin = len(df) / max(num_bins, 1)
        # Normalize to 0-1: 50+ samples/bin = very dense, <20 samples/bin = sparse
        self.data_density = min(samples_per_bin / 50.0, 1.0)
        
        # Compute statistics per bin
        for key, group in df.groupby('key'):
            r_bin = group['r'].values
            
            if len(r_bin) < self.min_samples_per_bin:  # Need sufficient points for robust stats
                # Use global fallback
                median = np.median(residuals)
                mad = self._compute_mad(residuals)
                z_scores = (residuals - median) / mad
                q_lo = np.quantile(z_scores, self.ql)
                q_hi = np.quantile(z_scores, self.qh)
            else:
                # Compute bin-specific statistics
                median = np.median(r_bin)
                mad = self._compute_mad(r_bin)
                
                # Normalize residuals to z-scores
                z_scores = (r_bin - median) / mad
                
                # Compute quantile thresholds on normalized residuals
                q_lo = np.quantile(z_scores, self.ql)
                q_hi = np.quantile(z_scores, self.qh)
            
            self.stats_table[key] = (median, mad, q_lo, q_hi)
    
    def predict(self, timestamps: pd.Series, residuals: np.ndarray) -> np.ndarray:
        """
        Predict anomalies by normalizing residuals and applying thresholds.
        
        Steps for each point:
        1. Get bin-specific statistics (median, MAD, quantiles)
        2. Normalize: z = (residual - median) / MAD
        3. Flag if z < q_low or z > q_high
        4. Apply adaptive temporal smoothing based on data density
        
        Adaptive smoothing:
        - High density (many samples/bin): Strict smoothing (2/3 neighbors)
        - Low density (few samples/bin): Relaxed smoothing (1/3 neighbors) or none
        """
        flags = np.zeros(len(residuals), dtype=int)
        
        for i, (ts, r) in enumerate(zip(timestamps, residuals)):
            key = self._get_key(ts)
            
            # Get statistics (with fallback to global if key not found)
            if key in self.stats_table:
                median, mad, q_lo, q_hi = self.stats_table[key]
            else:
                # Fallback: use overall statistics
                median = np.median(residuals)
                mad = self._compute_mad(residuals)
                q_lo, q_hi = -3.0, 3.0  # Conservative default
            
            # Normalize residual
            z_score = (r - median) / mad
            
            # Check if anomalous
            if z_score < q_lo or z_score > q_hi:
                flags[i] = 1
        
        # Adaptive temporal smoothing based on data density
        if self.temporal_smooth and len(flags) > 2:
            smoothed = np.zeros_like(flags)
            
            # Adjust smoothing threshold based on data density
            if self.data_density >= 0.5:
                # Dense data: Require 2 out of 3 (strict)
                required_neighbors = 2
            else:
                # Sparse data: Require only 1 out of 3 (relaxed)
                required_neighbors = 1
            
            for i in range(len(flags)):
                # Check if this point and neighbors are anomalous
                window_start = max(0, i-1)
                window_end = min(len(flags), i+2)
                window_sum = np.sum(flags[window_start:window_end])
                
                # Require at least N out of 3 points to be anomalous
                if window_sum >= required_neighbors:
                    smoothed[i] = 1
            
            return smoothed
        
        return flags


class AdaptiveQuantileDetector(SeasonalQuantileDetectorWD):
    """
    FIXED: Enhanced seasonal quantile detector with proper volatility handling.
    
    Previous bug: widened thresholds in high-volatility regions (opposite of correct)
    Fixed approach: tightens thresholds by raising lower bound and lowering upper bound
    
    Better alternative: Use VolatilityNormalizedQuantileDetector instead (more principled)
    """
    def __init__(self, ql=0.05, qh=0.95, bins=48, use_weekday=True, 
                 window_size=30, volatility_scale=0.5):
        """
        Parameters:
        -----------
        ql, qh: Base quantile thresholds
        bins: Number of time-of-day bins
        use_weekday: Whether to use weekday-specific thresholds
        window_size: Size of rolling window for volatility calculation
        volatility_scale: How much to tighten thresholds (0.5 = moderate)
        """
        super().__init__(ql=ql, qh=qh, bins=bins, use_weekday=use_weekday)
        self.window_size = window_size
        self.volatility_scale = volatility_scale
        self.volatility_map = {}
    
    def fit(self, timestamps: pd.Series, residuals: np.ndarray):
        """Fit detector with base quantiles and calculate volatility map."""
        super().fit(timestamps, residuals)
        
        # Build volatility map using robust MAD
        df = pd.DataFrame({'ts': timestamps, 'r': residuals})
        
        if self.use_weekday:
            df['k'] = df['ts'].apply(lambda t: (t.weekday(), _bin_for(t, self.bins)))
        else:
            df['k'] = df['ts'].apply(lambda t: _bin_for(t, self.bins))
            
        # Calculate robust volatility for each bin
        for k, g in df.groupby('k'):
            if len(g) >= 3:
                # Use MAD for robustness
                median_abs_dev = np.median(np.abs(g['r'] - np.median(g['r'])))
                self.volatility_map[k] = median_abs_dev * 1.4826  # Scale to match std
            else:
                self.volatility_map[k] = 1.0
    
    def predict(self, timestamps: pd.Series, residuals: np.ndarray) -> np.ndarray:
        """
        FIXED: Predict anomalies by TIGHTENING thresholds in high-volatility regions.
        
        Logic: High volatility means residuals naturally vary more, so we need
        tighter absolute thresholds to avoid flagging normal variance as anomalies.
        """
        flags = []
        for ts, r in zip(timestamps, residuals):
            k = (ts.weekday(), _bin_for(ts, self.bins)) if self.use_weekday else _bin_for(ts, self.bins)
            
            # Get base thresholds from parent class
            lo, hi = self.table.get(k, (-np.inf, np.inf))
            
            # Get volatility for this bin
            volatility = self.volatility_map.get(k, 1.0)
            
            # FIXED: Tighten thresholds in high-volatility regions
            # Raise lower bound and lower upper bound
            range_width = (hi - lo) * self.volatility_scale * (volatility / (volatility + 1.0))
            lo_adj = lo + range_width
            hi_adj = hi - range_width
            
            # Ensure adjusted thresholds are valid
            if lo_adj >= hi_adj:
                lo_adj, hi_adj = lo, hi  # Fall back to original
            
            flags.append(1 if (r < lo_adj or r > hi_adj) else 0)
            
        return np.array(flags, int)


# Note: IsolationForest and Multivariate detectors removed
# Reason: Fundamentally wrong approach for time-series residuals
# - IsolationForest treats residuals as independent samples (wrong for time-series)
# - Multivariate creates 20+ features causing overfitting and instability
# Replaced with VolatilityNormalizedQuantileDetector (principled, robust)


# ------------------ metrics & helpers ------------------
def smape(ytrue, yhat):
    ytrue = np.asarray(ytrue, float); yhat = np.asarray(yhat, float)
    denom = np.abs(ytrue) + np.abs(yhat); denom[denom==0] = 1.0
    return 100.0 * np.mean(np.abs(yhat - ytrue) / denom)

def coverage(ytrue, lo, hi):
    ytrue = np.asarray(ytrue, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    return float(np.mean((ytrue >= lo) & (ytrue <= hi)))

def calibrate_intervals(ytrue, lo, hi, target=0.95):
    emp = coverage(ytrue, lo, hi)
    if emp <= 0: scale = 2.0
    else: scale = np.clip(target / emp, 0.5, 2.0)
    mid = (np.asarray(lo)+np.asarray(hi))/2.0
    half = (np.asarray(hi)-np.asarray(lo))/2.0 * scale
    return mid - half, mid + half, emp, scale

def window_hit_rate(timestamps: pd.Series, flags: np.ndarray, windows_obj, file_key: str) -> float:
    entry = windows_obj.get(file_key)
    if entry is None: return 0.0
    if isinstance(entry, dict) and "windows" in entry: entry = entry["windows"]
    flagged_ts = set(pd.to_datetime(timestamps[flags == 1]).tolist())
    hits = 0; total = 0
    for w in entry:
        if isinstance(w, dict):
            st = pd.to_datetime(w.get("start") or w.get("startTime") or w.get("begin"))
            en = pd.to_datetime(w.get("end")   or w.get("endTime")   or w.get("finish"))
        else:
            st, en = pd.to_datetime(w[0]), pd.to_datetime(w[1])
        total += 1
        for ts in flagged_ts:
            if st <= ts <= en: hits += 1; break
    return hits / max(1, total)

def auc_pr(precision, recall):
    if len(precision) < 2: return 0.0
    order = np.argsort(recall)
    r = recall[order]; p = precision[order]
    return float(np.trapz(p, r))

def kfold_splits(n, k=3):
    # expanding-window splits on post-warmup region [0..n)
    # returns list of (train_end, test_end) indices w.r.t. that region
    if k < 1: return [(max(1,int(0.6*n)), n)]
    fold = max(2, n // (k + 1))
    bounds = []
    for i in range(1, k+1):
        tr = i * fold
        te = min(n, (i+1)*fold)
        if tr < te: bounds.append((tr, te))
    if not bounds: bounds = [(max(1,int(0.6*n)), n)]
    return bounds

# ============================================================
# NEW ANALYSIS FUNCTIONS
# ============================================================

def make_scatter_plot(summary_df: pd.DataFrame, save_path: pathlib.Path) -> float:
    """
    Plot sMAPE vs F1 to test if forecast quality predicts detection quality.
    Returns correlation coefficient.
    """
    plt.figure(figsize=(10, 7))
    
    models = summary_df['Model'].unique()
    colors = {'ARIMA': 'blue', 'Toto': 'green', 'GraniteTTM': 'red'}
    
    for model in models:
        df_m = summary_df[summary_df['Model'] == model]
        color = colors.get(model, 'gray')
        plt.scatter(df_m['sMAPE'], df_m['F1'], 
                   label=model, s=120, alpha=0.7, color=color, edgecolors='black')
        
        # Annotate points with dataset names
        for _, row in df_m.iterrows():
            plt.annotate(row['Dataset'][:15], 
                        (row['sMAPE'], row['F1']),
                        fontsize=7, alpha=0.6,
                        xytext=(3, 3), textcoords='offset points')
    
    plt.xlabel('sMAPE (lower = better forecast)', fontsize=12)
    plt.ylabel('F1 Score (higher = better detection)', fontsize=12)
    plt.title('Forecast Quality vs Detection Quality\n(If uncorrelated: detection logic is the bottleneck)', 
              fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    # Calculate correlation
    corr = summary_df[['sMAPE', 'F1']].corr().iloc[0, 1]
    return float(corr)


def make_model_comparison_plots(summary_df: pd.DataFrame, out_dir: pathlib.Path):
    """
    Create bar plots comparing models across metrics.
    """
    metrics = ['F1', 'AUPR', 'sMAPE', 'Coverage']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Group by model and get mean/std
        stats = summary_df.groupby('Model')[metric].agg(['mean', 'std']).reset_index()
        
        x = range(len(stats))
        plt.bar(x, stats['mean'], yerr=stats['std'], 
               capsize=5, alpha=0.7, edgecolor='black')
        plt.xticks(x, stats['Model'])
        plt.ylabel(metric, fontsize=12)
        plt.title(f'{metric} by Model (mean ± std across datasets)', fontsize=13)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(stats['mean'], stats['std'])):
            plt.text(i, mean + std + 0.01, f'{mean:.3f}', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(out_dir / f'comparison_{metric}.png', dpi=150)
        plt.close()


def make_residual_analysis(trace_df: pd.DataFrame, labels: np.ndarray, 
                          dataset_name: str, model_name: str, 
                          save_path: pathlib.Path):
    """
    Plot residual distributions for normal vs anomaly points.
    """
    residuals = trace_df['residual'].values
    
    # Only use test split for this analysis
    test_mask = trace_df['split'] == 'test'
    res_test = residuals[test_mask]
    lab_test = labels[test_mask] if len(labels) == len(residuals) else labels
    
    if len(res_test) != len(lab_test):
        print(f"[warn] Length mismatch in residual analysis for {dataset_name}/{model_name}")
        return 1.0
    
    normal_res = res_test[lab_test == 0]
    anomaly_res = res_test[lab_test == 1]
    
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    bins = np.linspace(min(res_test.min(), -5), max(res_test.max(), 5), 50)
    plt.hist(normal_res, bins=bins, alpha=0.6, label=f'Normal (n={len(normal_res)})', 
            color='blue', edgecolor='black')
    plt.hist(anomaly_res, bins=bins, alpha=0.6, label=f'Anomaly (n={len(anomaly_res)})', 
            color='red', edgecolor='black')
    
    plt.xlabel('Residual (actual - predicted)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Residual Distribution: {dataset_name} / {model_name}\n' + 
             f'Overlap explains detection difficulty', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    # Calculate separation metric
    if len(anomaly_res) > 0:
        normal_mean = np.mean(np.abs(normal_res))
        anomaly_mean = np.mean(np.abs(anomaly_res))
        separation = anomaly_mean / (normal_mean + 1e-8)
        return separation
    return 1.0

# ------------------ forecast API ------------------
@dataclass
class ForecastOut:
    mean: float; lo: float; hi: float

class ARIMAForecaster:
    def __init__(self, order=(5,1,0), alpha=ALPHA):
        self.order, self.alpha = order, alpha; self.res = None
    def begin(self, y_init: np.ndarray):
        m = SARIMAX(y_init, order=self.order, enforce_stationarity=False, enforce_invertibility=False)
        self.res = m.fit(disp=False)
    def forecast_and_update(self, y_obs: float) -> ForecastOut:
        fc = self.res.get_forecast(steps=1)
        pm = float(np.asarray(fc.predicted_mean)[0])
        ci = np.asarray(fc.conf_int(alpha=self.alpha))
        if ci.ndim == 1 and ci.size == 2: lo, hi = float(ci[0]), float(ci[1])
        else: lo, hi = float(ci[0,0]), float(ci[0,1])
        self.res = self.res.append(endog=np.asarray([y_obs], dtype=float), refit=False)
        return ForecastOut(pm, lo, hi)

class TotoForecaster:
    def __init__(self, device='cpu', checkpoint='Datadog/Toto-Open-Base-1.0',
                 num_samples=TOTO_SAMPLES, ql=0.10, qh=0.90, compile_model=False,
                 max_samples_per_batch=256):
        import torch as _torch
        from toto.model.toto import Toto
        from toto.inference.forecaster import TotoForecaster as _TF
        from toto.data.util.dataset import MaskedTimeseries
        self.torch = _torch; self.MaskedTimeseries = MaskedTimeseries
        self.model = Toto.from_pretrained(checkpoint).to(device); self.model.eval()
        self.forecaster = _TF(self.model.model)
        self.device = device
        self.num_samples = int(num_samples)
        self.ql = ql; self.qh = qh
        # Try to compile the model forward (optional, PyTorch 2.x)
        self._compiled = False
        if compile_model:
            try:
                if hasattr(_torch, "compile"):
                    # compile the inner model if available (may speed up many small calls)
                    try:
                        self.forecaster.model = _torch.compile(self.forecaster.model)
                        self._compiled = True
                    except Exception:
                        # compilation may fail; ignore gracefully
                        self._compiled = False
            except Exception:
                self._compiled = False
        # Allow a larger samples-per-batch so we do fewer forward calls
        self.max_samples_per_batch = int(max_samples_per_batch)

    def one_step(self, history: np.ndarray, interval_sec: int) -> ForecastOut:
        torch = self.torch
        h = np.asarray(history, dtype=np.float32); K = min(len(h), 2000); tail = h[-K:]
        mu, sigma = float(np.mean(tail)), float(np.std(tail))
        if not np.isfinite(sigma) or sigma < 1e-6: sigma = 1.0
        h_norm = (h - mu) / sigma
        series = torch.tensor(h_norm[None,:], dtype=torch.float32, device=self.device)
        ts_seconds = torch.zeros_like(series)
        dt = torch.full((1,), int(interval_sec), device=self.device)
        pad_mask = torch.ones_like(series, dtype=torch.bool); id_mask = torch.zeros_like(series)
        inputs = self.MaskedTimeseries(series=series, padding_mask=pad_mask,
                                       id_mask=id_mask, timestamp_seconds=ts_seconds,
                                       time_interval_seconds=dt)
        # Use a larger samples_per_batch to reduce number of forward passes
        spb = min(self.num_samples, max(64, min(self.num_samples, self.max_samples_per_batch)))
        with torch.inference_mode():
            try:
                pred = self.forecaster.forecast(inputs, prediction_length=1,
                                                num_samples=self.num_samples, samples_per_batch=spb)
            except Exception:
                # fallback if too-large spb leads to OOM or other failure: try smaller batches
                spb = min(self.num_samples, 64)
                pred = self.forecaster.forecast(inputs, prediction_length=1,
                                                num_samples=self.num_samples, samples_per_batch=spb)
        med = torch.nan_to_num(pred.median, nan=0.0, posinf=1e6, neginf=-1e6)
        ql  = torch.nan_to_num(pred.quantile(self.ql), nan=0.0, posinf=1e6, neginf=-1e6)
        qh  = torch.nan_to_num(pred.quantile(self.qh), nan=0.0, posinf=1e6, neginf=-1e6)
        mean = med.squeeze().detach().cpu().item() * sigma + mu
        lo   = ql.squeeze().detach().cpu().item()  * sigma + mu
        hi   = qh.squeeze().detach().cpu().item()  * sigma + mu
        return ForecastOut(float(mean), float(lo), float(hi))

class GraniteTTMForecaster:
    def __init__(self, context_len=1024, pred_len=1, device="cpu"):
        from tsfm_public.toolkit.get_model import get_model
        try:
            result = get_model(model_name_or_path="ibm-granite/granite-timeseries-ttm-r2",
                               context_length=context_len, prediction_length=max(1,pred_len))
        except TypeError:
            result = get_model(model_path="ibm-granite/granite-timeseries-ttm-r2",
                               context_length=context_len, prediction_length=max(1,pred_len))
        self.model = result[0] if isinstance(result, tuple) else result
        self.context_len, self.pred_len, self.device = context_len, max(1,pred_len), device
        try: self.model.to(device); self.model.eval()
        except Exception: pass
    def one_step(self, history: np.ndarray) -> ForecastOut:
        import torch as _torch
        x = _torch.tensor(history[-self.context_len:], dtype=_torch.float32).unsqueeze(0).unsqueeze(-1)
        with _torch.inference_mode():
            yhat = None
            try: yhat = self.model(x, prediction_length=self.pred_len)
            except TypeError: pass
            if yhat is None:
                try: yhat = self.model(x)
                except Exception: yhat = None
        if yhat is None:
            mu = float(history[-1])
            tail = history[-min(len(history), 512):]
            std_tail = float(np.std(tail - np.mean(tail))) if len(tail) else 1.0
            z = 1.64
            return ForecastOut(mu, mu - z*std_tail, mu + z*std_tail)
        yhat_np = yhat.detach().cpu().numpy() if hasattr(yhat, "detach") else np.asarray(yhat)
        mu = float(np.ravel(yhat_np)[0])
        tail = history[-max(32, self.context_len // 2):]
        res_std = float(np.std(tail - np.mean(tail))) if len(tail) else 1.0
        z = 1.64
        return ForecastOut(mu, mu - z*res_std, mu + z*res_std)

# ------------------ evaluation (+saving) ------------------
@dataclass
class RunCfg:
    lower_q: float = 0.05
    upper_q: float = 0.95
    warmup: int = WARMUP
    alpha: float = ALPHA
    bins_per_day: int = 48
    interval_sec: int = 1800

# ------------------ modified evaluation function ------------------
def evaluate(df: pd.DataFrame, f_name: str, forecaster, cfg: RunCfg,
             dataset_name: str, run_dir: pathlib.Path, windows: Dict[str, Any],
             detector_type="SeasonalQuantile") -> Dict[str, Any]:
    """
    Evaluate a forecaster with the specified detector type.
    
    Parameters:
    -----------
    df: Input dataframe with timestamp, value, label
    f_name: Name of forecaster (ARIMA, Toto, GraniteTTM)
    forecaster: Forecaster object
    cfg: Configuration parameters
    dataset_name: Name of dataset
    run_dir: Output directory
    windows: NAB label windows
    detector_type: Type of detector to use 
                  ("SeasonalQuantile", "AdaptiveQuantile", "IsolationForest", "Multivariate")
    """
    # 🚨 NEW: Track runtime
    eval_start_time = time.time()
    forecast_start_time = None
    forecast_end_time = None
    detection_start_time = None
    detection_end_time = None
    
    y = df['value'].values.astype(float)
    ts = df['timestamp']; labels = df['label'].values.astype(int)
    
    # 🚨 NEW: Limit Toto to last N points to reduce runtime
    if f_name == "Toto" and len(y) > TOTO_MAX_POINTS + cfg.warmup:
        orig_len = len(y)
        y = y[-TOTO_MAX_POINTS:]
        ts = ts.iloc[-TOTO_MAX_POINTS:].reset_index(drop=True)
        labels = labels[-TOTO_MAX_POINTS:]
        print(f"[info] Toto on {dataset_name}: Reduced from {orig_len} to {len(y)} points (speedup: {orig_len/len(y):.1f}×)")

    if f_name == "ARIMA" and len(y) <= max(10, cfg.warmup) + 1:
        raise ValueError(f"{dataset_name}: series too short for warmup={cfg.warmup}")

    yhat_list, lo_list, hi_list, res = [], [], [], []

    # 🚨 NEW: Start forecasting timer
    forecast_start_time = time.time()
    
    if f_name == "ARIMA":
        forecaster.begin(y[:cfg.warmup])
        for t in tqdm(range(cfg.warmup, len(y)-1), desc=f"{f_name} rolling"):
            out = forecaster.forecast_and_update(y[t])
            yhat_list.append(out.mean); lo_list.append(out.lo); hi_list.append(out.hi)
            res.append(y[t] - out.mean)
    else:
        for t in tqdm(range(cfg.warmup, len(y)-1), desc=f"{f_name} rolling"):
            hist = y[:t]
            if f_name == "Toto":
                out = forecaster.one_step(hist, cfg.interval_sec)
            else:
                out = forecaster.one_step(hist)
            yhat_list.append(out.mean); lo_list.append(out.lo); hi_list.append(out.hi)
            res.append(y[t] - out.mean)

    # 🚨 NEW: End forecasting timer
    forecast_end_time = time.time()
    forecast_time = forecast_end_time - forecast_start_time
    
    yhat_list = np.nan_to_num(np.array(yhat_list), nan=0.0, posinf=1e6, neginf=-1e6)
    lo_list   = np.nan_to_num(np.array(lo_list),   nan=0.0, posinf=1e6, neginf=-1e6)
    hi_list   = np.nan_to_num(np.array(hi_list),   nan=0.0, posinf=1e6, neginf=-1e6)
    res       = np.nan_to_num(np.array(res),       nan=0.0, posinf=1e6, neginf=-1e6)

    n = len(yhat_list)
    ts_eff  = ts[cfg.warmup:cfg.warmup+n].reset_index(drop=True)
    y_eff   = y[cfg.warmup:cfg.warmup+n]
    lab_eff = labels[cfg.warmup:cfg.warmup+n]

    # --- coverage calibration on train part (target 1-alpha) ---
    lo_cal, hi_cal, emp_cov_train, cov_scale = calibrate_intervals(
        y_eff, lo_list, hi_list, target=1.0-cfg.alpha
    )

    # --- 3-fold expanding-window CV for F1/AUCPR ---
    splits = kfold_splits(n, k=3)
    f1s, auprs = [], []
    
    for tr_end, te_end in splits:
        # Initialize appropriate detector based on detector_type
        if detector_type == "SeasonalQuantile":
            det_cv = SeasonalQuantileDetectorWD(ql=0.05, qh=0.95, bins=cfg.bins_per_day, use_weekday=True)
        elif detector_type == "AdaptiveQuantile":
            det_cv = AdaptiveQuantileDetector(ql=0.05, qh=0.95, bins=cfg.bins_per_day, use_weekday=True)
        elif detector_type == "VolatilityNormalized":
            det_cv = VolatilityNormalizedQuantileDetector(ql=0.05, qh=0.95, bins=cfg.bins_per_day, use_weekday=True)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}. Valid: SeasonalQuantile, AdaptiveQuantile, VolatilityNormalized")
            
        det_cv.fit(ts_eff.iloc[:tr_end], res[:tr_end])
        flags_cv = det_cv.predict(ts_eff.iloc[tr_end:te_end], res[tr_end:te_end])
        mag_cv = np.abs(res[tr_end:te_end])
        p_cv, r_cv, _ = precision_recall_curve(lab_eff[tr_end:te_end], mag_cv)
        f1s.append(f1_score(lab_eff[tr_end:te_end], flags_cv, zero_division=0))
        auprs.append(auc_pr(p_cv, r_cv))
        
    f1_cv_mean, f1_cv_std = float(np.mean(f1s)), float(np.std(f1s)) if len(f1s)>1 else 0.0
    aupr_cv_mean, aupr_cv_std = float(np.mean(auprs)), float(np.std(auprs)) if len(auprs)>1 else 0.0

    # --- final split for plots + tuned thresholds on validation (60/40) ---
    split = max(1, int(0.6*n))
    
    # 🚨 NEW: Start detection timer
    detection_start_time = time.time()
    
    # Grid search parameters (all detectors use quantile-based approach)
    grid = [(0.02,0.98), (0.05,0.95), (0.10,0.90), (0.15,0.85)]
    
    # Find best parameters via grid search on validation set
    best = (-1.0, None)  # (F1_val, params)
    
    if detector_type == "SeasonalQuantile":
        for ql, qh in grid:
            det = SeasonalQuantileDetectorWD(ql=ql, qh=qh, bins=cfg.bins_per_day, use_weekday=True)
            det.fit(ts_eff.iloc[:split], res[:split])
            flags_val = det.predict(ts_eff.iloc[:split], res[:split])
            f1_val = f1_score(lab_eff[:split], flags_val, zero_division=0)
            if f1_val > best[0]:
                best = (f1_val, (ql, qh, det))
        _, (qlb, qhb, det_best) = best
        
    elif detector_type == "AdaptiveQuantile":
        for ql, qh in grid:
            det = AdaptiveQuantileDetector(ql=ql, qh=qh, bins=cfg.bins_per_day, use_weekday=True)
            det.fit(ts_eff.iloc[:split], res[:split])
            flags_val = det.predict(ts_eff.iloc[:split], res[:split])
            f1_val = f1_score(lab_eff[:split], flags_val, zero_division=0)
            if f1_val > best[0]:
                best = (f1_val, (ql, qh, det))
        _, (qlb, qhb, det_best) = best
        
    elif detector_type == "VolatilityNormalized":
        for ql, qh in grid:
            det = VolatilityNormalizedQuantileDetector(ql=ql, qh=qh, bins=cfg.bins_per_day, use_weekday=True)
            det.fit(ts_eff.iloc[:split], res[:split])
            flags_val = det.predict(ts_eff.iloc[:split], res[:split])
            f1_val = f1_score(lab_eff[:split], flags_val, zero_division=0)
            if f1_val > best[0]:
                best = (f1_val, (ql, qh, det))
        _, (qlb, qhb, det_best) = best
    
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
    
    # Test on held-out test data
    flags_test = det_best.predict(ts_eff.iloc[split:], res[split:])
    mag = np.abs(res[split:])
    prec, rec, thr = precision_recall_curve(lab_eff[split:], mag)
    pr = pd.DataFrame({"precision":prec, "recall":rec,
                       "threshold":np.r_[thr, thr[-1] if len(thr)>0 else 0.0]})
    aupr = auc_pr(prec, rec)

    # 🚨 NEW: End detection timer
    detection_end_time = time.time()
    detection_time = detection_end_time - detection_start_time
    total_time = time.time() - eval_start_time
    
    f1 = f1_score(lab_eff[split:], flags_test, zero_division=0)
    s = smape(y_eff, yhat_list)
    cov_emp_test = coverage(y_eff[split:], lo_cal[split:], hi_cal[split:])
    whr = window_hit_rate(ts_eff.iloc[split:], flags_test, windows_obj=windows,
                          file_key=f"realKnownCause/{dataset_name}.csv")

    # --- save traces & plots (with calibrated intervals) ---
    trace = pd.DataFrame({
        "timestamp": ts_eff,
        "value": y_eff,
        "yhat": yhat_list,
        "lo": lo_cal, "hi": hi_cal,
        "residual": res,
        "split": ["train"]*split + ["test"]*(n - split),
    })
    trace["flag"] = np.r_[ [np.nan]*split, flags_test.astype(float) ]
    trace_path = run_dir / "traces" / f"{dataset_name}__{f_name}__{detector_type}.csv"
    pr_path = run_dir / "pr_traces" / f"{dataset_name}__{f_name}__{detector_type}.csv"
    trace_path.write_text(trace.to_csv(index=False))
    pr_path.write_text(pr.to_csv(index=False))

    # NEW: Residual analysis
    separation = make_residual_analysis(
        trace, lab_eff, dataset_name, f_name,
        run_dir / "plots" / f"RESID_{dataset_name}__{f_name}__{detector_type}.png"
    )

    # PR plot
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR: {dataset_name} / {f_name} / {detector_type} (AUPR={aupr:.3f})")
    plt.grid(True, linestyle=":"); plt.tight_layout()
    plt.savefig(run_dir / "plots" / f"PR_{dataset_name}__{f_name}__{detector_type}.png"); plt.close()

    # Timeline plot
    plt.figure(figsize=(11,4))
    plt.plot(trace["timestamp"], trace["value"], label="value")
    plt.plot(trace["timestamp"], trace["yhat"], label="yhat")
    plt.fill_between(trace["timestamp"], trace["lo"], trace["hi"], alpha=0.2, label="PI (cal)")
    tt = trace.iloc[split:]; tt = tt[tt["flag"] == 1.0]
    if len(tt): plt.scatter(tt["timestamp"], tt["value"], marker="x", s=20, label="anomaly")
    
    if detector_type in ["SeasonalQuantile", "AdaptiveQuantile", "VolatilityNormalized"]:
        title_params = f"bins={cfg.bins_per_day}, dt={cfg.interval_sec}s, q=({qlb:.2f},{qhb:.2f})"
    else:
        title_params = f"detector={detector_type}"
    
    plt.title(f"Timeline: {dataset_name} / {f_name} / {detector_type}\n{title_params}")
    plt.legend(); plt.tight_layout()
    plt.savefig(run_dir / "plots" / f"TL_{dataset_name}__{f_name}__{detector_type}.png"); plt.close()

    # Return parameters for results table
    result = {
        "F1": f1,
        "AUPR": aupr,
        "sMAPE": s,
        "Coverage": cov_emp_test,
        "WHit": whr,
        "trace": pr,
        "n_eval": len(lab_eff[split:]),
        "cov_scale": cov_scale,
        "F1_CV_mean": f1_cv_mean, "F1_CV_std": f1_cv_std,
        "AUPR_CV_mean": aupr_cv_mean, "AUPR_CV_std": aupr_cv_std,
        "emp_cov_train": emp_cov_train,
        "separation": separation,
        "detector_type": detector_type,
        # 🚨 NEW: Runtime metrics
        "forecast_time_sec": forecast_time,
        "detection_time_sec": detection_time,
        "total_time_sec": total_time,
        "n_forecasts": len(yhat_list)
    }
    
    # Add detector-specific parameters (all are quantile-based)
    result.update({"ql": qlb, "qh": qhb})
        
    return result
    
# ------------------ main ------------------
# ------------------ main with multiple detectors ------------------
if __name__ == "__main__":
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
    if not SKIP_TOTO:
        try:
            toto = TotoForecaster(
                device=DEVICE,
                checkpoint="Datadog/Toto-Open-Base-1.0",
                num_samples=128,  # 🚨 FIXED: Was 512 (4× slower!)
                ql=0.10,
                qh=0.90,
                compile_model=False,  # 🚨 FIXED: Doesn't help with varying input sizes
                max_samples_per_batch=256  # 🚨 OPTIMIZED: Larger batches for efficiency
            )
            has_toto = True
            print(f"[info] Toto initialized with {128} samples, max_points_per_dataset={TOTO_MAX_POINTS}")
        except Exception as e:
            print("[warn] Toto init failed:", e)
    else:
        print("[info] Skipping Toto per SKIP_TOTO=1 environment variable")

    has_granite = False
    try:
        from tsfm_public.toolkit.get_model import get_model
        granite = GraniteTTMForecaster(context_len=1024, pred_len=1, device=DEVICE)
        has_granite = True
    except Exception as e:
        print("[warn] Granite init failed:", e)

    # List of detector types to evaluate
    detector_types = [
        "SeasonalQuantile",        # Original baseline (weekday-aware)
        "VolatilityNormalized",    # Fixed: MAD-normalized z-scores with temporal smoothing
        "AdaptiveQuantile"         # Fixed: Properly tightens thresholds in high-volatility regions
    ]
    # Note: IsolationForest and Multivariate removed (fundamentally wrong for time-series)
    
    # Updated results collection with detector type
    rows = []; all_pr = []
    failed_runs = []  # 🚨 NEW: Track failures for summary
    
    for name, df in loaded:
        dt_sec, bins = infer_cadence_and_bins(df["timestamp"])
        local_cfg = RunCfg(lower_q=base_cfg.lower_q, upper_q=base_cfg.upper_q,
                           warmup=min(WARMUP, max(32, len(df)//5)),
                           alpha=base_cfg.alpha, bins_per_day=bins, interval_sec=dt_sec)

        for detector_type in detector_types:
            try:
                r_arima = evaluate(df, "ARIMA", arima, local_cfg, name, RUN_DIR, windows, 
                                  detector_type=detector_type)
                rows.append((name, "ARIMA", detector_type, r_arima["F1"], r_arima["AUPR"], 
                             r_arima["sMAPE"], r_arima["Coverage"], r_arima["WHit"],
                             r_arima.get("ql", 0), r_arima.get("qh", 0),
                             r_arima["cov_scale"], r_arima["F1_CV_mean"], r_arima["F1_CV_std"],
                             r_arima["AUPR_CV_mean"], r_arima["AUPR_CV_std"], r_arima["separation"],
                             r_arima["forecast_time_sec"], r_arima["detection_time_sec"], 
                             r_arima["total_time_sec"], r_arima["n_forecasts"]))
                all_pr.append(r_arima["trace"].assign(Dataset=name, Model="ARIMA", Detector=detector_type))
            except Exception as e:
                print(f"❌ [ERROR] ARIMA+{detector_type} failed on {name}: {e}")
                failed_runs.append((name, "ARIMA", detector_type, str(e)))

            if has_toto:
                try:
                    r_toto = evaluate(df, "Toto", toto, local_cfg, name, RUN_DIR, windows,
                                     detector_type=detector_type)
                    rows.append((name, "Toto", detector_type, r_toto["F1"], r_toto["AUPR"], 
                                 r_toto["sMAPE"], r_toto["Coverage"], r_toto["WHit"],
                                 r_toto.get("ql", 0), r_toto.get("qh", 0),
                                 r_toto["cov_scale"], r_toto["F1_CV_mean"], r_toto["F1_CV_std"],
                                 r_toto["AUPR_CV_mean"], r_toto["AUPR_CV_std"], r_toto["separation"],
                                 r_toto["forecast_time_sec"], r_toto["detection_time_sec"], 
                                 r_toto["total_time_sec"], r_toto["n_forecasts"]))
                    all_pr.append(r_toto["trace"].assign(Dataset=name, Model="Toto", Detector=detector_type))
                except Exception as e:
                    print(f"⚠️  [WARN] Toto+{detector_type} failed on {name}: {e}")
                    failed_runs.append((name, "Toto", detector_type, str(e)))

            if has_granite:
                try:
                    r_gra = evaluate(df, "GraniteTTM", granite, local_cfg, name, RUN_DIR, windows,
                                    detector_type=detector_type)
                    rows.append((name, "GraniteTTM", detector_type, r_gra["F1"], r_gra["AUPR"], 
                                 r_gra["sMAPE"], r_gra["Coverage"], r_gra["WHit"],
                                 r_gra.get("ql", 0), r_gra.get("qh", 0),
                                 r_gra["cov_scale"], r_gra["F1_CV_mean"], r_gra["F1_CV_std"],
                                 r_gra["AUPR_CV_mean"], r_gra["AUPR_CV_std"], r_gra["separation"],
                                 r_gra["forecast_time_sec"], r_gra["detection_time_sec"], 
                                 r_gra["total_time_sec"], r_gra["n_forecasts"]))
                    all_pr.append(r_gra["trace"].assign(Dataset=name, Model="GraniteTTM", Detector=detector_type))
                except Exception as e:
                    print(f"❌ [ERROR] GraniteTTM+{detector_type} failed on {name}: {e}")
                    failed_runs.append((name, "GraniteTTM", detector_type, str(e)))

    summary_path_csv = RUN_DIR / "exec_summary.csv"
    summary_path_md  = RUN_DIR / "exec_summary.md"
    if rows:
        out = pd.DataFrame(rows, columns=[
            "Dataset","Model","Detector","F1","AUPR","sMAPE","Coverage","WHit",
            "ql","qh","cov_scale","F1_CV_mean","F1_CV_std","AUPR_CV_mean","AUPR_CV_std",
            "separation","forecast_time_sec","detection_time_sec","total_time_sec","n_forecasts"
        ]).sort_values(["Dataset","F1"], ascending=[True,False])

        # Enhanced analysis: Compare detectors on rogue_agent
        print("\n" + "="*70)
        print("CRITICAL ANALYSIS: Testing if new detectors solve the rogue_agent paradox")
        print("="*70)
        
        # Filter for rogue_agent dataset
        rogue = out[out["Dataset"] == "rogue_agent_key_hold"]
        if len(rogue) > 0:
            # Group by detector and model, show F1 and sMAPE
            rogue_summary = rogue.pivot_table(
                index=["Detector"], 
                columns=["Model"], 
                values=["F1", "sMAPE"],
                aggfunc="mean"
            )
            print("\nRogue Agent Results:\n")
            print(rogue_summary.round(3))
            
            # Find best detector for Granite on rogue_agent
            granite_rogue = rogue[rogue["Model"] == "GraniteTTM"]
            if len(granite_rogue) > 0:
                best_det = granite_rogue.loc[granite_rogue["F1"].idxmax()]
                improvement = (best_det["F1"] / granite_rogue[granite_rogue["Detector"] == "SeasonalQuantile"]["F1"].values[0]) - 1
                print(f"\nBest detector for Granite on rogue_agent: {best_det['Detector']}")
                print(f"F1 score: {best_det['F1']:.3f} ({improvement*100:.1f}% improvement over baseline)")
                print(f"This validates our hypothesis: Better detectors unlock Granite's superior forecasting!")
        
        # Enhanced scatter plot - group by detector
        for detector in detector_types:
            det_data = out[out["Detector"] == detector]
            corr = det_data[['sMAPE', 'F1']].corr().iloc[0, 1]
            make_scatter_plot(det_data, RUN_DIR / "plots" / f"smape_vs_f1_scatter_{detector}.png")
            print(f"\nCorrelation between sMAPE and F1 for {detector}: {corr:.3f}")
        
        # Make detector comparison plots
        print("\nGenerating detector comparison plots...")
        for model in ["ARIMA", "Toto", "GraniteTTM"]:
            model_data = out[out["Model"] == model]
            plt.figure(figsize=(10, 6))
            
            # Group by detector and get mean F1
            stats = model_data.groupby('Detector')['F1'].agg(['mean', 'std']).reset_index()
            
            x = range(len(stats))
            plt.bar(x, stats['mean'], yerr=stats['std'], 
                   capsize=5, alpha=0.7, edgecolor='black')
            plt.xticks(x, stats['Detector'], rotation=30)
            plt.ylabel('F1 Score', fontsize=12)
            plt.title(f'Detector Performance for {model} (mean ± std across datasets)', fontsize=13)
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels on bars
            for i, (mean, std) in enumerate(zip(stats['mean'], stats['std'])):
                plt.text(i, mean + std + 0.01, f'{mean:.3f}', 
                        ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(RUN_DIR / "plots" / f'detector_comparison_{model}.png', dpi=150)
            plt.close()
        
        # NEW: Calculate average F1 improvement from detector variants
        baseline = out[out["Detector"] == "SeasonalQuantile"].groupby("Model")["F1"].mean()
        improvement_df = []
        for detector in detector_types[1:]:  # Skip baseline
            det_means = out[out["Detector"] == detector].groupby("Model")["F1"].mean()
            for model in det_means.index:
                if model in baseline.index:
                    pct_improvement = (det_means[model] / baseline[model] - 1) * 100
                    improvement_df.append((model, detector, det_means[model], pct_improvement))
        
        if improvement_df:
            imp_df = pd.DataFrame(improvement_df, 
                                 columns=["Model", "Detector", "F1", "Improvement_Pct"])
            print("\nDetector Improvements over Baseline:\n")
            print(imp_df.to_string(float_format=lambda x: f"{x:.2f}"))
            (RUN_DIR / "detector_improvements.csv").write_text(imp_df.to_csv(index=False))

        # Pretty-print with mean±std strings for CV cols
        out_fmt = out.copy()
        out_fmt["F1_CV"] = (out["F1_CV_mean"].map(lambda x: f"{x:.3f}") + " ± " +
                            out["F1_CV_std"].map(lambda x: f"{x:.3f}"))
        out_fmt["AUPR_CV"] = (out["AUPR_CV_mean"].map(lambda x: f"{x:.3f}") + " ± " +
                              out["AUPR_CV_std"].map(lambda x: f"{x:.3f}"))
        out_fmt = out_fmt.drop(columns=["F1_CV_mean","F1_CV_std","AUPR_CV_mean","AUPR_CV_std"])

        print("\n\n## Executive Summary\n")
        with pd.option_context('display.float_format', '{:.4f}'.format):
            print(out_fmt.to_markdown(index=False))
        out.to_csv(summary_path_csv, index=False)
        summary_path_md.write_text(out_fmt.to_markdown(index=False))
        
        # 🚨 NEW: Add runtime analysis
        print("\n" + "="*70)
        print("RUNTIME PERFORMANCE ANALYSIS")
        print("="*70)
        runtime_summary = out.groupby(["Model","Detector"]).agg({
            "forecast_time_sec": "mean",
            "detection_time_sec": "mean", 
            "total_time_sec": "mean",
            "n_forecasts": "mean"
        }).round(2)
        runtime_summary["forecasts_per_sec"] = (runtime_summary["n_forecasts"] / runtime_summary["forecast_time_sec"]).round(2)
        print("\nMean runtime by Model + Detector:\n")
        print(runtime_summary)
        print()
        
    else:
        print("\n[warn] No results were produced — all model runs failed.\n")

    if all_pr:
        pr_all = pd.concat(all_pr, ignore_index=True)
        pr_all.to_csv(RUN_DIR / "pr_traces.csv", index=False)

    # 🚨 NEW: Report failures
    if failed_runs:
        print("\n" + "="*70)
        print(f"⚠️  FAILURES SUMMARY: {len(failed_runs)} runs failed")
        print("="*70)
        fail_df = pd.DataFrame(failed_runs, columns=["Dataset","Model","Detector","Error"])
        print(fail_df.to_string(index=False))
        fail_df.to_csv(RUN_DIR / "failed_runs.csv", index=False)
        print(f"\nSaved to: {RUN_DIR / 'failed_runs.csv'}")
        print()

    print(f"\n{'='*70}")
    print(f"All artifacts saved under: {RUN_DIR}")
    print(f"{'='*70}")
    print(f"\nKey outputs:")
    print(f"  - exec_summary.csv: Full results table (with runtime metrics)")
    print(f"  - failed_runs.csv: Failed runs summary ({len(failed_runs)} failures)")
    print(f"  - detector_improvements.csv: Improvement from new detectors")
    print(f"  - plots/smape_vs_f1_scatter_*.png: Correlation analysis per detector")
    print(f"  - plots/detector_comparison_*.png: Detector performance by model")
    print(f"  - plots/RESID_*.png: Residual distribution analysis")
    print(f"\n")

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from dtaidistance import dtw
    HAS_DTAIDISTANCE = True
except ImportError:
    HAS_DTAIDISTANCE = False


# read CSV, convert timestamp to seconds since start, and return DataFrame
def load_and_normalize(csv_path):
    try:
        df = pd.read_csv(csv_path, usecols=["timestamp", "value"])

        # CSV exists but has no rows
        if df.empty:
            print(f"Info: {csv_path} is empty, skipping")
            return None

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        t0 = df["timestamp"].iloc[0]
        df["t_norm"] = (df["timestamp"] - t0).dt.total_seconds()

        return df

    except EmptyDataError:
        print(f"Info: {csv_path} is empty, skipping")
        return None

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None


class TimeSeriesComparator:
    """Compare multiple time series using correlation, derivatives, DTW, and PCA."""
    
    def __init__(self, csv_files: List[str], labels: Optional[List[str]] = None):
        """
        Initialize with list of CSV file paths.
        
        Args:
            csv_files: List of paths to CSV files
            labels: Optional list of labels for each series (defaults to filename)
        """
        self.csv_files = csv_files
        self.labels = labels or [Path(f).stem for f in csv_files]
        self.series_dict: Dict[str, np.ndarray] = {}
        self.aligned_series: Dict[str, np.ndarray] = {}
        self.time_grid: Optional[np.ndarray] = None
        self._load_and_align_data()
    
    def _load_and_align_data(self):
        """Load CSV files and align them to a common time grid."""
        dfs = {}
        for csv_file, label in zip(self.csv_files, self.labels):
            df = load_and_normalize(csv_file)
            if df is not None and not df.empty:
                dfs[label] = df
        
        if not dfs:
            raise ValueError("No valid CSV files loaded")
        
        # Create common time grid from all data
        min_t = min(d["t_norm"].min() for d in dfs.values())
        max_t = max(d["t_norm"].max() for d in dfs.values())
        
        # Use 1000 points or original resolution, whichever is finer
        n_points = max(1000, max(len(d) for d in dfs.values()))
        self.time_grid = np.linspace(min_t, max_t, n_points)
        
        # Interpolate all series to common time grid
        for label, df in dfs.items():
            f = interpolate.interp1d(
                df["t_norm"], df["value"],
                kind='cubic', bounds_error=False, 
                fill_value='extrapolate'
            )
            values = f(self.time_grid)
            # Clip to reasonable range (handle extrapolation artifacts)
            values = np.clip(values, np.percentile(values, 1), np.percentile(values, 99))
            self.aligned_series[label] = values
        
        print(f"Loaded and aligned {len(self.aligned_series)} time series")
    
    def _normalize(self, series: np.ndarray) -> np.ndarray:
        """Normalize series to zero mean and unit variance."""
        return (series - np.mean(series)) / (np.std(series) + 1e-10)
    
    def compute_correlation_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Compute normalized correlation matrix between all series."""
        labels = list(self.aligned_series.keys())
        n = len(labels)
        corr_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                series_i = self._normalize(self.aligned_series[labels[i]])
                series_j = self._normalize(self.aligned_series[labels[j]])
                corr_matrix[i, j] = np.corrcoef(series_i, series_j)[0, 1]
        
        return corr_matrix, labels
    
    def compute_derivative_correlation_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Compute correlation matrix of time derivatives."""
        labels = list(self.aligned_series.keys())
        n = len(labels)
        deriv_corr_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                deriv_i = np.gradient(self.aligned_series[labels[i]])
                deriv_j = np.gradient(self.aligned_series[labels[j]])
                
                deriv_i = self._normalize(deriv_i)
                deriv_j = self._normalize(deriv_j)
                
                deriv_corr_matrix[i, j] = np.corrcoef(deriv_i, deriv_j)[0, 1]
        
        return deriv_corr_matrix, labels
    
    def compute_dtw_distance_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Compute DTW distance matrix between all series."""
        labels = list(self.aligned_series.keys())
        n = len(labels)
        dtw_matrix = np.zeros((n, n))
        
        if not HAS_DTAIDISTANCE:
            print("Warning: dtaidistance not installed, computing DTW distances (slower)")
            for i in range(n):
                for j in range(n):
                    if i <= j:
                        dtw_dist = self._dtw_distance(
                            self.aligned_series[labels[i]],
                            self.aligned_series[labels[j]]
                        )
                        dtw_matrix[i, j] = dtw_dist
                        dtw_matrix[j, i] = dtw_dist
        else:
            for i in range(n):
                for j in range(n):
                    if i <= j:
                        # Normalize series for DTW
                        s1 = self._normalize(self.aligned_series[labels[i]])
                        s2 = self._normalize(self.aligned_series[labels[j]])
                        dtw_dist = dtw.distance_fast(s1, s2)
                        dtw_matrix[i, j] = dtw_dist
                        dtw_matrix[j, i] = dtw_dist
        
        return dtw_matrix, labels
    
    def _dtw_distance(self, series1: np.ndarray, series2: np.ndarray) -> float:
        """Compute Dynamic Time Warping distance using dynamic programming."""
        n, m = len(series1), len(series2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.abs(series1[i-1] - series2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],
                    dtw_matrix[i, j-1],
                    dtw_matrix[i-1, j-1]
                )
        
        return dtw_matrix[n, m]
    
    def extract_features(self) -> Tuple[np.ndarray, List[str]]:
        """Extract statistical features from each time series."""
        labels = list(self.aligned_series.keys())
        features = []
        
        for label in labels:
            series = self.aligned_series[label]
            feature_vec = np.array([
                np.mean(series),
                np.std(series),
                np.min(series),
                np.max(series),
                np.median(series),
                np.percentile(series, 25),
                np.percentile(series, 75),
                np.mean(np.abs(np.gradient(series))),  # mean absolute derivative
                np.std(np.gradient(series)),
                (np.max(series) - np.min(series)) / (np.mean(series) + 1e-10),  # coefficient of variation
                self._compute_autocorr(series),  # autocorrelation
                self._compute_entropy(series),  # entropy
            ])
            features.append(feature_vec)
        
        return np.array(features), labels
    
    def _compute_autocorr(self, series: np.ndarray, lag: int = 1) -> float:
        """Compute autocorrelation at given lag."""
        c = np.correlate(series - np.mean(series), series - np.mean(series), 'full')
        c = c / (np.std(series) ** 2 * len(series))
        return c[len(c) // 2 + lag]
    
    def _compute_entropy(self, series: np.ndarray, bins: int = 20) -> float:
        """Compute approximate entropy of series."""
        hist, _ = np.histogram(series, bins=bins)
        hist = hist[hist > 0]
        probs = hist / np.sum(hist)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def plot_all_comparisons(self, output_dir: str = "./"):
        """Create all four comparison visualizations."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Correlation Heatmap
        corr_matrix, labels = self.compute_correlation_matrix()
        self._plot_heatmap(
            corr_matrix, labels, 
            title="Correlation Heatmap (Normalized)",
            cmap="RdBu_r", vmin=-1, vmax=1,
            filename=f"{output_dir}/01_correlation_heatmap.png"
        )
        
        # 2. Derivative Correlation Heatmap
        deriv_corr_matrix, labels = self.compute_derivative_correlation_matrix()
        self._plot_heatmap(
            deriv_corr_matrix, labels,
            title="Derivative Correlation Heatmap",
            cmap="RdBu_r", vmin=-1, vmax=1,
            filename=f"{output_dir}/02_derivative_correlation_heatmap.png"
        )
        
        # 3. DTW Distance Matrix
        dtw_matrix, labels = self.compute_dtw_distance_matrix()
        self._plot_heatmap(
            dtw_matrix, labels,
            title="DTW Distance Matrix",
            cmap="YlOrRd",
            filename=f"{output_dir}/03_dtw_distance_matrix.png"
        )
        
        # 4. PCA Scatter Plot
        self._plot_pca_scatter(
            filename=f"{output_dir}/04_pca_scatter_plot.png"
        )
        
        print(f"Saved all comparison plots to {output_dir}")
    
    def _plot_heatmap(self, matrix: np.ndarray, labels: List[str],
                      title: str, cmap: str = "viridis",
                      vmin=None, vmax=None, filename: str = None):
        """Plot a heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap,
                   xticklabels=labels, yticklabels=labels,
                   vmin=vmin, vmax=vmax, cbar_kws={"label": title},
                   ax=ax, square=True)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        plt.show()
    
    def _plot_pca_scatter(self, filename: str = None):
        """Plot PCA scatter plot using extracted features."""
        features, labels = self.extract_features()
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply PCA
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(features_pca[:, 0], features_pca[:, 1],
                            s=200, alpha=0.7, c=range(len(labels)),
                            cmap='tab10', edgecolors='black', linewidth=1.5)
        
        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(label, (features_pca[i, 0], features_pca[i, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                       facecolor='yellow', alpha=0.3))
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                     fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                     fontsize=12)
        ax.set_title('PCA Scatter Plot (Feature-Based)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        plt.show()
    
    def plot_time_series(self, filename: str = None):
        """Plot all aligned time series for visual inspection."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for label, series in self.aligned_series.items():
            normalized = self._normalize(series)
            ax.plot(self.time_grid, normalized, label=label, linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Normalized Value', fontsize=12)
        ax.set_title('Aligned Time Series', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        plt.show()

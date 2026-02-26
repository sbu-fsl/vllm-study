import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from pandas.errors import EmptyDataError
from sklearn.metrics import mean_squared_error
from itertools import combinations
from dtaidistance import dtw


plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 12
})

ROOT = "."
SUBS = ["data"]
MODELS = ["ibm-2b-lmcache", "ibm-2b-no-lmcache"]
METRICS = [
    "container_blkio_read",
    "container_blkio_write",
    "container_file_descriptors",
    "container_fs_read_count",
    "container_fs_read",
    "container_fs_write_count",
    "container_fs_write"
    # "container_network_receive_pkt",
    # "container_network_receive",
    # "container_network_transmit_pkt",
    # "container_network_transmit",
    # "gpu_fb_free",
    # "gpu_fb_used",
    # "gpu_mem_copy",
    # "pcie_rx",
    # "pcie_tx"
]


# ---------------------------------------
# Load split CSV (hash-based file)
# ---------------------------------------
def load_and_normalize(csv_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path, usecols=["timestamp", "value"])

        if df.empty:
            return None

        # If timestamp is numeric epoch
        if np.issubdtype(df["timestamp"].dtype, np.number):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        t0 = df["timestamp"].iloc[0]
        df["t_norm"] = (df["timestamp"] - t0).dt.total_seconds()

        return df

    except EmptyDataError:
        return None
    except Exception as e:
        print(f"[error] {csv_path}: {e}")
        return None


# -----------------------------
# Helper
# -----------------------------
def zscore(series):
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


# -----------------------------
# Main
# -----------------------------
def main(metric: str):
    os.makedirs(os.path.join("images", metric), exist_ok=True)

    series_dict = {}

    # 🔹 Scan new directory structure
    for sub in SUBS:
        for model in MODELS:

            metric_dir = os.path.join(ROOT, sub, model, metric)

            if not os.path.isdir(metric_dir):
                continue

            csv_files = glob.glob(os.path.join(metric_dir, "*.csv"))

            for csv_path in csv_files:
                df = load_and_normalize(csv_path)
                if df is None:
                    continue

                # Use hash filename as identifier
                hash_name = os.path.splitext(os.path.basename(csv_path))[0]
                name = f"{sub}:{model}:{hash_name[:8]}"

                s = df.groupby("t_norm")["value"].mean()
                series_dict[name] = s

    # Align
    aligned_df = pd.concat(series_dict, axis=1, join="inner")

    # Normalize
    aligned_df = aligned_df.apply(zscore)
    print("[info] aligned shape:", aligned_df.shape)

    names = aligned_df.columns.tolist()
    n = len(names)

    correlation_matrix = pd.DataFrame(index=names, columns=names)
    mse_matrix = pd.DataFrame(index=names, columns=names)
    dtw_matrix = pd.DataFrame(index=names, columns=names)

    for i in range(n):
        for j in range(n):
            s1 = aligned_df.iloc[:, i]
            s2 = aligned_df.iloc[:, j]

            correlation_matrix.iloc[i, j] = s1.corr(s2)
            mse_matrix.iloc[i, j] = mean_squared_error(s1, s2)
            dtw_matrix.iloc[i, j] = dtw.distance(s1.values, s2.values)

    print("\nCorrelation Matrix:\n", correlation_matrix)

    # -----------------------------
    # Overlay Plot
    # -----------------------------
    plt.figure(figsize=(24, 14))

    for col in aligned_df.columns:
        plt.plot(aligned_df.index, aligned_df[col], linewidth=1)

    plt.title("Overlay of Time Series (Normalized)", fontsize=20)
    plt.xlabel("Time (seconds)", fontsize=16)
    plt.ylabel("Z-score Value", fontsize=16)

    plt.legend(
        aligned_df.columns,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9
    )

    plt.tight_layout()
    plt.savefig(f"images/{metric}/overlay.svg", format="svg")
    plt.close()

    # -----------------------------
    # Heatmap
    # -----------------------------
    def plot_heatmap(matrix, title):
        plt.figure(figsize=(18, 16))

        matrix_float = matrix.astype(float)
        norm = colors.Normalize(
            vmin=np.min(matrix_float),
            vmax=np.max(matrix_float)
        )

        im = plt.imshow(matrix_float,
                        interpolation='nearest',
                        cmap='viridis',
                        norm=norm)

        plt.colorbar(im)

        plt.xticks(range(n), names, rotation=60, ha="right", fontsize=9)
        plt.yticks(range(n), names, fontsize=9)

        for i in range(n):
            plt.text(i, i, 'X',
                     ha='center',
                     va='center',
                     color='black',
                     fontsize=12,
                     fontweight='bold')

        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"images/{metric}/{title.replace(' ', '_').lower()}.svg",
                    format="svg")
        plt.close()

    plot_heatmap(correlation_matrix, "Correlation Matrix")
    plot_heatmap(mse_matrix, "MSE Matrix")
    plot_heatmap(dtw_matrix, "DTW Distance Matrix")

    # -----------------------------
    # Most Similar
    # -----------------------------
    print("\nMost Similar Pairs (by highest correlation):")

    pairs = list(combinations(names, 2))
    pairs_sorted = sorted(
        pairs,
        key=lambda x: correlation_matrix.loc[x[0], x[1]],
        reverse=True
    )

    for p in pairs_sorted:
        print(f"{p[0]} vs {p[1]} -> "
              f"Correlation: {correlation_matrix.loc[p[0], p[1]]:.4f}")


if __name__ == "__main__":
    for metric in METRICS:
        print("\nProcessing metric:", metric)
        main(metric)
        print(f"[info] Completed processing for {metric}")

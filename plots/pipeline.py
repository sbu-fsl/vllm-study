import os.path
import os
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

ROOT = "results"
SUBS = [
    "ansh",
    "tahsin"
]
MODELS = [
    "facebook-opt-125m",
    "facebook-opt-350m",
    "google-gemma-2b",
    "ibm-granite-33-8b",
    "ibm-granite-40-h-1b",
    "ibm-granite-40-h-tiny",
    "ibm-granite-40-micro",
    "meta-llama-31-8b",
    "qwen-3-06b",
    "qwen-3-8b"
]
METRICS = [
    "container_blkio_read"
    # "container_blkio_write",
    # "container_file_descriptors",
    # "container_fs_read_count",
    # "container_fs_read",
    # "container_fs_write_count",
    # "container_fs_write",
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

# read CSV, convert timestamp to seconds since start, and return DataFrame
def load_and_normalize(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, usecols=["timestamp", "value"])

        # CSV exists but has no rows
        if df.empty:
            print(f"[info] {csv_path} is empty, skipping")
            return None

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        t0 = df["timestamp"].iloc[0]
        df["t_norm"] = (df["timestamp"] - t0).dt.total_seconds()

        return df
    except EmptyDataError:
        print(f"[info] {csv_path} is empty, skipping")
        return None
    except Exception as e:
        print(f"[error] {csv_path}: {e}")
        return None
    
# -----------------------------
# Helper Functions
# -----------------------------
def zscore(series):
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std

def main(metric: str):
    os.makedirs(os.path.join("images", metric), exist_ok=True)

    # load the datasets into DataFrame
    dfs = {}
    for sub in SUBS:
        for model in MODELS:
            path = os.path.join(ROOT, sub, model, metric + ".csv")
            df = load_and_normalize(path)
            if df is not None:
                dfs[sub + ":" + model] = df

    series_dict = {}

    for name, df in dfs.items():
        s = df.set_index("t_norm")["value"]  # use only value column
        series_dict[name] = s

    aligned_df = pd.concat(series_dict, axis=1, join="inner")
    aligned_df.columns = dfs.keys()

    # normalize
    aligned_df = aligned_df.apply(zscore)
    print("[info] aligned shape:", aligned_df.shape)

    # -----------------------------
    # Compute Metrics
    # -----------------------------
    names = aligned_df.columns.tolist()
    n = len(names)

    correlation_matrix = pd.DataFrame(index=names, columns=names)
    mse_matrix = pd.DataFrame(index=names, columns=names)
    dtw_matrix = pd.DataFrame(index=names, columns=names)

    for i in range(n):
        for j in range(n):
            s1 = aligned_df.iloc[:, i]
            s2 = aligned_df.iloc[:, j]

            # Correlation
            correlation_matrix.iloc[i, j] = s1.corr(s2)

            # MSE
            mse_matrix.iloc[i, j] = mean_squared_error(s1, s2)

            # DTW
            dtw_matrix.iloc[i, j] = dtw.distance(s1.values, s2.values)

    print("\nCorrelation Matrix:\n", correlation_matrix)
    print("\nMSE Matrix:\n", mse_matrix)
    print("\nDTW Matrix:\n", dtw_matrix)

    # -----------------------------
    # Visualization
    # -----------------------------

    # 1️⃣ Overlay Plot (Large SVG)
    plt.figure(figsize=(24, 14))  # BIG figure

    for col in aligned_df.columns:
        plt.plot(aligned_df.index, aligned_df[col], linewidth=1)

    plt.title("Overlay of Time Series (Normalized)", fontsize=20)
    plt.xlabel("Time (seconds)", fontsize=16)
    plt.ylabel("Z-score Value", fontsize=16)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # move legend outside (critical for 20 items)
    plt.legend(
        aligned_df.columns,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig("images/" + metric + "/overlay.svg", format="svg")
    plt.close()

    # 2️⃣ Heatmap Function
    def plot_heatmap(matrix, title):
        n = matrix.shape[0]

        plt.figure(figsize=(18, 16))

        norm = colors.Normalize(vmin=np.min(matrix), vmax=np.max(matrix))

        im = plt.imshow(matrix.astype(float),
                        interpolation='nearest',
                        cmap='viridis',
                        norm=norm)

        plt.colorbar(im, fraction=0.046, pad=0.04)

        plt.xticks(range(n), names, rotation=60, ha="right", fontsize=10)
        plt.yticks(range(n), names, fontsize=10)

        # add grid lines
        plt.gca().set_xticks(np.arange(-.5, n, 1), minor=True)
        plt.gca().set_yticks(np.arange(-.5, n, 1), minor=True)
        plt.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
        plt.gca().tick_params(which='minor', bottom=False, left=False)

        # add X on diagonal
        for i in range(n):
            plt.text(i, i, 'X',
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=14,
                    fontweight='bold')

        plt.title(title, fontsize=18)

        plt.tight_layout()
        plt.savefig("images/" + metric + "/" + title.replace(" ", "_").lower() + ".svg", format="svg")
        plt.close()

    plot_heatmap(correlation_matrix, "Correlation Matrix")
    plot_heatmap(mse_matrix, "MSE Matrix")
    plot_heatmap(dtw_matrix, "DTW Distance Matrix")

    # -----------------------------
    # Summary Interpretation
    # -----------------------------
    print("\nMost Similar Pairs (by highest correlation):")
    pairs = list(combinations(names, 2))
    pairs_sorted = sorted(
        pairs,
        key=lambda x: correlation_matrix.loc[x[0], x[1]],
        reverse=True
    )

    for p in pairs_sorted:
        print(f"{p[0]} vs {p[1]} -> Correlation: {correlation_matrix.loc[p[0], p[1]]:.4f}")


if __name__ == "__main__":
    for metric in METRICS:
        main(metric)

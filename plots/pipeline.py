import os.path
import pandas as pd
import matplotlib.pyplot as plt
from pandas.errors import EmptyDataError
from sklearn.metrics import mean_squared_error
from itertools import combinations
from dtaidistance import dtw


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
METRIC = "container_fs_write"

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
    return (series - series.mean()) / series.std()


if __name__ == "__main__":
    # load the datasets into DataFrame
    dfs = {}
    for sub in SUBS:
        for model in MODELS:
            path = os.path.join(ROOT, sub, model, METRIC + ".csv")
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

    # 1️⃣ Overlay Plot
    plt.figure()
    for col in aligned_df.columns:
        plt.plot(aligned_df.index, aligned_df[col], label=col)

    plt.title("Overlay of Time Series (Normalized)")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


    # 2️⃣ Heatmap Function
    def plot_heatmap(matrix, title):
        plt.figure()
        plt.imshow(matrix.astype(float), interpolation='nearest')
        plt.colorbar()
        plt.xticks(range(n), names, rotation=45)
        plt.yticks(range(n), names)
        plt.title(title)
        plt.tight_layout()
        plt.show()

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

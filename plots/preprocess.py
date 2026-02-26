import os
import os.path
import hashlib
import pandas as pd


ROOT = "."
SUBS = [
    "data"
]
MODELS = [
    "ibm-2b-lmcache",
    "ibm-2b-no-lmcache"
]
METRICS = [
    "container_blkio_read",
    "container_blkio_write",
    "container_file_descriptors",
    "container_fs_read_count",
    "container_fs_read",
    "container_fs_write_count",
    "container_fs_write",
    "container_network_receive_pkt",
    "container_network_receive",
    "container_network_transmit_pkt",
    "container_network_transmit",
    "gpu_fb_free",
    "gpu_fb_used",
    "gpu_mem_copy",
    "pcie_rx",
    "pcie_tx"
]


def compute_row_hash(row: pd.Series, exclude_cols: set) -> str:
    """
    Compute stable hash from all columns except excluded ones.
    """
    items = []

    for col in sorted(row.index):
        if col in exclude_cols:
            continue
        value = "" if pd.isna(row[col]) else str(row[col])
        items.append(f"{col}={value}")

    joined = "|".join(items)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# read CSV
def load_and_normalize(csv_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)

        if df.empty:
            print(f"[info] {csv_path} is empty, skipping")
            return None

        if "timestamp" not in df.columns or "value" not in df.columns:
            print(f"[error] {csv_path} missing required columns")
            return None

        return df

    except Exception as e:
        print(f"[error] {csv_path}: {e}")
        return None


def split_by_hash(df: pd.DataFrame, output_dir: str):
    """
    Split dataframe into multiple CSV files based on hash of non-time/value columns.
    """

    exclude_cols = {"timestamp", "value"}

    # Compute hash column
    df["_hash"] = df.apply(
        lambda row: compute_row_hash(row, exclude_cols),
        axis=1
    )

    # Group by hash
    grouped = df.groupby("_hash")

    for h, group in grouped:
        out_df = group[["timestamp", "value"]].copy()
        out_df = out_df.sort_values("timestamp")

        output_path = os.path.join(output_dir, f"{h}.csv")
        out_df.to_csv(output_path, index=False)

        print(f"[info] wrote {output_path} ({len(out_df)} rows)")


def main(metric: str):
    for sub in SUBS:
        for model in MODELS:
            path = os.path.join(ROOT, sub, model, metric + ".csv")
            df = load_and_normalize(path)

            if df is None:
                continue

            output_dir = os.path.join(ROOT, sub, model, metric)
            os.makedirs(output_dir, exist_ok=True)

            print(f"[info] splitting {path}")
            split_by_hash(df, output_dir)


if __name__ == "__main__":
    for metric in METRICS:
        print("\nProcessing metric:", metric)
        main(metric)
        print(f"[info] Completed processing for {metric}")

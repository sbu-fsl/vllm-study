import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


plt.style.use("default")


def load_and_normalize(csv_path, label):
    try:
        df = pd.read_csv(csv_path, usecols=["timestamp", "value"])

        if df.empty:
            print(f"Info: {csv_path} is empty, skipping")
            return None

        # Ensure integer timestamps (epoch seconds)
        df["timestamp"] = df["timestamp"].astype("int64")

        df = df.sort_values("timestamp").reset_index(drop=True)

        # Normalize using raw integers (NO datetime)
        t0 = df["timestamp"].iloc[0]
        df["t_norm"] = df["timestamp"] - t0

        df["source"] = label

        # Optional debug export
        # df.to_csv(f'{csv_path}.normalized', index=False)

        dup = df["t_norm"].duplicated().sum()
        if dup > 0:
            print(f"WARNING: {dup} duplicate timestamps in {csv_path}")

        return df

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None


def main(csv_files, metric_name):
    os.makedirs("images", exist_ok=True)

    plt.figure(figsize=(10, 5))
    plotted_any = False

    for csv_path in csv_files:

        if not os.path.isfile(csv_path):
            print(f"Warning: {csv_path} not found, skipping")
            continue

        # Label = model + short hash
        model_name = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
        hash_name = os.path.splitext(os.path.basename(csv_path))[0][:8]

        label = f"{model_name}:{hash_name}"

        df = load_and_normalize(csv_path, label)
        if df is None:
            continue

        plt.plot(
            df["t_norm"],
            df["value"],
            label=label,
            linestyle='-',
            linewidth=1.2
        )

        plotted_any = True

    if not plotted_any:
        print("Nothing to plot.")
        return

    plt.title(f"Values of {metric_name}")
    plt.xlabel("Time since start (seconds)")
    plt.ylabel("Value")
    plt.legend(fontsize=8)
    plt.tight_layout()

    output_path = f"images/{metric_name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot normalized metric values from multiple CSV files"
    )

    parser.add_argument("metric_name")
    parser.add_argument("csv_files", nargs="+")

    args = parser.parse_args()

    main(args.csv_files, args.metric_name)

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

        df.to_csv(f'{csv_path}.normalized', index=False)

        # Detect true duplicates
        dup = df["t_norm"].duplicated().sum()
        if dup > 0:
            print(f"WARNING: {dup} duplicate timestamps in {csv_path}")

        return df

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None


def main(directories, metric_name):
    os.makedirs("images", exist_ok=True)

    plt.figure(figsize=(10, 5))
    plotted_any = False

    for directory in directories:
        csv_path = os.path.join(directory, metric_name + ".csv")

        if not os.path.isfile(csv_path):
            print(f"Warning: {csv_path} not found, skipping")
            continue

        label = os.path.basename(os.path.normpath(directory))

        df = load_and_normalize(csv_path, label)
        if df is None:
            continue

        plt.plot(
            df["t_norm"],
            df["value"],
            label=label,
            linestyle='-',
            linewidth=1.5
        )

        plotted_any = True

    if not plotted_any:
        print("Nothing to plot.")
        return

    plt.title(f"Values of {metric_name}")
    plt.xlabel("Time since start (seconds)")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

    output_path = f"images/{metric_name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot normalized metric values from multiple directories"
    )

    parser.add_argument("metric_name")
    parser.add_argument("directories", nargs="+")

    args = parser.parse_args()
    main(args.directories, args.metric_name)

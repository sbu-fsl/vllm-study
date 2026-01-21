import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_and_normalize(csv_path):
    # read only needed columns
    df = pd.read_csv(csv_path, usecols=["timestamp", "value"])

    # parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # normalize time so each series starts at 0
    t0 = df["timestamp"].iloc[0]
    df["t_norm"] = (df["timestamp"] - t0).dt.total_seconds()

    return df


def main(directories, metric_name):
    plt.figure()

    for directory in directories:
        csv_path = os.path.join(directory, metric_name + ".csv")

        if not os.path.isfile(csv_path):
            print(f"Warning: {csv_path} not found, skipping")
            continue

        # legend label = directory name
        label = os.path.basename(os.path.normpath(directory))

        df = load_and_normalize(csv_path)

        plt.plot(df["t_norm"], df["value"], label=label)

    plt.title(f"Values of {metric_name}")
    plt.xlabel("Time since start (seconds)")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot normalized metric values from multiple directories"
    )

    parser.add_argument(
        "metric_name",
        help="Metric CSV filename (e.g. metric1.csv)"
    )

    parser.add_argument(
        "directories",
        nargs="+",
        help="Directories containing the metric CSV (e.g. data/facebook-opt data/llama)"
    )

    args = parser.parse_args()
    main(args.directories, args.metric_name)

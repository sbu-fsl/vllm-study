import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.errors import EmptyDataError


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


def main(directories, metric_name):
    plt.figure()
    plotted_any = False

    for directory in directories:
        csv_path = os.path.join(directory, metric_name + ".csv")

        if not os.path.isfile(csv_path):
            print(f"Warning: {csv_path} not found, skipping")
            continue

        label = os.path.basename(os.path.normpath(directory))

        df = load_and_normalize(csv_path)
        if df is None:
            continue

        plt.plot(df["t_norm"], df["value"], label=label)
        plotted_any = True

    if not plotted_any:
        print("Nothing to plot.")
        return

    plt.title(f"Values of {metric_name}")
    plt.xlabel("Time since start (seconds)")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/{metric_name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot normalized metric values from multiple directories"
    )

    parser.add_argument(
        "metric_name",
        help="Metric CSV filename (without .csv)"
    )

    parser.add_argument(
        "directories",
        nargs="+",
        help="Directories containing the metric CSV"
    )

    args = parser.parse_args()
    main(args.directories, args.metric_name)

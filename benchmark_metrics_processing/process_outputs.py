import os
import json
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Augment CSV with GPU metrics from JSON files.")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file")
    parser.add_argument("--output_csv", required=True, help="Path to the output augmented CSV file")
    parser.add_argument("--runs_dir", required=True, help="Directory containing output runs with metrics JSONs")
    return parser.parse_args()

def main():
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    runs_dir = Path(args.runs_dir)

    # Read CSV and drop rows with any missing values
    df = pd.read_csv(input_csv)
    df = df.dropna()
    full_gpu_rows = []

    for index, row in df.iterrows():
        run_name = row["run_name"]
        run_dir = runs_dir / run_name

        if not run_dir.exists():
            print(f"[!] Skipping: run folder not found for {run_name}")
            continue

        try:
            start_iso = datetime.strptime(row["start_date"].split(",")[0], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%SZ")
            end_iso = datetime.strptime(row["end_date"].split(",")[0], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception as e:
            print(f"[!] Date parsing error for {run_name}: {e}")
            continue

        expected_json_filename = f"{start_iso}_{end_iso}.json"
        row_data = {}

        for metric_folder in run_dir.iterdir():
            if not metric_folder.is_dir():
                continue

            json_file = metric_folder / expected_json_filename
            if not json_file.exists():
                continue

            try:
                with open(json_file) as f:
                    data = json.load(f)

                results = data.get("data", {}).get("result", [])
                for gpu_idx, gpu_metric in enumerate(results):
                    values = gpu_metric.get("values", [])
                    col_name = f"{metric_folder.name}_GPU{gpu_idx}"
                    row_data[col_name] = json.dumps(values)

            except Exception as e:
                print(f"[!] Error reading {json_file}: {e}")
                continue

        full_gpu_rows.append(row_data)

    # Combine and save
    gpu_df = pd.DataFrame(full_gpu_rows)
    df_combined = pd.concat([df.reset_index(drop=True), gpu_df], axis=1)
    df_combined.to_csv(output_csv, index=False)
    print(f"[✓] Augmented CSV saved to: {output_csv}")

if __name__ == "__main__":
    main()

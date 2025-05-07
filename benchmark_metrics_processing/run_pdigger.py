import argparse
import pandas as pd
import json
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

# === Utility: Convert CSV-style datetime to ISO8601 ===
def convert_date_format(date_str):
    try:
        dt = datetime.strptime(date_str.split(',')[0], "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
        raise ValueError(f"Invalid date format: {date_str} -> {e}")

# === CLI Argument Parser ===
parser = argparse.ArgumentParser(description="Run Prometheus Digger for experiment runs.")
parser.add_argument('--csv', required=True, help="Path to the experiment CSV file")
parser.add_argument('--config', required=True, help="Path to the base config.json")
parser.add_argument('--out', default="output", help="Directory where pdigger writes by default")
parser.add_argument('--renamed-out', default="output_runs", help="Directory to store renamed output folders")
parser.add_argument('--prom', default="./pdigger", help="Path to pdigger executable")
args = parser.parse_args()

# === Prepare Paths ===
CSV_FILE = Path(args.csv)
CONFIG_FILE = Path(args.config)
DEFAULT_OUTPUT_DIR = Path(args.out)
RENAMED_OUTPUT_ROOT = Path(args.renamed_out)
PROM_COMMAND = args.prom

# === Step 1: Load CSV ===
df = pd.read_csv(CSV_FILE)

# === Step 2: Build pdigger once ===
print("[⚙️ ] Building Prometheus Digger...")
try:
    subprocess.run(["go", "build", "-o", PROM_COMMAND], check=True)
    subprocess.run(["chmod", "+x", PROM_COMMAND], check=True)
    print("[✓] pdigger built and ready.")
except subprocess.CalledProcessError as e:
    print(f"[❌] Failed to build pdigger: {e}")
    exit(1)

# === Step 3: Process each run ===
for index, row in df.iterrows():
    try:
        run_name = row["run_name"]
        start_iso = convert_date_format(row["start_date"])
        end_iso = convert_date_format(row["end_date"])

        # Load and update config.json
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        config["from"] = start_iso
        config["to"] = end_iso

        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)

        # Run Prometheus Digger
        print(f"\n[→] Running Prometheus Digger for: {run_name}")
        subprocess.run([PROM_COMMAND], check=True)

        # Move output to renamed folder
        renamed_output_path = RENAMED_OUTPUT_ROOT / str(run_name)
        renamed_output_path.parent.mkdir(parents=True, exist_ok=True)

        if DEFAULT_OUTPUT_DIR.exists():
            if renamed_output_path.exists():
                shutil.rmtree(renamed_output_path)
            shutil.move(str(DEFAULT_OUTPUT_DIR), str(renamed_output_path))
            print(f"[✓] Output moved to: {renamed_output_path}")
        else:
            print(f"[!] Warning: Output directory '{DEFAULT_OUTPUT_DIR}' not found.")

    except Exception as e:
        print(f"[❌] Skipping run '{row.get('run_name', 'UNKNOWN')}': {e}")

print("\n✅ All valid runs processed.")


# Benchmark Metrics Processing Utilities

This folder provides scripts to process and augment benchmark experiment results with Prometheus GPU/CPU/memory metrics.

## Scripts

### 1. `process_outputs.py`

Augments the original benchmark CSV file with additional GPU metrics extracted from Prometheus output JSONs.

**Usage:**

```bash
python process_outputs.py --input_csv path/to/benchmark_results.csv --output_csv path/to/augmented_results.csv --runs_dir path/to/output_runs/
```

**Arguments:**

- `--input_csv`: Path to the original benchmark CSV.
- `--output_csv`: Path where the augmented CSV will be saved.
- `--runs_dir`: Directory containing output run folders with metrics JSON files.

### 2. `run_pdigger.py`

Automates running Prometheus Digger (pdigger) across all benchmark runs by adjusting the time window based on each run's start and end timestamps.

**Usage:**

```bash
python run_pdigger.py --csv path/to/benchmark_results.csv --config path/to/base_config.json --out output --renamed-out output_runs --prom ./pdigger
```

**Arguments:**

- `--csv`: Path to the benchmark CSV file containing run metadata.
- `--config`: Path to the base Prometheus Digger config.json file.
- `--out`: Temporary output directory used by pdigger (default: `output`).
- `--renamed-out`: Directory where per-run outputs will be organized (default: `output_runs`).
- `--prom`: Path to the built pdigger executable.

## Requirements

- Python 3.8+
- pandas
- Prometheus Digger (pdigger) built from source (Go compiler required)

## Notes

- Make sure Prometheus Digger (pdigger) is built successfully before running `run_pdigger.py`.
- Ensure Prometheus server retention policy is long enough to query past GPU/CPU metrics.

## Folder Structure

```
suny-ibm-multicloud-gpus/
├── benchmark_metrics_utils/
│   ├── process_outputs.py
│   ├── run_pdigger.py
│   └── README.md
├── (other files)
```

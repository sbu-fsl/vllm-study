# Quick Setup Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

## Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/sbu-fsl/SUNY-iBM-multicloud-gpus.git
   cd SUNY-iBM-multicloud-gpus
   git checkout timeseries-benchmark-release
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Benchmark

### Option 1: Full Benchmark (recommended for first run)
```bash
python main_timeseries_benchmark.py
```

This will:
- Download NAB datasets (if not present)
- Run all three models (ARIMA, TOTO, Granite)
- Test all detectors (SeasonalQuantile, AdaptiveQuantile, VolatilityNormalized)
- Generate results in `runs/` directory

### Option 2: Quick Test (single dataset)
```bash
python main_timeseries_benchmark.py --quick-test
```

### Option 3: vLLM Production Analysis
```bash
python vllm_anomaly_analysis.py --dataset vllm_datasets/
```

## Understanding Results

Results are saved in the `runs/` directory with timestamp:
```
runs/TSB_YYYY-MM-DD_HH-MM-SS_<config>/
├── exec_summary.csv       # Performance metrics
├── plots/                 # Visualization
└── logs/                  # Execution logs
```

## Common Issues

### 1. CUDA/GPU Issues
If you don't have a GPU, models will automatically fall back to CPU. For TOTO and Granite, this will be slower.

### 2. Memory Errors
Reduce batch size or number of Monte Carlo samples:
```bash
python main_timeseries_benchmark.py --toto-samples 64
```

### 3. Dataset Download Failures
Manually download NAB datasets:
```bash
git clone https://github.com/numenta/NAB.git
cp -r NAB/data dataset/
```

## Configuration

Edit `main_timeseries_benchmark.py` to customize:
- Models to evaluate
- Datasets to test
- Detector configurations
- Output directories

## Next Steps

1. Review results in `runs/latest/exec_summary.csv`
2. Check visualizations in `runs/latest/plots/`
3. Read `README.md` for detailed documentation
4. Explore `scripts/` for phase-specific execution

## Support

For issues or questions:
- Open a GitHub issue
- Check `README.md` for detailed documentation
- Review example outputs in `results/` directory

## Citation

If you use this code, please cite:
```bibtex
@misc{timeseries_benchmark_2025,
  author = {SUNY Stony Brook Research Team},
  title = {Time-Series Anomaly Detection Benchmark},
  year = {2025},
  publisher = {GitHub}
}
```

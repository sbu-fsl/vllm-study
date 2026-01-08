# Time-Series Anomaly Detection Benchmark

A comprehensive benchmarking framework for evaluating foundation models on time-series anomaly detection tasks.

## Overview

This repository contains a complete implementation of a four-phase evaluation framework comparing ARIMA, DataDog TOTO, and IBM Granite TTM models across diverse anomaly detection scenarios. The work includes novel detection algorithms, production validation, and systematic performance analysis.

## Key Features

- **Multiple Models**: ARIMA baseline, DataDog TOTO, IBM Granite TTM
- **Novel Detectors**: VolatilityNormalized, AdaptiveQuantile, SeasonalQuantile algorithms
- **Production Validation**: Real vLLM deployment testing with 420 test cases
- **Comprehensive Benchmarking**: 15 NAB datasets + custom infrastructure metrics
- **Performance Analysis**: Energy consumption, CO₂ emissions, GPU metrics tracking

## Repository Structure

```
.
├── main_timeseries_benchmark.py    # Main benchmark execution
├── vllm_anomaly_analysis.py        # vLLM production analysis
├── vllm_anomaly_detection_report.py # Report generation
├── vllm_benchmark.py                # vLLM benchmarking utilities
├── analyze_phase2_results.py       # Phase 2 analysis scripts
├── enhanced_analysis.py             # Advanced analytics
├── pattern_characterization.py     # Pattern analysis
├── publication_report_generator.py # Report generation
├── requirements.txt                # Python dependencies
├── scripts/                        # Execution scripts
├── utils/                          # Utility functions
├── dataset/                        # NAB datasets
├── vllm_datasets/                  # vLLM test data
├── results/                        # Experimental results
└── runs/                           # Experiment run logs
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd timeseries_benchmark_release

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Full Benchmark

```bash
python main_timeseries_benchmark.py
```

### Run vLLM Production Analysis

```bash
python vllm_anomaly_analysis.py
```

### Analyze Phase 2 Results

```bash
python analyze_phase2_results.py
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy, Pandas, Matplotlib
- Statsmodels (for ARIMA)
- Transformers (for foundation models)
- See `requirements.txt` for full list

## Key Results

### Model Performance

| Model | Mean F1 | Speedup | Best Use Case |
|-------|---------|---------|---------------|
| **TOTO** | 0.262 | 1× (baseline) | High accuracy scenarios |
| **Granite TTM** | 0.216 | 340× benchmark, 34.6× production | Low-latency, high-throughput |
| **ARIMA** | 0.201 | Fast | Simple patterns, interpretability |

### Detector Performance

- **VolatilityNormalized**: Best for subtle anomalies and heteroscedastic data
- **AdaptiveQuantile**: Best for high-volatility patterns
- **SeasonalQuantile**: Best for spike detection

### Production Validation

- 420 test cases across GPU, memory, network, disk metrics
- 100% detection rate on critical GPU metrics
- 42.6% overall detection rate (expected for subtle changes)

## Experimental Phases

### Phase 0: Validation
- Single dataset validation
- Detector implementation verification
- Baseline establishment

### Phase 1: Initial Comparison
- 6 representative datasets
- Model comparison
- Pattern identification

### Phase 2: Comprehensive Benchmark
- 15 NAB datasets
- Full detector evaluation
- Statistical validation

### Phase 3: Production Validation
- Real vLLM deployment
- 420 production test cases
- Metric-specific analysis

## Citation

If you use this code or data in your research, please cite:

```bibtex
@misc{timeseries_benchmark_2025,
  author = {SUNY Stony Brook Research Team},
  title = {Time-Series Anomaly Detection Benchmark},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/sbu-fsl/SUNY-iBM-multicloud-gpus}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **SUNY-IBM AI Collaborative Research Alliance** - Compute resources
- **Stony Brook University** - Research infrastructure and support
- Built on systems research at Stony Brook University

## Contact

For questions or collaboration:
- Open an issue on GitHub
- SUNY-IBM AI Research Alliance

## Related Work

- NAB (Numenta Anomaly Benchmark)
- IBM Granite Time-Series Model (TTM-R2)
- DataDog TOTO Forecaster
- vLLM: Fast LLM Serving Framework

# TIMM-Imagenet Distributed Benchmark - SUNY-IBM Multicloud GPU Project

This folder adds a **highly scalable distributed benchmark** framework using **Timm library models** and **PyTorch DistributedDataParallel (DDP)** over the ImageNet dataset.

## Project Overview

The script `timm_imagenet_distributed_benchmark.py` benchmarks dozens of deep learning models from the Timm model zoo under different distributed training configurations.  
It measures:
- Training time per epoch
- Maximum GPU memory usage
- Model validation accuracy
- Validation loss
- Number of parameters
- Activation functions used
- Other model and experiment metadata

The experiments are designed to simulate realistic distributed workloads on multi-GPU cloud setups.

## How to Run

**Simple one-line execution:**

```bash
bash run_experiments.sh
```

**Manual execution:**

```bash
python timm_imagenet_distributed_benchmark.py --csv_file timm_imagenet_experiments.csv
```

**Arguments:**
- `--csv_file`: (Optional) Name of the CSV file where experiment results will be saved.  
  Default: `experiment_log.csv`

**Output:**
- **Summary CSV file:**  
  e.g., `timm_imagenet_experiments.csv`
- **Training logs:**  
  Stored in the `logs/` directory

**Collected metrics include:**
- Start and end timestamps
- Distributed setup information
- Model name, batch size, number of epochs
- Input shape used (224x224, 384x384, 512x512)
- GPU type and GPU count
- Epoch duration (seconds)
- Maximum GPU memory usage (MB)
- Validation accuracy (%)
- Validation loss
- Number of parameters
- Activation functions
- Any experiment errors (captured separately)

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- timm (latest)
- ImageNet-1K Dataset (or fallback to dummy data for testing)

## Notes

- Experiments automatically adjust for small (224x224), medium (384x384), and large (512x512) input models.
- Batch sizes are dynamically chosen based on target GPU memory utilization (~20% and ~90%).
- If ImageNet dataset is unavailable, fallback dummy datasets will be used.

## Folder Structure

```
suny-ibm-multicloud-gpus/
├── timm_imagenet_benchmarks/
│   ├── timm_imagenet_distributed_benchmark.py
│   ├── run_experiments.sh
│   └── README.md
├── (other files)
```

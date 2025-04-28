# SUNY-IBM Multicloud GPU - Imagenet Distributed Benchmark

This folder adds distributed benchmarking experiments for various ImageNet models on multi-GPU systems using PyTorch DistributedDataParallel (DDP).

## Project Overview

The script `imagenet_distributed_benchmark.py` runs distributed training experiments across Torchvision and Hugging Face models on ImageNet subsets.  
It measures:
- Training time per epoch
- Maximum GPU memory usage
- Training and validation behavior under different batch sizes, number of epochs, and GPU counts.

## How to Run

**Simple one-line execution:**

```bash
bash run_experiments.sh
```

**Manual execution:**

```bash
python imagenet_distributed_benchmark.py --logfile ImagenetNet_Experiments.csv
```

**Arguments:**
- `--logfile`: (Optional) Name of the CSV file where experiment results will be saved.  
  Default: `experiment_log.csv`

**Output:**
- **Summary CSV file:**  
  e.g., `ImagenetNet_Experiments.csv`
- **Training logs:**  
  Stored in the `logs/` directory

**Collected metrics include:**
- Start and end timestamps
- Distributed setup info
- Model name, batch size, number of epochs
- GPU type and GPU count
- Epoch duration (seconds)
- Maximum GPU memory usage (MB)
- Any experiment errors (captured separately)

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- transformers (Hugging Face)

## Notes

- Experiments run on small subsets (10,000 train and 100 validation samples) for quicker benchmarking.
- If ImageNet data is unavailable, dummy datasets are automatically used.

## Folder Structure

```
suny-ibm-multicloud-gpus/
├── imagenet_benchmarks/
│   ├── imagenet_distributed_benchmark.py
│   ├── run_experiments.sh
│   └── README.md
├── (other files)
```

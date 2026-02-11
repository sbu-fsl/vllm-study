#!/usr/bin/env python3
from pathlib import Path
from pipeline import TimeSeriesComparator

# Test with just 2 CSV files first
dataset_dir = Path("/Users/anajafizadeh/projects/sunyibm/vllm-study/timeseries-models/vllm_datasets")
csv_files = sorted(list(dataset_dir.glob("*.csv")))[:2]

print(f"Testing with {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"  - {f.name}")

comparator = TimeSeriesComparator(
    csv_files=[str(f) for f in csv_files],
    labels=[f.stem for f in csv_files]
)

print("\nTimeSeriesComparator initialized successfully!")
print(f"Loaded {len(comparator.aligned_series)} series")

# Test correlation matrix
print("\n--- Testing correlation matrix ---")
corr_matrix, labels = comparator.compute_correlation_matrix()
print(f"Correlation matrix shape: {corr_matrix.shape}")
print(f"Correlation values:\n{corr_matrix}")

print("\n✓ All tests passed!")

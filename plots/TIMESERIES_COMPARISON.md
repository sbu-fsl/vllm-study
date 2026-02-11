# Time Series Comparison Tools

This directory now contains enhanced tools for comparing multiple time series using four complementary visualization methods.

## Overview

The `TimeSeriesComparator` class in `pipeline.py` provides comprehensive analysis tools for comparing time series data from different workloads or conditions:

### Four Visualization Methods

1. **Correlation Heatmap (Normalized)**
   - Shows Pearson correlation coefficients between all pairs of time series
   - Range: -1 to 1 (1 = perfect positive correlation, -1 = perfect negative)
   - Use case: Identifies which workloads show similar overall behavior

2. **Derivative Correlation Heatmap**
   - Computes correlations of time derivatives (rates of change)
   - Reveals dynamic similarities beyond static values
   - Use case: Identifies workloads with similar change patterns

3. **DTW Distance Matrix**
   - Dynamic Time Warping distances between all series
   - Accounts for temporal misalignments
   - Use case: Groups workloads with similar temporal patterns regardless of timing

4. **PCA Scatter Plot (Feature-Based)**
   - Extracts 12 statistical features from each time series:
     - Mean, std, min, max, median, Q1, Q3
     - Mean absolute derivative, derivative std dev
     - Coefficient of variation
     - Autocorrelation
     - Entropy
   - Reduces to 2D using PCA for visualization
   - Use case: Clusters workloads based on comprehensive feature profiles

## Usage

### Basic Example: Compare Multiple CSV Files

```python
from pipeline import TimeSeriesComparator

# Create comparator with your CSV files
csv_files = [
    "path/to/metric_workload1.csv",
    "path/to/metric_workload2.csv", 
    "path/to/metric_workload3.csv",
]

labels = ["Workload 1", "Workload 2", "Workload 3"]

comparator = TimeSeriesComparator(csv_files, labels=labels)

# Generate all four visualizations
comparator.plot_all_comparisons(output_dir="./results")
```

### Expected Outputs

- `00_aligned_time_series.png` - Original normalized time series
- `01_correlation_heatmap.png` - Pearson correlations
- `02_derivative_correlation_heatmap.png` - Rate-of-change correlations
- `03_dtw_distance_matrix.png` - Temporal pattern distances
- `04_pca_scatter_plot.png` - Feature-based clustering

### Advanced Usage

```python
# Get individual matrices for custom analysis
corr_matrix, labels = comparator.compute_correlation_matrix()
deriv_corr_matrix, labels = comparator.compute_derivative_correlation_matrix()
dtw_matrix, labels = comparator.compute_dtw_distance_matrix()

# Extract features for other analyses
features, labels = comparator.extract_features()
```

## CSV File Format

Your CSV files should contain two columns:
- `timestamp`: ISO format timestamp (e.g., "2025-10-02 00:22:34")
- `value`: numeric metric value

Example:
```csv
timestamp,value
2025-10-02 00:22:34,42.5
2025-10-02 00:22:35,43.1
2025-10-02 00:22:36,42.8
```

## Interpreting Results

### Correlation Heatmap
- Values close to 1 (red): Highly correlated workloads
- Values close to 0 (white): Independent workloads
- Values close to -1 (blue): Negatively correlated workloads

### DTW Distance Matrix
- Smaller values: More similar temporal patterns
- Values toward edges appear in warm colors (yellow/orange/red)
- Helps identify workloads with different temporal signatures

### PCA Scatter Plot
- Proximity indicates feature similarity
- Clusters represent similar workload characteristics
- Explained variance shown on axes

## Running Scripts

```bash
# Run the example comparison
python3 compare_timeseries.py

# Or test the pipeline
python3 test_pipeline.py
```

## Required Packages

- pandas
- matplotlib
- scipy
- scikit-learn
- seaborn  
- dtaidistance (optional, for faster DTW computation)

Install with:
```bash
pip install -r requirements.txt
```

## Tips for Best Results

1. **Align workloads**: Ensure time ranges are comparable
2. **Normalize values**: Apply z-score normalization if metrics have different scales
3. **Handle missing data**: CSV files with empty rows are automatically skipped
4. **Compare similar metrics**: Best results when comparing same metric under different loads
5. **Sufficient data points**: Aim for 500+ data points per series

## Architecture

- `pipeline.py`: Core `TimeSeriesComparator` class with all methods
- `compare_timeseries.py`: Example usage script
- `test_pipeline.py`: Quick verification script
- `requirements.txt`: Dependencies

# Time Series Comparison Suite - Complete Guide

## 📊 Overview

You now have a complete suite of tools to compare multiple time series from different workloads using **four complementary visualization methods**:

1. **Correlation Heatmap (Normalized)** - Shows overall similarity
2. **Derivative Correlation Heatmap** - Shows rate-of-change similarity
3. **DTW Distance Matrix** - Shows temporal pattern similarity
4. **PCA Scatter Plot (Feature-Based)** - Shows feature-based clustering

---

## 🚀 Quick Start

### Option 1: Use the Jupyter Notebook (Recommended for Interactive Exploration)

```bash
cd /Users/anajafizadeh/projects/sunyibm/vllm-study/plots
jupyter notebook timeseries_comparison.ipynb
```

Then:
1. Modify the file selection in the "Load and Explore CSV Files" section
2. Run all cells sequentially
3. Explore the generated visualizations

### Option 2: Use the Python Class Directly

```python
from pipeline import TimeSeriesComparator
from pathlib import Path

# Select your CSV files
csv_files = [
    "/path/to/metric_workload1.csv",
    "/path/to/metric_workload2.csv",
    "/path/to/metric_workload3.csv",
]

# Create comparator
comparator = TimeSeriesComparator(csv_files, labels=["WL1", "WL2", "WL3"])

# Generate all visualizations
comparator.plot_all_comparisons(output_dir="./results")
```

### Option 3: Run Example Scripts

```bash
# See comprehensive examples
python3 examples.py

# Run quick test
python3 test_pipeline.py

# Or run the main comparison
python3 compare_timeseries.py
```

---

## 📁 Files in This Directory

### Core Implementation
- **`pipeline.py`** - Main `TimeSeriesComparator` class with all methods
  - `load_and_normalize()` - Load and normalize CSV files
  - `compute_correlation_matrix()` - Pearson correlation
  - `compute_derivative_correlation_matrix()` - Rate-of-change correlation
  - `compute_dtw_distance_matrix()` - Dynamic Time Warping distances
  - `extract_features()` - Extract 12 statistical features
  - `plot_*()` - Visualization methods

### Examples and Notebooks
- **`timeseries_comparison.ipynb`** - Interactive Jupyter notebook with full workflow
- **`compare_timeseries.py`** - Example script for batch processing
- **`examples.py`** - Multiple detailed examples
- **`test_pipeline.py`** - Quick verification script

### Documentation
- **`TIMESERIES_COMPARISON.md`** - Detailed technical documentation
- **`README.md`** - Requirements and setup (if exists)

---

## 🔍 Understanding the Four Visualization Methods

### 1. Correlation Heatmap (Normalized)

**What it shows:** Pearson correlation coefficient between each pair of time series

**Values:**
- **1.0 (Red):** Perfect positive correlation - series move together
- **0.0 (White):** No correlation - series move independently
- **-1.0 (Blue):** Perfect negative correlation - series move oppositely

**When to use:** Identify which workloads have similar overall behavior

**Example interpretation:**
```
CPU_Usage  <-> GPU_Util: 0.85  → Strong correlation
CPU_Usage  <-> Memory:   0.42  → Weak correlation
GPU_Power  <-> GPU_Temp: 0.92  → Very strong correlation
```

---

### 2. Derivative Correlation Heatmap

**What it shows:** Correlation of **rate of change** (first derivatives) between series

**Key insight:** Captures how similarly two series **change over time**, even if their absolute values differ

**When to use:** Identify workloads with similar **dynamics** or **behavior patterns**

**Example interpretation:**
```
Two series can have:
- Low correlation in values but HIGH derivative correlation
  → Similar patterns but shifted/scaled
- High correlation in values but LOW derivative correlation
  → Similar averages but different behaviors
```

---

### 3. DTW Distance Matrix

**What it shows:** Dynamic Time Warping distances between all pairs

**Why DTW matters:**
- Standard distance metrics assume 1-to-1 time correspondence
- DTW allows for temporal stretching/compression
- Captures series that have similar shapes but different timing

**Values:**
- **Small distances (Yellow):** Very similar temporal patterns
- **Large distances (Red/Orange):** Very different temporal patterns

**When to use:** Identify workloads with similar **temporal signatures** regardless of speed

**Example:**
```
Series A: Slow ramp up, plateau
Series B: Fast ramp up, plateau
→ Low correlation (different speeds)
→ High DTW similarity (same shape)
```

---

### 4. PCA Scatter Plot (Feature-Based)

**What it shows:** 2D projection (PCA) of 12 extracted features per series

**Features extracted:**
- Mean, Std Dev, Min, Max, Median
- Q1 (25th percentile), Q3 (75th percentile)
- Mean absolute derivative
- Std dev of derivative
- Coefficient of variation
- Plus 2 more advanced features

**Interpretation:**
- **Proximity** = feature similarity
- **Clusters** = workloads with similar statistical profiles
- **Axis labels** show explained variance

**When to use:** High-level clustering to identify workload groups

---

## 📊 Output Files

When you run the comparison, you get:

1. **00_aligned_time_series.png**
   - All normalized series plotted together
   - Verify alignment and magnitude relationships

2. **01_correlation_heatmap.png**
   - Heatmap of Pearson correlations
   - Matrix form for easy reference

3. **02_derivative_correlation_heatmap.png**
   - Heatmap of derivative correlations
   - Shows rate-of-change patterns

4. **03_dtw_distance_matrix.png**
   - Heatmap of DTW distances
   - Shows temporal pattern differences

5. **04_pca_scatter_plot.png**
   - 2D scatter plot from PCA
   - Each series labeled with coordinates

---

## 💡 Practical Use Cases

### Use Case 1: Compare Same Metric Under Different Workloads

```python
csv_files = [
    "CPU_Usage_light_load.csv",
    "CPU_Usage_medium_load.csv",
    "CPU_Usage_heavy_load.csv",
]

comparator = TimeSeriesComparator(csv_files, labels=["Light", "Medium", "Heavy"])
comparator.plot_all_comparisons()

# Insight: Do heavier loads show correlations with higher resource usage?
```

### Use Case 2: Identify Resource Coupling

```python
csv_files = [
    "CPU_Usage.csv",
    "Memory_Usage.csv",
    "IO_Throughput.csv",
    "Network_Throughput.csv",
]

comparator = TimeSeriesComparator(csv_files)
comparator.plot_all_comparisons()

# Insight: Which resources are tightly coupled?
```

### Use Case 3: Validate Consistency

```python
csv_files = [
    "gpu_temp_trial1.csv",
    "gpu_temp_trial2.csv",
    "gpu_temp_trial3.csv",
]

comparator = TimeSeriesComparator(csv_files, labels=["Trial 1", "Trial 2", "Trial 3"])

# Insight: Are experimental results reproducible/consistent?
```

---

## 🛠️ Advanced Usage

### Extract Correlation Matrix Only

```python
from pipeline import TimeSeriesComparator

comparator = TimeSeriesComparator(csv_files)
corr_matrix, labels = comparator.compute_correlation_matrix()

# Use for custom analysis
print(corr_matrix)  # numpy array
```

### Get Feature Vectors

```python
features, labels = comparator.extract_features()
# features: (n_series, 12) numpy array
# Can use for clustering, classification, etc.
```

### Plot Only Specific Visualizations

```python
# Just correlation
corr_matrix, labels = comparator.compute_correlation_matrix()
comparator._plot_heatmap(corr_matrix, labels, title="My Correlations")

# Just time series
comparator.plot_time_series(filename="my_series.png")

# Just PCA
comparator._plot_pca_scatter(filename="my_pca.png")
```

---

## 📋 CSV Format Requirements

Your CSV files should have:

```csv
timestamp,value
2025-10-02 00:22:34,42.5
2025-10-02 00:22:35,43.1
2025-10-02 00:22:36,42.8
...
```

**Required columns:**
- `timestamp` - ISO format timestamp
- `value` - Numeric metric value

**Automatic handling:**
- Empty files are skipped
- Series are automatically aligned to common time grid
- Values are z-score normalized
- Temporal gaps are interpolated

---

## 🐛 Troubleshooting

### Issue: "No valid CSV files loaded"
**Solution:** Check that your CSV files have `timestamp` and `value` columns

### Issue: "dtaidistance not installed"
**Solution:** Install with `pip install dtaidistance`
(Optional - slower DTW will be used if not available)

### Issue: Memory error with large datasets
**Solution:** Reduce number of files or time grid resolution:
```python
# In pipeline.py, line ~90:
n_points = 500  # Reduce from 1000
```

### Issue: Visualization looks strange
**Solution:** Check data range:
```python
for name, series in comparator.aligned_series.items():
    print(f"{name}: min={series.min():.2f}, max={series.max():.2f}")
```

---

## 📚 Required Packages

```
pandas
matplotlib
scipy
scikit-learn
seaborn
dtaidistance (optional, for faster DTW)
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🎯 Key Takeaways

1. **Correlation Heatmap** → Overall similarity
2. **Derivative Correlation** → Dynamic similarity
3. **DTW Distance** → Temporal pattern similarity
4. **PCA Plot** → Feature-based clustering

**Best practice:** Use all four together! They provide complementary views:
- Correlation = what values do
- Derivatives = how values change
- DTW = what patterns emerge
- PCA = what features matter

---

## 📞 Need Help?

- Check `TIMESERIES_COMPARISON.md` for technical details
- See `examples.py` for code templates
- Review `timeseries_comparison.ipynb` for step-by-step walkthrough

---

## 🎓 Mathematical Background

### Pearson Correlation
$$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}$$

### Dynamic Time Warping
$$DTW(A, B) = \min_{w} \sqrt{\sum_{i,j \in w} (a_i - b_j)^2}$$

### PCA
Finds orthogonal directions of maximum variance:
$$PC_k = \arg\max_v \text{Var}(X \cdot v)$$

---

Last updated: February 11, 2026

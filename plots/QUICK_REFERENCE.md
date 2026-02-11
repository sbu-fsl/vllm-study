# Time Series Comparison - Quick Reference Card

## 🎯 The 4 Methods at a Glance

| Method | What It Shows | Good For | Range | Color |
|--------|---------------|----------|-------|-------|
| **Correlation** | Overall similarity | Finding correlated behaviors | -1 to 1 | Blue/White/Red |
| **Derivative Corr** | Rate-of-change similarity | Finding similar dynamics | -1 to 1 | Blue/White/Red |
| **DTW Distance** | Temporal pattern similarity | Finding shape matches | 0 to ∞ | Yellow/Orange/Red |
| **PCA Scatter** | Feature clustering | Grouping workloads | 2D points | Colored dots |

---

## 🚀 Getting Started in 30 Seconds

### Step 1: Open Jupyter Notebook
```bash
cd /Users/anajafizadeh/projects/sunyibm/vllm-study/plots
jupyter notebook timeseries_comparison.ipynb
```

### Step 2: Select CSV Files
```python
csv_files = [  # Your CSV paths here
    "/path/to/file1.csv",
    "/path/to/file2.csv",
]
```

### Step 3: Run All Cells
Click "Run All" in the Jupyter menu

### Step 4: View Results
Look in `results/` folder for 5 PNG files

---

## 💻 Using the Python Class

```python
from pipeline import TimeSeriesComparator

# 1. Create
comparator = TimeSeriesComparator(csv_files, labels=["Series A", "Series B"])

# 2. Generate all visualizations
comparator.plot_all_comparisons(output_dir="./results")

# 3. Or get individual matrices
corr, labels = comparator.compute_correlation_matrix()
dtw, labels = comparator.compute_dtw_distance_matrix()
features, labels = comparator.extract_features()
```

---

## 📊 Reading the Heatmaps

### Correlation Heatmap
```
1.0   ████ Perfect match
0.5   ████ Moderate correlation
0.0   ████ No correlation
-0.5  ████ Moderate inverse
-1.0  ████ Perfect inverse
```

**Diagonal:** Always 1.0 (series correlates with itself)
**Off-diagonal:** Correlation between different series

### DTW Distance Heatmap
```
Yellow  ████ Similar patterns
Orange  ████ Somewhat different
Red     ████ Very different
```

**Diagonal:** Always 0 (distance from self is 0)
**Off-diagonal:** Distance between different series

---

## 🎨 What Each Output File Means

1. **00_aligned_time_series.png** - Raw data check
   - All series on same plot
   - Verify they look "reasonable"

2. **01_correlation_heatmap.png** - Value similarity
   - Strong diagonal block = similar values
   - Blue regions = opposite behaviors

3. **02_derivative_correlation_heatmap.png** - Behavior similarity
   - High values = same change patterns
   - Different from (1)? → Scaling issues or shifts

4. **03_dtw_distance_matrix.png** - Temporal similarity
   - Yellow cluster = similar shapes
   - Red regions = different shapes

5. **04_pca_scatter_plot.png** - Feature clusters
   - Close points = similar workloads
   - Spread = diverse workloads

---

## ✅ Interpretation Guide

### All 4 Methods Agree (All Show Similarity)
→ **Highly similar workloads**
- Same behavior, same dynamics, same patterns
- Action: May be redundant

### High Correlation, High DTW, Low Derivative
→ **Static similarity, different rates of change**
- Same average values but different speeds
- Action: Investigate why speeds differ

### Low Correlation, High Derivative
→ **Dynamic similarity despite different values**
- Different scales but similar patterns
- Action: Check normalization/scaling

### All Methods Show Low Similarity
→ **Fundamentally different workloads**
- No overlap in behavior
- Action: Keep all distinct types

---

## 🔧 Customization

### Change Number of Files
In notebook, modify:
```python
N_FILES = 6  # Change this number
selected_files = csv_files[:N_FILES]
```

### Specify Exact Files
```python
selected_files = [
    DATASET_DIR / "CPU_Usage.csv",
    DATASET_DIR / "Memory_Usage.csv",
]
```

### Extract More Features
In pipeline.py `extract_features()`, add more to the list:
```python
features = np.array([
    # ... existing 12 features ...
    np.percentile(series, 90),  # Add P90
    np.percentile(series, 10),  # Add P10
])
```

### Adjust DTW Resolution
For faster computation, in pipeline.py line ~90:
```python
n_points = 500  # Reduce from 1000 for speed
```

---

## 📈 Common Patterns

### Pattern 1: Clear Clusters in PCA
→ Natural grouping of workloads
- Use cluster labels for workflow optimization

### Pattern 2: Diagonal-Dominant Correlation
→ Most series are independent
- Good diversity in workload mix

### Pattern 3: One Series Outlier (PCA)
→ One workload is fundamentally different
- May warrant separate optimization

### Pattern 4: High DTW, Low Correlation
→ Same shapes at different scales
- Normalization successful, patterns similar

---

## 🐍 Python Snippets

### Get highest correlated pair
```python
from pipeline import TimeSeriesComparator
comp = TimeSeriesComparator(files)
corr, labels = comp.compute_correlation_matrix()
i, j = np.unravel_index(np.argmax(np.triu(corr, k=1)), corr.shape)
print(f"{labels[i]} ↔ {labels[j]}: {corr[i,j]:.3f}")
```

### Find outlier workload
```python
features, labels = comp.extract_features()
from sklearn.decomposition import PCA
pca = PCA(2)
projected = pca.fit_transform(features)
distances = np.sqrt((projected**2).sum(axis=1))
print(f"Outlier: {labels[np.argmax(distances)]}")
```

### Export correlation matrix to CSV
```python
corr, labels = comp.compute_correlation_matrix()
pd.DataFrame(corr, index=labels, columns=labels).to_csv("correlations.csv")
```

---

## 🔗 Key Formulas

**Pearson Correlation:** $r = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$ ranges from -1 to 1

**Derivative:** $\frac{dX}{dt} = X[i+1] - X[i]$ (rate of change)

**DTW:** Minimum cost path accounting for temporal warping

**PCA:** Find directions of maximum variance in feature space

---

## 📞 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Blank/strange heatmap | Check CSV has `timestamp` and `value` columns |
| Memory error | Reduce `n_points` or use fewer files |
| Import error | `pip install -r requirements.txt` |
| DTW very slow | Install dtaidistance: `pip install dtaidistance` |
| NaN values in output | Check for empty/malformed rows in CSV |

---

## 📁 File Locations

```
plots/
├── pipeline.py                      ← Core class
├── timeseries_comparison.ipynb      ← Interactive notebook ✨
├── compare_timeseries.py            ← Batch processing
├── examples.py                      ← Code examples
├── COMPLETE_GUIDE.md                ← Full documentation
├── TIMESERIES_COMPARISON.md         ← Technical details
└── results/                         ← Output images
    ├── 00_aligned_time_series.png
    ├── 01_correlation_heatmap.png
    ├── 02_derivative_correlation_heatmap.png
    ├── 03_dtw_distance_matrix.png
    └── 04_pca_scatter_plot.png
```

---

## 🎓 Learn More

- **Local:** See `COMPLETE_GUIDE.md` for comprehensive documentation
- **Methods:** See `TIMESERIES_COMPARISON.md` for technical details
- **Examples:** See `examples.py` for code templates
- **Interactive:** Open `timeseries_comparison.ipynb` in Jupyter

---

**Last Updated:** February 11, 2026
**Status:** Ready to use ✓

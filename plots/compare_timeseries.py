#!/usr/bin/env python3
"""
Example script to compare multiple time series using correlation, derivatives, DTW, and PCA.
"""

from pathlib import Path
from pipeline import TimeSeriesComparator


def main():
    """
    Compare time series from vllm_datasets directory.
    
    This example compares all CSV files in the datasets folder,
    or you can specify specific metrics to compare different workloads.
    """
    
    # Directory containing CSV files
    dataset_dir = Path("/Users/anajafizadeh/projects/sunyibm/vllm-study/timeseries-models/vllm_datasets")
    
    # Option 1: Compare all CSV files
    print("=" * 80)
    print("OPTION 1: Comparing all available metrics")
    print("=" * 80)
    
    csv_files = sorted(list(dataset_dir.glob("*.csv")))
    
    if csv_files:
        # Limit to first 10 for faster processing (can adjust as needed)
        csv_files = csv_files[:10]
        
        print(f"\nFound {len(csv_files)} CSV files. Using first {len(csv_files)}...")
        for f in csv_files:
            print(f"  - {f.name}")
        
        comparator = TimeSeriesComparator(
            csv_files=[str(f) for f in csv_files],
            labels=[f.stem for f in csv_files]
        )
        
        # Plot aligned time series first (for verification)
        print("\nPlotting original aligned time series...")
        comparator.plot_time_series(
            filename="/Users/anajafizadeh/projects/sunyibm/vllm-study/plots/00_aligned_time_series.png"
        )
        
        # Generate all comparison visualizations
        print("\nGenerating comparison visualizations...")
        output_dir = "/Users/anajafizadeh/projects/sunyibm/vllm-study/plots"
        comparator.plot_all_comparisons(output_dir=output_dir)
    else:
        print("No CSV files found!")
    
    # Option 2: Compare specific metrics (same metric type, different workloads)
    print("\n" + "=" * 80)
    print("OPTION 2: Example - comparing same metric under different workloads")
    print("=" * 80)
    print("""
To compare the same metric (e.g., CPU_Usage) under different workloads/conditions:

    comparator = TimeSeriesComparator(
        csv_files=[
            "path/to/metric_workload1.csv",
            "path/to/metric_workload2.csv",
            "path/to/metric_workload3.csv",
        ],
        labels=["Workload 1", "Workload 2", "Workload 3"]
    )
    
    comparator.plot_all_comparisons(output_dir="./results")

This will generate:
    1. 01_correlation_heatmap.png - Shows how similar the workloads are
    2. 02_derivative_correlation_heatmap.png - Shows how similar their changes are
    3. 03_dtw_distance_matrix.png - Shows temporal alignment differences
    4. 04_pca_scatter_plot.png - Shows feature-based clustering of workloads
    5. 00_aligned_time_series.png - Visualizes the aligned raw time series
    """)


if __name__ == "__main__":
    main()

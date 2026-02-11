#!/usr/bin/env python3
"""
Quick Start Guide: Compare Time Series from Different Workloads

This script demonstrates how to use TimeSeriesComparator to compare
time series data from different workloads or experimental conditions.
"""

from pathlib import Path
from pipeline import TimeSeriesComparator


def example_1_compare_all_metrics():
    """
    Example 1: Compare ALL available metrics from the dataset.
    Useful for understanding which metrics behave similarly.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Compare All Metrics")
    print("="*80)
    
    dataset_dir = Path("/Users/anajafizadeh/projects/sunyibm/vllm-study/timeseries-models/vllm_datasets")
    csv_files = sorted(list(dataset_dir.glob("*.csv")))
    
    print(f"\nFound {len(csv_files)} CSV files:")
    for i, f in enumerate(csv_files[:5], 1):
        print(f"  {i}. {f.name}")
    if len(csv_files) > 5:
        print(f"  ... and {len(csv_files) - 5} more")
    
    # For faster processing, use a subset
    csv_files = csv_files[:8]
    
    try:
        comparator = TimeSeriesComparator(
            csv_files=[str(f) for f in csv_files],
            labels=[f.stem for f in csv_files]
        )
        
        print("\n✓ Loaded and aligned all time series")
        print(f"  Common time grid: {len(comparator.time_grid)} points")
        print(f"  Time range: {comparator.time_grid[0]:.1f}s to {comparator.time_grid[-1]:.1f}s")
        
        # Generate visualizations
        output_dir = Path("/Users/anajafizadeh/projects/sunyibm/vllm-study/plots/results_all_metrics")
        output_dir.mkdir(exist_ok=True)
        
        comparator.plot_all_comparisons(output_dir=str(output_dir))
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_2_compare_same_metric_different_points():
    """
    Example 2: If you have multiple measurements of the SAME metric 
    (e.g., CPU usage at different times or under different conditions),
    compare them to see how the metric behaves.
    
    Since we have one snapshot, we'll use related CPU/Memory metrics.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Compare Related Metrics (Resource Usage)")
    print("="*80)
    
    dataset_dir = Path("/Users/anajafizadeh/projects/sunyibm/vllm-study/timeseries-models/vllm_datasets")
    
    # Select metrics related to system resources
    resource_metrics = [
        "CPU_Usage-data-2025-10-02_00_22_34.csv",
        "Memory_Usage-data-2025-10-02_00_22_44.csv",
        "GPU_Utilization-data-2025-10-02_00_24_51.csv",
        "GPU_Memory_Utilization-data-2025-10-02_00_25_08.csv",
    ]
    
    csv_files = [dataset_dir / m for m in resource_metrics if (dataset_dir / m).exists()]
    
    if len(csv_files) < 2:
        print("✗ Not enough resource metrics available")
        return
    
    print(f"\nComparing {len(csv_files)} resource metrics:")
    for f in csv_files:
        print(f"  - {f.stem}")
    
    try:
        comparator = TimeSeriesComparator(
            csv_files=[str(f) for f in csv_files],
            labels=[f.stem for f in csv_files]
        )
        
        print("\n✓ Loaded and aligned resource metrics")
        
        output_dir = Path("/Users/anajafizadeh/projects/sunyibm/vllm-study/plots/results_resources")
        output_dir.mkdir(exist_ok=True)
        
        comparator.plot_all_comparisons(output_dir=str(output_dir))
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_3_compare_performance_metrics():
    """
    Example 3: Compare performance-related metrics.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Compare Performance Metrics")
    print("="*80)
    
    dataset_dir = Path("/Users/anajafizadeh/projects/sunyibm/vllm-study/timeseries-models/vllm_datasets")
    
    performance_metrics = [
        "Request_Inference_Time_Rate-data-2025-10-02_00_25_49.csv",
        "Time_Per_Output_Token_Sum-data-2025-10-02_00_25_54.csv",
        "Time_To_First_Token_Sum-data-2025-10-02_00_25_43.csv",
        "Token_Generations_Rate-data-2025-10-02_00_25_29.csv",
    ]
    
    csv_files = [dataset_dir / m for m in performance_metrics if (dataset_dir / m).exists()]
    
    if len(csv_files) < 2:
        print("✗ Not enough performance metrics available")
        return
    
    print(f"\nComparing {len(csv_files)} performance metrics:")
    for f in csv_files:
        print(f"  - {f.stem}")
    
    try:
        comparator = TimeSeriesComparator(
            csv_files=[str(f) for f in csv_files],
            labels=[f.stem for f in csv_files]
        )
        
        print("\n✓ Loaded and aligned performance metrics")
        
        output_dir = Path("/Users/anajafizadeh/projects/sunyibm/vllm-study/plots/results_performance")
        output_dir.mkdir(exist_ok=True)
        
        comparator.plot_all_comparisons(output_dir=str(output_dir))
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_4_custom_comparison():
    """
    Example 4: Template for custom comparison.
    Modify this to compare specific CSV files of your choice.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Comparison Template")
    print("="*80)
    
    print("""
To compare specific CSV files:

    from pipeline import TimeSeriesComparator
    
    csv_files = [
        "/path/to/experiment_1.csv",
        "/path/to/experiment_2.csv",
        "/path/to/experiment_3.csv",
    ]
    
    labels = ["Experiment 1", "Experiment 2", "Experiment 3"]
    
    comparator = TimeSeriesComparator(csv_files, labels=labels)
    comparator.plot_all_comparisons(output_dir="./my_results")
    
Each output file shows different aspects of similarity:
    - Correlation Heatmap: Overall similarity
    - Derivative Correlation: Rate-of-change similarity
    - DTW Distance: Temporal pattern similarity
    - PCA Plot: Feature-based clustering
    """)


def main():
    """Run all examples."""
    
    print("\n" + "="*80)
    print("TIME SERIES COMPARISON EXAMPLES")
    print("="*80)
    print("""
These examples show how to compare time series data from different workloads 
using 4 complementary visualization methods:

1. Correlation Heatmap (Normalized)
   - Shows which series are correlated
   - Values: -1 (opposite) to 1 (identical)

2. Derivative Correlation Heatmap
   - Shows which series have similar rate-of-change patterns
   - Useful for understanding behavioral similarities

3. DTW Distance Matrix
   - Shows temporal pattern distances
   - Accounts for timing differences and scales
   
4. PCA Scatter Plot (Feature-Based)
   - Shows which series cluster together based on 12 statistical features
   - Reveals high-level behavioral groupings
    """)
    
    # Run examples
    try:
        example_1_compare_all_metrics()
    except Exception as e:
        print(f"Example 1 error: {e}")
    
    try:
        example_2_compare_same_metric_different_points()
    except Exception as e:
        print(f"Example 2 error: {e}")
    
    try:
        example_3_compare_performance_metrics()
    except Exception as e:
        print(f"Example 3 error: {e}")
    
    example_4_custom_comparison()
    
    print("\n" + "="*80)
    print("GUIDE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

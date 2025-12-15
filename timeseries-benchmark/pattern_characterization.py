"""
Pattern Characterization Analysis
Automatically classifies time series patterns and identifies model strengths
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.tsa.seasonal import STL
from scipy import stats as sp_stats
from scipy.signal import periodogram


def load_dataset_traces(run_dir):
    """Load original dataset values (not residuals)"""
    run_dir = Path(run_dir)
    datasets = {}
    
    for csv_file in (run_dir / "datasets").glob("*.csv"):
        name = csv_file.stem
        df = pd.read_csv(csv_file)
        datasets[name] = df
    
    return datasets


def characterize_timeseries(y, name="series", fs=1.0):
    """
    Extract comprehensive pattern features from time series
    
    Returns:
        dict with pattern characteristics
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]  # Remove NaN/inf
    
    if len(y) < 50:
        return None
    
    features = {'name': name}
    
    # 1. Basic statistics
    features['length'] = len(y)
    features['mean'] = float(np.mean(y))
    features['std'] = float(np.std(y))
    features['cv'] = features['std'] / (abs(features['mean']) + 1e-8)  # Coefficient of variation
    
    # 2. Trend strength (linear regression slope)
    x = np.arange(len(y))
    slope, intercept, r_value, _, _ = sp_stats.linregress(x, y)
    features['trend_slope'] = float(slope)
    features['trend_strength'] = float(r_value ** 2)  # R²
    
    # 3. Volatility metrics
    diff = np.diff(y)
    features['volatility'] = float(np.std(diff))
    features['volatility_ratio'] = features['volatility'] / (features['std'] + 1e-8)
    
    # 4. Stationarity (Augmented Dickey-Fuller conceptually - use simple proxy)
    # Ratio of short-term to long-term variance
    window_short = min(50, len(y) // 10)
    window_long = min(200, len(y) // 3)
    
    var_short = np.mean([np.var(y[i:i+window_short]) 
                        for i in range(0, len(y)-window_short, window_short)])
    var_long = np.var(y)
    features['stationarity_proxy'] = float(var_short / (var_long + 1e-8))
    
    # 5. Seasonality detection via STL decomposition
    try:
        # Auto-detect period (naive: use periodogram peak)
        freqs, power = periodogram(y, fs=fs)
        if len(power) > 10:
            peak_idx = np.argmax(power[1:]) + 1  # Ignore DC component
            period = int(fs / freqs[peak_idx])
            period = np.clip(period, 7, len(y) // 3)  # Reasonable bounds
        else:
            period = 48  # Default daily with 30min sampling
        
        # STL decomposition
        stl = STL(y, seasonal=period, robust=True)
        result = stl.fit()
        
        # Seasonality strength: 1 - Var(residual) / Var(seasonal + residual)
        var_seasonal = np.var(result.seasonal)
        var_resid = np.var(result.resid)
        features['seasonality_strength'] = float(max(0, 1 - var_resid / (var_seasonal + var_resid + 1e-8)))
        
        # Trend strength from STL
        var_trend = np.var(result.trend)
        features['trend_strength_stl'] = float(max(0, 1 - var_resid / (var_trend + var_resid + 1e-8)))
        
        features['stl_period'] = int(period)
        features['stl_success'] = True
        
    except Exception as e:
        features['seasonality_strength'] = 0.0
        features['trend_strength_stl'] = features['trend_strength']
        features['stl_period'] = 0
        features['stl_success'] = False
    
    # 6. Autocorrelation
    # Lag-1 autocorrelation
    if len(y) > 1:
        features['autocorr_lag1'] = float(np.corrcoef(y[:-1], y[1:])[0, 1])
    else:
        features['autocorr_lag1'] = 0.0
    
    # 7. Irregularity / Complexity
    # Approximate entropy (simple version)
    features['range_ratio'] = float((np.max(y) - np.min(y)) / (features['std'] + 1e-8))
    
    # Number of zero-crossings (after normalization)
    y_norm = (y - np.mean(y)) / (np.std(y) + 1e-8)
    zero_crossings = np.sum(np.diff(np.sign(y_norm)) != 0)
    features['zero_crossings_rate'] = float(zero_crossings / len(y))
    
    # 8. Outlier presence (using IQR method)
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    outlier_mask = (y < q1 - 1.5*iqr) | (y > q3 + 1.5*iqr)
    features['outlier_fraction'] = float(np.mean(outlier_mask))
    
    # 9. Pattern classification (rule-based)
    features['pattern_type'] = classify_pattern(features)
    
    return features


def classify_pattern(features):
    """
    Classify time series into pattern categories based on features
    """
    # Strong seasonality
    if features['seasonality_strength'] > 0.6:
        if features['trend_strength_stl'] > 0.3:
            return "seasonal_with_trend"
        else:
            return "seasonal_stationary"
    
    # Strong trend, weak seasonality
    elif features['trend_strength_stl'] > 0.5:
        return "trending"
    
    # High volatility, low autocorrelation
    elif features['volatility_ratio'] > 1.5 and features['autocorr_lag1'] < 0.3:
        return "volatile_irregular"
    
    # High autocorrelation, low volatility
    elif features['autocorr_lag1'] > 0.8:
        return "smooth_persistent"
    
    # Low autocorrelation, stationary
    elif features['stationarity_proxy'] > 0.8:
        return "noise_like"
    
    else:
        return "mixed_pattern"


def analyze_patterns(run_dir, summary_df):
    """
    Main pattern analysis function
    """
    run_dir = Path(run_dir)
    output_dir = run_dir / "enhanced_analysis"
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("PATTERN CHARACTERIZATION ANALYSIS")
    print("="*80)
    
    # Load datasets
    datasets = load_dataset_traces(run_dir)
    print(f"\n📂 Loaded {len(datasets)} datasets")
    
    # Characterize each dataset
    pattern_features = []
    for name, df in datasets.items():
        print(f"  Analyzing {name}...")
        features = characterize_timeseries(df['value'].values, name=name)
        if features:
            pattern_features.append(features)
    
    pattern_df = pd.DataFrame(pattern_features)
    pattern_df.to_csv(output_dir / "pattern_features.csv", index=False)
    
    print("\n📊 PATTERN CLASSIFICATIONS:")
    print(pattern_df[['name', 'pattern_type', 'seasonality_strength', 
                      'trend_strength_stl', 'volatility_ratio']].to_string(index=False))
    
    # Merge with performance results
    merged = pattern_df.merge(
        summary_df[['Dataset', 'Model', 'F1', 'sMAPE']], 
        left_on='name', right_on='Dataset', how='inner'
    )
    
    # VISUALIZATION 1: Pattern features heatmap
    plot_pattern_heatmap(pattern_df, output_dir)
    
    # VISUALIZATION 2: Model performance by pattern type
    plot_performance_by_pattern(merged, output_dir)
    
    # VISUALIZATION 3: Feature importance for prediction
    analyze_feature_importance(merged, output_dir)
    
    # ANALYSIS: Which model for which pattern?
    recommendations = generate_model_recommendations(merged)
    
    return pattern_df, merged, recommendations


def plot_pattern_heatmap(pattern_df, output_dir):
    """
    Visualize pattern features as heatmap
    """
    # Select key features for visualization
    feature_cols = [
        'seasonality_strength', 'trend_strength_stl', 'volatility_ratio',
        'autocorr_lag1', 'stationarity_proxy', 'outlier_fraction',
        'zero_crossings_rate'
    ]
    
    # Normalize features to [0, 1] for visualization
    data = pattern_df[feature_cols].values
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(data_norm.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    ax.set_xticks(range(len(pattern_df)))
    ax.set_xticklabels(pattern_df['name'], rotation=45, ha='right')
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels([col.replace('_', ' ').title() for col in feature_cols])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Feature Value', rotation=270, labelpad=20)
    
    # Add values as text
    for i in range(len(pattern_df)):
        for j in range(len(feature_cols)):
            text = ax.text(i, j, f'{data_norm[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Time-Series Pattern Features\n(Normalized to [0,1] for comparison)', 
                 fontsize=13, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pattern_features_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved pattern_features_heatmap.png")


def plot_performance_by_pattern(merged_df, output_dir):
    """
    Show which models work best for each pattern type
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: F1 by pattern type
    ax = axes[0]
    
    pattern_types = merged_df['pattern_type'].unique()
    models = merged_df['Model'].unique()
    
    x = np.arange(len(pattern_types))
    width = 0.25
    colors_map = {'ARIMA': 'blue', 'Toto': 'green', 'GraniteTTM': 'red'}
    
    for i, model in enumerate(models):
        means = []
        for pt in pattern_types:
            subset = merged_df[(merged_df['pattern_type'] == pt) & (merged_df['Model'] == model)]
            means.append(subset['F1'].mean() if len(subset) > 0 else 0)
        
        ax.bar(x + i*width, means, width, label=model, 
               color=colors_map.get(model, 'gray'), alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Pattern Type', fontsize=11)
    ax.set_ylabel('Mean F1 Score', fontsize=11)
    ax.set_title('Model Performance by Pattern Type', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([pt.replace('_', '\n') for pt in pattern_types], fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: sMAPE by pattern type
    ax = axes[1]
    
    for i, model in enumerate(models):
        means = []
        for pt in pattern_types:
            subset = merged_df[(merged_df['pattern_type'] == pt) & (merged_df['Model'] == model)]
            means.append(subset['sMAPE'].mean() if len(subset) > 0 else 0)
        
        ax.bar(x + i*width, means, width, label=model,
               color=colors_map.get(model, 'gray'), alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Pattern Type', fontsize=11)
    ax.set_ylabel('Mean sMAPE (lower = better)', fontsize=11)
    ax.set_title('Forecast Accuracy by Pattern Type', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([pt.replace('_', '\n') for pt in pattern_types], fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_pattern_type.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved performance_by_pattern_type.png")


def analyze_feature_importance(merged_df, output_dir):
    """
    Correlate pattern features with model performance
    """
    feature_cols = [
        'seasonality_strength', 'trend_strength_stl', 'volatility_ratio',
        'autocorr_lag1', 'stationarity_proxy'
    ]
    
    # Calculate correlations for each model
    results = []
    
    for model in merged_df['Model'].unique():
        model_data = merged_df[merged_df['Model'] == model]
        
        for feature in feature_cols:
            corr_f1 = model_data[[feature, 'F1']].corr().iloc[0, 1]
            corr_smape = model_data[[feature, 'sMAPE']].corr().iloc[0, 1]
            
            results.append({
                'Model': model,
                'Feature': feature,
                'Corr_F1': corr_f1,
                'Corr_sMAPE': corr_smape
            })
    
    corr_df = pd.DataFrame(results)
    corr_df.to_csv(output_dir / 'feature_correlations.csv', index=False)
    
    # Visualization: Heatmap of correlations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # F1 correlations
    ax = axes[0]
    pivot_f1 = corr_df.pivot(index='Feature', columns='Model', values='Corr_F1')
    sns.heatmap(pivot_f1, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation with F1 Score\n(Positive = feature helps detection)', 
                 fontsize=11)
    ax.set_xlabel('')
    
    # sMAPE correlations
    ax = axes[1]
    pivot_smape = corr_df.pivot(index='Feature', columns='Model', values='Corr_sMAPE')
    sns.heatmap(pivot_smape, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation with sMAPE\n(Negative = feature helps forecasting)', 
                 fontsize=11)
    ax.set_xlabel('')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved feature_importance.png")
    
    # Print insights
    print("\n🔍 KEY FEATURE INSIGHTS:")
    
    # Find strongest positive correlations with F1
    top_f1 = corr_df.nlargest(3, 'Corr_F1')
    print("\n  Top features positively correlated with F1:")
    for _, row in top_f1.iterrows():
        print(f"    {row['Model']:12s} + {row['Feature']:25s} = {row['Corr_F1']:+.3f}")
    
    # Find strongest negative correlations with sMAPE
    top_smape = corr_df.nsmallest(3, 'Corr_sMAPE')
    print("\n  Top features negatively correlated with sMAPE (good for forecasting):")
    for _, row in top_smape.iterrows():
        print(f"    {row['Model']:12s} + {row['Feature']:25s} = {row['Corr_sMAPE']:+.3f}")
    
    return corr_df


def generate_model_recommendations(merged_df):
    """
    Create model selection recommendations based on pattern analysis
    """
    print("\n" + "="*80)
    print("MODEL SELECTION RECOMMENDATIONS")
    print("="*80)
    
    recommendations = {}
    
    for pattern_type in merged_df['pattern_type'].unique():
        pattern_data = merged_df[merged_df['pattern_type'] == pattern_type]
        
        # Find best model for F1
        best_f1 = pattern_data.groupby('Model')['F1'].mean().sort_values(ascending=False)
        winner_f1 = best_f1.index[0]
        
        # Find best model for sMAPE (lower is better)
        best_smape = pattern_data.groupby('Model')['sMAPE'].mean().sort_values()
        winner_smape = best_smape.index[0]
        
        recommendations[pattern_type] = {
            'best_detection': winner_f1,
            'f1_score': best_f1.iloc[0],
            'best_forecast': winner_smape,
            'smape': best_smape.iloc[0],
            'n_datasets': len(pattern_data['name'].unique())
        }
        
        print(f"\n📌 {pattern_type.upper().replace('_', ' ')}")
        print(f"   Datasets: {len(pattern_data['name'].unique())}")
        print(f"   Best Detection:  {winner_f1:12s} (F1 = {best_f1.iloc[0]:.3f})")
        print(f"   Best Forecast:   {winner_smape:12s} (sMAPE = {best_smape.iloc[0]:.3f})")
        
        if winner_f1 == winner_smape:
            print(f"   → Recommendation: Use {winner_f1} (wins both metrics)")
        else:
            print(f"   → Trade-off: {winner_f1} for detection, {winner_smape} for forecasting")
    
    # Save recommendations
    rec_df = pd.DataFrame.from_dict(recommendations, orient='index').reset_index()
    rec_df.columns = ['pattern_type', 'best_detection', 'f1_score', 
                     'best_forecast', 'smape', 'n_datasets']
    
    return recommendations


# ============================================================
# MAIN EXECUTION
# ============================================================

def run_pattern_analysis(run_dir):
    """
    Run complete pattern characterization analysis
    """
    run_dir = Path(run_dir)
    
    # Load existing results
    summary = pd.read_csv(run_dir / "exec_summary.csv")
    
    # Run pattern analysis
    pattern_df, merged_df, recommendations = analyze_patterns(run_dir, summary)
    
    print("\n" + "="*80)
    print("✅ PATTERN ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  - pattern_features.csv")
    print(f"  - pattern_features_heatmap.png")
    print(f"  - performance_by_pattern_type.png")
    print(f"  - feature_importance.png")
    print(f"  - feature_correlations.csv")
    
    return pattern_df, merged_df, recommendations


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pattern_characterization.py <path_to_TSB_run_directory>")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    run_pattern_analysis(run_dir)

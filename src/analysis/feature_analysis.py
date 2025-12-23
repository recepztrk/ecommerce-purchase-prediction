"""
Phase 1: Comprehensive Feature Analysis

Detailed analysis of all 24 features:
- Profiling (distribution, stats, outliers)
- Target correlation
- Feature importance (from v3.0 model)
- Feature interactions (multicollinearity)
- Noise detection
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from datetime import datetime

import lightgbm as lgb
from sklearn.metrics import mutual_info_score
from scipy.stats import spearmanr

print("="*80)
print("PHASE 1: COMPREHENSIVE FEATURE ANALYSIS")
print("="*80)
print(f"Start: {datetime.now().strftime('%H:%M:%S')}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n1. Loading data...")

train_df = pd.read_parquet('data/v3/train_sessions_v3.parquet')
val_df = pd.read_parquet('data/v3/val_sessions_v3.parquet')
test_df = pd.read_parquet('data/v3/test_sessions_v3.parquet')

# Combine for analysis
all_data = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)

print(f"  Total samples: {len(all_data):,}")
print(f"  Target distribution: {all_data['target'].value_counts().to_dict()}")

# Get feature list
exclude_cols = ['user_session', 'user_id', 'session_start', 'session_end', 'target']
features = [c for c in all_data.columns if c not in exclude_cols]

print(f"  Features to analyze: {len(features)}")

# ============================================================================
# STEP 2: FEATURE PROFILING
# ============================================================================
print("\n2. Feature Profiling...")

profiling_results = []

for feat in features:
    data = all_data[feat]
    
    profile = {
        'feature': feat,
        'dtype': str(data.dtype),
        'missing': data.isnull().sum(),
        'missing_pct': data.isnull().sum() / len(data) * 100,
        'unique': data.nunique(),
        'mean': data.mean() if np.issubdtype(data.dtype, np.number) else None,
        'std': data.std() if np.issubdtype(data.dtype, np.number) else None,
        'min': data.min() if np.issubdtype(data.dtype, np.number) else None,
        'max': data.max() if np.issubdtype(data.dtype, np.number) else None,
        'median': data.median() if np.issubdtype(data.dtype, np.number) else None,
        'skew': data.skew() if np.issubdtype(data.dtype, np.number) else None,
        'zeros': (data == 0).sum() if np.issubdtype(data.dtype, np.number) else None,
        'zeros_pct': (data == 0).sum() / len(data) * 100 if np.issubdtype(data.dtype, np.number) else None,
    }
    
    profiling_results.append(profile)

profiling_df = pd.DataFrame(profiling_results)

print("\nProfiling Summary:")
print(f"  Features with >5% missing: {(profiling_df['missing_pct'] > 5).sum()}")
print(f"  Features with >50% zeros: {(profiling_df['zeros_pct'] > 50).sum()}")
print(f"  High skew features (|skew|>2): {(profiling_df['skew'].abs() > 2).sum()}")

# ============================================================================
# STEP 3: TARGET CORRELATION ANALYSIS
# ============================================================================
print("\n3. Target Correlation Analysis...")

correlation_results = []

for feat in features:
    # Pearson correlation
    pearson_corr = all_data[feat].corr(all_data['target'])
    
    # Spearman correlation (rank-based, catches non-linear)
    spearman_corr, _ = spearmanr(all_data[feat], all_data['target'])
    
    # Mutual information (non-linear dependency)
    mi = mutual_info_score(
        all_data['target'], 
        pd.qcut(all_data[feat], q=10, duplicates='drop')
    )
    
    correlation_results.append({
        'feature': feat,
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'mutual_info': mi,
    })

correlation_df = pd.DataFrame(correlation_results).sort_values('pearson', key=abs, ascending=False)

print("\nTop 10 Correlated Features:")
for _, row in correlation_df.head(10).iterrows():
    print(f"  {row['feature']:30s} Pearson: {row['pearson']:+.4f}  Spearman: {row['spearman']:+.4f}  MI: {row['mutual_info']:.4f}")

print("\nWeakest Correlated Features:")
for _, row in correlation_df.tail(5).iterrows():
    print(f"  {row['feature']:30s} Pearson: {row['pearson']:+.4f}  Spearman: {row['spearman']:+.4f}  MI: {row['mutual_info']:.4f}")

# ============================================================================
# STEP 4: FEATURE IMPORTANCE (from v3.0 model)
# ============================================================================
print("\n4. Feature Importance Analysis...")

try:
    model = lgb.Booster(model_file='models/lightgbm_v3.txt')
    
    importances = model.feature_importance()
    feature_names = model.feature_name()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'importance_pct': importances / importances.sum() * 100
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:8.0f} ({row['importance_pct']:.2f}%)")
    
    print("\nBottom 10 Important Features:")
    for _, row in importance_df.tail(10).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:8.0f} ({row['importance_pct']:.2f}%)")
    
    # Noise features (<1% importance)
    noise_features = importance_df[importance_df['importance_pct'] < 1.0]
    print(f"\nNoise Features (importance <1%): {len(noise_features)}")
    for feat in noise_features['feature']:
        print(f"  - {feat}")
    
except Exception as e:
    print(f"  Could not load v3.0 model: {e}")
    importance_df = None

# ============================================================================
# STEP 5: FEATURE INTERACTION ANALYSIS
# ============================================================================
print("\n5. Feature Interaction Analysis...")

# Correlation matrix
numeric_features = all_data[features].select_dtypes(include=[np.number]).columns
corr_matrix = all_data[numeric_features].corr()

# Find high correlation pairs
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.85:  # High correlation threshold
            high_corr_pairs.append({
                'feature1': corr_matrix.columns[i],
                'feature2': corr_matrix.columns[j],
                'correlation': corr_val
            })

high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', key=abs, ascending=False)

print(f"\nHigh Correlation Pairs (|corr| > 0.85): {len(high_corr_df)}")
for _, row in high_corr_df.iterrows():
    print(f"  {row['feature1']:25s} ↔ {row['feature2']:25s} ({row['correlation']:+.3f})")

# ============================================================================
# STEP 6: GENERATE VISUALIZATIONS
# ============================================================================
print("\n6. Generating Visualizations...")

# 6.1 Correlation Heatmap
plt.figure(figsize=(16, 14))
sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/feature_analysis/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Correlation heatmap saved")

# 6.2 Feature Importance
if importance_df is not None:
    plt.figure(figsize=(12, 10))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('reports/feature_analysis/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Feature importance plot saved")

# 6.3 Target Correlation
plt.figure(figsize=(12, 10))
top_corr = correlation_df.head(15)
plt.barh(range(len(top_corr)), top_corr['pearson'])
plt.yticks(range(len(top_corr)), top_corr['feature'])
plt.xlabel('Pearson Correlation with Target', fontsize=12)
plt.title('Top 15 Features by Target Correlation', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('reports/feature_analysis/target_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Target correlation plot saved")

# ============================================================================
# STEP 7: GENERATE REPORT
# ============================================================================
print("\n7. Generating Analysis Report...")

report = f"""# Feature Analysis Report - v3.0

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Dataset:** {len(all_data):,} sessions  
**Features Analyzed:** {len(features)}

---

## Executive Summary

### Key Findings

**Strong Features (Keep):**
{chr(10).join([f"- **{row['feature']}** (Corr: {row['pearson']:+.3f}, Importance: {importance_df[importance_df['feature']==row['feature']]['importance_pct'].iloc[0]:.1f}%)" for _, row in correlation_df.head(5).iterrows() if importance_df is not None])}

**Weak Features (Potential Removal):**
{chr(10).join([f"- **{row['feature']}** (Importance: {row['importance_pct']:.2f}%)" for _, row in (importance_df.tail(5).iterrows() if importance_df is not None else [])])}

**Redundant Pairs:**
{chr(10).join([f"- {row['feature1']} ↔ {row['feature2']} ({row['correlation']:+.3f})" for _, row in high_corr_df.head(5).iterrows()])}

---

## Detailed Analysis

### 1. Feature Profiling

{profiling_df.to_markdown(index=False)}

### 2. Target Correlation

{correlation_df.to_markdown(index=False)}

### 3. Feature Importance

{importance_df.to_markdown(index=False) if importance_df is not None else "Model not available"}

### 4. High Correlation Pairs

{high_corr_df.to_markdown(index=False) if len(high_corr_df) > 0 else "No high correlation pairs found"}

---

## Recommendations

### Remove (Noise Features)
{chr(10).join([f"- `{feat}`" for feat in (noise_features['feature'] if importance_df is not None and len(noise_features) > 0 else [])])}

### Remove (Redundant Features)  
{chr(10).join([f"- `{row['feature2']}` (redundant with {row['feature1']})" for _, row in high_corr_df.head(5).iterrows()])}

### Investigate Further
- Features with high skew (potential transformation needed)
- Features with >50% zeros (potential indicator variables)

---

## Visualizations

![Correlation Heatmap](feature_analysis/correlation_heatmap.png)

![Feature Importance](feature_analysis/feature_importance.png)

![Target Correlation](feature_analysis/target_correlation.png)

---

**Next Steps:**
1. Remove noise and redundant features
2. Engineer new interaction features
3. Create clean dataset
4. Train optimized model
"""

with open('reports/feature_analysis_report.md', 'w') as f:
    f.write(report)

print("  ✓ Report saved to reports/feature_analysis_report.md")

# Save analysis results
analysis_results = {
    'profiling': profiling_df,
    'correlation': correlation_df,
    'importance': importance_df,
    'high_corr_pairs': high_corr_df,
    'timestamp': datetime.now().isoformat()
}

with open('reports/feature_analysis/analysis_results.pkl', 'wb') as f:
    pickle.dump(analysis_results, f)

print("  ✓ Analysis results saved")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PHASE 1 COMPLETE: FEATURE ANALYSIS")
print("="*80)

print(f"\nFeature Quality Summary:")
print(f"  Total features: {len(features)}")
if importance_df is not None:
    print(f"  Noise features (<1% importance): {len(noise_features)}")
print(f"  Redundant pairs (corr >0.85): {len(high_corr_df)}")
print(f"  Strong features (corr >0.15): {(correlation_df['pearson'].abs() > 0.15).sum()}")

print(f"\nFiles Generated:")
print(f"  ✓ reports/feature_analysis_report.md")
print(f"  ✓ reports/feature_analysis/correlation_heatmap.png")
print(f"  ✓ reports/feature_analysis/feature_importance.png")
print(f"  ✓ reports/feature_analysis/target_correlation.png")
print(f"  ✓ reports/feature_analysis/analysis_results.pkl")

print(f"\nEnd: {datetime.now().strftime('%H:%M:%S')}")
print("="*80)

"""
Detailed Metrics Analysis for All Phase 3 Optimized Models

Calculate comprehensive metrics for:
- Mac: ExtraTrees, LightGBM, XGBoost
- Colab: LightGBM, XGBoost

Metrics: Train/Val/Test AUC, F1, Precision, Recall, Gap
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import pickle

print("="*80)
print("DETAILED METRICS ANALYSIS - ALL OPTIMIZED MODELS")
print("="*80)

# Load data
print("\nLoading data...")
train_df = pd.read_parquet('data/v3_final/train_sessions_final.parquet')
val_df = pd.read_parquet('data/v3_final/val_sessions_final.parquet')
test_df = pd.read_parquet('data/v3_final/test_sessions_final.parquet')

exclude_cols = ['user_session', 'user_id', 'session_start', 'session_end', 'target']
features = [c for c in train_df.columns if c not in exclude_cols]

X_train, y_train = train_df[features].values, train_df['target'].values
X_val, y_val = val_df[features].values, val_df['target'].values
X_test, y_test = test_df[features].values, test_df['target'].values

print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")

def calculate_metrics(model, model_name, source):
    """Calculate comprehensive metrics for a model"""
    print(f"\n{'-'*80}")
    print(f"{model_name} ({source})")
    print(f"{'-'*80}")
    
    # Predictions
    if 'lightgbm' in model_name.lower():
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
    else:  # ExtraTrees, XGBoost
        train_pred = model.predict_proba(X_train)[:, 1]
        val_pred = model.predict_proba(X_val)[:, 1]
        test_pred = model.predict_proba(X_test)[:, 1]
    
    # AUC scores
    train_auc = roc_auc_score(y_train, train_pred)
    val_auc = roc_auc_score(y_val, val_pred)
    test_auc = roc_auc_score(y_test, test_pred)
    
    # Convert to binary for classification metrics (threshold 0.5)
    train_pred_binary = (train_pred > 0.5).astype(int)
    val_pred_binary = (val_pred > 0.5).astype(int)
    test_pred_binary = (test_pred > 0.5).astype(int)
    
    # Classification metrics
    train_f1 = f1_score(y_train, train_pred_binary)
    val_f1 = f1_score(y_val, val_pred_binary)
    test_f1 = f1_score(y_test, test_pred_binary)
    
    train_precision = precision_score(y_train, train_pred_binary)
    val_precision = precision_score(y_val, val_pred_binary)
    test_precision = precision_score(y_test, test_pred_binary)
    
    train_recall = recall_score(y_train, train_pred_binary)
    val_recall = recall_score(y_val, val_pred_binary)
    test_recall = recall_score(y_test, test_pred_binary)
    
    # Gaps
    train_test_gap = abs(train_auc - test_auc) / train_auc * 100
    
    # Print results
    print(f"\n  AUC Scores:")
    print(f"    Train: {train_auc:.4f}")
    print(f"    Val:   {val_auc:.4f}")
    print(f"    Test:  {test_auc:.4f}")
    print(f"    Gap:   {train_test_gap:.2f}%")
    
    print(f"\n  F1 Score:")
    print(f"    Train: {train_f1:.4f}")
    print(f"    Val:   {val_f1:.4f}")
    print(f"    Test:  {test_f1:.4f}")
    
    print(f"\n  Precision:")
    print(f"    Train: {train_precision:.4f}")
    print(f"    Val:   {val_precision:.4f}")
    print(f"    Test:  {test_precision:.4f}")
    
    print(f"\n  Recall:")
    print(f"    Train: {train_recall:.4f}")
    print(f"    Val:   {val_recall:.4f}")
    print(f"    Test:  {test_recall:.4f}")
    
    # Test confusion matrix
    cm = confusion_matrix(y_test, test_pred_binary)
    print(f"\n  Test Confusion Matrix:")
    print(f"    TN: {cm[0,0]:>8,}  FP: {cm[0,1]:>8,}")
    print(f"    FN: {cm[1,0]:>8,}  TP: {cm[1,1]:>8,}")
    
    return {
        'model': model_name,
        'source': source,
        'train_auc': train_auc,
        'val_auc': val_auc,
        'test_auc': test_auc,
        'train_test_gap': train_test_gap,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'test_f1': test_f1,
        'train_precision': train_precision,
        'val_precision': val_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'val_recall': val_recall,
        'test_recall': test_recall,
        'test_tn': int(cm[0,0]),
        'test_fp': int(cm[0,1]),
        'test_fn': int(cm[1,0]),
        'test_tp': int(cm[1,1])
    }

results = []

# ============================================================================
# 1. Mac ExtraTrees
# ============================================================================
print("\n" + "="*80)
print("LOADING MAC MODELS")
print("="*80)

model_et = joblib.load('models/best_extratrees.pkl')
results.append(calculate_metrics(model_et, 'ExtraTrees', 'Mac'))

# ============================================================================
# 2. Mac LightGBM
# ============================================================================
model_lgb_mac = lgb.Booster(model_file='models/best_lightgbm.txt')
results.append(calculate_metrics(model_lgb_mac, 'LightGBM', 'Mac'))

# ============================================================================
# 3. Mac XGBoost
# ============================================================================
model_xgb_mac = joblib.load('models/best_xgboost.pkl')
results.append(calculate_metrics(model_xgb_mac, 'XGBoost', 'Mac'))

# ============================================================================
# 4. Colab LightGBM
# ============================================================================
print("\n" + "="*80)
print("LOADING COLAB MODELS")
print("="*80)

model_lgb_colab = lgb.Booster(model_file='models/best_lightgbm_colab.txt')
results.append(calculate_metrics(model_lgb_colab, 'LightGBM', 'Colab'))

# ============================================================================
# 5. Colab XGBoost
# ============================================================================
model_xgb_colab = joblib.load('models/best_xgboost_colab.pkl')
results.append(calculate_metrics(model_xgb_colab, 'XGBoost', 'Colab'))

# ============================================================================
# COMPARISON TABLE
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON")
print("="*80)

results_df = pd.DataFrame(results)
results_df['model_id'] = results_df['source'] + ' ' + results_df['model']

# Sort by test AUC
results_df = results_df.sort_values('test_auc', ascending=False)

print("\n📊 AUC SCORES:")
print(results_df[['model_id', 'train_auc', 'val_auc', 'test_auc', 'train_test_gap']].to_string(index=False))

print("\n📊 CLASSIFICATION METRICS (Test Set):")
print(results_df[['model_id', 'test_f1', 'test_precision', 'test_recall']].to_string(index=False))

print("\n📊 DETAILED COMPARISON:")
for i, row in results_df.iterrows():
    rank = list(results_df.index).index(i) + 1
    emoji = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][rank-1]
    print(f"\n{emoji} {rank}. {row['model_id']}")
    print(f"   Test AUC: {row['test_auc']:.4f}  Gap: {row['train_test_gap']:.2f}%")
    print(f"   F1: {row['test_f1']:.4f}  Precision: {row['test_precision']:.4f}  Recall: {row['test_recall']:.4f}")

# Save results
results_df.to_csv('reports/phase3_detailed_metrics.csv', index=False)
with open('reports/phase3_detailed_metrics.pkl', 'wb') as f:
    pickle.dump(results_df, f)

print("\n✓ Results saved to reports/phase3_detailed_metrics.csv")

# ============================================================================
# v3.0 BASELINE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("IMPROVEMENT vs v3.0 BASELINE")
print("="*80)

v3_baseline = {
    'test_auc': 0.7619,
    'test_f1': 0.69,
    'train_test_gap': 11.0
}

best_model = results_df.iloc[0]

print(f"\nv3.0 Baseline:")
print(f"  Test AUC: {v3_baseline['test_auc']:.4f}")
print(f"  F1 Score: {v3_baseline['test_f1']:.2f}")
print(f"  Gap:      {v3_baseline['train_test_gap']:.1f}%")

print(f"\nBest Model ({best_model['model_id']}):")
print(f"  Test AUC: {best_model['test_auc']:.4f}  (+{(best_model['test_auc'] - v3_baseline['test_auc']):.4f}, {(best_model['test_auc'] - v3_baseline['test_auc']) / v3_baseline['test_auc'] * 100:+.2f}%)")
print(f"  F1 Score: {best_model['test_f1']:.4f}  ({best_model['test_f1'] - v3_baseline['test_f1']:+.4f})")
print(f"  Gap:      {best_model['train_test_gap']:.2f}%  ({best_model['train_test_gap'] - v3_baseline['train_test_gap']:+.2f}%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

"""
Phase 4: Weighted Ensemble - Top 3 Models

Combine strengths:
- Mac ExtraTrees: Best AUC (0.7751)
- Colab XGBoost: Best Precision (0.61)
- Colab LightGBM: Best Recall (0.85) & F1 (0.68)

Strategy: Weighted voting with optimal weights found via grid search
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from itertools import product
import pickle

print("="*80)
print("PHASE 4: WEIGHTED ENSEMBLE - TOP 3 MODELS")
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

# ============================================================================
# Load Models
# ============================================================================
print("\n" + "="*80)
print("LOADING TOP 3 MODELS")
print("="*80)

print("\n1. Mac ExtraTrees (Best AUC: 0.7751)...")
model_et = joblib.load('models/best_extratrees.pkl')
print("   ✓ Loaded")

print("\n2. Colab XGBoost (Best Precision: 0.61)...")
model_xgb = joblib.load('models/best_xgboost_colab.pkl')
print("   ✓ Loaded")

print("\n3. Colab LightGBM (Best Recall: 0.85)...")
model_lgb = lgb.Booster(model_file='models/best_lightgbm_colab.txt')
print("   ✓ Loaded")

# ============================================================================
# Generate Predictions
# ============================================================================
print("\n" + "="*80)
print("GENERATING PREDICTIONS")
print("="*80)

print("\nGenerating predictions for all datasets...")

# Train predictions
print("  Train set...")
train_pred_et = model_et.predict_proba(X_train)[:, 1]
train_pred_xgb = model_xgb.predict_proba(X_train)[:, 1]
train_pred_lgb = model_lgb.predict(X_train)

# Validation predictions
print("  Validation set...")
val_pred_et = model_et.predict_proba(X_val)[:, 1]
val_pred_xgb = model_xgb.predict_proba(X_val)[:, 1]
val_pred_lgb = model_lgb.predict(X_val)

# Test predictions
print("  Test set...")
test_pred_et = model_et.predict_proba(X_test)[:, 1]
test_pred_xgb = model_xgb.predict_proba(X_test)[:, 1]
test_pred_lgb = model_lgb.predict(X_test)

print("✓ All predictions generated")

# ============================================================================
# Find Optimal Weights (Grid Search on Validation)
# ============================================================================
print("\n" + "="*80)
print("FINDING OPTIMAL WEIGHTS (Grid Search)")
print("="*80)

print("\nSearching weight combinations...")
print("  Weights: ExtraTrees, XGBoost, LightGBM")
print("  Range: 0.0 to 1.0 (step 0.1)")
print("  Constraint: Sum = 1.0")

best_auc = 0
best_weights = None
best_f1 = 0

# Grid search over weight combinations
weight_range = np.arange(0, 1.1, 0.1)
total_combinations = 0
tested = 0

for w1 in weight_range:
    for w2 in weight_range:
        w3 = 1.0 - w1 - w2
        if w3 < 0 or w3 > 1.0:
            continue
        
        total_combinations += 1
        
        # Ensemble prediction on validation
        val_ensemble = w1 * val_pred_et + w2 * val_pred_xgb + w3 * val_pred_lgb
        
        # Calculate metrics
        auc = roc_auc_score(y_val, val_ensemble)
        f1 = f1_score(y_val, (val_ensemble > 0.5).astype(int))
        
        tested += 1
        
        # Update best
        if auc > best_auc:
            best_auc = auc
            best_weights = (w1, w2, w3)
            best_f1 = f1

print(f"\n  Total combinations tested: {tested}")
print(f"\n  Best weights found:")
print(f"    ExtraTrees: {best_weights[0]:.1f}")
print(f"    XGBoost:    {best_weights[1]:.1f}")
print(f"    LightGBM:   {best_weights[2]:.1f}")
print(f"\n  Validation performance:")
print(f"    AUC: {best_auc:.4f}")
print(f"    F1:  {best_f1:.4f}")

# ============================================================================
# Test Ensemble with Optimal Weights
# ============================================================================
print("\n" + "="*80)
print("TESTING ENSEMBLE ON ALL DATASETS")
print("="*80)

w_et, w_xgb, w_lgb = best_weights

# Train ensemble
train_ensemble = w_et * train_pred_et + w_xgb * train_pred_xgb + w_lgb * train_pred_lgb
train_ensemble_binary = (train_ensemble > 0.5).astype(int)

# Val ensemble
val_ensemble = w_et * val_pred_et + w_xgb * val_pred_xgb + w_lgb * val_pred_lgb
val_ensemble_binary = (val_ensemble > 0.5).astype(int)

# Test ensemble
test_ensemble = w_et * test_pred_et + w_xgb * test_pred_xgb + w_lgb * test_pred_lgb
test_ensemble_binary = (test_ensemble > 0.5).astype(int)

# Calculate comprehensive metrics
def calculate_all_metrics(y_true, y_pred, y_pred_binary, dataset_name):
    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    print(f"\n{dataset_name} Set:")
    print(f"  AUC:       {auc:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN: {cm[0,0]:>8,}  FP: {cm[0,1]:>8,}")
    print(f"    FN: {cm[1,0]:>8,}  TP: {cm[1,1]:>8,}")
    
    return {
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'cm': cm
    }

train_metrics = calculate_all_metrics(y_train, train_ensemble, train_ensemble_binary, "Train")
val_metrics = calculate_all_metrics(y_val, val_ensemble, val_ensemble_binary, "Val")
test_metrics = calculate_all_metrics(y_test, test_ensemble, test_ensemble_binary, "Test")

# Calculate gap
train_test_gap = abs(train_metrics['auc'] - test_metrics['auc']) / train_metrics['auc'] * 100
print(f"\nTrain-Test Gap: {train_test_gap:.2f}%")

# ============================================================================
# Comparison with v3.0 and Best Single Model
# ============================================================================
print("\n" + "="*80)
print("PERFORMANCE COMPARISON")
print("="*80)

v3_baseline = {
    'test_auc': 0.7619,
    'test_f1': 0.69,
    'gap': 11.0
}

best_single = {
    'name': 'Mac ExtraTrees',
    'test_auc': 0.7751,
    'test_f1': 0.6702,
    'gap': 13.6
}

print(f"\nv3.0 Baseline:")
print(f"  Test AUC: {v3_baseline['test_auc']:.4f}")
print(f"  F1 Score: {v3_baseline['test_f1']:.2f}")
print(f"  Gap:      {v3_baseline['gap']:.1f}%")

print(f"\nBest Single Model ({best_single['name']}):")
print(f"  Test AUC: {best_single['test_auc']:.4f}  ({(best_single['test_auc'] - v3_baseline['test_auc']):.4f}, {(best_single['test_auc'] - v3_baseline['test_auc']) / v3_baseline['test_auc'] * 100:+.2f}%)")
print(f"  F1 Score: {best_single['test_f1']:.4f}  ({best_single['test_f1'] - v3_baseline['test_f1']:+.4f})")
print(f"  Gap:      {best_single['gap']:.1f}%  ({best_single['gap'] - v3_baseline['gap']:+.1f}%)")

print(f"\n🎯 ENSEMBLE (Weighted Top 3):")
print(f"  Test AUC: {test_metrics['auc']:.4f}  ({(test_metrics['auc'] - v3_baseline['test_auc']):.4f}, {(test_metrics['auc'] - v3_baseline['test_auc']) / v3_baseline['test_auc'] * 100:+.2f}%)")
print(f"  F1 Score: {test_metrics['f1']:.4f}  ({test_metrics['f1'] - v3_baseline['test_f1']:+.4f})")
print(f"  Gap:      {train_test_gap:.2f}%  ({train_test_gap - v3_baseline['gap']:+.2f}%)")

# Improvement vs best single
print(f"\nEnsemble vs Best Single Model:")
print(f"  AUC: {(test_metrics['auc'] - best_single['test_auc']):.4f} ({(test_metrics['auc'] - best_single['test_auc']) / best_single['test_auc'] * 100:+.2f}%)")
print(f"  F1:  {test_metrics['f1'] - best_single['test_f1']:+.4f}")

# ============================================================================
# Save Results
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save weights
weights_info = {
    'weights': {
        'extratrees': w_et,
        'xgboost': w_xgb,
        'lightgbm': w_lgb
    },
    'val_auc': best_auc,
    'val_f1': best_f1,
    'test_auc': test_metrics['auc'],
    'test_f1': test_metrics['f1'],
    'test_precision': test_metrics['precision'],
    'test_recall': test_metrics['recall']
}

with open('models/ensemble_phase4_weights.json', 'w') as f:
    import json
    json.dump(weights_info, f, indent=2)
print("✓ Weights saved: models/ensemble_phase4_weights.json")

# Save predictions
np.save('models/ensemble_phase4_test_pred.npy', test_ensemble)
print("✓ Predictions saved: models/ensemble_phase4_test_pred.npy")

# Save results
results = {
    'weights': weights_info,
    'train_metrics': train_metrics,
    'val_metrics': val_metrics,
    'test_metrics': test_metrics,
    'train_test_gap': train_test_gap
}

with open('reports/phase4_ensemble_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("✓ Results saved: reports/phase4_ensemble_results.pkl")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "="*80)
print("🎉 PHASE 4 COMPLETE: ENSEMBLE RESULTS")
print("="*80)

print(f"\nOptimal Weights:")
print(f"  ExtraTrees: {w_et:.1f} (Best AUC strength)")
print(f"  XGBoost:    {w_xgb:.1f} (Precision strength)")
print(f"  LightGBM:   {w_lgb:.1f} (Recall strength)")

print(f"\n📊 Final Performance:")
print(f"  Test AUC:       {test_metrics['auc']:.4f}")
print(f"  Test F1:        {test_metrics['f1']:.4f}")
print(f"  Test Precision: {test_metrics['precision']:.4f}")
print(f"  Test Recall:    {test_metrics['recall']:.4f}")
print(f"  Train-Test Gap: {train_test_gap:.2f}%")

print(f"\n🏆 vs v3.0 Baseline:")
print(f"  AUC improvement: {(test_metrics['auc'] - v3_baseline['test_auc']) / v3_baseline['test_auc'] * 100:+.2f}%")
print(f"  F1 improvement:  {test_metrics['f1'] - v3_baseline['test_f1']:+.4f}")

if test_metrics['auc'] > v3_baseline['test_auc'] and test_metrics['f1'] > v3_baseline['test_f1']:
    print("\n✅ SUCCESS: Ensemble beats v3.0 in BOTH AUC and F1!")
elif test_metrics['auc'] > v3_baseline['test_auc']:
    print("\n⚠️  PARTIAL: Ensemble beats v3.0 in AUC but not F1")
else:
    print("\n❌ WARNING: Ensemble doesn't beat v3.0 baseline")

print("\n" + "="*80)

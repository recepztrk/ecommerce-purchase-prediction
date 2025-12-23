"""
Phase 4b: Alternative Ensemble Methods

Method 1: Equal Weights (0.33, 0.33, 0.33)
Method 2: Stacking with Meta-Learner

Goal: Beat v3.0 baseline (AUC 0.7619, F1 0.69)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import pickle

print("="*80)
print("PHASE 4b: ALTERNATIVE ENSEMBLE METHODS")
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

print(f"  Data loaded: {X_train.shape}")

# Load models and predictions (reuse from previous run)
print("\nLoading models...")
model_et = joblib.load('models/best_extratrees.pkl')
model_xgb = joblib.load('models/best_xgboost_colab.pkl')
model_lgb = lgb.Booster(model_file='models/best_lightgbm_colab.txt')

print("\nGenerating predictions...")
# Validation predictions
val_pred_et = model_et.predict_proba(X_val)[:, 1]
val_pred_xgb = model_xgb.predict_proba(X_val)[:, 1]
val_pred_lgb = model_lgb.predict(X_val)

# Test predictions
test_pred_et = model_et.predict_proba(X_test)[:, 1]
test_pred_xgb = model_xgb.predict_proba(X_test)[:, 1]
test_pred_lgb = model_lgb.predict(X_test)

print("✓ Predictions ready")

# Helper function
def evaluate_ensemble(test_pred, name):
    test_pred_binary = (test_pred > 0.5).astype(int)
    
    auc = roc_auc_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred_binary)
    precision = precision_score(y_test, test_pred_binary)
    recall = recall_score(y_test, test_pred_binary)
    cm = confusion_matrix(y_test, test_pred_binary)
    
    print(f"\n{name}:")
    print(f"  Test AUC:       {auc:.4f}")
    print(f"  Test F1:        {f1:.4f}")
    print(f"  Test Precision: {precision:.4f}")
    print(f"  Test Recall:    {recall:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN: {cm[0,0]:>8,}  FP: {cm[0,1]:>8,}")
    print(f"    FN: {cm[1,0]:>8,}  TP: {cm[1,1]:>8,}")
    
    return {
        'name': name,
        'test_auc': auc,
        'test_f1': f1,
        'test_precision': precision,
        'test_recall': recall
    }

results = []

# ============================================================================
# METHOD 1: EQUAL WEIGHTS
# ============================================================================
print("\n" + "="*80)
print("METHOD 1: EQUAL WEIGHTS (0.33, 0.33, 0.33)")
print("="*80)

equal_test = (test_pred_et + test_pred_xgb + test_pred_lgb) / 3
results.append(evaluate_ensemble(equal_test, "Equal Weights Ensemble"))

# ============================================================================
# METHOD 2: STACKING WITH META-LEARNER
# ============================================================================
print("\n" + "="*80)
print("METHOD 2: STACKING (LogisticRegression Meta-Learner)")
print("="*80)

print("\nTraining meta-learner on validation set...")

# Stack validation predictions as features
val_stack = np.column_stack([val_pred_et, val_pred_xgb, val_pred_lgb])

# Train meta-learner
meta_learner = LogisticRegression(max_iter=1000, random_state=42)
meta_learner.fit(val_stack, y_val)

print(f"  Meta-learner coefficients:")
print(f"    ExtraTrees: {meta_learner.coef_[0][0]:.4f}")
print(f"    XGBoost:    {meta_learner.coef_[0][1]:.4f}")
print(f"    LightGBM:   {meta_learner.coef_[0][2]:.4f}")

# Predict on test
test_stack = np.column_stack([test_pred_et, test_pred_xgb, test_pred_lgb])
stacking_test = meta_learner.predict_proba(test_stack)[:, 1]

results.append(evaluate_ensemble(stacking_test, "Stacking Ensemble"))

# Save stacking model
joblib.dump(meta_learner, 'models/stacking_meta_learner.pkl')
print("\n✓ Meta-learner saved: models/stacking_meta_learner.pkl")

# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON")
print("="*80)

# Add baseline and best single
v3_baseline = {
    'name': 'v3.0 Baseline',
    'test_auc': 0.7619,
    'test_f1': 0.69,
    'test_precision': 0.65,
    'test_recall': 0.98
}

best_single = {
    'name': 'Mac ExtraTrees',
    'test_auc': 0.7751,
    'test_f1': 0.6702,
    'test_precision': 0.5942,
    'test_recall': 0.7686
}

all_results = [v3_baseline, best_single] + results
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('test_auc', ascending=False)

print("\n📊 FINAL RANKING (by Test AUC):")
print(results_df.to_string(index=False))

# Find best overall
print("\n" + "="*80)
print("BEST MODEL ANALYSIS")
print("="*80)

best_auc_model = results_df.iloc[0]
print(f"\n🏆 Best AUC: {best_auc_model['name']}")
print(f"   AUC: {best_auc_model['test_auc']:.4f}")
print(f"   F1:  {best_auc_model['test_f1']:.4f}")

# Check if any beat v3.0 in BOTH metrics
print("\n🎯 Models that beat v3.0 in BOTH AUC AND F1:")
winners = results_df[(results_df['test_auc'] > v3_baseline['test_auc']) & 
                     (results_df['test_f1'] > v3_baseline['test_f1'])]

if len(winners) > 0:
    print(winners[['name', 'test_auc', 'test_f1']].to_string(index=False))
    print("\n✅ SUCCESS! We have a winner!")
else:
    print("  None. ❌")
    print("\n📌 Recommendation: Use v3.0 baseline (best balanced)")

# Save results
results_df.to_csv('reports/phase4b_alternative_ensemble_results.csv', index=False)
with open('reports/phase4b_results.pkl', 'wb') as f:
    pickle.dump({
        'results': results_df,
        'equal_weights_pred': equal_test,
        'stacking_pred': stacking_test,
        'meta_learner_coef': meta_learner.coef_[0]
    }, f)

print("\n✓ Results saved")

# ============================================================================
# FINAL RECOMMENDATION
# ============================================================================
print("\n" + "="*80)
print("🎯 FINAL RECOMMENDATION")
print("="*80)

if len(winners) > 0:
    best = winners.iloc[0]
    print(f"\nUSE: {best['name']}")
    print(f"  Test AUC: {best['test_auc']:.4f} ({(best['test_auc'] - v3_baseline['test_auc']) / v3_baseline['test_auc'] * 100:+.2f}%)")
    print(f"  Test F1:  {best['test_f1']:.4f} ({best['test_f1'] - v3_baseline['test_f1']:+.4f})")
else:
    print(f"\nUSE: v3.0 Baseline")
    print(f"  Test AUC: {v3_baseline['test_auc']:.4f}")
    print(f"  Test F1:  {v3_baseline['test_f1']:.2f}")
    print("\nWhy? Most balanced (high F1, low gap)")

print("\n" + "="*80)
print("PHASE 4b COMPLETE")
print("="*80)

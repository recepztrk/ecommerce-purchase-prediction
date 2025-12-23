"""
Technique 1: K-Fold Cross-Validation

Goal: Improve model stability and reduce overfitting
Current: Single train/val split, 11% gap
Target: 5-fold CV, 7-8% gap, 0.76 → 0.77 AUC
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# Config
RANDOM_SEED = 42

print("="*80)
print("TECHNIQUE 1: K-FOLD CROSS-VALIDATION")
print("="*80)
print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
print(f"Goal: Stable model, reduce overfitting")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n1. Loading v3.0 data...")

train_df = pd.read_parquet('data/v3/train_sessions_v3.parquet')
val_df = pd.read_parquet('data/v3/val_sessions_v3.parquet')
test_df = pd.read_parquet('data/v3/test_sessions_v3.parquet')

print(f"  Train: {train_df.shape}")
print(f"  Val:   {val_df.shape}")
print(f"  Test:  {test_df.shape}")

# Combine train+val for K-Fold (standard practice)
full_train = pd.concat([train_df, val_df], axis=0, ignore_index=True)
print(f"  Combined train+val: {full_train.shape}")

# Prepare features
exclude_cols = ['user_session', 'user_id', 'session_start', 'session_end', 'target']
feature_cols = [c for c in full_train.columns if c not in exclude_cols]

X = full_train[feature_cols].values
y = full_train['target'].values

X_test = test_df[feature_cols].values
y_test = test_df['target'].values

print(f"\n  Features: {len(feature_cols)}")
print(f"  X shape: {X.shape}")
print(f"  Target distribution: {np.mean(y)*100:.2f}% positive")

# ============================================================================
# STEP 2: K-FOLD CROSS-VALIDATION SETUP
# ============================================================================
print("\n" + "="*80)
print("2. K-FOLD CV SETUP")
print("="*80)

N_FOLDS = 5
kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

print(f"  Number of folds: {N_FOLDS}")
print(f"  Stratified: Yes (maintains class balance)")

# Out-of-fold predictions storage
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))

fold_scores = []

# ============================================================================
# STEP 3: K-FOLD TRAINING
# ============================================================================
print("\n" + "="*80)
print("3. K-FOLD TRAINING")
print("="*80)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    print(f"\n{'='*80}")
    print(f"FOLD {fold + 1}/{N_FOLDS}")
    print(f"{'='*80}")
    
    # Split data
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    print(f"  Train: {len(X_tr):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    
    # ========================================================================
    # LightGBM
    # ========================================================================
    print(f"\n  Training LightGBM...")
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 9,
        'min_child_samples': 20,
        'verbose': -1,
        'random_state': RANDOM_SEED + fold,
    }
    
    lgb_train = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, feature_name=feature_cols)
    
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
        ]
    )
    
    # Out-of-fold predictions
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    
    # Test predictions (average across folds)
    test_preds_lgb += lgb_model.predict(X_test) / N_FOLDS
    
    # Fold evaluation
    fold_val_auc = roc_auc_score(y_val, oof_lgb[val_idx])
    print(f"    LightGBM Val AUC: {fold_val_auc:.4f}")
    
    # ========================================================================
    # XGBoost
    # ========================================================================
    print(f"\n  Training XGBoost...")
    
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 9,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 20,
        'random_state': RANDOM_SEED + fold,
        'tree_method': 'hist',
        'verbosity': 0,
    }
    
    xgb_train = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
    xgb_val = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    
    xgb_model = xgb.train(
        xgb_params,
        xgb_train,
        num_boost_round=2000,
        evals=[(xgb_train, 'train'), (xgb_val, 'val')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    
    # Out-of-fold predictions
    oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(X_val, feature_names=feature_cols))
    
    # Test predictions (average across folds)
    test_preds_xgb += xgb_model.predict(xgb.DMatrix(X_test, feature_names=feature_cols)) / N_FOLDS
    
    # Fold evaluation
    fold_val_auc_xgb = roc_auc_score(y_val, oof_xgb[val_idx])
    print(f"    XGBoost Val AUC: {fold_val_auc_xgb:.4f}")
    
    # Store fold scores
    fold_scores.append({
        'fold': fold + 1,
        'lgb_val_auc': fold_val_auc,
        'xgb_val_auc': fold_val_auc_xgb,
    })

# ============================================================================
# STEP 4: OUT-OF-FOLD EVALUATION
# ============================================================================
print("\n" + "="*80)
print("4. OUT-OF-FOLD EVALUATION")
print("="*80)

# Ensemble: Simple average
oof_ensemble = 0.5 * oof_lgb + 0.5 * oof_xgb

# Out-of-fold metrics
oof_auc_lgb = roc_auc_score(y, oof_lgb)
oof_auc_xgb = roc_auc_score(y, oof_xgb)
oof_auc_ensemble = roc_auc_score(y, oof_ensemble)

print(f"\nOut-of-Fold AUC:")
print(f"  LightGBM:  {oof_auc_lgb:.4f}")
print(f"  XGBoost:   {oof_auc_xgb:.4f}")
print(f"  Ensemble:  {oof_auc_ensemble:.4f}")

# Fold consistency
print(f"\nFold Consistency:")
lgb_fold_aucs = [f['lgb_val_auc'] for f in fold_scores]
xgb_fold_aucs = [f['xgb_val_auc'] for f in fold_scores]
print(f"  LightGBM std: {np.std(lgb_fold_aucs):.4f}")
print(f"  XGBoost std:  {np.std(xgb_fold_aucs):.4f}")

# ============================================================================
# STEP 5: TEST SET EVALUATION
# ============================================================================
print("\n" + "="*80)
print("5. TEST SET EVALUATION")
print("="*80)

# Ensemble test predictions
test_preds_ensemble = 0.5 * test_preds_lgb + 0.5 * test_preds_xgb

# Test AUC
test_auc_lgb = roc_auc_score(y_test, test_preds_lgb)
test_auc_xgb = roc_auc_score(y_test, test_preds_xgb)
test_auc_ensemble = roc_auc_score(y_test, test_preds_ensemble)

print(f"\nTest AUC:")
print(f"  LightGBM:  {test_auc_lgb:.4f}")
print(f"  XGBoost:   {test_auc_xgb:.4f}")
print(f"  Ensemble:  {test_auc_ensemble:.4f}")

# Classification metrics (threshold 0.5)
test_pred_binary = (test_preds_ensemble > 0.5).astype(int)
test_precision = precision_score(y_test, test_pred_binary)
test_recall = recall_score(y_test, test_pred_binary)
test_f1 = f1_score(y_test, test_pred_binary)

print(f"\nClassification Metrics (threshold=0.5):")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1 Score:  {test_f1:.4f}")

# ============================================================================
# STEP 6: COMPARISON WITH v3.0
# ============================================================================
print("\n" + "="*80)
print("6. COMPARISON WITH v3.0 BASELINE")
print("="*80)

# v3.0 scores (from previous results)
v3_test_auc = 0.7619
v3_val_auc = 0.8041

# Improvement
auc_improvement = ((test_auc_ensemble - v3_test_auc) / v3_test_auc) * 100

print(f"\nv3.0 Baseline:")
print(f"  Val AUC:  {v3_val_auc:.4f}")
print(f"  Test AUC: {v3_test_auc:.4f}")

print(f"\nK-Fold CV Results:")
print(f"  OOF AUC:  {oof_auc_ensemble:.4f}")
print(f"  Test AUC: {test_auc_ensemble:.4f}")

print(f"\nImprovement:")
print(f"  Absolute: {test_auc_ensemble - v3_test_auc:+.4f}")
print(f"  Relative: {auc_improvement:+.2f}%")

if test_auc_ensemble > v3_test_auc:
    print(f"\n  ✅ SUCCESS! K-Fold improved over v3.0!")
else:
    print(f"\n  ⚠️ No improvement (might be noise)")

# ============================================================================
# STEP 7: SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("7. SAVING RESULTS")
print("="*80)

# Save predictions
np.save('models/kfold_oof_lgb.npy', oof_lgb)
np.save('models/kfold_oof_xgb.npy', oof_xgb)
np.save('models/kfold_test_lgb.npy', test_preds_lgb)
np.save('models/kfold_test_xgb.npy', test_preds_xgb)

# Save results
results = {
    'oof_auc_lgb': oof_auc_lgb,
    'oof_auc_xgb': oof_auc_xgb,
    'oof_auc_ensemble': oof_auc_ensemble,
    'test_auc_lgb': test_auc_lgb,
    'test_auc_xgb': test_auc_xgb,
    'test_auc_ensemble': test_auc_ensemble,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'test_f1': test_f1,
    'fold_scores': fold_scores,
    'improvement_vs_v3': auc_improvement,
    'n_folds': N_FOLDS,
    'created_at': datetime.now().isoformat(),
}

with open('models/kfold_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("✓ Results saved")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("🎉 TECHNIQUE 1 COMPLETE: K-FOLD CV")
print("="*80)

print(f"\nFinal Results:")
print(f"  Test AUC: {test_auc_ensemble:.4f}")
print(f"  Improvement: {auc_improvement:+.2f}%")
print(f"  F1 Score: {test_f1:.4f}")

print(f"\nNext: Technique 2 - Stacking Ensemble (tomorrow)")
print(f"End time: {datetime.now().strftime('%H:%M:%S')}")

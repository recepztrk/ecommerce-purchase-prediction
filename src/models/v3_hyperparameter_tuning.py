"""
Hyperparameter Tuning for v3.0 LightGBM

Goal: Improve v3.0 baseline (0.7619 AUC) with better hyperparameters
Method: Optuna optimization (50 trials)
Expected: +0.5-0.8% AUC
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import optuna
import pickle

print("="*80)
print("HYPERPARAMETER TUNING - v3.0 LightGBM")
print("="*80)
print("\nGoal: Improve v3.0 baseline (AUC 0.7619, F1 0.69)")

# Load v3.0 data
print("\nLoading v3.0 data...")
train_df = pd.read_parquet('data/v3/train_sessions_v3.parquet')
val_df = pd.read_parquet('data/v3/val_sessions_v3.parquet')
test_df = pd.read_parquet('data/v3/test_sessions_v3.parquet')

exclude_cols = ['user_session', 'user_id', 'session_start', 'session_end', 'target']
features = [c for c in train_df.columns if c not in exclude_cols]

X_train, y_train = train_df[features].values, train_df['target'].values
X_val, y_val = val_df[features].values, val_df['target'].values
X_test, y_test = test_df[features].values, test_df['target'].values

print(f"  Train: {X_train.shape}")
print(f"  Features: {len(features)}")

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# OPTUNA OPTIMIZATION
# ============================================================================
print("\n" + "="*80)
print("RUNNING OPTUNA (50 trials)")
print("="*80)

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'verbose': -1,
        'random_state': 42
    }
    
    model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train),
        num_boost_round=1000,
        valid_sets=[lgb.Dataset(X_val, label=y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    
    pred = model.predict(X_val)
    return roc_auc_score(y_val, pred)

study = optuna.create_study(direction='maximize', study_name='v3.0_LightGBM')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\n✅ Optimization Complete!")
print(f"\n  Best Val AUC: {study.best_value:.4f}")
print(f"\n  Best Parameters:")
for key, value in study.best_params.items():
    print(f"    {key:20s}: {value}")

# ============================================================================
# TRAIN FINAL MODEL & EVALUATE
# ============================================================================
print("\n" + "="*80)
print("TRAINING FINAL MODEL WITH BEST PARAMS")
print("="*80)

best_params = study.best_params.copy()
best_params.update({
    'objective': 'binary',
    'metric': 'auc',
    'verbose': -1,
    'random_state': 42
})

final_model = lgb.train(
    best_params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=2000,
    valid_sets=[lgb.Dataset(X_val, label=y_val)],
    callbacks=[lgb.early_stopping(150, verbose=False)]
)

print(f"\n  Training iterations: {final_model.num_trees()}")

# Comprehensive evaluation
def evaluate_model(model, X, y, dataset_name):
    pred = model.predict(X)
    pred_binary = (pred > 0.5).astype(int)
    
    auc = roc_auc_score(y, pred)
    f1 = f1_score(y, pred_binary)
    precision = precision_score(y, pred_binary)
    recall = recall_score(y, pred_binary)
    cm = confusion_matrix(y, pred_binary)
    
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
        'recall': recall
    }

train_metrics = evaluate_model(final_model, X_train, y_train, "Train")
val_metrics = evaluate_model(final_model, X_val, y_val, "Val")
test_metrics = evaluate_model(final_model, X_test, y_test, "Test")

# Calculate gap
train_test_gap = abs(train_metrics['auc'] - test_metrics['auc']) / train_metrics['auc'] * 100
print(f"\nTrain-Test Gap: {train_test_gap:.2f}%")

# ============================================================================
# COMPARISON WITH v3.0 BASELINE
# ============================================================================
print("\n" + "="*80)
print("COMPARISON WITH v3.0 BASELINE")
print("="*80)

v3_baseline = {
    'auc': 0.7619,
    'f1': 0.69,
    'precision': 0.65,
    'recall': 0.98,
    'gap': 11.0
}

print(f"\nv3.0 Baseline:")
print(f"  Test AUC: {v3_baseline['auc']:.4f}")
print(f"  F1:       {v3_baseline['f1']:.2f}")
print(f"  Precision: {v3_baseline['precision']:.2f}")
print(f"  Recall:    {v3_baseline['recall']:.2f}")
print(f"  Gap:       {v3_baseline['gap']:.1f}%")

print(f"\nv3.0 Optimized (Hyperparameter Tuned):")
print(f"  Test AUC: {test_metrics['auc']:.4f}  ({test_metrics['auc'] - v3_baseline['auc']:+.4f}, {(test_metrics['auc'] - v3_baseline['auc'])/v3_baseline['auc']*100:+.2f}%)")
print(f"  F1:       {test_metrics['f1']:.4f}  ({test_metrics['f1'] - v3_baseline['f1']:+.4f})")
print(f"  Precision: {test_metrics['precision']:.4f}  ({test_metrics['precision'] - v3_baseline['precision']:+.4f})")
print(f"  Recall:    {test_metrics['recall']:.4f}  ({test_metrics['recall'] - v3_baseline['recall']:+.4f})")
print(f"  Gap:       {train_test_gap:.2f}%  ({train_test_gap - v3_baseline['gap']:+.2f}%)")

# Check if improvement
if test_metrics['auc'] > v3_baseline['auc'] and test_metrics['f1'] >= v3_baseline['f1']:
    print("\n✅ SUCCESS! Improved in BOTH AUC and F1!")
elif test_metrics['auc'] > v3_baseline['auc']:
    print("\n⚠️  PARTIAL: Improved AUC but F1 decreased")
else:
    print("\n❌ FAILED: No improvement over baseline")

# Save results
final_model.save_model('models/v3_lightgbm_optimized.txt')
print("\n✓ Model saved: models/v3_lightgbm_optimized.txt")

results = {
    'best_params': study.best_params,
    'val_auc': study.best_value,
    'train_metrics': train_metrics,
    'val_metrics': val_metrics,
    'test_metrics': test_metrics,
    'gap': train_test_gap,
    'vs_baseline': {
        'auc_diff': test_metrics['auc'] - v3_baseline['auc'],
        'f1_diff': test_metrics['f1'] - v3_baseline['f1']
    }
}

with open('reports/v3_hyperparameter_tuning_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("✓ Results saved: reports/v3_hyperparameter_tuning_results.pkl")

print("\n" + "="*80)
print("HYPERPARAMETER TUNING COMPLETE")
print("="*80)

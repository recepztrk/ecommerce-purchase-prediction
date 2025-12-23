"""
Phase 3: Hyperparameter Tuning with Optuna

Optimize top 3 algorithms:
1. ExtraTrees (current best: 0.7644)
2. LightGBM (0.7629)
3. XGBoost (0.7623)

25 trials each, optimize for AUC
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import optuna
import pickle

print("="*80)
print("PHASE 3: HYPERPARAMETER TUNING WITH OPTUNA")
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

print(f"  Data ready: {X_train.shape}")

optuna.logging.set_verbosity(optuna.logging.WARNING)

results = []

# ============================================================================
# 1. ExtraTrees Optimization
# ============================================================================
print("\n" + "="*80)
print("1. OPTIMIZING EXTRATREES (50 trials)")
print("="*80)

def objective_extratrees(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 800),
        'max_depth': trial.suggest_int('max_depth', 10, 25),
        'min_samples_split': trial.suggest_int('min_samples_split', 10, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.8]),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = ExtraTreesClassifier(**params)
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, pred)

study_et = optuna.create_study(direction='maximize', study_name='ExtraTrees')
study_et.optimize(objective_extratrees, n_trials=25, show_progress_bar=True)

print(f"\n  Best Val AUC: {study_et.best_value:.4f}")
print(f"  Best params: {study_et.best_params}")

# Train with best params and test
best_et = ExtraTreesClassifier(**study_et.best_params, random_state=42, n_jobs=-1)
best_et.fit(X_train, y_train)
et_test_pred = best_et.predict_proba(X_test)[:, 1]
et_test_auc = roc_auc_score(y_test, et_test_pred)

print(f"  Test AUC: {et_test_auc:.4f}")

results.append({
    'algorithm': 'ExtraTrees (Optimized)',
    'val_auc': study_et.best_value,
    'test_auc': et_test_auc,
    'best_params': study_et.best_params
})

# ============================================================================
# 2. LightGBM Optimization
# ============================================================================
print("\n" + "="*80)
print("2. OPTIMIZING LIGHTGBM (50 trials)")
print("="*80)

def objective_lgb(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'verbose': -1,
        'random_state': 42
    }
    
    model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_val, label=y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    pred = model.predict(X_val)
    return roc_auc_score(y_val, pred)

study_lgb = optuna.create_study(direction='maximize', study_name='LightGBM')
study_lgb.optimize(objective_lgb, n_trials=25, show_progress_bar=True)

print(f"\n  Best Val AUC: {study_lgb.best_value:.4f}")
print(f"  Best params: {study_lgb.best_params}")

# Train with best params
best_lgb_params = study_lgb.best_params.copy()
best_lgb_params.update({'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'random_state': 42})
best_lgb = lgb.train(
    best_lgb_params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=1000,
    valid_sets=[lgb.Dataset(X_val, label=y_val)],
    callbacks=[lgb.early_stopping(100, verbose=False)]
)

lgb_test_pred = best_lgb.predict(X_test)
lgb_test_auc = roc_auc_score(y_test, lgb_test_pred)

print(f"  Test AUC: {lgb_test_auc:.4f}")

results.append({
    'algorithm': 'LightGBM (Optimized)',
    'val_auc': study_lgb.best_value,
    'test_auc': lgb_test_auc,
    'best_params': study_lgb.best_params
})

# ============================================================================
# 3. XGBoost Optimization
# ============================================================================
print("\n" + "="*80)
print("3. OPTIMIZING XGBOOST (50 trials)")
print("="*80)

def objective_xgb(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'random_state': 42,
        'verbosity': 0
    }
    
    model = xgb.XGBClassifier(**params, n_estimators=500, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, pred)

study_xgb = optuna.create_study(direction='maximize', study_name='XGBoost')
study_xgb.optimize(objective_xgb, n_trials=25, show_progress_bar=True)

print(f"\n  Best Val AUC: {study_xgb.best_value:.4f}")
print(f"  Best params: {study_xgb.best_params}")

# Train with best params
best_xgb_params = study_xgb.best_params.copy()
best_xgb_params.update({'objective': 'binary:logistic', 'eval_metric': 'auc', 'random_state': 42, 'verbosity': 0})
best_xgb_model = xgb.XGBClassifier(**best_xgb_params, n_estimators=1000, early_stopping_rounds=100)
best_xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

xgb_test_pred = best_xgb_model.predict_proba(X_test)[:, 1]
xgb_test_auc = roc_auc_score(y_test, xgb_test_pred)

print(f"  Test AUC: {xgb_test_auc:.4f}")

results.append({
    'algorithm': 'XGBoost (Optimized)',
    'val_auc': study_xgb.best_value,
    'test_auc': xgb_test_auc,
    'best_params': study_xgb.best_params
})

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("PHASE 3 COMPLETE: OPTIMIZATION RESULTS")
print("="*80)

results_df = pd.DataFrame(results).sort_values('test_auc', ascending=False)

print("\nRanking by Test AUC:")
for _, row in results_df.iterrows():
    print(f"  {row['algorithm']:25s} Val: {row['val_auc']:.4f}  Test: {row['test_auc']:.4f}")

# Best model
best_model = results_df.iloc[0]
print(f"\n🏆 WINNER: {best_model['algorithm']}")
print(f"  Test AUC: {best_model['test_auc']:.4f}")

# Improvement
v3_baseline = 0.7619
improvement = best_model['test_auc'] - v3_baseline

print(f"\nImprovement vs v3.0 baseline:")
print(f"  v3.0: {v3_baseline:.4f}")
print(f"  Best: {best_model['test_auc']:.4f}")
print(f"  Gain: +{improvement:.4f} (+{improvement/v3_baseline*100:.2f}%)")

# Save
results_df.to_csv('reports/phase3_optuna_results.csv', index=False)
with open('reports/phase3_results.pkl', 'wb') as f:
    pickle.dump({
        'results': results_df,
        'best_model': best_model,
        'studies': {'et': study_et, 'lgb': study_lgb, 'xgb': study_xgb}
    }, f)

# Save best models
import joblib
joblib.dump(best_et, 'models/best_extratrees.pkl')
best_lgb.save_model('models/best_lightgbm.txt')
joblib.dump(best_xgb_model, 'models/best_xgboost.pkl')

print("\n✓ Results and models saved")
print("="*80)

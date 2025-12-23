"""
Phase 2: Algorithm Testing

Test 5 different algorithms on clean dataset:
1. LightGBM (baseline - already done: 0.7629)
2. XGBoost
3. Random Forest
4. ExtraTrees
5. HistGradientBoosting

Compare and pick best 2-3 for hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
import time

print("="*80)
print("PHASE 2: ALGORITHM TESTING")
print("="*80)

# Load data
print("\nLoading clean dataset...")
train_df = pd.read_parquet('data/v3_final/train_sessions_final.parquet')
val_df = pd.read_parquet('data/v3_final/val_sessions_final.parquet')
test_df = pd.read_parquet('data/v3_final/test_sessions_final.parquet')

exclude_cols = ['user_session', 'user_id', 'session_start', 'session_end', 'target']
features = [c for c in train_df.columns if c not in exclude_cols]

X_train, y_train = train_df[features].values, train_df['target'].values
X_val, y_val = val_df[features].values, val_df['target'].values
X_test, y_test = test_df[features].values, test_df['target'].values

print(f"  Features: {len(features)}")
print(f"  Train: {X_train.shape}")

results = []

# ============================================================================
# 1. LightGBM (baseline - already known)
# ============================================================================
print("\n" + "="*80)
print("1. LIGHTGBM (Baseline)")
print("="*80)

print("  Using previous result: 0.7629 AUC")
results.append({
    'algorithm': 'LightGBM',
    'test_auc': 0.7629,
    'test_f1': 0.6862,
    'train_time': 'N/A (cached)'
})

# ============================================================================
# 2. XGBoost
# ============================================================================
print("\n" + "="*80)
print("2. XGBOOST")
print("="*80)

start = time.time()

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 9,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 1000,
    'early_stopping_rounds': 100,
    'random_state': 42,
    'verbosity': 0
}

xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, (xgb_pred > 0.5).astype(int))

train_time = time.time() - start

print(f"  Test AUC: {xgb_auc:.4f}")
print(f"  Test F1:  {xgb_f1:.4f}")
print(f"  Time: {train_time:.1f}s")

results.append({
    'algorithm': 'XGBoost',
    'test_auc': xgb_auc,
    'test_f1': xgb_f1,
    'train_time': f'{train_time:.1f}s'
})

# ============================================================================
# 3. Random Forest
# ============================================================================
print("\n" + "="*80)
print("3. RANDOM FOREST")
print("="*80)

start = time.time()

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, (rf_pred > 0.5).astype(int))

train_time = time.time() - start

print(f"  Test AUC: {rf_auc:.4f}")
print(f"  Test F1:  {rf_f1:.4f}")
print(f"  Time: {train_time:.1f}s")

results.append({
    'algorithm': 'Random Forest',
    'test_auc': rf_auc,
    'test_f1': rf_f1,
    'train_time': f'{train_time:.1f}s'
})

# ============================================================================
# 4. ExtraTrees
# ============================================================================
print("\n" + "="*80)
print("4. EXTRATREES")
print("="*80)

start = time.time()

et_model = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

et_model.fit(X_train, y_train)

et_pred = et_model.predict_proba(X_test)[:, 1]
et_auc = roc_auc_score(y_test, et_pred)
et_f1 = f1_score(y_test, (et_pred > 0.5).astype(int))

train_time = time.time() - start

print(f"  Test AUC: {et_auc:.4f}")
print(f"  Test F1:  {et_f1:.4f}")
print(f"  Time: {train_time:.1f}s")

results.append({
    'algorithm': 'ExtraTrees',
    'test_auc': et_auc,
    'test_f1': et_f1,
    'train_time': f'{train_time:.1f}s'
})

# ============================================================================
# 5. HistGradientBoosting
# ============================================================================
print("\n" + "="*80)
print("5. HISTGRADIENTBOOSTING")
print("="*80)

start = time.time()

hgb_model = HistGradientBoostingClassifier(
    max_iter=1000,
    learning_rate=0.03,
    max_depth=9,
    min_samples_leaf=20,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=100,
    random_state=42,
    verbose=0
)

hgb_model.fit(X_train, y_train)

hgb_pred = hgb_model.predict_proba(X_test)[:, 1]
hgb_auc = roc_auc_score(y_test, hgb_pred)
hgb_f1 = f1_score(y_test, (hgb_pred > 0.5).astype(int))

train_time = time.time() - start

print(f"  Test AUC: {hgb_auc:.4f}")
print(f"  Test F1:  {hgb_f1:.4f}")
print(f"  Time: {train_time:.1f}s")

results.append({
    'algorithm': 'HistGradientBoosting',
    'test_auc': hgb_auc,
    'test_f1': hgb_f1,
    'train_time': f'{train_time:.1f}s'
})

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("PHASE 2 COMPLETE: ALGORITHM COMPARISON")
print("="*80)

results_df = pd.DataFrame(results).sort_values('test_auc', ascending=False)

print("\nRanking by Test AUC:")
for i, row in results_df.iterrows():
    print(f"  {row['algorithm']:25s} AUC: {row['test_auc']:.4f}  F1: {row['test_f1']:.4f}  Time: {row['train_time']}")

# Best algorithms
best_3 = results_df.head(3)
print(f"\nTop 3 Algorithms for Phase 3 (Hyperparameter Tuning):")
for i, row in best_3.iterrows():
    print(f"  {i+1}. {row['algorithm']} (AUC: {row['test_auc']:.4f})")

# Save
results_df.to_csv('reports/phase2_algorithm_comparison.csv', index=False)
print(f"\n✓ Results saved: reports/phase2_algorithm_comparison.csv")

print("\n" + "="*80)
print("NEXT: Phase 3 - Hyperparameter Tuning (Top 2-3 algorithms)")
print("="*80)

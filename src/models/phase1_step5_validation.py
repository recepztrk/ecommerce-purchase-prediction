"""
Phase 1 Step 5: Final Validation

Train model with final 24 clean features and validate improvement
"""

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score

print("="*80)
print("PHASE 1 STEP 5: FINAL VALIDATION")
print("="*80)

# Load final clean data
print("\n1. Loading final clean dataset...")
train_df = pd.read_parquet('data/v3_final/train_sessions_final.parquet')
val_df = pd.read_parquet('data/v3_final/val_sessions_final.parquet')
test_df = pd.read_parquet('data/v3_final/test_sessions_final.parquet')

exclude_cols = ['user_session', 'user_id', 'session_start', 'session_end', 'target']
features = [c for c in train_df.columns if c not in exclude_cols]

print(f"  Final features: {len(features)}")

X_train, y_train = train_df[features], train_df['target']
X_val, y_val = val_df[features], val_df['target']
X_test, y_test = test_df[features], test_df['target']

# 2. Train final model
print("\n2. Training LightGBM with 24 clean features...")

params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 127,
    'learning_rate': 0.03,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 9,
    'min_child_samples': 20,
    'verbose': -1,
    'random_seed': 42
}

model = lgb.train(
    params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=1000,
    valid_sets=[lgb.Dataset(X_val, label=y_val)],
    callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(100)]
)

# 3. Evaluate
test_pred = model.predict(X_test)
val_pred = model.predict(X_val)
train_pred = model.predict(X_train)

train_auc = roc_auc_score(y_train, train_pred)
val_auc = roc_auc_score(y_val, val_pred)
test_auc = roc_auc_score(y_test, test_pred)

test_f1 = f1_score(y_test, (test_pred > 0.5).astype(int))

gap = abs(train_auc - test_auc) / train_auc * 100

print(f"\n" + "="*80)
print("PHASE 1 COMPLETE: FINAL RESULTS")
print(f"="*80)

print(f"\nPerformance (24 clean features):")
print(f"  Train AUC: {train_auc:.4f}")
print(f"  Val AUC:   {val_auc:.4f}")
print(f"  Test AUC:  {test_auc:.4f}")
print(f"  Test F1:   {test_f1:.4f}")
print(f"  Gap:       {gap:.2f}%")

# Comparison with v3.0
v3_auc = 0.7619
v3_f1 = 0.69
improvement_auc = test_auc - v3_auc
improvement_f1 = test_f1 - v3_f1

print(f"\nComparison with v3.0:")
print(f"  v3.0 Test AUC: {v3_auc:.4f}")
print(f"  Phase 1 AUC:   {test_auc:.4f}")
print(f"  Improvement:   {improvement_auc:+.4f} ({improvement_auc/v3_auc*100:+.2f}%)")

print(f"\n  v3.0 F1:       {v3_f1:.4f}")
print(f"  Phase 1 F1:    {test_f1:.4f}")
print(f"  Improvement:   {improvement_f1:+.4f}")

# Feature list
print(f"\nFinal 24 Features:")
for i, f in enumerate(features, 1):
    importance_pct = model.feature_importance()[i-1] / model.feature_importance().sum() * 100
    print(f"  {i:2d}. {f:30s} ({importance_pct:5.2f}%)")

# Save
model.save_model('models/lgb_phase1_final.txt')
print(f"\n✓ Model saved: models/lgb_phase1_final.txt")

print(f"\n{'='*80}")
print("NEXT: Phase 2 - Algorithm Testing")
print(f"{'='*80}")
print("Ready to test 5 different algorithms with this clean dataset")
print(f"='*80")

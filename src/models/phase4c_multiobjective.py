"""
Phase 4c: Multi-Objective Ensemble Optimization

COMPREHENSIVE APPROACH:
- Optimize for ALL metrics: AUC, F1, Precision, Recall, Gap
- Test multiple objective functions
- Find best balanced ensemble

Objectives to test:
1. AUC only (baseline comparison)
2. F1 only
3. AUC + F1 (balanced)
4. AUC + F1 + Precision
5. AUC + F1 + Recall
6. Composite score (weighted combo of all)
7. Minimize Gap while maximizing AUC
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
print("PHASE 4c: MULTI-OBJECTIVE ENSEMBLE OPTIMIZATION")
print("="*80)
print("\nOptimizing for ALL metrics: AUC, F1, Precision, Recall, Gap")

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

# Load models
print("\nLoading models...")
model_et = joblib.load('models/best_extratrees.pkl')
model_xgb = joblib.load('models/best_xgboost_colab.pkl')
model_lgb = lgb.Booster(model_file='models/best_lightgbm_colab.txt')

# Generate predictions
print("\nGenerating predictions...")
train_pred_et = model_et.predict_proba(X_train)[:, 1]
train_pred_xgb = model_xgb.predict_proba(X_train)[:, 1]
train_pred_lgb = model_lgb.predict(X_train)

val_pred_et = model_et.predict_proba(X_val)[:, 1]
val_pred_xgb = model_xgb.predict_proba(X_val)[:, 1]
val_pred_lgb = model_lgb.predict(X_val)

test_pred_et = model_et.predict_proba(X_test)[:, 1]
test_pred_xgb = model_xgb.predict_proba(X_test)[:, 1]
test_pred_lgb = model_lgb.predict(X_test)

print("✓ Predictions ready")

# ============================================================================
# COMPREHENSIVE EVALUATION FUNCTION
# ============================================================================
def evaluate_all_metrics(y_true_train, train_pred, y_true_val, val_pred, y_true_test, test_pred):
    """Calculate ALL metrics for comprehensive evaluation"""
    
    # Binary predictions
    train_pred_binary = (train_pred > 0.5).astype(int)
    val_pred_binary = (val_pred > 0.5).astype(int)
    test_pred_binary = (test_pred > 0.5).astype(int)
    
    # Train metrics
    train_auc = roc_auc_score(y_true_train, train_pred)
    train_f1 = f1_score(y_true_train, train_pred_binary)
    train_precision = precision_score(y_true_train, train_pred_binary)
    train_recall = recall_score(y_true_train, train_pred_binary)
    
    # Val metrics
    val_auc = roc_auc_score(y_true_val, val_pred)
    val_f1 = f1_score(y_true_val, val_pred_binary)
    val_precision = precision_score(y_true_val, val_pred_binary)
    val_recall = recall_score(y_true_val, val_pred_binary)
    
    # Test metrics
    test_auc = roc_auc_score(y_true_test, test_pred)
    test_f1 = f1_score(y_true_test, test_pred_binary)
    test_precision = precision_score(y_true_test, test_pred_binary)
    test_recall = recall_score(y_true_test, test_pred_binary)
    
    # Gap
    train_test_gap = abs(train_auc - test_auc) / train_auc * 100
    
    return {
        'train_auc': train_auc,
        'train_f1': train_f1,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'val_auc': val_auc,
        'val_f1': val_f1,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'test_auc': test_auc,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'gap': train_test_gap
    }

# ============================================================================
# MULTI-OBJECTIVE GRID SEARCH
# ============================================================================
print("\n" + "="*80)
print("MULTI-OBJECTIVE GRID SEARCH")
print("="*80)

print("\nTesting different objective functions...")

weight_range = np.arange(0, 1.1, 0.1)
best_results = {}

# ============================================================================
# Objective 1: Maximize AUC only (baseline)
# ============================================================================
print("\n1. Objective: Maximize AUC")
best_score = 0
best_weights = None

for w1 in weight_range:
    for w2 in weight_range:
        w3 = 1.0 - w1 - w2
        if w3 < 0 or w3 > 1.0:
            continue
        
        val_ensemble = w1 * val_pred_et + w2 * val_pred_xgb + w3 * val_pred_lgb
        score = roc_auc_score(y_val, val_ensemble)
        
        if score > best_score:
            best_score = score
            best_weights = (w1, w2, w3)

print(f"   Best weights: ET={best_weights[0]:.1f}, XGB={best_weights[1]:.1f}, LGB={best_weights[2]:.1f}")
print(f"   Val AUC: {best_score:.4f}")

# Evaluate on all sets
train_ens = best_weights[0] * train_pred_et + best_weights[1] * train_pred_xgb + best_weights[2] * train_pred_lgb
val_ens = best_weights[0] * val_pred_et + best_weights[1] * val_pred_xgb + best_weights[2] * val_pred_lgb
test_ens = best_weights[0] * test_pred_et + best_weights[1] * test_pred_xgb + best_weights[2] * test_pred_lgb

best_results['AUC_only'] = {
    'weights': best_weights,
    'metrics': evaluate_all_metrics(y_train, train_ens, y_val, val_ens, y_test, test_ens)
}

# ============================================================================
# Objective 2: Maximize F1
# ============================================================================
print("\n2. Objective: Maximize F1")
best_score = 0
best_weights = None

for w1 in weight_range:
    for w2 in weight_range:
        w3 = 1.0 - w1 - w2
        if w3 < 0 or w3 > 1.0:
            continue
        
        val_ensemble = w1 * val_pred_et + w2 * val_pred_xgb + w3 * val_pred_lgb
        score = f1_score(y_val, (val_ensemble > 0.5).astype(int))
        
        if score > best_score:
            best_score = score
            best_weights = (w1, w2, w3)

print(f"   Best weights: ET={best_weights[0]:.1f}, XGB={best_weights[1]:.1f}, LGB={best_weights[2]:.1f}")
print(f"   Val F1: {best_score:.4f}")

train_ens = best_weights[0] * train_pred_et + best_weights[1] * train_pred_xgb + best_weights[2] * train_pred_lgb
val_ens = best_weights[0] * val_pred_et + best_weights[1] * val_pred_xgb + best_weights[2] * val_pred_lgb
test_ens = best_weights[0] * test_pred_et + best_weights[1] * test_pred_xgb + best_weights[2] * test_pred_lgb

best_results['F1_only'] = {
    'weights': best_weights,
    'metrics': evaluate_all_metrics(y_train, train_ens, y_val, val_ens, y_test, test_ens)
}

# ============================================================================
# Objective 3: Balanced AUC + F1
# ============================================================================
print("\n3. Objective: Balanced (AUC + F1)")
best_score = 0
best_weights = None

for w1 in weight_range:
    for w2 in weight_range:
        w3 = 1.0 - w1 - w2
        if w3 < 0 or w3 > 1.0:
            continue
        
        val_ensemble = w1 * val_pred_et + w2 * val_pred_xgb + w3 * val_pred_lgb
        auc = roc_auc_score(y_val, val_ensemble)
        f1 = f1_score(y_val, (val_ensemble > 0.5).astype(int))
        score = (auc + f1) / 2  # Equal weight
        
        if score > best_score:
            best_score = score
            best_weights = (w1, w2, w3)

print(f"   Best weights: ET={best_weights[0]:.1f}, XGB={best_weights[1]:.1f}, LGB={best_weights[2]:.1f}")
print(f"   Val (AUC+F1)/2: {best_score:.4f}")

train_ens = best_weights[0] * train_pred_et + best_weights[1] * train_pred_xgb + best_weights[2] * train_pred_lgb
val_ens = best_weights[0] * val_pred_et + best_weights[1] * val_pred_xgb + best_weights[2] * val_pred_lgb
test_ens = best_weights[0] * test_pred_et + best_weights[1] * test_pred_xgb + best_weights[2] * test_pred_lgb

best_results['AUC_F1_balanced'] = {
    'weights': best_weights,
    'metrics': evaluate_all_metrics(y_train, train_ens, y_val, val_ens, y_test, test_ens)
}

# ============================================================================
# Objective 4: AUC + F1 + Precision
# ============================================================================
print("\n4. Objective: AUC + F1 + Precision")
best_score = 0
best_weights = None

for w1 in weight_range:
    for w2 in weight_range:
        w3 = 1.0 - w1 - w2
        if w3 < 0 or w3 > 1.0:
            continue
        
        val_ensemble = w1 * val_pred_et + w2 * val_pred_xgb + w3 * val_pred_lgb
        val_bin = (val_ensemble > 0.5).astype(int)
        auc = roc_auc_score(y_val, val_ensemble)
        f1 = f1_score(y_val, val_bin)
        prec = precision_score(y_val, val_bin)
        score = (auc + f1 + prec) / 3
        
        if score > best_score:
            best_score = score
            best_weights = (w1, w2, w3)

print(f"   Best weights: ET={best_weights[0]:.1f}, XGB={best_weights[1]:.1f}, LGB={best_weights[2]:.1f}")
print(f"   Val (AUC+F1+Prec)/3: {best_score:.4f}")

train_ens = best_weights[0] * train_pred_et + best_weights[1] * train_pred_xgb + best_weights[2] * train_pred_lgb
val_ens = best_weights[0] * val_pred_et + best_weights[1] * val_pred_xgb + best_weights[2] * val_pred_lgb
test_ens = best_weights[0] * test_pred_et + best_weights[1] * test_pred_xgb + best_weights[2] * test_pred_lgb

best_results['AUC_F1_Precision'] = {
    'weights': best_weights,
    'metrics': evaluate_all_metrics(y_train, train_ens, y_val, val_ens, y_test, test_ens)
}

# ============================================================================
# Objective 5: AUC + F1 + Recall
# ============================================================================
print("\n5. Objective: AUC + F1 + Recall")
best_score = 0
best_weights = None

for w1 in weight_range:
    for w2 in weight_range:
        w3 = 1.0 - w1 - w2
        if w3 < 0 or w3 > 1.0:
            continue
        
        val_ensemble = w1 * val_pred_et + w2 * val_pred_xgb + w3 * val_pred_lgb
        val_bin = (val_ensemble > 0.5).astype(int)
        auc = roc_auc_score(y_val, val_ensemble)
        f1 = f1_score(y_val, val_bin)
        rec = recall_score(y_val, val_bin)
        score = (auc + f1 + rec) / 3
        
        if score > best_score:
            best_score = score
            best_weights = (w1, w2, w3)

print(f"   Best weights: ET={best_weights[0]:.1f}, XGB={best_weights[1]:.1f}, LGB={best_weights[2]:.1f}")
print(f"   Val (AUC+F1+Rec)/3: {best_score:.4f}")

train_ens = best_weights[0] * train_pred_et + best_weights[1] * train_pred_xgb + best_weights[2] * train_pred_lgb
val_ens = best_weights[0] * val_pred_et + best_weights[1] * val_pred_xgb + best_weights[2] * val_pred_lgb
test_ens = best_weights[0] * test_pred_et + best_weights[1] * test_pred_xgb + best_weights[2] * test_pred_lgb

best_results['AUC_F1_Recall'] = {
    'weights': best_weights,
    'metrics': evaluate_all_metrics(y_train, train_ens, y_val, val_ens, y_test, test_ens)
}

# ============================================================================
# Objective 6: Composite Score (ALL metrics weighted)
# ============================================================================
print("\n6. Objective: Composite (0.3*AUC + 0.3*F1 + 0.2*Prec + 0.2*Rec)")
best_score = 0
best_weights = None

for w1 in weight_range:
    for w2 in weight_range:
        w3 = 1.0 - w1 - w2
        if w3 < 0 or w3 > 1.0:
            continue
        
        val_ensemble = w1 * val_pred_et + w2 * val_pred_xgb + w3 * val_pred_lgb
        val_bin = (val_ensemble > 0.5).astype(int)
        auc = roc_auc_score(y_val, val_ensemble)
        f1 = f1_score(y_val, val_bin)
        prec = precision_score(y_val, val_bin)
        rec = recall_score(y_val, val_bin)
        score = 0.3*auc + 0.3*f1 + 0.2*prec + 0.2*rec
        
        if score > best_score:
            best_score = score
            best_weights = (w1, w2, w3)

print(f"   Best weights: ET={best_weights[0]:.1f}, XGB={best_weights[1]:.1f}, LGB={best_weights[2]:.1f}")
print(f"   Val Composite: {best_score:.4f}")

train_ens = best_weights[0] * train_pred_et + best_weights[1] * train_pred_xgb + best_weights[2] * train_pred_lgb
val_ens = best_weights[0] * val_pred_et + best_weights[1] * val_pred_xgb + best_weights[2] * val_pred_lgb
test_ens = best_weights[0] * test_pred_et + best_weights[1] * test_pred_xgb + best_weights[2] * test_pred_lgb

best_results['Composite'] = {
    'weights': best_weights,
    'metrics': evaluate_all_metrics(y_train, train_ens, y_val, val_ens, y_test, test_ens)
}

# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE RESULTS - ALL OBJECTIVES")
print("="*80)

# Create comparison table
comparison_data = []
for obj_name, obj_data in best_results.items():
    metrics = obj_data['metrics']
    weights = obj_data['weights']
    comparison_data.append({
        'Objective': obj_name,
        'Weights_ET': weights[0],
        'Weights_XGB': weights[1],
        'Weights_LGB': weights[2],
        'Test_AUC': metrics['test_auc'],
        'Test_F1': metrics['test_f1'],
        'Test_Precision': metrics['test_precision'],
        'Test_Recall': metrics['test_recall'],
        'Gap': metrics['gap']
    })

results_df = pd.DataFrame(comparison_data)

print("\n📊 Test Set Performance (All Objectives):")
print(results_df[['Objective', 'Test_AUC', 'Test_F1', 'Test_Precision', 'Test_Recall', 'Gap']].to_string(index=False))

print("\n📊 Optimal Weights for Each Objective:")
print(results_df[['Objective', 'Weights_ET', 'Weights_XGB', 'Weights_LGB']].to_string(index=False))

# Add v3.0 and best single for comparison
v3_metrics = {
    'Objective': 'v3.0 Baseline',
    'Test_AUC': 0.7619,
    'Test_F1': 0.69,
    'Test_Precision': 0.65,
    'Test_Recall': 0.98,
    'Gap': 11.0
}

extratrees_metrics = {
    'Objective': 'ExtraTrees (best single)',
    'Test_AUC': 0.7751,
    'Test_F1': 0.6702,
    'Test_Precision': 0.5942,
    'Test_Recall': 0.7686,
    'Gap': 13.6
}

all_results = pd.concat([
    pd.DataFrame([v3_metrics, extratrees_metrics]),
    results_df[['Objective', 'Test_AUC', 'Test_F1', 'Test_Precision', 'Test_Recall', 'Gap']]
], ignore_index=True)

print("\n" + "="*80)
print("FINAL COMPARISON (Including Baselines)")
print("="*80)
print(all_results.to_string(index=False))

# Find best for each metric
print("\n" + "="*80)
print("BEST MODEL FOR EACH METRIC")
print("="*80)

best_auc = all_results.loc[all_results['Test_AUC'].idxmax()]
best_f1 = all_results.loc[all_results['Test_F1'].idxmax()]
best_prec = all_results.loc[all_results['Test_Precision'].idxmax()]
best_rec = all_results.loc[all_results['Test_Recall'].idxmax()]
best_gap = all_results.loc[all_results['Gap'].idxmin()]

print(f"\n🏆 Best AUC: {best_auc['Objective']:30s} {best_auc['Test_AUC']:.4f}")
print(f"🏆 Best F1:  {best_f1['Objective']:30s} {best_f1['Test_F1']:.4f}")
print(f"🏆 Best Precision: {best_prec['Objective']:25s} {best_prec['Test_Precision']:.4f}")
print(f"🏆 Best Recall:    {best_rec['Objective']:25s} {best_rec['Test_Recall']:.4f}")
print(f"🏆 Best Gap:       {best_gap['Objective']:25s} {best_gap['Gap']:.2f}%")

# Save results
results_df.to_csv('reports/phase4c_multiobjective_results.csv', index=False)
with open('reports/phase4c_results.pkl', 'wb') as f:
    pickle.dump(best_results, f)

print("\n✓ Results saved")
print("\n" + "="*80)
print("PHASE 4c COMPLETE")
print("="*80)

"""
Improved model training with hyperparameter optimization and ensemble.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
import json
sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import lightgbm as lgb
import xgboost as xgb

from src.utils.config import (
    TRAIN_PROCESSED, VAL_PROCESSED, TEST_PROCESSED,
    TARGET_COL, MODELS_DIR, RANDOM_SEED
)
from src.features.build import get_feature_columns


def load_data():
    """Load processed data."""
    print("Loading data...")
    train_df = pd.read_parquet(TRAIN_PROCESSED)
    val_df = pd.read_parquet(VAL_PROCESSED)
    test_df = pd.read_parquet(TEST_PROCESSED)
    
    feature_cols = get_feature_columns(train_df)
    
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_COL].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_COL].values
    
    print(f"Train: {X_train.shape}")
    print(f"Val: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    print(f"Features: {len(feature_cols)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def train_lightgbm_optimized(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols):
    """
    Train LightGBM with optimized hyperparameters.
    """
    print("\n" + "="*80)
    print("LIGHTGBM - OPTIMIZED HYPERPARAMETERS")
    print("="*80)
    
    # Optimized parameters (based on typical best practices)
    params = {
        'objective': 'binary',
        'metric': ['auc', 'binary_logloss'],
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 7,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_cols)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=200)
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )
    
    print(f"Best iteration: {model.best_iteration}")
    
    # Predictions
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Metrics
    train_auc = roc_auc_score(y_train, y_train_pred)
    val_auc = roc_auc_score(y_val, y_val_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)
    
    print(f"\nROC-AUC:")
    print(f"  Train: {train_auc:.4f}")
    print(f"  Val:   {val_auc:.4f}")
    print(f"  Test:  {test_auc:.4f}")
    
    # Save
    model_path = MODELS_DIR / "lightgbm_v2.txt"
    model.save_model(str(model_path))
    
    return {
        'model': model,
        'train_pred': y_train_pred,
        'val_pred': y_val_pred,
        'test_pred': y_test_pred,
        'train_auc': train_auc,
        'val_auc': val_auc,
        'test_auc': test_auc,
    }


def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train XGBoost model.
    """
    print("\n" + "="*80)
    print("XGBOOST")
    print("="*80)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 7,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'tree_method': 'hist',
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=evals,
        early_stopping_rounds=100,
        verbose_eval=200
    )
    
    print(f"Best iteration: {model.best_iteration}")
    
    # Predictions
    y_train_pred = model.predict(dtrain, iteration_range=(0, model.best_iteration))
    y_val_pred = model.predict(dval, iteration_range=(0, model.best_iteration))
    y_test_pred = model.predict(dtest, iteration_range=(0, model.best_iteration))
    
    # Metrics
    train_auc = roc_auc_score(y_train, y_train_pred)
    val_auc = roc_auc_score(y_val, y_val_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)
    
    print(f"\nROC-AUC:")
    print(f"  Train: {train_auc:.4f}")
    print(f"  Val:   {val_auc:.4f}")
    print(f"  Test:  {test_auc:.4f}")
    
    # Save
    model_path = MODELS_DIR / "xgboost_v2.json"
    model.save_model(str(model_path))
    
    return {
        'model': model,
        'train_pred': y_train_pred,
        'val_pred': y_val_pred,
        'test_pred': y_test_pred,
        'train_auc': train_auc,
        'val_auc': val_auc,
        'test_auc': test_auc,
    }


def create_ensemble(lgb_results, xgb_results, y_train, y_val, y_test):
    """
    Create weighted ensemble of models.
    """
    print("\n" + "="*80)
    print("ENSEMBLE MODEL")
    print("="*80)
    
    # Find optimal weights on validation set
    best_auc = 0
    best_weights = None
    
    print("Finding optimal weights...")
    for lgb_weight in np.linspace(0.3, 0.7, 9):
        xgb_weight = 1 - lgb_weight
        
        val_pred = lgb_weight * lgb_results['val_pred'] + xgb_weight * xgb_results['val_pred']
        auc = roc_auc_score(y_val, val_pred)
        
        if auc > best_auc:
            best_auc = auc
            best_weights = (lgb_weight, xgb_weight)
    
    print(f"Optimal weights: LightGBM={best_weights[0]:.2f}, XGBoost={best_weights[1]:.2f}")
    
    # Create ensemble predictions
    train_pred = best_weights[0] * lgb_results['train_pred'] + best_weights[1] * xgb_results['train_pred']
    val_pred = best_weights[0] * lgb_results['val_pred'] + best_weights[1] * xgb_results['val_pred']
    test_pred = best_weights[0] * lgb_results['test_pred'] + best_weights[1] * xgb_results['test_pred']
    
    # Metrics
    train_auc = roc_auc_score(y_train, train_pred)
    val_auc = roc_auc_score(y_val, val_pred)
    test_auc = roc_auc_score(y_test, test_pred)
    
    print(f"\nEnsemble ROC-AUC:")
    print(f"  Train: {train_auc:.4f}")
    print(f"  Val:   {val_auc:.4f}")
    print(f"  Test:  {test_auc:.4f}")
    
    # Save weights
    weights_path = MODELS_DIR / "ensemble_weights.json"
    with open(weights_path, 'w') as f:
        json.dump({'lgb': best_weights[0], 'xgb': best_weights[1]}, f)
    
    return {
        'weights': best_weights,
        'train_pred': train_pred,
        'val_pred': val_pred,
        'test_pred': test_pred,
        'train_auc': train_auc,
        'val_auc': val_auc,
        'test_auc': test_auc,
    }


def main():
    """Main training pipeline."""
    print("="*80)
    print("IMPROVED MODEL TRAINING v2.0")
    print("="*80)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_data()
    
    # Train models
    lgb_results = train_lightgbm_optimized(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols)
    xgb_results = train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
    ensemble_results = create_ensemble(lgb_results, xgb_results, y_train, y_val, y_test)
    
    # Summary
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    summary = pd.DataFrame({
        'Model': ['LightGBM v2', 'XGBoost', 'Ensemble'],
        'Train_AUC': [
            lgb_results['train_auc'],
            xgb_results['train_auc'],
            ensemble_results['train_auc']
        ],
        'Val_AUC': [
            lgb_results['val_auc'],
            xgb_results['val_auc'],
            ensemble_results['val_auc']
        ],
        'Test_AUC': [
            lgb_results['test_auc'],
            xgb_results['test_auc'],
            ensemble_results['test_auc']
        ],
    })
    
    print("\n", summary.to_string(index=False))
    
    # Save summary
    summary_path = MODELS_DIR / "model_comparison_v2.csv"
    summary.to_csv(summary_path, index=False)
    
    # Save all results
    results = {
        'lgb': lgb_results,
        'xgb': xgb_results,
        'ensemble': ensemble_results,
    }
    
    results_path = MODELS_DIR / "training_results_v2.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✓ Results saved to {MODELS_DIR}")
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)
    
    return results


if __name__ == "__main__":
    main()


"""
Model training v3: With improved data quality.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import lightgbm as lgb
import xgboost as xgb

from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR, TARGET_COL, RANDOM_SEED


def load_data_v3():
    """Load v3 processed data."""
    print("Loading v3 data...")
    
    v3_dir = PROCESSED_DATA_DIR / "v3"
    
    train_df = pd.read_parquet(v3_dir / "train_features_v3.parquet")
    val_df = pd.read_parquet(v3_dir / "val_features_v3.parquet")
    test_df = pd.read_parquet(v3_dir / "test_features_v3.parquet")
    
    # Get feature columns (exclude identifiers and target)
    exclude_cols = ['user_session', 'user_id', 'target', 'session_start', 'session_end']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_COL].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_COL].values
    
    print(f"Train: {X_train.shape}, target: {y_train.mean():.2%}")
    print(f"Val: {X_val.shape}, target: {y_val.mean():.2%}")
    print(f"Test: {X_test.shape}, target: {y_test.mean():.2%}")
    print(f"Features: {len(feature_cols)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def train_lightgbm_v3(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols):
    """Train LightGBM v3."""
    print("\n" + "="*80)
    print("LIGHTGBM v3")
    print("="*80)
    
    params = {
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
        lgb.log_evaluation(period=100)
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
    print(f"  Test:  {test_auc:.4f} ⭐")
    
    # Save
    model_path = MODELS_DIR / "lightgbm_v3.txt"
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


def train_xgboost_v3(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train XGBoost v3."""
    print("\n" + "="*80)
    print("XGBOOST v3")
    print("="*80)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 9,
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
        verbose_eval=100
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
    print(f"  Test:  {test_auc:.4f} ⭐")
    
    # Save
    model_path = MODELS_DIR / "xgboost_v3.json"
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


def main():
    """Main training pipeline."""
    print("="*80)
    print("MODEL TRAINING v3.0 - WITH IMPROVED DATA")
    print("="*80)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_data_v3()
    
    # Train models
    lgb_results = train_lightgbm_v3(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols)
    xgb_results = train_xgboost_v3(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Ensemble
    print("\n" + "="*80)
    print("ENSEMBLE v3 (50/50)")
    print("="*80)
    
    ens_train = 0.5 * lgb_results['train_pred'] + 0.5 * xgb_results['train_pred']
    ens_val = 0.5 * lgb_results['val_pred'] + 0.5 * xgb_results['val_pred']
    ens_test = 0.5 * lgb_results['test_pred'] + 0.5 * xgb_results['test_pred']
    
    ens_train_auc = roc_auc_score(y_train, ens_train)
    ens_val_auc = roc_auc_score(y_val, ens_val)
    ens_test_auc = roc_auc_score(y_test, ens_test)
    
    print(f"ROC-AUC:")
    print(f"  Train: {ens_train_auc:.4f}")
    print(f"  Val:   {ens_val_auc:.4f}")
    print(f"  Test:  {ens_test_auc:.4f} ⭐")
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON - ALL VERSIONS")
    print("="*80)
    
    summary = pd.DataFrame({
        'Version': ['v1.0 LightGBM', 'v2.0 LightGBM', 'v2.0 XGBoost', 'v2.0 Ensemble',
                    'v3.0 LightGBM', 'v3.0 XGBoost', 'v3.0 Ensemble'],
        'Test_AUC': [0.5936, 0.6107, 0.6098, 0.6107,
                     lgb_results['test_auc'], xgb_results['test_auc'], ens_test_auc],
        'Val_AUC': [0.6492, 0.6596, 0.6578, 0.6593,
                    lgb_results['val_auc'], xgb_results['val_auc'], ens_val_auc],
    })
    
    summary['Improvement_vs_v1'] = ((summary['Test_AUC'] - 0.5936) / 0.5936 * 100).round(2)
    
    print("\n", summary.to_string(index=False))
    
    # Save results
    summary.to_csv(MODELS_DIR / "version_comparison_v3.csv", index=False)
    
    results = {
        'lgb': lgb_results,
        'xgb': xgb_results,
        'ensemble': {
            'train_pred': ens_train,
            'val_pred': ens_val,
            'test_pred': ens_test,
            'train_auc': ens_train_auc,
            'val_auc': ens_val_auc,
            'test_auc': ens_test_auc,
        }
    }
    
    with open(MODELS_DIR / "training_results_v3.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print("\n" + "="*80)
    print("✓ Training complete!")
    print("="*80)


if __name__ == "__main__":
    main()


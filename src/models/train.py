"""
Model training module: Baseline and advanced models.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
import json
sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score
)
import lightgbm as lgb

from src.utils.config import (
    TRAIN_PROCESSED, VAL_PROCESSED, TEST_PROCESSED,
    TARGET_COL, MODELS_DIR, RANDOM_SEED
)
from src.features.build import get_feature_columns


def load_data():
    """Load processed data."""
    print("Loading processed data...")
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
    
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Val: X={X_val.shape}, y={y_val.shape}")
    print(f"Test: X={X_test.shape}, y={y_test.shape}")
    print(f"Features: {len(feature_cols)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def evaluate_model(y_true, y_pred_proba, threshold=0.5):
    """
    Evaluate model performance.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        dict of metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    return metrics


def train_naive_baseline(y_train, y_val, y_test):
    """
    Naive baseline: Predict majority class.
    
    Returns:
        dict with predictions and metrics
    """
    print("\n" + "="*80)
    print("NAIVE BASELINE (Majority Class)")
    print("="*80)
    
    # Predict positive class probability = training set positive rate
    pos_rate = y_train.mean()
    print(f"Training positive rate: {pos_rate:.4f}")
    
    # Constant predictions
    y_train_pred = np.full_like(y_train, pos_rate, dtype=float)
    y_val_pred = np.full_like(y_val, pos_rate, dtype=float)
    y_test_pred = np.full_like(y_test, pos_rate, dtype=float)
    
    # Evaluate
    train_metrics = evaluate_model(y_train, y_train_pred)
    val_metrics = evaluate_model(y_val, y_val_pred)
    test_metrics = evaluate_model(y_test, y_test_pred)
    
    print("\nMetrics:")
    print(f"  Train - ROC-AUC: {train_metrics['roc_auc']:.4f}, PR-AUC: {train_metrics['pr_auc']:.4f}")
    print(f"  Val   - ROC-AUC: {val_metrics['roc_auc']:.4f}, PR-AUC: {val_metrics['pr_auc']:.4f}")
    print(f"  Test  - ROC-AUC: {test_metrics['roc_auc']:.4f}, PR-AUC: {test_metrics['pr_auc']:.4f}")
    
    return {
        'model_name': 'naive_baseline',
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
    }


def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train Logistic Regression baseline.
    
    Returns:
        dict with model, predictions, and metrics
    """
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION BASELINE")
    print("="*80)
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with class weighting
    print("Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict_proba(X_train_scaled)[:, 1]
    y_val_pred = model.predict_proba(X_val_scaled)[:, 1]
    y_test_pred = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate
    train_metrics = evaluate_model(y_train, y_train_pred)
    val_metrics = evaluate_model(y_val, y_val_pred)
    test_metrics = evaluate_model(y_test, y_test_pred)
    
    print("\nMetrics:")
    print(f"  Train - ROC-AUC: {train_metrics['roc_auc']:.4f}, PR-AUC: {train_metrics['pr_auc']:.4f}, F1: {train_metrics['f1']:.4f}")
    print(f"  Val   - ROC-AUC: {val_metrics['roc_auc']:.4f}, PR-AUC: {val_metrics['pr_auc']:.4f}, F1: {val_metrics['f1']:.4f}")
    print(f"  Test  - ROC-AUC: {test_metrics['roc_auc']:.4f}, PR-AUC: {test_metrics['pr_auc']:.4f}, F1: {test_metrics['f1']:.4f}")
    
    # Save model
    model_path = MODELS_DIR / "logistic_regression.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\nModel saved to: {model_path}")
    
    return {
        'model_name': 'logistic_regression',
        'model': model,
        'scaler': scaler,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred,
    }


def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols):
    """
    Train LightGBM model.
    
    Returns:
        dict with model, predictions, and metrics
    """
    print("\n" + "="*80)
    print("LIGHTGBM MODEL")
    print("="*80)
    
    # Create datasets
    print("Creating LightGBM datasets...")
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_cols)
    
    # Parameters
    params = {
        'objective': 'binary',
        'metric': ['auc', 'binary_logloss'],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': -1,
        'min_child_samples': 20,
        'scale_pos_weight': 1.0,  # Can adjust for imbalance
        'verbose': -1,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
    }
    
    # Train with early stopping
    print("Training LightGBM...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100)
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )
    
    print(f"\nBest iteration: {model.best_iteration}")
    
    # Predictions
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Evaluate
    train_metrics = evaluate_model(y_train, y_train_pred)
    val_metrics = evaluate_model(y_val, y_val_pred)
    test_metrics = evaluate_model(y_test, y_test_pred)
    
    print("\nMetrics:")
    print(f"  Train - ROC-AUC: {train_metrics['roc_auc']:.4f}, PR-AUC: {train_metrics['pr_auc']:.4f}, F1: {train_metrics['f1']:.4f}")
    print(f"  Val   - ROC-AUC: {val_metrics['roc_auc']:.4f}, PR-AUC: {val_metrics['pr_auc']:.4f}, F1: {val_metrics['f1']:.4f}")
    print(f"  Test  - ROC-AUC: {test_metrics['roc_auc']:.4f}, PR-AUC: {test_metrics['pr_auc']:.4f}, F1: {test_metrics['f1']:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 features:")
    print(feature_importance.head(20).to_string(index=False))
    
    # Save model
    model_path = MODELS_DIR / "lightgbm_model.txt"
    model.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    # Save feature importance
    fi_path = MODELS_DIR / "feature_importance.csv"
    feature_importance.to_csv(fi_path, index=False)
    
    return {
        'model_name': 'lightgbm',
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred,
        'feature_importance': feature_importance,
    }


def main():
    """Main training pipeline."""
    print("="*80)
    print("MODEL TRAINING PIPELINE")
    print("="*80)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_data()
    
    # Train models
    results = {}
    
    # 1. Naive baseline
    results['naive'] = train_naive_baseline(y_train, y_val, y_test)
    
    # 2. Logistic Regression
    results['logreg'] = train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # 3. LightGBM
    results['lgbm'] = train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols)
    
    # Summary table
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    summary = []
    for model_name, result in results.items():
        summary.append({
            'Model': result['model_name'],
            'Val_ROC_AUC': result['val_metrics']['roc_auc'],
            'Val_PR_AUC': result['val_metrics']['pr_auc'],
            'Val_F1': result['val_metrics']['f1'],
            'Test_ROC_AUC': result['test_metrics']['roc_auc'],
            'Test_PR_AUC': result['test_metrics']['pr_auc'],
            'Test_F1': result['test_metrics']['f1'],
        })
    
    summary_df = pd.DataFrame(summary)
    print("\n", summary_df.to_string(index=False))
    
    # Save summary
    summary_path = MODELS_DIR / "model_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    
    # Save all results
    results_path = MODELS_DIR / "training_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)


if __name__ == "__main__":
    main()


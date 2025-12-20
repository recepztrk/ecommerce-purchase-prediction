"""
Evaluation module: Detailed model evaluation, threshold optimization, error analysis.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report, roc_auc_score, average_precision_score
)

from src.utils.config import (
    MODELS_DIR, REPORTS_DIR, VAL_PROCESSED, TEST_PROCESSED, TARGET_COL
)
from src.features.build import get_feature_columns


def load_results():
    """Load training results."""
    results_path = MODELS_DIR / "training_results.pkl"
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results


def load_val_test_data():
    """Load validation and test data."""
    val_df = pd.read_parquet(VAL_PROCESSED)
    test_df = pd.read_parquet(TEST_PROCESSED)
    
    feature_cols = get_feature_columns(val_df)
    
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_COL].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_COL].values
    
    return X_val, y_val, X_test, y_test, val_df, test_df


def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: 'f1', 'precision', or 'recall'
    
    Returns:
        optimal threshold, best score
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    thresholds = np.linspace(0.1, 0.9, 81)
    scores = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        scores.append(score)
    
    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]


def plot_roc_pr_curves(results, y_val, y_test):
    """
    Plot ROC and PR curves for all models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ROC curve - Validation
    ax = axes[0, 0]
    for model_name, result in results.items():
        if 'y_val_pred' in result:
            fpr, tpr, _ = roc_curve(y_val, result['y_val_pred'])
            auc = roc_auc_score(y_val, result['y_val_pred'])
            ax.plot(fpr, tpr, label=f"{result['model_name']} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Validation Set')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ROC curve - Test
    ax = axes[0, 1]
    for model_name, result in results.items():
        if 'y_test_pred' in result:
            fpr, tpr, _ = roc_curve(y_test, result['y_test_pred'])
            auc = roc_auc_score(y_test, result['y_test_pred'])
            ax.plot(fpr, tpr, label=f"{result['model_name']} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Test Set')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PR curve - Validation
    ax = axes[1, 0]
    baseline = y_val.mean()
    ax.axhline(baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')
    for model_name, result in results.items():
        if 'y_val_pred' in result:
            precision, recall, _ = precision_recall_curve(y_val, result['y_val_pred'])
            ap = average_precision_score(y_val, result['y_val_pred'])
            ax.plot(recall, precision, label=f"{result['model_name']} (AP={ap:.3f})")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve - Validation Set')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PR curve - Test
    ax = axes[1, 1]
    baseline = y_test.mean()
    ax.axhline(baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')
    for model_name, result in results.items():
        if 'y_test_pred' in result:
            precision, recall, _ = precision_recall_curve(y_test, result['y_test_pred'])
            ap = average_precision_score(y_test, result['y_test_pred'])
            ax.plot(recall, precision, label=f"{result['model_name']} (AP={ap:.3f})")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve - Test Set')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = REPORTS_DIR / "roc_pr_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved ROC/PR curves to: {save_path}")
    plt.close()


def plot_confusion_matrices(results, y_val, y_test):
    """
    Plot confusion matrices for best model.
    """
    # Use LightGBM results
    lgbm_result = results['lgbm']
    
    # Find optimal threshold on validation set
    opt_thresh, opt_f1 = find_optimal_threshold(y_val, lgbm_result['y_val_pred'], metric='f1')
    print(f"\nOptimal threshold (F1): {opt_thresh:.3f} (F1={opt_f1:.3f})")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (y_true, y_pred_proba, split_name) in enumerate([
        (y_val, lgbm_result['y_val_pred'], 'Validation'),
        (y_test, lgbm_result['y_test_pred'], 'Test')
    ]):
        y_pred = (y_pred_proba >= opt_thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {split_name} (thresh={opt_thresh:.2f})')
        ax.set_xticklabels(['No Purchase', 'Purchase'])
        ax.set_yticklabels(['No Purchase', 'Purchase'])
    
    plt.tight_layout()
    save_path = REPORTS_DIR / "confusion_matrices.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrices to: {save_path}")
    plt.close()
    
    return opt_thresh


def analyze_errors(y_true, y_pred_proba, df, threshold, split_name='Test'):
    """
    Analyze prediction errors.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        df: Original DataFrame with features
        threshold: Classification threshold
        split_name: Name of the split
    """
    print(f"\n{'='*80}")
    print(f"ERROR ANALYSIS - {split_name} Set")
    print(f"{'='*80}")
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Create error DataFrame
    error_df = df.copy()
    error_df['y_true'] = y_true
    error_df['y_pred'] = y_pred
    error_df['y_pred_proba'] = y_pred_proba
    error_df['is_correct'] = (y_true == y_pred).astype(int)
    error_df['error_type'] = 'Correct'
    error_df.loc[(y_true == 1) & (y_pred == 0), 'error_type'] = 'False Negative'
    error_df.loc[(y_true == 0) & (y_pred == 1), 'error_type'] = 'False Positive'
    
    # Error distribution
    print("\nError distribution:")
    print(error_df['error_type'].value_counts())
    print(f"\nAccuracy: {error_df['is_correct'].mean():.4f}")
    
    # False Negatives analysis (missed purchases)
    fn_df = error_df[error_df['error_type'] == 'False Negative']
    if len(fn_df) > 0:
        print(f"\nFalse Negatives (Missed Purchases): {len(fn_df)}")
        print("Characteristics:")
        print(f"  Avg events: {fn_df['n_events'].mean():.2f} (vs {error_df['n_events'].mean():.2f} overall)")
        print(f"  Avg price: {fn_df['price_mean'].mean():.2f} (vs {error_df['price_mean'].mean():.2f} overall)")
        print(f"  Avg duration: {fn_df['session_duration_seconds'].mean():.2f}s (vs {error_df['session_duration_seconds'].mean():.2f}s overall)")
        print(f"  Avg prediction prob: {fn_df['y_pred_proba'].mean():.3f}")
    
    # False Positives analysis (false alarms)
    fp_df = error_df[error_df['error_type'] == 'False Positive']
    if len(fp_df) > 0:
        print(f"\nFalse Positives (False Alarms): {len(fp_df)}")
        print("Characteristics:")
        print(f"  Avg events: {fp_df['n_events'].mean():.2f} (vs {error_df['n_events'].mean():.2f} overall)")
        print(f"  Avg price: {fp_df['price_mean'].mean():.2f} (vs {error_df['price_mean'].mean():.2f} overall)")
        print(f"  Avg duration: {fp_df['session_duration_seconds'].mean():.2f}s (vs {error_df['session_duration_seconds'].mean():.2f}s overall)")
        print(f"  Avg prediction prob: {fp_df['y_pred_proba'].mean():.3f}")
    
    # Save error analysis
    error_summary = error_df.groupby('error_type').agg({
        'n_events': ['mean', 'std'],
        'price_mean': ['mean', 'std'],
        'session_duration_seconds': ['mean', 'std'],
        'y_pred_proba': ['mean', 'std'],
    }).round(3)
    
    save_path = REPORTS_DIR / f"error_analysis_{split_name.lower()}.csv"
    error_summary.to_csv(save_path)
    print(f"\nError analysis saved to: {save_path}")


def plot_feature_importance(results):
    """
    Plot feature importance for LightGBM.
    """
    lgbm_result = results['lgbm']
    fi = lgbm_result['feature_importance'].head(20)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(fi)), fi['importance'])
    plt.yticks(range(len(fi)), fi['feature'])
    plt.xlabel('Importance (Gain)')
    plt.title('Top 20 Feature Importance - LightGBM')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    save_path = REPORTS_DIR / "feature_importance.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved feature importance plot to: {save_path}")
    plt.close()


def main():
    """Main evaluation pipeline."""
    print("="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Load results and data
    results = load_results()
    X_val, y_val, X_test, y_test, val_df, test_df = load_val_test_data()
    
    # Plot ROC and PR curves
    print("\nGenerating ROC and PR curves...")
    plot_roc_pr_curves(results, y_val, y_test)
    
    # Plot confusion matrices and find optimal threshold
    print("\nGenerating confusion matrices...")
    opt_thresh = plot_confusion_matrices(results, y_val, y_test)
    
    # Error analysis
    analyze_errors(y_test, results['lgbm']['y_test_pred'], test_df, opt_thresh, 'Test')
    analyze_errors(y_val, results['lgbm']['y_val_pred'], val_df, opt_thresh, 'Validation')
    
    # Feature importance
    print("\nGenerating feature importance plot...")
    plot_feature_importance(results)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print("\nBest Model: LightGBM")
    print(f"Optimal Threshold: {opt_thresh:.3f}")
    print("\nTest Set Performance:")
    lgbm_metrics = results['lgbm']['test_metrics']
    print(f"  ROC-AUC: {lgbm_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:  {lgbm_metrics['pr_auc']:.4f}")
    print(f"  F1 (default 0.5): {lgbm_metrics['f1']:.4f}")
    
    # Recalculate with optimal threshold
    y_pred_opt = (results['lgbm']['y_test_pred'] >= opt_thresh).astype(int)
    from sklearn.metrics import f1_score, precision_score, recall_score
    print(f"  F1 (optimal {opt_thresh:.2f}): {f1_score(y_test, y_pred_opt):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_opt):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_opt):.4f}")
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()


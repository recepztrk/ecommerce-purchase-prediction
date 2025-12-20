"""
Feature engineering module: Build additional features from session data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import (
    TRAIN_PROCESSED, VAL_PROCESSED, TEST_PROCESSED,
    EXCLUDE_COLS, TARGET_COL, RANDOM_SEED
)


def create_advanced_features(df, train_stats=None, is_train=True):
    """
    Create advanced features from session data.
    
    Args:
        df: Session-level DataFrame
        train_stats: Statistics from training set (for validation/test)
        is_train: Whether this is training data
    
    Returns:
        DataFrame with additional features, train_stats dict
    """
    df = df.copy()
    
    # 1. Price-based features
    df['price_per_event'] = df['price_sum'] / (df['n_events'] + 1)
    df['price_range'] = df['price_max'] - df['price_min']
    df['price_cv'] = df['price_std'] / (df['price_mean'] + 1)  # Coefficient of variation
    
    # 2. Diversity features
    df['product_diversity'] = df['n_unique_products'] / (df['n_events'] + 1)
    df['brand_diversity'] = df['n_unique_brands'] / (df['n_events'] + 1)
    df['category_diversity'] = (
        df['cat_0_nunique'] + df['cat_1_nunique'] + 
        df['cat_2_nunique'] + df['cat_3_nunique']
    ) / 4.0
    
    # 3. Time-based features
    df['hour_range'] = df['ts_hour_max'] - df['ts_hour_min']
    df['is_night_session'] = ((df['ts_hour_mean'] >= 22) | (df['ts_hour_mean'] <= 6)).astype(int)
    df['is_weekend'] = (df['ts_weekday_mean'] >= 5).astype(int)
    
    # 4. Session intensity features
    df['events_per_minute'] = df['n_events'] / (df['session_duration_seconds'] / 60 + 1)
    df['is_quick_session'] = (df['session_duration_seconds'] < 60).astype(int)
    df['is_long_session'] = (df['session_duration_seconds'] > 600).astype(int)
    
    # 5. Behavioral patterns
    df['high_event_count'] = (df['n_events'] > 10).astype(int)
    df['single_product_focus'] = (df['n_unique_products'] == 1).astype(int)
    
    # 6. Statistical features
    if is_train:
        # Calculate statistics from training data
        train_stats = {
            'price_mean_global': df['price_mean'].mean(),
            'price_std_global': df['price_mean'].std(),
            'n_events_mean': df['n_events'].mean(),
            'n_events_std': df['n_events'].std(),
        }
    
    # Normalize features using train statistics
    if train_stats is not None:
        df['price_vs_global'] = (
            df['price_mean'] - train_stats['price_mean_global']
        ) / (train_stats['price_std_global'] + 1)
        
        df['events_vs_global'] = (
            df['n_events'] - train_stats['n_events_mean']
        ) / (train_stats['n_events_std'] + 1)
    
    return df, train_stats


def get_feature_columns(df):
    """
    Get list of feature columns (excluding target and identifiers).
    
    Args:
        df: DataFrame
    
    Returns:
        List of feature column names
    """
    exclude = set(EXCLUDE_COLS)
    feature_cols = [col for col in df.columns if col not in exclude]
    return feature_cols


def prepare_features(save=True):
    """
    Prepare features for all splits.
    
    Args:
        save: Whether to save processed data
    
    Returns:
        dict with train, val, test DataFrames and feature_cols
    """
    print("="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    # Load session data
    print("\nLoading session data...")
    train_df = pd.read_parquet(TRAIN_PROCESSED)
    val_df = pd.read_parquet(VAL_PROCESSED)
    test_df = pd.read_parquet(TEST_PROCESSED)
    
    print(f"Train: {train_df.shape}")
    print(f"Val: {val_df.shape}")
    print(f"Test: {test_df.shape}")
    
    # Create features
    print("\nCreating advanced features...")
    train_df, train_stats = create_advanced_features(train_df, is_train=True)
    val_df, _ = create_advanced_features(val_df, train_stats=train_stats, is_train=False)
    test_df, _ = create_advanced_features(test_df, train_stats=train_stats, is_train=False)
    
    print(f"Train after feature engineering: {train_df.shape}")
    print(f"Val after feature engineering: {val_df.shape}")
    print(f"Test after feature engineering: {test_df.shape}")
    
    # Get feature columns
    feature_cols = get_feature_columns(train_df)
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols[:10]}... (showing first 10)")
    
    # Handle infinite values
    for df in [train_df, val_df, test_df]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Save
    if save:
        train_df.to_parquet(TRAIN_PROCESSED, index=False)
        val_df.to_parquet(VAL_PROCESSED, index=False)
        test_df.to_parquet(TEST_PROCESSED, index=False)
        print(f"\nSaved processed data with features.")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'feature_cols': feature_cols,
        'train_stats': train_stats
    }


def main():
    """Main execution function."""
    result = prepare_features(save=True)
    
    print("\n" + "="*80)
    print("Feature engineering complete!")
    print("="*80)
    
    # Show feature importance preview
    train_df = result['train']
    feature_cols = result['feature_cols']
    
    print(f"\nFeature statistics (train set):")
    print(train_df[feature_cols].describe().T[['mean', 'std', 'min', 'max']].head(15))


if __name__ == "__main__":
    main()


"""
Data preparation module: Load event-level data and aggregate to session-level.
Implements memory-efficient processing and leakage prevention.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import (
    TRAIN_FILE, VAL_FILE, TEST_FILE,
    TRAIN_PROCESSED, VAL_PROCESSED, TEST_PROCESSED,
    RANDOM_SEED
)


def load_parquet_memory_safe(filepath, columns=None):
    """
    Load parquet file with memory optimization.
    
    Args:
        filepath: Path to parquet file
        columns: List of columns to load (None = all)
    
    Returns:
        DataFrame with optimized dtypes
    """
    df = pd.read_parquet(filepath, columns=columns, engine='pyarrow')
    
    # Optimize dtypes
    for col in df.select_dtypes(include=['object']).columns:
        if col not in ['event_time', 'timestamp']:
            df[col] = df[col].astype('category')
    
    # Downcast numeric columns
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df


def aggregate_to_session_level(df, verbose=True):
    """
    Aggregate event-level data to session-level.
    
    Key transformations:
    - Target: 1 if session contains any 'purchase' event, 0 otherwise
    - Features: Session-level aggregations (NO event_type used as feature)
    
    Args:
        df: Event-level DataFrame
        verbose: Print progress info
    
    Returns:
        Session-level DataFrame
    """
    if verbose:
        print(f"Input shape: {df.shape}")
        print(f"Unique sessions: {df['user_session'].nunique()}")
        print(f"Event type distribution:\n{df['event_type'].value_counts()}")
    
    # Create target: session has purchase event
    session_target = df.groupby('user_session')['event_type'].apply(
        lambda x: 1 if 'purchase' in x.values else 0
    ).reset_index()
    session_target.columns = ['user_session', 'target']
    
    if verbose:
        print(f"\nSession-level target distribution:")
        print(session_target['target'].value_counts())
        print(f"Positive rate: {session_target['target'].mean():.4f}")
    
    # Drop null sessions (if any)
    df = df[df['user_session'].notna()].copy()
    
    # Basic session info (first event in session)
    session_info = df.groupby('user_session').first()[['user_id']].reset_index()
    
    # Aggregate features (WITHOUT using event_type)
    agg_dict = {
        'product_id': ['count', 'nunique'],  # Total events, unique products
        'price': ['mean', 'std', 'min', 'max', 'sum'],  # Price statistics
        'brand': 'nunique',  # Unique brands
        'cat_0': 'nunique',
        'cat_1': 'nunique',
        'cat_2': 'nunique',
        'cat_3': 'nunique',
        'ts_hour': ['mean', 'std', 'min', 'max'],  # Time features
        'ts_weekday': ['mean', 'min', 'max'],
        'ts_day': ['mean'],
        'ts_month': ['mean'],
    }
    
    # Convert price to numeric if it's not
    if df['price'].dtype == 'object' or df['price'].dtype.name == 'category':
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    session_agg = df.groupby('user_session').agg(agg_dict)
    
    # Flatten column names
    session_agg.columns = ['_'.join(col).strip() if col[1] else col[0] 
                           for col in session_agg.columns.values]
    session_agg = session_agg.reset_index()
    
    # Rename for clarity
    rename_dict = {
        'product_id_count': 'n_events',
        'product_id_nunique': 'n_unique_products',
        'brand_nunique': 'n_unique_brands',
    }
    session_agg.rename(columns=rename_dict, inplace=True)
    
    # Time-based features: session duration
    time_stats = df.groupby('user_session')['timestamp'].agg(['min', 'max']).reset_index()
    time_stats['session_duration_seconds'] = (
        time_stats['max'] - time_stats['min']
    ).dt.total_seconds()
    time_stats = time_stats[['user_session', 'session_duration_seconds']]
    
    # Event rate (events per second)
    session_agg = session_agg.merge(time_stats, on='user_session', how='left')
    session_agg['event_rate'] = session_agg['n_events'] / (
        session_agg['session_duration_seconds'] + 1  # Avoid division by zero
    )
    
    # Merge all
    session_df = session_info.merge(session_agg, on='user_session', how='left')
    session_df = session_df.merge(session_target, on='user_session', how='left')
    
    # Fill NaN values (only for numeric columns)
    numeric_cols = session_df.select_dtypes(include=[np.number]).columns
    session_df[numeric_cols] = session_df[numeric_cols].fillna(0)
    
    if verbose:
        print(f"\nOutput shape: {session_df.shape}")
        print(f"Memory usage: {session_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return session_df


def prepare_all_splits(save=True):
    """
    Prepare train, validation, and test splits.
    
    Args:
        save: Whether to save processed data
    
    Returns:
        dict with 'train', 'val', 'test' DataFrames
    """
    splits = {}
    
    for split_name, filepath, output_path in [
        ('train', TRAIN_FILE, TRAIN_PROCESSED),
        ('val', VAL_FILE, VAL_PROCESSED),
        ('test', TEST_FILE, TEST_PROCESSED)
    ]:
        print(f"\n{'='*80}")
        print(f"Processing {split_name.upper()} split")
        print(f"{'='*80}")
        
        # Load event-level data
        df_events = load_parquet_memory_safe(filepath)
        
        # Aggregate to session-level
        df_sessions = aggregate_to_session_level(df_events, verbose=True)
        
        # Save
        if save:
            df_sessions.to_parquet(output_path, index=False, engine='pyarrow')
            print(f"Saved to: {output_path}")
        
        splits[split_name] = df_sessions
        
        # Clear memory
        del df_events
    
    return splits


def validate_splits(splits):
    """
    Validate that splits don't have overlapping sessions (leakage check).
    
    Args:
        splits: dict with 'train', 'val', 'test' DataFrames
    """
    print(f"\n{'='*80}")
    print("SPLIT VALIDATION (Leakage Check)")
    print(f"{'='*80}")
    
    train_sessions = set(splits['train']['user_session'])
    val_sessions = set(splits['val']['user_session'])
    test_sessions = set(splits['test']['user_session'])
    
    train_val_overlap = train_sessions & val_sessions
    train_test_overlap = train_sessions & test_sessions
    val_test_overlap = val_sessions & test_sessions
    
    print(f"Train sessions: {len(train_sessions)}")
    print(f"Val sessions: {len(val_sessions)}")
    print(f"Test sessions: {len(test_sessions)}")
    print(f"\nOverlap check:")
    print(f"  Train-Val overlap: {len(train_val_overlap)} sessions")
    print(f"  Train-Test overlap: {len(train_test_overlap)} sessions")
    print(f"  Val-Test overlap: {len(val_test_overlap)} sessions")
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("\n⚠️  WARNING: Session overlap detected! Potential leakage.")
    else:
        print("\n✓ No session overlap. Splits are clean.")
    
    # Target distribution
    print(f"\nTarget distribution:")
    for split_name in ['train', 'val', 'test']:
        target_mean = splits[split_name]['target'].mean()
        print(f"  {split_name}: {target_mean:.4f}")


def main():
    """Main execution function."""
    print("E-Commerce Purchase Prediction - Data Preparation")
    print("="*80)
    
    # Prepare all splits
    splits = prepare_all_splits(save=True)
    
    # Validate splits
    validate_splits(splits)
    
    print(f"\n{'='*80}")
    print("Data preparation complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()


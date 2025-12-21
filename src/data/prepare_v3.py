"""
Data preparation v3: Improved session handling and filtering.
Addresses key issues: single-event sessions, outliers, session merging.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import (
    TRAIN_FILE, VAL_FILE, TEST_FILE,
    PROCESSED_DATA_DIR, RANDOM_SEED
)


def merge_user_sessions(df, time_threshold_minutes=30):
    """
    Merge sessions from same user within time threshold.
    
    Args:
        df: Event-level DataFrame
        time_threshold_minutes: Max time gap to merge sessions
    
    Returns:
        DataFrame with merged sessions
    """
    print(f"Merging sessions within {time_threshold_minutes} minutes...")
    
    # Sort by user and time
    df = df.sort_values(['user_id', 'timestamp']).copy()
    
    # Calculate time gap between consecutive events for same user
    df['time_diff'] = df.groupby('user_id')['timestamp'].diff()
    df['time_diff_minutes'] = df['time_diff'].dt.total_seconds() / 60
    
    # Mark new session when gap > threshold or different user
    df['new_session'] = (
        (df['time_diff_minutes'] > time_threshold_minutes) | 
        (df['time_diff_minutes'].isna())
    ).astype(int)
    
    # Create new session ID
    df['merged_session_id'] = df.groupby('user_id')['new_session'].cumsum()
    df['merged_session'] = df['user_id'].astype(str) + '_' + df['merged_session_id'].astype(str)
    
    # Update user_session column
    original_sessions = df['user_session'].nunique()
    df['user_session'] = df['merged_session']
    merged_sessions = df['user_session'].nunique()
    
    print(f"  Original sessions: {original_sessions:,}")
    print(f"  Merged sessions: {merged_sessions:,}")
    print(f"  Reduction: {(1 - merged_sessions/original_sessions)*100:.1f}%")
    
    # Drop temporary columns
    df = df.drop(['time_diff', 'time_diff_minutes', 'new_session', 'merged_session_id', 'merged_session'], axis=1)
    
    return df


def filter_quality_sessions(df_sessions, min_events=2, max_duration_hours=2):
    """
    Filter out low-quality sessions.
    
    Args:
        df_sessions: Session-level DataFrame
        min_events: Minimum number of events
        max_duration_hours: Maximum session duration
    
    Returns:
        Filtered DataFrame
    """
    print(f"\nFiltering sessions...")
    print(f"  Original: {len(df_sessions):,} sessions")
    
    # Filter: minimum events
    mask_events = df_sessions['n_events'] >= min_events
    print(f"  After min_events>={min_events}: {mask_events.sum():,} ({mask_events.mean()*100:.1f}%)")
    
    # Filter: maximum duration (remove outliers)
    max_duration_seconds = max_duration_hours * 3600
    mask_duration = df_sessions['session_duration_seconds'] <= max_duration_seconds
    print(f"  After max_duration<={max_duration_hours}h: {mask_duration.sum():,} ({mask_duration.mean()*100:.1f}%)")
    
    # Combined filter
    mask_combined = mask_events & mask_duration
    df_filtered = df_sessions[mask_combined].copy()
    
    print(f"  Final: {len(df_filtered):,} sessions ({len(df_filtered)/len(df_sessions)*100:.1f}%)")
    print(f"  Removed: {len(df_sessions) - len(df_filtered):,} sessions")
    
    # Check target balance
    print(f"\n  Target distribution after filtering:")
    print(f"    Positive: {df_filtered['target'].mean()*100:.2f}%")
    
    return df_filtered


def create_session_aggregations(df_events):
    """
    Create rich session-level features from events.
    Improved version with better handling.
    """
    print("Creating session aggregations...")
    
    # Remove null sessions
    df_events = df_events[df_events['user_session'].notna()].copy()
    
    # Create target: session has purchase
    session_target = df_events.groupby('user_session', observed=True)['event_type'].apply(
        lambda x: 1 if 'purchase' in x.values else 0
    ).reset_index()
    session_target.columns = ['user_session', 'target']
    
    # Basic info
    session_info = df_events.groupby('user_session', observed=True).agg({
        'user_id': 'first',
        'timestamp': ['min', 'max'],
    }).reset_index()
    session_info.columns = ['user_session', 'user_id', 'session_start', 'session_end']
    session_info['session_duration_seconds'] = (
        session_info['session_end'] - session_info['session_start']
    ).dt.total_seconds()
    
    # Convert price to numeric
    if df_events['price'].dtype == 'object' or df_events['price'].dtype.name == 'category':
        df_events['price'] = pd.to_numeric(df_events['price'], errors='coerce')
    
    # Aggregate features
    agg_dict = {
        'product_id': ['count', 'nunique'],
        'price': ['mean', 'std', 'min', 'max', 'sum'],
        'brand': 'nunique',
        'cat_0': 'nunique',
        'cat_1': 'nunique',
        'cat_2': 'nunique',
        'cat_3': 'nunique',
        'ts_hour': ['mean', 'std', 'min', 'max'],
        'ts_weekday': ['mean', 'min', 'max'],
        'ts_day': 'mean',
        'ts_month': 'mean',
    }
    
    session_agg = df_events.groupby('user_session', observed=True).agg(agg_dict)
    session_agg.columns = ['_'.join(col).strip() if col[1] else col[0] 
                           for col in session_agg.columns.values]
    session_agg = session_agg.reset_index()
    
    # Rename columns
    rename_dict = {
        'product_id_count': 'n_events',
        'product_id_nunique': 'n_unique_products',
        'brand_nunique': 'n_unique_brands',
    }
    session_agg.rename(columns=rename_dict, inplace=True)
    
    # Merge all
    session_df = session_info.merge(session_agg, on='user_session', how='left')
    session_df = session_df.merge(session_target, on='user_session', how='left')
    
    # Add derived features
    session_df['event_rate'] = session_df['n_events'] / (session_df['session_duration_seconds'] + 1)
    session_df['product_diversity'] = session_df['n_unique_products'] / session_df['n_events']
    
    # Fill NaN
    numeric_cols = session_df.select_dtypes(include=[np.number]).columns
    session_df[numeric_cols] = session_df[numeric_cols].fillna(0)
    session_df.replace([np.inf, -np.inf], 0, inplace=True)
    
    return session_df


def process_split_v3(event_file, output_file, split_name, 
                     merge_sessions=True, filter_sessions=True):
    """
    Process a single split with v3 improvements.
    """
    print(f"\n{'='*80}")
    print(f"Processing {split_name.upper()} - v3")
    print(f"{'='*80}")
    
    # Load event data
    print(f"Loading {event_file}...")
    df_events = pd.read_parquet(event_file, engine='pyarrow')
    print(f"  Events: {len(df_events):,}")
    print(f"  Sessions (original): {df_events['user_session'].nunique():,}")
    
    # Step 1: Merge sessions
    if merge_sessions:
        df_events = merge_user_sessions(df_events, time_threshold_minutes=30)
        print(f"  Sessions (after merge): {df_events['user_session'].nunique():,}")
    
    # Step 2: Create session aggregations
    df_sessions = create_session_aggregations(df_events)
    print(f"  Sessions created: {len(df_sessions):,}")
    
    # Step 3: Filter quality sessions
    if filter_sessions:
        df_sessions = filter_quality_sessions(
            df_sessions, 
            min_events=2,  # At least 2 events
            max_duration_hours=2  # Max 2 hours
        )
    
    # Step 4: Save
    df_sessions.to_parquet(output_file, index=False, engine='pyarrow')
    print(f"✓ Saved to {output_file}")
    print(f"  Final shape: {df_sessions.shape}")
    print(f"  Memory: {df_sessions.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df_sessions


def main():
    """Main execution."""
    print("="*80)
    print("DATA PREPARATION v3.0 - IMPROVED QUALITY")
    print("="*80)
    
    # Create output directory
    v3_dir = PROCESSED_DATA_DIR / "v3"
    v3_dir.mkdir(exist_ok=True)
    
    results = {}
    
    # Process each split
    for split_name, event_file in [
        ('train', TRAIN_FILE),
        ('val', VAL_FILE),
        ('test', TEST_FILE),
    ]:
        output_file = v3_dir / f"{split_name}_sessions_v3.parquet"
        
        df = process_split_v3(
            event_file=event_file,
            output_file=output_file,
            split_name=split_name,
            merge_sessions=True,
            filter_sessions=True
        )
        
        results[split_name] = df
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY - v3 vs v2 Comparison")
    print(f"{'='*80}")
    
    for split_name in ['train', 'val', 'test']:
        v2_file = PROCESSED_DATA_DIR / f"{split_name}_sessions.parquet"
        v3_file = v3_dir / f"{split_name}_sessions_v3.parquet"
        
        df_v2 = pd.read_parquet(v2_file)
        df_v3 = results[split_name]
        
        print(f"\n{split_name.upper()}:")
        print(f"  v2: {len(df_v2):,} sessions → v3: {len(df_v3):,} sessions")
        print(f"  Reduction: {(1 - len(df_v3)/len(df_v2))*100:.1f}%")
        print(f"  v2 target: {df_v2['target'].mean()*100:.2f}% → v3: {df_v3['target'].mean()*100:.2f}%")
    
    print(f"\n{'='*80}")
    print("v3 Data preparation complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()


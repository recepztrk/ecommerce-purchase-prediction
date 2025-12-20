"""
Advanced feature engineering module.
Creates sophisticated features from event sequences and patterns.
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


def create_sequence_features(df_events):
    """
    Create features from event sequences within sessions.
    
    Args:
        df_events: Event-level DataFrame
    
    Returns:
        Session-level DataFrame with sequence features
    """
    # Sort by session and time
    df_events = df_events.sort_values(['user_session', 'timestamp'])
    
    # Convert price to numeric
    if df_events['price'].dtype == 'object' or df_events['price'].dtype.name == 'category':
        df_events['price'] = pd.to_numeric(df_events['price'], errors='coerce')
    
    features = []
    
    for session_id, group in df_events.groupby('user_session', observed=True):
        if len(group) < 1:
            continue
            
        feature_dict = {'user_session': session_id}
        
        # Basic counts
        n_events = len(group)
        feature_dict['n_events_seq'] = n_events
        
        if n_events == 1:
            # Single event session - fill with defaults
            feature_dict.update({
                'first_to_last_time': 0,
                'time_acceleration': 0,
                'price_first': group.iloc[0]['price'],
                'price_last': group.iloc[0]['price'],
                'price_trend': 0,
                'price_volatility': 0,
                'category_switches': 0,
                'brand_switches': 0,
                'hour_first': group.iloc[0]['ts_hour'],
                'hour_last': group.iloc[0]['ts_hour'],
                'hour_consistency': 1.0,
                'time_gap_mean': 0,
                'time_gap_std': 0,
                'max_price_position': 0,
                'repeat_products': 0,
                'price_ascending': 0,
                'focus_score': 1.0,
                'exploration_score': 0,
                'decisiveness_score': 1.0,
            })
        else:
            # Multiple events - calculate sequence features
            
            # 1. Time-based features
            times = group['timestamp'].values
            time_diffs = np.diff(times).astype('timedelta64[s]').astype(float)
            
            feature_dict['first_to_last_time'] = (times[-1] - times[0]).astype('timedelta64[s]').astype(float)
            
            # Time acceleration (are events getting faster or slower?)
            if len(time_diffs) > 1:
                first_half_mean = np.mean(time_diffs[:len(time_diffs)//2])
                second_half_mean = np.mean(time_diffs[len(time_diffs)//2:])
                if first_half_mean > 0:
                    feature_dict['time_acceleration'] = (first_half_mean - second_half_mean) / first_half_mean
                else:
                    feature_dict['time_acceleration'] = 0
            else:
                feature_dict['time_acceleration'] = 0
            
            # 2. Price trajectory
            prices = group['price'].values
            feature_dict['price_first'] = prices[0]
            feature_dict['price_last'] = prices[-1]
            
            # Price trend (slope of prices over time)
            if len(prices) > 1:
                price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
                feature_dict['price_trend'] = price_trend
                feature_dict['price_volatility'] = np.std(prices) / (np.mean(prices) + 1)
            else:
                feature_dict['price_trend'] = 0
                feature_dict['price_volatility'] = 0
            
            # 3. Category and brand switches
            if 'cat_0' in group.columns:
                cat_switches = (group['cat_0'].values[1:] != group['cat_0'].values[:-1]).sum()
                feature_dict['category_switches'] = cat_switches
            else:
                feature_dict['category_switches'] = 0
            
            if 'brand' in group.columns:
                brand_switches = (group['brand'].values[1:] != group['brand'].values[:-1]).sum()
                feature_dict['brand_switches'] = brand_switches
            else:
                feature_dict['brand_switches'] = 0
            
            # 4. Temporal patterns
            hours = group['ts_hour'].values
            feature_dict['hour_first'] = hours[0]
            feature_dict['hour_last'] = hours[-1]
            feature_dict['hour_consistency'] = 1.0 - (np.std(hours) / 24.0)  # How consistent is the time?
            
            # Time gaps between events
            feature_dict['time_gap_mean'] = np.mean(time_diffs)
            feature_dict['time_gap_std'] = np.std(time_diffs) if len(time_diffs) > 1 else 0
            
            # 5. Product patterns
            # Position of max price
            max_price_idx = np.argmax(prices)
            feature_dict['max_price_position'] = max_price_idx / (len(prices) - 1) if len(prices) > 1 else 0
            
            # Repeat products
            product_ids = group['product_id'].values
            unique_products = len(set(product_ids))
            feature_dict['repeat_products'] = n_events - unique_products
            
            # Are prices ascending?
            feature_dict['price_ascending'] = int(np.all(prices[1:] >= prices[:-1]))
            
            # 6. Behavioral scores
            # Focus score: How focused on few products?
            feature_dict['focus_score'] = 1.0 - (unique_products / n_events)
            
            # Exploration score: How much switching?
            total_switches = feature_dict['category_switches'] + feature_dict['brand_switches']
            feature_dict['exploration_score'] = total_switches / max(n_events - 1, 1)
            
            # Decisiveness score: Fast decisions (short session with purchases)
            avg_time_per_event = feature_dict['first_to_last_time'] / max(n_events - 1, 1)
            feature_dict['decisiveness_score'] = 1.0 / (1.0 + avg_time_per_event / 60)  # Normalize by minute
        
        features.append(feature_dict)
    
    return pd.DataFrame(features)


def merge_advanced_features(session_df, advanced_df):
    """
    Merge advanced features with existing session features.
    
    Args:
        session_df: Existing session-level DataFrame
        advanced_df: Advanced features DataFrame
    
    Returns:
        Merged DataFrame
    """
    # Merge on user_session
    merged = session_df.merge(advanced_df, on='user_session', how='left')
    
    # Fill NaN with 0
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    merged[numeric_cols] = merged[numeric_cols].fillna(0)
    
    # Handle infinite values
    merged.replace([np.inf, -np.inf], 0, inplace=True)
    
    return merged


def build_advanced_features(save=True):
    """
    Build advanced features for all splits.
    
    Args:
        save: Whether to save processed data
    
    Returns:
        dict with train, val, test DataFrames
    """
    print("="*80)
    print("ADVANCED FEATURE ENGINEERING")
    print("="*80)
    
    results = {}
    
    for split_name, event_file, session_file in [
        ('train', TRAIN_FILE, TRAIN_PROCESSED),
        ('val', VAL_FILE, VAL_PROCESSED),
        ('test', TEST_FILE, TEST_PROCESSED)
    ]:
        print(f"\n{'='*80}")
        print(f"Processing {split_name.upper()} split")
        print(f"{'='*80}")
        
        # Load event-level data
        print(f"Loading event data from {event_file}...")
        df_events = pd.read_parquet(event_file, engine='pyarrow')
        print(f"  Events: {len(df_events):,}")
        
        # Load existing session data
        print(f"Loading session data from {session_file}...")
        df_sessions = pd.read_parquet(session_file, engine='pyarrow')
        print(f"  Sessions: {len(df_sessions):,}")
        print(f"  Existing features: {len(df_sessions.columns)}")
        
        # Create advanced features
        print("Creating advanced sequence features...")
        advanced_features = create_sequence_features(df_events)
        print(f"  New features: {len(advanced_features.columns) - 1}")  # -1 for user_session
        
        # Merge with existing features
        print("Merging features...")
        df_merged = merge_advanced_features(df_sessions, advanced_features)
        print(f"  Total features: {len(df_merged.columns)}")
        print(f"  Memory: {df_merged.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Save
        if save:
            df_merged.to_parquet(session_file, index=False, engine='pyarrow')
            print(f"✓ Saved to {session_file}")
        
        results[split_name] = df_merged
        
        # Clear memory
        del df_events
    
    print("\n" + "="*80)
    print("Advanced feature engineering complete!")
    print("="*80)
    
    # Show sample features
    print("\nNew features added:")
    new_cols = [col for col in results['train'].columns if col not in 
                ['user_session', 'user_id', 'target', 'n_events', 'n_unique_products']]
    for i, col in enumerate(new_cols[:20], 1):
        print(f"  {i:2d}. {col}")
    
    if len(new_cols) > 20:
        print(f"  ... and {len(new_cols) - 20} more")
    
    return results


def main():
    """Main execution function."""
    results = build_advanced_features(save=True)
    
    print("\n" + "="*80)
    print("Feature statistics (train set):")
    print("="*80)
    
    train_df = results['train']
    feature_cols = [col for col in train_df.columns if col not in 
                    ['user_session', 'user_id', 'target']]
    
    print(f"\nTotal features: {len(feature_cols)}")
    print("\nSample statistics:")
    print(train_df[feature_cols].describe().T[['mean', 'std', 'min', 'max']].head(15))


if __name__ == "__main__":
    main()


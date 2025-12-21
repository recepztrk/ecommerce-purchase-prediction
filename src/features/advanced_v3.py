"""
Advanced feature engineering v3: Event-level, user, and product features.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import TRAIN_FILE, VAL_FILE, TEST_FILE, PROCESSED_DATA_DIR, RANDOM_SEED


def create_event_sequence_features(df_events):
    """
    Create rich features from event sequences.
    """
    print("Creating event sequence features...")
    
    df_events = df_events.sort_values(['user_session', 'timestamp'])
    
    # Convert price
    if df_events['price'].dtype == 'object' or df_events['price'].dtype.name == 'category':
        df_events['price'] = pd.to_numeric(df_events['price'], errors='coerce')
    
    features_list = []
    
    for session_id, group in df_events.groupby('user_session', observed=True):
        if len(group) < 2:
            continue
            
        feat = {'user_session': session_id}
        n = len(group)
        
        # Time-based features
        times = group['timestamp'].values
        time_diffs = np.diff(times).astype('timedelta64[s]').astype(float)
        
        feat['time_gap_mean'] = np.mean(time_diffs)
        feat['time_gap_std'] = np.std(time_diffs) if n > 2 else 0
        feat['time_gap_max'] = np.max(time_diffs)
        feat['time_gap_min'] = np.min(time_diffs)
        
        # Time acceleration
        if n > 3:
            first_half = time_diffs[:len(time_diffs)//2]
            second_half = time_diffs[len(time_diffs)//2:]
            feat['time_acceleration'] = np.mean(first_half) - np.mean(second_half)
        else:
            feat['time_acceleration'] = 0
        
        # Price trajectory
        prices = group['price'].values
        feat['price_first'] = prices[0]
        feat['price_last'] = prices[-1]
        feat['price_trend'] = np.polyfit(range(n), prices, 1)[0] if n > 1 else 0
        feat['price_volatility'] = np.std(prices) / (np.mean(prices) + 1)
        feat['price_range'] = np.max(prices) - np.min(prices)
        
        # Price patterns
        feat['price_increasing'] = int(np.all(prices[1:] >= prices[:-1]))
        feat['price_decreasing'] = int(np.all(prices[1:] <= prices[:-1]))
        
        # Product patterns
        products = group['product_id'].values
        feat['unique_products'] = len(set(products))
        feat['repeat_product_ratio'] = 1 - (feat['unique_products'] / n)
        
        # Most viewed product (frequency)
        product_counts = pd.Series(products).value_counts()
        feat['max_product_views'] = product_counts.iloc[0] if len(product_counts) > 0 else 1
        feat['max_product_ratio'] = feat['max_product_views'] / n
        
        # Category switches
        if 'cat_0' in group.columns:
            cat_switches = (group['cat_0'].values[1:] != group['cat_0'].values[:-1]).sum()
            feat['category_switches'] = cat_switches
            feat['category_switch_rate'] = cat_switches / (n - 1)
        else:
            feat['category_switches'] = 0
            feat['category_switch_rate'] = 0
        
        # Brand switches
        if 'brand' in group.columns:
            brand_switches = (group['brand'].values[1:] != group['brand'].values[:-1]).sum()
            feat['brand_switches'] = brand_switches
            feat['brand_switch_rate'] = brand_switches / (n - 1)
        else:
            feat['brand_switches'] = 0
            feat['brand_switch_rate'] = 0
        
        # Temporal patterns
        hours = group['ts_hour'].values
        feat['hour_first'] = hours[0]
        feat['hour_last'] = hours[-1]
        feat['hour_std'] = np.std(hours)
        feat['hour_range'] = np.max(hours) - np.min(hours)
        
        # Behavioral scores
        feat['focus_score'] = 1 - (feat['unique_products'] / n)  # How focused?
        feat['exploration_score'] = (feat['category_switches'] + feat['brand_switches']) / (n - 1)  # How much exploring?
        
        # Decision speed
        total_time = (times[-1] - times[0]).astype('timedelta64[s]').astype(float)
        feat['decision_speed'] = n / (total_time / 60 + 1)  # Events per minute
        
        features_list.append(feat)
    
    return pd.DataFrame(features_list)


def create_product_features(df_events, df_sessions_train=None, is_train=True):
    """
    Create product-level features (popularity, conversion rate, etc.)
    """
    print("Creating product features...")
    
    if is_train:
        # Calculate product statistics from training data
        product_stats = df_events.groupby('product_id').agg({
            'user_session': 'nunique',  # How many sessions viewed this product
            'event_type': lambda x: (x == 'purchase').sum(),  # How many purchases
        }).reset_index()
        product_stats.columns = ['product_id', 'product_popularity', 'product_purchases']
        product_stats['product_conversion_rate'] = product_stats['product_purchases'] / (product_stats['product_popularity'] + 1)
        
        # Brand statistics
        brand_stats = df_events.groupby('brand').agg({
            'user_session': 'nunique',
            'event_type': lambda x: (x == 'purchase').sum(),
        }).reset_index()
        brand_stats.columns = ['brand', 'brand_popularity', 'brand_purchases']
        brand_stats['brand_conversion_rate'] = brand_stats['brand_purchases'] / (brand_stats['brand_popularity'] + 1)
        
        # Category statistics
        cat_stats = df_events.groupby('cat_0').agg({
            'user_session': 'nunique',
            'event_type': lambda x: (x == 'purchase').sum(),
        }).reset_index()
        cat_stats.columns = ['cat_0', 'category_popularity', 'category_purchases']
        cat_stats['category_conversion_rate'] = cat_stats['category_purchases'] / (cat_stats['category_popularity'] + 1)
        
        stats_dict = {
            'product': product_stats,
            'brand': brand_stats,
            'category': cat_stats
        }
    else:
        stats_dict = None
    
    # Aggregate per session
    session_product_features = []
    
    for session_id, group in df_events.groupby('user_session', observed=True):
        feat = {'user_session': session_id}
        
        if is_train or stats_dict is not None:
            # Product features
            products = group['product_id'].values
            if is_train:
                product_pops = [product_stats[product_stats['product_id'] == p]['product_popularity'].values[0] 
                               if len(product_stats[product_stats['product_id'] == p]) > 0 else 0 
                               for p in products]
                product_convs = [product_stats[product_stats['product_id'] == p]['product_conversion_rate'].values[0] 
                                if len(product_stats[product_stats['product_id'] == p]) > 0 else 0 
                                for p in products]
            else:
                product_pops = [stats_dict['product'][stats_dict['product']['product_id'] == p]['product_popularity'].values[0] 
                               if len(stats_dict['product'][stats_dict['product']['product_id'] == p]) > 0 else 0 
                               for p in products]
                product_convs = [stats_dict['product'][stats_dict['product']['product_id'] == p]['product_conversion_rate'].values[0] 
                                if len(stats_dict['product'][stats_dict['product']['product_id'] == p]) > 0 else 0 
                                for p in products]
            
            feat['avg_product_popularity'] = np.mean(product_pops) if len(product_pops) > 0 else 0
            feat['max_product_popularity'] = np.max(product_pops) if len(product_pops) > 0 else 0
            feat['avg_product_conversion'] = np.mean(product_convs) if len(product_convs) > 0 else 0
            feat['max_product_conversion'] = np.max(product_convs) if len(product_convs) > 0 else 0
        else:
            feat['avg_product_popularity'] = 0
            feat['max_product_popularity'] = 0
            feat['avg_product_conversion'] = 0
            feat['max_product_conversion'] = 0
        
        session_product_features.append(feat)
    
    return pd.DataFrame(session_product_features), stats_dict if is_train else None


def build_features_v3(split_name, event_file, session_file, train_stats=None):
    """
    Build v3 features for a split.
    """
    print(f"\n{'='*80}")
    print(f"Building v3 features for {split_name.upper()}")
    print(f"{'='*80}")
    
    is_train = (split_name == 'train')
    
    # Load data
    print(f"Loading event data...")
    df_events = pd.read_parquet(event_file)
    print(f"  Events: {len(df_events):,}")
    
    print(f"Loading session data...")
    df_sessions = pd.read_parquet(session_file)
    print(f"  Sessions: {len(df_sessions):,}")
    print(f"  Existing features: {len(df_sessions.columns)}")
    
    # Create sequence features
    sequence_features = create_event_sequence_features(df_events)
    print(f"  Sequence features: {len(sequence_features.columns) - 1}")
    
    # Create product features
    product_features, new_train_stats = create_product_features(
        df_events, 
        df_sessions_train=df_sessions if is_train else None,
        is_train=is_train
    )
    print(f"  Product features: {len(product_features.columns) - 1}")
    
    if is_train:
        train_stats = new_train_stats
    
    # Merge features
    print("Merging features...")
    df_merged = df_sessions.merge(sequence_features, on='user_session', how='left')
    df_merged = df_merged.merge(product_features, on='user_session', how='left')
    
    # Fill NaN
    numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
    df_merged[numeric_cols] = df_merged[numeric_cols].fillna(0)
    df_merged.replace([np.inf, -np.inf], 0, inplace=True)
    
    print(f"  Total features: {len(df_merged.columns)}")
    print(f"  Memory: {df_merged.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df_merged, train_stats


def main():
    """Main execution."""
    print("="*80)
    print("ADVANCED FEATURE ENGINEERING v3.0")
    print("="*80)
    
    v3_dir = PROCESSED_DATA_DIR / "v3"
    train_stats = None
    
    for split_name, event_file in [
        ('train', TRAIN_FILE),
        ('val', VAL_FILE),
        ('test', TEST_FILE),
    ]:
        session_file = v3_dir / f"{split_name}_sessions_v3.parquet"
        output_file = v3_dir / f"{split_name}_features_v3.parquet"
        
        df_features, train_stats = build_features_v3(
            split_name=split_name,
            event_file=event_file,
            session_file=session_file,
            train_stats=train_stats
        )
        
        # Save
        df_features.to_parquet(output_file, index=False)
        print(f"✓ Saved to {output_file}\n")
    
    print("="*80)
    print("v3 Feature engineering complete!")
    print("="*80)


if __name__ == "__main__":
    main()


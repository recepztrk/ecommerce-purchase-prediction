"""
Phase 2: Feature Engineering - Cleanup and New Features

Based on Phase 1 analysis:
- Remove 7 noise features (<1% importance)
- Remove 5 redundant features (high correlation)
- Add 15 smart new features

Output: Optimized feature set (24 → 27 features)
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("PHASE 2: FEATURE ENGINEERING - CLEANUP & NEW FEATURES")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Features to REMOVE (from Phase 1 analysis)
NOISE_FEATURES = [
    'ts_hour_std',       # 0.00% importance
    'cat_3_nunique',     # 0.00% importance  
    'n_unique_brands',   # 0.05% importance
    'cat_0_nunique',     # 0.14% importance
    'ts_hour_min',       # 0.70% importance
    'ts_hour_max',       # 0.89% importance
    'ts_weekday_min',    # 0.93% importance
]

REDUNDANT_FEATURES = [
    'ts_weekday_max',    # redundant with ts_weekday_mean (1.00 corr)
    'price_min',         # redundant with price_mean (0.95 corr)
    'price_max',         # redundant with price_mean (0.95 corr)
    'cat_1_nunique',     # redundant with cat_2_nunique (0.88 corr)
    'price_sum',         # redundant (can derive from price_mean * n_events)
]

FEATURES_TO_REMOVE = NOISE_FEATURES + REDUNDANT_FEATURES

print(f"\nFeatures to remove: {len(FEATURES_TO_REMOVE)}")
print("  Noise features (7):")
for f in NOISE_FEATURES:
    print(f"    - {f}")
print("  Redundant features (5):")
for f in REDUNDANT_FEATURES:
    print(f"    - {f}")

# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def add_interaction_features(df):
    """Add interaction features between existing features."""
    print("\n  Adding interaction features...")
    
    # 1. Price per product
    df['price_per_product'] = df['price_mean'] / (df['n_unique_products'] + 1)
    
    # 2. Price-product interaction
    df['price_product_score'] = df['price_mean'] * df['n_unique_products']
    
    # 3. Time engagement
    df['time_engagement_score'] = df['session_duration_seconds'] * df['n_events']
    
    # 4. Events per minute (improved event_rate)
    df['events_per_minute'] = df['n_events'] / (df['session_duration_seconds'] / 60 + 1)
    
    print(f"    Added: 4 interaction features")
    return df


def add_behavioral_features(df):
    """Add behavioral pattern features."""
    print("\n  Adding behavioral features...")
    
    # 1. Focused shopper (few products, many views)
    df['is_focused_shopper'] = (
        (df['n_unique_products'] <= 3) & (df['n_events'] >= 5)
    ).astype(int)
    
    # 2. Window shopper (many products, long session)
    df['is_window_shopper'] = (
        (df['n_unique_products'] >= 10) & (df['session_duration_seconds'] >= 300)
    ).astype(int)
    
    # 3. Decisive shopper (few events, short session)
    df['is_decisive'] = (
        (df['n_events'] <= 5) & (df['session_duration_seconds'] <= 120)
    ).astype(int)
    
    # 4. Price sensitivity (normalized price std)
    df['price_sensitivity'] = df['price_std'] / (df['price_mean'] + 1)
    
    print(f"    Added: 4 behavioral features")
    return df


def add_temporal_features(df):
    """Add temporal intelligence features."""
    print("\n  Adding temporal features...")
    
    # 1. Weekend shopping
    df['is_weekend'] = (df['ts_weekday_mean'] >= 5).astype(int)
    
    # 2. Peak hour shopping (18:00-22:00)
    df['is_peak_hour'] = (
        (df['ts_hour_mean'] >= 18) & (df['ts_hour_mean'] <= 22)
    ).astype(int)
    
    # 3. Morning shopping (6:00-12:00)
    df['is_morning'] = (
        (df['ts_hour_mean'] >= 6) & (df['ts_hour_mean'] <= 12)
    ).astype(int)
    
    # 4. Late night shopping (22:00-6:00)
    df['is_late_night'] = (
        (df['ts_hour_mean'] >= 22) | (df['ts_hour_mean'] <= 6)
    ).astype(int)
    
    print(f"    Added: 4 temporal features")
    return df


def add_category_features(df):
    """Add category intelligence features."""
    print("\n  Adding category features...")
    
    # 1. Category consistency
    df['category_consistency'] = df['cat_2_nunique'] / (df['n_unique_products'] + 1)
    
    # 2. Category focused (single or few categories)
    df['is_category_focused'] = (df['cat_2_nunique'] <= 2).astype(int)
    
    # 3. Product revisit rate (1 - diversity)
    df['product_revisit_rate'] = 1 - df['product_diversity']
    
    print(f"    Added: 3 category features")
    return df


def validate_features(df):
    """Validate new features for NaN, inf issues."""
    print("\n  Validating features...")
    
    # Check for NaN
    nan_counts = df.isnull().sum()
    if nan_counts.sum() > 0:
        print(f"    Warning: Found {nan_counts.sum()} NaN values")
        # Fill NaN with 0
        df = df.fillna(0)
    
    # Check for inf
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(df[numeric_cols].values)
    if inf_mask.sum() > 0:
        print(f"    Warning: Found {inf_mask.sum()} inf values")
        # Replace inf with 0
        df = df.replace([np.inf, -np.inf], 0)
    
    print(f"    ✓ Validation complete")
    return df


def create_optimized_features(df_raw):
    """Main function: cleanup + add new features."""
    print(f"\nProcessing {len(df_raw):,} sessions...")
    
    # Step 1: Remove bad features
    print(f"\n1. Removing {len(FEATURES_TO_REMOVE)} features...")
    df = df_raw.drop(columns=FEATURES_TO_REMOVE, errors='ignore')
    print(f"   Columns before: {len(df_raw.columns)}")
    print(f"   Columns after:  {len(df.columns)}")
    
    # Step 2: Add new features
    print(f"\n2. Adding new features...")
    df = add_interaction_features(df)
    df = add_behavioral_features(df)
    df = add_temporal_features(df)
    df = add_category_features(df)
    
    # Step 3: Validate
    print(f"\n3. Validation...")
    df = validate_features(df)
    
    # Summary
    new_feature_count = len(df.columns) - len(df_raw.columns) + len(FEATURES_TO_REMOVE)
    print(f"\nFeature Engineering Summary:")
    print(f"  Original features: {len(df_raw.columns) - 5}")  # -5 for meta+target
    print(f"  Removed: {len(FEATURES_TO_REMOVE)}")
    print(f"  Added: {new_feature_count}")
    print(f"  Final features: {len(df.columns) - 5}")
    
    return df


# ============================================================================
# PROCESS ALL SPLITS
# ============================================================================

# Create output directory
Path('data/v4').mkdir(exist_ok=True)

for split_name in ['train', 'val', 'test']:
    print(f"\n{'='*80}")
    print(f"PROCESSING: {split_name.upper()}")
    print(f"{'='*80}")
    
    # Load
    input_file = f'data/v3/{split_name}_sessions_v3.parquet'
    df_raw = pd.read_parquet(input_file)
    
    # Engineer features
    df_optimized = create_optimized_features(df_raw)
    
    # Save
    output_file = f'data/v4/{split_name}_sessions_v4.parquet'
    df_optimized.to_parquet(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")
    print(f"  Size: {Path(output_file).stat().st_size / 1024**2:.1f} MB")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PHASE 2 COMPLETE: FEATURE ENGINEERING")
print("="*80)

# Load final train for summary
train_final = pd.read_parquet('data/v4/train_sessions_v4.parquet')
exclude_cols = ['user_session', 'user_id', 'session_start', 'session_end', 'target']
final_features = [c for c in train_final.columns if c not in exclude_cols]

print(f"\nFinal Feature Set ({len(final_features)} features):")
print("\nKept from v3.0:")
kept_features = [f for f in final_features if f not in [
    'price_per_product', 'price_product_score', 'time_engagement_score', 
    'events_per_minute', 'is_focused_shopper', 'is_window_shopper',
    'is_decisive', 'price_sensitivity', 'is_weekend', 'is_peak_hour',
    'is_morning', 'is_late_night', 'category_consistency', 'is_category_focused',
    'product_revisit_rate'
]]
for f in kept_features:
    print(f"  ✓ {f}")

print(f"\nNew Features ({15}):")
new_features = [
    'price_per_product', 'price_product_score', 'time_engagement_score',
    'events_per_minute', 'is_focused_shopper', 'is_window_shopper',
    'is_decisive', 'price_sensitivity', 'is_weekend', 'is_peak_hour',
    'is_morning', 'is_late_night', 'category_consistency', 'is_category_focused',
    'product_revisit_rate'
]
for f in new_features:
    if f in final_features:
        print(f"  + {f}")

print(f"\nFiles created:")
print(f"  ✓ data/v4/train_sessions_v4.parquet")
print(f"  ✓ data/v4/val_sessions_v4.parquet")
print(f"  ✓ data/v4/test_sessions_v4.parquet")

print(f"\nReady for Phase 3: Model Training!")
print("="*80)

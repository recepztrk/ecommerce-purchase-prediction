"""Configuration and constants for the project."""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "archive"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42

# Data files
TRAIN_FILE = DATA_DIR / "train.parquet"
VAL_FILE = DATA_DIR / "val.parquet"
TEST_FILE = DATA_DIR / "test.parquet"

# Processed data files
TRAIN_PROCESSED = PROCESSED_DATA_DIR / "train_sessions.parquet"
VAL_PROCESSED = PROCESSED_DATA_DIR / "val_sessions.parquet"
TEST_PROCESSED = PROCESSED_DATA_DIR / "test_sessions.parquet"

# Columns to exclude from features (leakage prevention)
EXCLUDE_COLS = ['event_type', 'target', 'user_session', 'user_id', 'event_time', 'timestamp']

# Target column
TARGET_COL = 'target'


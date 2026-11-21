
from pathlib import Path
import pandas as pd

# Dataset paths
DATA_DIR = Path("Dataset")

TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"

# Check whether files exist
if not TRAIN_FILE.exists() or not TEST_FILE.exists():
    raise FileNotFoundError("Training or testing dataset not found!")

# Load datasets
print("Loading datasets...")

train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
test_df = pd.read_csv(TEST_FILE, low_memory=False)

# Remove unwanted spaces from column names
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# Dataset dimensions
print("\n========== DATASET SHAPE ==========")
print("Training:", train_df.shape)
print("Testing :", test_df.shape)

# Display first five records
print("\n========== FIRST FIVE RECORDS ==========")
print(train_df.head())

# Column information
print("\n========== COLUMN INFORMATION ==========")
print(train_df.dtypes)

# Missing values
print("\n========== MISSING VALUES ==========")
print(train_df.isnull().sum())

# Duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print("Training:", train_df.duplicated().sum())
print("Testing :", test_df.duplicated().sum())

# Binary classification distribution
print("\n========== BINARY CLASS DISTRIBUTION ==========")
if "label" in train_df.columns:
    print(train_df["label"].value_counts())

# Multiclass classification distribution
print("\n========== ATTACK CATEGORY DISTRIBUTION ==========")
if "attack_cat" in train_df.columns:
    print(train_df["attack_cat"].value_counts())

print("\nDataset understanding completed!")

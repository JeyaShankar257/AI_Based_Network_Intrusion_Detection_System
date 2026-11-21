from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("Dataset")
PROCESSED_DIR = Path("dataset_processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"

# --------------------------------------------------
# 2. LOAD DATASETS
# --------------------------------------------------

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
test_df = pd.read_csv(TEST_FILE, low_memory=False)

# --------------------------------------------------
# 3. CLEAN COLUMN NAMES
# --------------------------------------------------

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# Verify target column
if "label" not in train_df.columns:
    raise ValueError("Target column 'label' not found in training data.")

if "label" not in test_df.columns:
    raise ValueError("Target column 'label' not found in testing data.")

print("\nOriginal training shape:", train_df.shape)
print("Original testing shape:", test_df.shape)

# --------------------------------------------------
# 4. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X_train = train_df.drop(columns=["label", "attack_cat", "id"], errors="ignore")

y_train = train_df["label"]

X_test = test_df.drop(columns=["label", "attack_cat", "id"], errors="ignore")

y_test = test_df["label"]

# --------------------------------------------------
# 5. ALIGN TRAINING AND TESTING FEATURES
# --------------------------------------------------

X_test = X_test.reindex(columns=X_train.columns)

# --------------------------------------------------
# 6. CLEAN FEATURE VALUES
# --------------------------------------------------

for column in X_train.select_dtypes(include=["object"]).columns:

    X_train[column] = X_train[column].str.strip()
    X_test[column] = X_test[column].str.strip()

# --------------------------------------------------
# 7. CHECK TARGET VALUES
# --------------------------------------------------

print("\n========== TARGET DISTRIBUTION ==========")

print("Training:")
print(y_train.value_counts())

print("\nTesting:")
print(y_test.value_counts())

# --------------------------------------------------
# 8. SAVE PREPARED DATASETS
# --------------------------------------------------

X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)

y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

# --------------------------------------------------
# 9. FINAL SUMMARY
# --------------------------------------------------

print("\n========== PREPROCESSING SUMMARY ==========")

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

print("\nCategorical features:")
print(X_train.select_dtypes(include=["object"]).columns.tolist())

print("\nNumerical features:")
print(X_train.select_dtypes(include=["number"]).columns.tolist())

print("\nMissing values in training features:")
print(X_train.isnull().sum()[X_train.isnull().sum() > 0])

print("\nPrepared datasets saved successfully!")

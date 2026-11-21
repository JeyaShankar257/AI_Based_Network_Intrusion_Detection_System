from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("Dataset")
REPORT_DIR = Path("reports/figures")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("Loading training dataset...")

df = pd.read_csv(TRAIN_FILE, low_memory=False)

df.columns = df.columns.str.strip()

print("\n========== DATASET OVERVIEW ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

# --------------------------------------------------
# 3. MISSING VALUES
# --------------------------------------------------

print("\n========== MISSING VALUES ==========")

missing = df.isnull().sum()
missing = missing[missing > 0]

if missing.empty:
    print("No missing values found.")
else:
    print(missing)

# --------------------------------------------------
# 4. BINARY CLASS DISTRIBUTION
# --------------------------------------------------

if "label" in df.columns:

    print("\n========== BINARY CLASS DISTRIBUTION ==========")

    print(df["label"].value_counts())

    plt.figure(figsize=(7, 5))

    sns.countplot(data=df, x="label")

    plt.title("Normal vs Attack Traffic")
    plt.xlabel("Class (0 = Normal, 1 = Attack)")
    plt.ylabel("Number of Records")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "class_distribution.png")
    plt.show()

# --------------------------------------------------
# 5. ATTACK CATEGORY DISTRIBUTION
# --------------------------------------------------

if "attack_cat" in df.columns:

    print("\n========== ATTACK CATEGORY DISTRIBUTION ==========")

    print(df["attack_cat"].value_counts())

    plt.figure(figsize=(11, 6))

    sns.countplot(data=df, y="attack_cat", order=df["attack_cat"].value_counts().index)

    plt.title("Attack Category Distribution")
    plt.xlabel("Number of Records")
    plt.ylabel("Attack Category")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "attack_category_distribution.png")
    plt.show()

# --------------------------------------------------
# 6. NUMERICAL FEATURE DISTRIBUTION
# --------------------------------------------------

numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

# Exclude identifiers and target columns
numeric_features = [col for col in numeric_columns if col not in ["id", "label"]]

print("\n========== NUMERICAL FEATURES ==========")
print(numeric_features)

selected_features = numeric_features[:6]

if selected_features:

    df[selected_features].hist(figsize=(14, 9), bins=30)

    plt.suptitle("Numerical Feature Distributions")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "numerical_feature_distribution.png")
    plt.show()

# --------------------------------------------------
# 7. CATEGORICAL FEATURE ANALYSIS
# --------------------------------------------------

categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

print("\n========== CATEGORICAL FEATURES ==========")
print(categorical_columns)

for column in categorical_columns:

    print(f"\nTop values in {column}:")
    print(df[column].value_counts().head(10))

# --------------------------------------------------
# 8. CORRELATION ANALYSIS
# --------------------------------------------------

correlation_columns = [col for col in numeric_columns if col != "id"]

if len(correlation_columns) > 1:

    correlation = df[correlation_columns].corr()

    plt.figure(figsize=(16, 12))

    sns.heatmap(correlation, cmap="coolwarm", center=0, linewidths=0.2)

    plt.title("Numerical Feature Correlation")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "correlation_heatmap.png")
    plt.show()

# --------------------------------------------------
# 9. SAVE SUMMARY
# --------------------------------------------------

summary = df.describe(include="all").transpose()

summary.to_csv(REPORT_DIR / "dataset_summary.csv")

print("\n========== EDA COMPLETED ==========")
print("Charts saved in:", REPORT_DIR)

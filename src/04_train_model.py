from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("dataset_processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/figures")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD PREPROCESSED DATA
# --------------------------------------------------

print("Loading processed datasets...")

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# --------------------------------------------------
# 3. IDENTIFY FEATURE TYPES
# --------------------------------------------------

categorical_features = ["proto", "service", "state"]

numeric_features = [
    column for column in X_train.columns if column not in categorical_features
]

print("\nCategorical features:", categorical_features)
print("Numerical feature count:", len(numeric_features))

# --------------------------------------------------
# 4. NUMERICAL PREPROCESSING
# --------------------------------------------------

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

# --------------------------------------------------
# 5. CATEGORICAL PREPROCESSING
# --------------------------------------------------

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# --------------------------------------------------
# 6. COMBINE PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ]
)

# --------------------------------------------------
# 7. CREATE RANDOM FOREST MODEL
# --------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
)

# --------------------------------------------------
# 8. CREATE COMPLETE PIPELINE
# --------------------------------------------------

model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", random_forest)])

# --------------------------------------------------
# 9. TRAIN MODEL
# --------------------------------------------------

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed!")

# --------------------------------------------------
# 10. MAKE PREDICTIONS
# --------------------------------------------------

print("\nGenerating predictions...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# 11. MODEL EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\n========== CLASSIFICATION REPORT ==========")

report = classification_report(
    y_test, y_pred, target_names=["Normal", "Attack"], zero_division=0
)

print(report)

# --------------------------------------------------
# 12. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Attack"],
    yticklabels=["Normal", "Attack"],
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()
plt.savefig(REPORT_DIR / "random_forest_confusion_matrix.png")
plt.show()

# --------------------------------------------------
# 13. SAVE MODEL
# --------------------------------------------------

model_path = MODEL_DIR / "random_forest_nids.joblib"

joblib.dump(model, model_path)

print("\nModel saved:", model_path)

# --------------------------------------------------
# 14. SAVE EVALUATION REPORT
# --------------------------------------------------

metrics = pd.DataFrame(
    {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        "Score": [accuracy, precision, recall, f1, roc_auc],
    }
)

metrics.to_csv(Path("reports") / "model_metrics.csv", index=False)

with open(Path("reports") / "classification_report.txt", "w") as file:
    file.write(report)

print("\nEvaluation reports saved.")
print("\nRandom Forest training and evaluation completed!")

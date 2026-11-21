from pathlib import Path
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

DATA_DIR = Path("dataset_processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

print("Loading test datasets...")

X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

print("Loading trained model...")

model = joblib.load(MODEL_DIR / "random_forest_nids.joblib")

with open(MODEL_DIR / "classification_threshold.json", "r") as file:
    threshold_config = json.load(file)

threshold = threshold_config["threshold"]

print(f"\nSelected threshold: {threshold}")

# Generate attack probabilities
print("\nGenerating predictions...")

y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate both thresholds
results = []

for current_threshold in [0.50, threshold]:

    y_pred = (y_prob >= current_threshold).astype(int)

    results.append(
        {
            "threshold": current_threshold,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }
    )

results_df = pd.DataFrame(results)

print("\n========== TEST PERFORMANCE COMPARISON ==========")
print(results_df.to_string(index=False))

# Save comparison
results_df.to_csv(REPORT_DIR / "threshold_test_comparison.csv", index=False)

# Final predictions using selected threshold
y_pred = (y_prob >= threshold).astype(int)

print("\n========== TUNED MODEL CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        zero_division=0,
    )
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\n========== TUNED MODEL CONFUSION MATRIX ==========")
print(cm)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Attack"],
    yticklabels=["Normal", "Attack"],
)

plt.title(f"Random Forest - Tuned Threshold ({threshold})")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(REPORT_DIR / "figures" / "tuned_confusion_matrix.png", dpi=300)

plt.close()

# Save classification report
with open(REPORT_DIR / "tuned_classification_report.txt", "w") as file:

    file.write(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            zero_division=0,
        )
    )

print("\nEvaluation reports saved successfully!")
print("Tuned model evaluation completed!")

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score

DATA_DIR = Path("dataset_processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading processed datasets...")

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")

categorical_features = ["proto", "service", "state"]

numeric_features = [
    column for column in X_train.columns if column not in categorical_features
]


def create_model():

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


# Split training data into training and validation sets
X_train_part, X_validation, y_train_part, y_validation = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
)

print("\nTraining validation model...")

validation_model = create_model()
validation_model.fit(X_train_part, y_train_part)

# Obtain attack probabilities
y_prob = validation_model.predict_proba(X_validation)[:, 1]

print("\nEvaluating classification thresholds...")

results = []

for threshold in np.arange(0.10, 0.91, 0.01):

    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_validation, y_pred, zero_division=0)

    recall = recall_score(y_validation, y_pred, zero_division=0)

    f2 = fbeta_score(y_validation, y_pred, beta=2, zero_division=0)

    results.append(
        {
            "threshold": round(float(threshold), 2),
            "precision": precision,
            "recall": recall,
            "f2_score": f2,
        }
    )

results_df = pd.DataFrame(results)

# Prefer thresholds achieving at least 95% attack recall
eligible = results_df[results_df["recall"] >= 0.95]

if not eligible.empty:
    best_result = eligible.sort_values(by="precision", ascending=False).iloc[0]

    selection_reason = "Highest precision with recall >= 95%"

else:
    best_result = results_df.loc[results_df["f2_score"].idxmax()]

    selection_reason = "Highest F2 score (95% recall not achieved)"

best_threshold = float(best_result["threshold"])

print("\n========== THRESHOLD TUNING RESULTS ==========")

print(f"Selected threshold : {best_threshold:.2f}")
print(f"Precision          : {best_result['precision']:.4f}")
print(f"Recall             : {best_result['recall']:.4f}")
print(f"F2 Score           : {best_result['f2_score']:.4f}")
print(f"Selection reason   : {selection_reason}")

# Save threshold evaluation results
results_df.to_csv(REPORT_DIR / "threshold_evaluation.csv", index=False)

# Retrain final model using the complete training dataset
print("\nRetraining final model using complete training data...")

final_model = create_model()
final_model.fit(X_train, y_train)

# Save final model
joblib.dump(final_model, MODEL_DIR / "random_forest_nids.joblib")

# Save selected threshold
with open(MODEL_DIR / "classification_threshold.json", "w") as file:
    json.dump({"threshold": best_threshold}, file, indent=4)

print("\nFinal model saved successfully!")
print("Threshold saved successfully!")
print("\nThreshold tuning completed!")

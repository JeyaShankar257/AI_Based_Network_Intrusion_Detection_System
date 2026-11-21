from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = Path("models/random_forest_nids.joblib")
OUTPUT_DIR = Path("reports/figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading trained model...")

model = joblib.load(MODEL_PATH)

# Extract preprocessing and Random Forest components
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Extract feature importance
importance = classifier.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})

# Sort by importance
importance_df = importance_df.sort_values(by="importance", ascending=False)

# Display top 20 features
print("\n========== TOP 20 IMPORTANT FEATURES ==========")

print(importance_df.head(20).to_string(index=False))

# Save feature importance
importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

# Plot top 20 features
plt.figure(figsize=(12, 8))

sns.barplot(data=importance_df.head(20), x="importance", y="feature")

plt.title("Random Forest - Top 20 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=300)

plt.close()

print("\nFeature importance saved successfully!")

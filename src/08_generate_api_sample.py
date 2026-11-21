from pathlib import Path
import json
import pandas as pd

DATA_DIR = Path("dataset_processed")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load test dataset
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

# Select first test record
sample = X_test.iloc[0].to_dict()

# Convert NumPy values into standard Python values
sample = {
    key: value.item() if hasattr(value, "item") else value
    for key, value in sample.items()
}

# Save JSON input
output_path = REPORT_DIR / "api_sample.json"

with open(output_path, "w") as file:
    json.dump(sample, file, indent=4)

print("API sample generated successfully!")

print("\nActual class:")
print("Attack" if y_test.iloc[0] == 1 else "Normal")

print("\nJSON file saved:", output_path)
print("Number of features:", len(sample))

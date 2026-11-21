from pathlib import Path
import json

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from typing import Any

# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "random_forest_nids.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "classification_threshold.json"


# --------------------------------------------------
# FEATURE CONFIGURATION
# --------------------------------------------------

CATEGORICAL_FEATURES = ["proto", "service", "state"]

NUMERICAL_FEATURES = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------
# LOAD MODEL AND THRESHOLD
# --------------------------------------------------

print("Loading Random Forest model...")

model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as file:
    threshold_config = json.load(file)

CLASSIFICATION_THRESHOLD = float(threshold_config["threshold"])

# Check that the API feature list matches model input
model_features = list(model.named_steps["preprocessor"].feature_names_in_)

if set(model_features) != set(FEATURE_COLUMNS):
    raise RuntimeError(
        "API feature configuration does not match trained model features."
    )

print("Model loaded successfully!")
print(f"Classification threshold: {CLASSIFICATION_THRESHOLD}")


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="AI-Based Network Intrusion Detection System",
    description="API for classifying network traffic as normal or suspicious.",
    version="1.0.0",
)


# --------------------------------------------------
# HEALTH CHECK ENDPOINT
# --------------------------------------------------


@app.get("/")
def home():

    return {
        "message": "AI-Based Network Intrusion Detection System API",
        "status": "running",
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": True,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
    }


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------


@app.post("/predict")
def predict_traffic(payload: dict[str, Any]):

    # Check missing features
    missing_features = [
        feature for feature in FEATURE_COLUMNS if feature not in payload
    ]

    if missing_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required features",
                "missing_features": missing_features,
            },
        )

    # Check unexpected features
    extra_features = [feature for feature in payload if feature not in FEATURE_COLUMNS]

    if extra_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Unexpected features provided",
                "extra_features": extra_features,
            },
        )

    # Create DataFrame in the correct feature order
    input_df = pd.DataFrame([payload], columns=FEATURE_COLUMNS)

    # Validate numerical features
    for feature in NUMERICAL_FEATURES:

        try:
            input_df[feature] = pd.to_numeric(input_df[feature], errors="raise")
        except ValueError, TypeError:
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{feature}' must contain a numeric value.",
            )

    if input_df[NUMERICAL_FEATURES].isna().any().any():
        raise HTTPException(
            status_code=422, detail="Numerical features cannot contain null values."
        )

    # Validate categorical features
    for feature in CATEGORICAL_FEATURES:

        value = input_df.at[0, feature]

        if not isinstance(value, str) or not value.strip():
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{feature}' must contain a non-empty string.",
            )

        input_df.at[0, feature] = value.strip()

    # Generate prediction
    try:

        attack_probability = model.predict_proba(input_df)[0][1]

        prediction = int(attack_probability >= CLASSIFICATION_THRESHOLD)

    except Exception as error:

        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(error)}")

    # Prepare response
    if prediction == 1:
        result = "Attack"
        message = "Suspicious network traffic detected."
    else:
        result = "Normal"
        message = "Network traffic appears normal."

    return {
        "prediction": prediction,
        "classification": result,
        "attack_probability": round(float(attack_probability), 4),
        "threshold": CLASSIFICATION_THRESHOLD,
        "message": message,
    }

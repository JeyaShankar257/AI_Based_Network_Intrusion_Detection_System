import streamlit as st
import requests
import pandas as pd

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000"

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    """
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    .main-title {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #9CA3AF;
        font-size: 15px;
    }

    .status-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #12352B;
        color: #6EE7B7;
        margin-bottom: 10px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 15px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# SESSION STATE
# ==========================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

# ==========================================
# LOAD DATASET
# ==========================================


@st.cache_data
def load_dataset():
    X_test = pd.read_csv("dataset_processed/X_test.csv")
    y_test = pd.read_csv("dataset_processed/y_test.csv").squeeze("columns")

    return X_test, y_test


try:
    X_test, y_test = load_dataset()

except Exception as error:
    st.error(f"Dataset loading failed: {error}")
    st.stop()

# ==========================================
# API HEALTH CHECK
# ==========================================

api_connected = False
model_loaded = False
threshold = 0.50

try:
    response = requests.get(f"{API_URL}/health", timeout=5)

    response.raise_for_status()

    health = response.json()

    api_connected = health.get("status") == "healthy"
    model_loaded = health.get("model_loaded", False)
    threshold = health.get("classification_threshold", 0.50)

except requests.exceptions.RequestException:
    pass

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    st.title("🛡️ NIDS")

    st.caption("AI-Powered Network Security")

    st.divider()

    selected_page = st.radio(
        "NAVIGATION",
        [
            "Dashboard Overview",
            "Traffic Analysis",
            "Prediction History",
            "Model Information",
        ],
    )

    st.divider()

    st.subheader("System Status")

    if api_connected:
        st.success("API Connected")
    else:
        st.error("API Disconnected")

    if model_loaded:
        st.success("Model Loaded")
    else:
        st.error("Model Not Loaded")

    st.divider()

    st.caption("AI-Based Network Intrusion Detection System")
    st.caption("Version 1.0")

# ==========================================
# REUSABLE DASHBOARD SUMMARY
# ==========================================


def show_summary():

    history = st.session_state.prediction_history

    if not history:
        st.info("No traffic analyzed yet. " "Visit Traffic Analysis to begin.")
        return

    history_df = pd.DataFrame(history)

    total = len(history_df)

    attacks = (history_df["Classification"] == "Attack").sum()

    normal = (history_df["Classification"] == "Normal").sum()

    attack_rate = attacks / total * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Analyzed", total)
    col2.metric("Normal Traffic", normal)
    col3.metric("Attacks Detected", attacks)
    col4.metric("Attack Rate", f"{attack_rate:.2f}%")


# ==========================================
# PAGE 1: DASHBOARD OVERVIEW
# ==========================================

if selected_page == "Dashboard Overview":

    st.markdown(
        '<div class="main-title">Security Dashboard</div>', unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        "Monitor network traffic and identify suspicious activities."
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.subheader("System Overview")

    col1, col2 = st.columns(2)

    with col1:
        if api_connected:
            st.success("FastAPI Backend: Online")
        else:
            st.error("FastAPI Backend: Offline")

    with col2:
        if model_loaded:
            st.success("Random Forest Model: Ready")
        else:
            st.error("Random Forest Model: Unavailable")

    st.divider()

    st.subheader("Traffic Statistics")

    show_summary()

    history = st.session_state.prediction_history

    if history:

        history_df = pd.DataFrame(history)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Traffic Distribution")

            distribution = (
                history_df["Classification"]
                .value_counts()
                .reindex(["Normal", "Attack"], fill_value=0)
            )

            chart_df = pd.DataFrame(
                {"Traffic Type": distribution.index, "Records": distribution.values}
            )

            st.bar_chart(chart_df.set_index("Traffic Type"))

        with col2:

            st.subheader("Attack Probability")

            probability_df = history_df[
                ["Record Index", "Attack Probability (%)"]
            ].set_index("Record Index")

            st.line_chart(probability_df)

        st.divider()

        st.subheader("Recent Predictions")

        st.dataframe(
            history_df.tail(5).iloc[::-1], use_container_width=True, hide_index=True
        )

    else:

        st.divider()

        st.info(
            "Your dashboard is ready. "
            "Analyze network traffic to see statistics and charts."
        )

# ==========================================
# PAGE 2: TRAFFIC ANALYSIS
# ==========================================

elif selected_page == "Traffic Analysis":

    st.title("🔍 Network Traffic Analysis")

    st.write(
        "Select a network traffic record "
        "and analyze it using the trained Random Forest model."
    )

    st.divider()

    sample_index = st.number_input(
        "Select Traffic Record Index",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0,
        step=1,
    )

    sample = X_test.iloc[int(sample_index)]

    with st.expander("View Network Traffic Features"):

        st.dataframe(sample.to_frame(name="Value"), use_container_width=True)

    if st.button("Analyze Traffic", type="primary"):

        if not api_connected or not model_loaded:

            st.error("API or ML model is unavailable.")

        else:

            payload = sample.to_dict()

            try:

                response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)

                response.raise_for_status()

                result = response.json()

                actual_label = int(y_test.iloc[int(sample_index)])

                actual_class = "Attack" if actual_label == 1 else "Normal"

                history_entry = {
                    "Record Index": int(sample_index),
                    "Classification": result["classification"],
                    "Attack Probability (%)": round(
                        result["attack_probability"] * 100, 2
                    ),
                    "Actual Class": actual_class,
                    "Correct Prediction": (result["prediction"] == actual_label),
                }

                st.session_state.prediction_history.append(history_entry)

                st.session_state.latest_result = {
                    "result": result,
                    "actual_class": actual_class,
                    "record_index": int(sample_index),
                }

            except requests.exceptions.RequestException as error:

                st.error(f"Prediction failed: {error}")

    # Display latest prediction

    if st.session_state.latest_result is not None:

        latest = st.session_state.latest_result
        result = latest["result"]

        st.divider()

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            if result["prediction"] == 1:
                st.error("🚨 ATTACK DETECTED")
            else:
                st.success("✅ NORMAL TRAFFIC")

        with col2:

            st.metric(
                "Attack Probability", f"{result['attack_probability'] * 100:.2f}%"
            )

        st.info(result["message"])

        st.write(f"**Actual Dataset Label:** {latest['actual_class']}")

        actual_value = 1 if latest["actual_class"] == "Attack" else 0

        if result["prediction"] == actual_value:
            st.success("Prediction matches actual dataset label.")
        else:
            st.warning("Prediction differs from actual dataset label.")

        with st.expander("View Prediction Details"):
            st.json(result)

# ==========================================
# PAGE 3: PREDICTION HISTORY
# ==========================================

elif selected_page == "Prediction History":

    st.title("📋 Prediction History")

    st.write(
        "Review all network traffic records analyzed " "during the current session."
    )

    st.divider()

    history = st.session_state.prediction_history

    if history:

        history_df = pd.DataFrame(history)

        show_summary()

        st.divider()

        st.subheader("All Predictions")

        st.dataframe(history_df.iloc[::-1], use_container_width=True, hide_index=True)

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction History (CSV)",
            data=csv_data,
            file_name="nids_prediction_history.csv",
            mime="text/csv",
        )

        st.divider()

        if st.button("Clear Prediction History"):

            st.session_state.prediction_history = []
            st.session_state.latest_result = None

            st.rerun()

    else:

        st.info("No prediction history available.")

# ==========================================
# PAGE 4: MODEL INFORMATION
# ==========================================

elif selected_page == "Model Information":

    st.title("🧠 Model Information")

    st.write(
        "Technical information about the trained " "network intrusion detection model."
    )

    st.divider()

    st.subheader("Model Configuration")

    col1, col2 = st.columns(2)

    with col1:

        st.metric("Algorithm", "Random Forest")
        st.metric("Number of Estimators", "200")
        st.metric("Classification Threshold", f"{threshold:.2f}")

    with col2:

        st.metric("Dataset", "UNSW-NB15")
        st.metric("Total Features", "42")
        st.metric("Categorical Features", "3")

    st.write("**Numerical Features:** 39")

    st.divider()

    st.subheader("Model Evaluation Metrics")

    st.caption(
        "Metrics below are from the final model evaluated "
        "on the official test dataset at threshold 0.50."
    )

    metrics = {
        "Accuracy": 0.903029,
        "Precision": 0.987956,
        "Recall": 0.868109,
        "F1-Score": 0.924163,
        "ROC-AUC": 0.982664,
    }

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{metrics['Accuracy'] * 100:.2f}%")
    col2.metric("Precision", f"{metrics['Precision'] * 100:.2f}%")
    col3.metric("Recall", f"{metrics['Recall'] * 100:.2f}%")

    col1, col2 = st.columns(2)

    col1.metric("F1-Score", f"{metrics['F1-Score'] * 100:.2f}%")
    col2.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")

    st.divider()

    st.subheader("Technology Stack")

    st.markdown("""
    - **Programming Language:** Python
    - **Data Processing:** Pandas, NumPy
    - **Machine Learning:** Scikit-learn
    - **Algorithm:** Random Forest Classifier
    - **Backend:** FastAPI
    - **Frontend:** Streamlit
    - **Model Storage:** Joblib
    - **Dataset:** UNSW-NB15
    """)

    st.divider()

    st.info(
        "This application is a machine-learning-based "
        "network traffic classification prototype. "
        "It currently analyzes dataset records, not live network packets."
    )

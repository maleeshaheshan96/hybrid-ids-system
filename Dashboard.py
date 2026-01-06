import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# Lazy load Neural Network
# -----------------------------
@st.cache_resource
def load_nn_model(path):
    from tensorflow import keras
    return keras.models.load_model(path)

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Hybrid AI Intrusion Detection",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# CSS fixes (alignment)
# -----------------------------
st.markdown("""
<style>
.main, .block-container {
    padding: 1rem 2rem !important;
    margin: 0 auto !important;
    max-width: 1200px !important;
    width: 100% !important;
}

.element-container,
.stMarkdown,
.stDataFrame,
.stMetric,
.stButton,
.stSelectbox,
.stFileUploader {
    width: 100% !important;
}

[data-testid="metric-container"] {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    margin: 0.5rem 0;
}

.stButton > button {
    width: 100%;
}

h1, h2, h3 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Configuration
# -----------------------------
DATASET_CHOICE = "CICIDS"
MODEL_DIR = f"trained_models_{DATASET_CHOICE.lower()}"

# -----------------------------
# Load lightweight components
# -----------------------------
@st.cache_resource
def load_hybrid_system():
    try:
        rf_model = joblib.load(f"{MODEL_DIR}/rf_model.pkl")
        scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
        feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
        ensemble_config = joblib.load(f"{MODEL_DIR}/hybrid_ensemble.pkl")
        results_summary = joblib.load(f"{MODEL_DIR}/results_summary.pkl")

        return rf_model, scaler, label_encoder, feature_names, ensemble_config, results_summary
    except Exception as e:
        st.error(f"Error loading system: {e}")
        return None, None, None, None, None, None

# -----------------------------
# Hybrid prediction
# -----------------------------
def hybrid_predict(rf_model, nn_model, X_data, strategy="Simple Average"):
    rf_proba = rf_model.predict_proba(X_data)
    nn_proba = nn_model.predict(X_data, verbose=0)

    ensemble_proba = (rf_proba + nn_proba) / 2
    predictions = np.argmax(ensemble_proba, axis=1)
    confidence = np.max(ensemble_proba, axis=1)

    return predictions, confidence

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_data(df, feature_names, scaler):
    drop_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'id', 'Label']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())

    for f in feature_names:
        if f not in df.columns:
            df[f] = 0

    df = df[feature_names]
    return scaler.transform(df)

# -----------------------------
# Load system
# -----------------------------
rf_model, scaler, label_encoder, feature_names, ensemble_config, results_summary = load_hybrid_system()
nn_model = None

if rf_model is None:
    st.stop()

class_mapping = {i: c for i, c in enumerate(label_encoder.classes_)}

# -----------------------------
# UI Header
# -----------------------------
st.title("Hybrid AI-Powered Intrusion Detection System")
st.markdown("**ML + DL Ensemble | Near Real-Time Threat Detection**")
st.markdown("---")

# -----------------------------
# Architecture Summary
# -----------------------------
st.subheader("System Architecture")
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Random Forest Accuracy", f"{results_summary['hybrid_ensemble']['individual_accuracies']['random_forest']:.1%}")

with c2:
    st.metric("Neural Network Accuracy", f"{results_summary['hybrid_ensemble']['individual_accuracies']['neural_network']:.1%}")

with c3:
    st.metric("Hybrid Ensemble Accuracy", f"{results_summary['hybrid_ensemble']['ensemble_accuracy']:.1%}")

st.markdown("---")

# -----------------------------
# File Upload
# -----------------------------
st.subheader("Upload Network Traffic CSV")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    if nn_model is None:
        with st.spinner("Loading Neural Network model..."):
            nn_model = load_nn_model(f"{MODEL_DIR}/nn_model.keras")

    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.info(f"File: {uploaded_file.name} | Size: {file_size_mb:.1f} MB")

    if st.button("🚀 Analyze with Hybrid AI"):
        start_time = time.time()

        try:
            with st.spinner("Preprocessing data..."):
                df = pd.read_csv(uploaded_file, low_memory=False)
                X = preprocess_data(df, feature_names, scaler)

            with st.spinner("Running inference..."):
                preds, conf = hybrid_predict(rf_model, nn_model, X, ensemble_config['best_strategy'])
                labels = [class_mapping[p] for p in preds]

            elapsed = time.time() - start_time

            st.success(f"Analysis completed in {elapsed:.2f} seconds")

            st.markdown("---")
            st.subheader("Results Summary")

            total = len(labels)
            threats = sum(1 for l in labels if l != "Benign")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Flows", f"{total:,}")
            c2.metric("Threats Detected", f"{threats:,}")
            c3.metric("Threat Rate", f"{(threats/total)*100:.2f}%")
            c4.metric("Avg Confidence", f"{np.mean(conf):.3f}")

            st.markdown("---")
            st.subheader("Attack Distribution")
            st.bar_chart(pd.Series(labels).value_counts())

            st.markdown("---")
            st.subheader("Sample Predictions")
            st.dataframe(pd.DataFrame({
                "Prediction": labels[:100],
                "Confidence": conf[:100]
            }))

        except Exception as e:
            st.error(f"Analysis failed: {e}")

else:
    st.info("Upload a CSV file to begin detection.")
    with st.expander("Expected Data Format"):
        st.write(", ".join(feature_names[:10]) + " ...")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("**Hybrid ML+DL IDS | MSc Research Project**")

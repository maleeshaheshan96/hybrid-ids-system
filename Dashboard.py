import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# Set page config with centered layout
st.set_page_config(
    page_title="Hybrid AI Intrusion Detection", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Comprehensive CSS fix for alignment
st.markdown("""
<style>
    /* Reset all margins and padding */
    .main, .block-container {
        padding: 1rem 2rem !important;
        margin: 0 auto !important;
        max-width: 1200px !important;
        width: 100% !important;
    }
    
    /* Ensure full width for all content */
    .element-container, 
    .stMarkdown, 
    .stDataFrame,
    .stMetric,
    .stButton,
    .stSelectbox,
    .stFileUploader {
        width: 100% !important;
    }
    
    /* Fix column layout */
    .css-12oz5g7 {
        flex: 1 1 0% !important;
        width: auto !important;
    }
    
    /* Center metrics */
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Fix button alignment */
    .stButton > button {
        width: 100%;
        margin: 0 auto;
        display: block;
    }
    
    /* Center titles and headers */
    h1, h2, h3 {
        text-align: center;
        width: 100%;
    }
    
    /* Fix dataframe width */
    .dataframe {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
DATASET_CHOICE = "CICIDS"
MODEL_DIR = f"trained_models_{DATASET_CHOICE.lower()}"

@st.cache_resource
def load_hybrid_system():
    try:
        rf_model = joblib.load(f"{MODEL_DIR}/rf_model.pkl")
        nn_model = keras.models.load_model(f"{MODEL_DIR}/nn_model.keras")
        ensemble_config = joblib.load(f"{MODEL_DIR}/hybrid_ensemble.pkl")
        scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
        feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
        results_summary = joblib.load(f"{MODEL_DIR}/results_summary.pkl")
        
        return rf_model, nn_model, ensemble_config, scaler, label_encoder, feature_names, results_summary
    except Exception as e:
        st.error(f"Error loading hybrid system: {e}")
        return None, None, None, None, None, None, None

def hybrid_predict(rf_model, nn_model, X_data, strategy='Simple Average'):
    rf_proba = rf_model.predict_proba(X_data)
    nn_proba = nn_model.predict(X_data, verbose=0)
    
    if strategy == 'Simple Average':
        ensemble_proba = (rf_proba + nn_proba) / 2
    else:
        ensemble_proba = (rf_proba + nn_proba) / 2
    
    predictions = np.argmax(ensemble_proba, axis=1)
    confidence = np.max(ensemble_proba, axis=1)
    
    return predictions, confidence, ensemble_proba

def preprocess_data(chunk, feature_names, scaler):
    non_feature_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'id', 'Label']
    for col in non_feature_cols:
        if col in chunk.columns:
            chunk = chunk.drop(col, axis=1)
    
    categorical_cols = chunk.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
    
    chunk = chunk.replace([np.inf, -np.inf], np.nan)
    chunk = chunk.fillna(chunk.median())
    
    missing_features = [f for f in feature_names if f not in chunk.columns]
    if missing_features:
        for feature in missing_features:
            chunk[feature] = 0
    
    chunk = chunk.reindex(columns=feature_names, fill_value=0)
    
    try:
        X_scaled = scaler.transform(chunk)
        return X_scaled
    except Exception as e:
        st.error(f"Scaling error: {e}")
        return None

def filter_valid_attack_types(class_mapping):
    """
    Filter out data artifacts from attack type display.
    Excludes classes that are data labeling errors (e.g., 'Label' class with only 3 samples).
    """
    excluded_classes = ['label']  # Add more class names here if needed (case-insensitive)
    
    valid_attacks = {}
    for class_id, class_name in class_mapping.items():
        if class_name.lower() not in excluded_classes:
            valid_attacks[class_id] = class_name
    
    return valid_attacks

# Load system
rf_model, nn_model, ensemble_config, scaler, label_encoder, feature_names, results_summary = load_hybrid_system()

if rf_model is None:
    st.stop()

# Create class mapping
class_mapping = {}
for i, class_name in enumerate(label_encoder.classes_):
    class_mapping[i] = class_name

# UPDATED: Filter out data artifacts for display
valid_attack_types = filter_valid_attack_types(class_mapping)

# Main Interface
st.title("Hybrid AI-Powered Intrusion Detection System")
st.markdown("**Advanced ML+DL Ensemble | Real-time Threat Detection**")
st.markdown("---")

# System Overview
st.subheader("System Architecture")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🌳 Random Forest**")
    st.write("Classical Machine Learning")
    if 'hybrid_ensemble' in results_summary:
        rf_acc = results_summary['hybrid_ensemble']['individual_accuracies']['random_forest']
        st.metric("RF Accuracy", f"{rf_acc:.1%}")

with col2:
    st.markdown("**🧠 Neural Network**")
    st.write("Deep Learning Component")
    if 'hybrid_ensemble' in results_summary:
        nn_acc = results_summary['hybrid_ensemble']['individual_accuracies']['neural_network']
        st.metric("NN Accuracy", f"{nn_acc:.1%}")

with col3:
    st.markdown("**⚖️ Hybrid Ensemble**")
    st.write("ML+DL Integration")
    if 'hybrid_ensemble' in results_summary:
        hybrid_acc = results_summary['hybrid_ensemble']['ensemble_accuracy']
        st.metric("Ensemble Accuracy", f"{hybrid_acc:.1%}")

st.markdown("---")

# Detectable Attacks - UPDATED: Use filtered attack types
st.subheader("Detectable Attack Types")
attack_names = list(valid_attack_types.values())  # Changed from class_mapping to valid_attack_types
cols = st.columns(len(attack_names))
for i, name in enumerate(attack_names):
    with cols[i]:
        st.write(f"**{name}**")

st.markdown("---")

# File Upload
st.subheader("Upload Network Traffic Data")
uploaded_file = st.file_uploader("Choose a CSV file containing network flow data", type=["csv"])

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.info(f"**File:** {uploaded_file.name} | **Size:** {file_size_mb:.1f} MB")
    
    if st.button("🚀 Analyze with Hybrid AI", type="primary"):
        start_time = time.time()
        
        try:
            with st.spinner("Loading and preprocessing data..."):
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df):,} network flows")
                
                with st.expander("Sample of Uploaded Data"):
                    st.dataframe(df.head(), use_container_width=True)
                
                X_processed = preprocess_data(df, feature_names, scaler)
                
                if X_processed is not None:
                    with st.spinner("Running Hybrid AI Analysis..."):
                        predictions, confidence, all_probabilities = hybrid_predict(
                            rf_model, nn_model, X_processed, 
                            strategy=ensemble_config['best_strategy']
                        )
                        
                        pred_labels = [class_mapping.get(pred, f"Unknown_{pred}") for pred in predictions]
                        processing_time = time.time() - start_time
                    
                    st.success(f"Analysis complete in {processing_time:.1f} seconds!")
                    st.markdown("---")
                    
                    # Results Summary
                    st.subheader("Threat Detection Results")
                    
                    total_samples = len(predictions)
                    threats_detected = len([p for p in pred_labels if p != "Benign"])
                    avg_confidence = np.mean(confidence)
                    threat_rate = (threats_detected/total_samples)*100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Flows", f"{total_samples:,}")
                    with col2:
                        st.metric("Threats Detected", f"{threats_detected:,}")
                    with col3:
                        st.metric("Threat Rate", f"{threat_rate:.1f}%")
                    with col4:
                        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                    
                    st.markdown("---")
                    
                    # Analysis Results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Attack Distribution")
                        attack_counts = pd.Series(pred_labels).value_counts()
                        st.bar_chart(attack_counts, use_container_width=True)
                        
                        st.write("**Breakdown:**")
                        for attack, count in attack_counts.items():
                            percentage = (count / total_samples) * 100
                            if attack == "Benign":
                                st.write(f"🟢 {attack}: {count:,} ({percentage:.1f}%)")
                            else:
                                st.write(f"🔴 {attack}: {count:,} ({percentage:.1f}%)")
                    
                    with col2:
                        st.subheader("Confidence Distribution")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.hist(confidence, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                        ax.set_xlabel("Prediction Confidence")
                        ax.set_ylabel("Number of Samples")
                        ax.set_title("Hybrid AI Confidence Distribution")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
                    
                    st.markdown("---")
                    
                    # High Confidence Threats
                    st.subheader("High-Confidence Threat Analysis")
                    high_conf_threshold = 0.9
                    high_conf_threats = []
                    
                    for i, (pred, conf) in enumerate(zip(pred_labels, confidence)):
                        if pred != "Benign" and conf >= high_conf_threshold:
                            high_conf_threats.append({
                                'Row': i+1,
                                'Attack_Type': pred,
                                'Confidence': f"{conf:.3f}"
                            })
                    
                    if high_conf_threats:
                        threat_df = pd.DataFrame(high_conf_threats)
                        st.write(f"🚨 **{len(high_conf_threats)} high-confidence threats detected** (≥{high_conf_threshold:.0%} confidence)")
                        st.dataframe(threat_df, use_container_width=True)
                    else:
                        st.info("No high-confidence threats detected.")
                    
                    st.markdown("---")
                    
                    # Sample Predictions
                    st.subheader("Sample Predictions")
                    results_df = pd.DataFrame({
                        'Row_ID': range(1, min(101, total_samples + 1)),
                        'Prediction': pred_labels[:100],
                        'Confidence': [f"{c:.3f}" for c in confidence[:100]]
                    })
                    st.dataframe(results_df, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Export Results
                    st.subheader("Export Results")
                    
                    full_results = pd.DataFrame({
                        'Row_ID': range(1, total_samples + 1),
                        'Prediction': pred_labels,
                        'Confidence': confidence,
                        'Processing_Time': processing_time,
                        'Model_Type': 'Hybrid_ML_DL_Ensemble'
                    })
                    
                    csv_data = full_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Complete Results (CSV)",
                        data=csv_data,
                        file_name=f"hybrid_ai_predictions_{int(time.time())}.csv",
                        mime="text/csv"
                    )
                    
                    st.markdown("---")
                    
                    # System Performance
                    st.subheader("System Performance")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        processing_speed = total_samples / processing_time
                        st.metric("Processing Speed", f"{processing_speed:,.0f} flows/sec")
                    
                    with col2:
                        st.metric("Total Time", f"{processing_time:.2f} seconds")
                    
                    with col3:
                        st.metric("Model Type", "Hybrid ML+DL")
                
                else:
                    st.error("Failed to preprocess data. Please check file format.")
        
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.info("Please ensure your CSV contains network flow features.")

else:
    st.info("Upload a CSV file to begin hybrid AI threat detection")
    
    with st.expander("Expected Data Format"):
        st.write("Your CSV should contain network flow features such as:")
        st.code("""
        - Flow Duration, Tot Fwd Pkts, Tot Bwd Pkts
        - Flow Byts/s, Flow Pkts/s, Fwd Pkts/s
        - Dst Port, Protocol, Flow IAT Mean
        - Bwd Pkts/s, Flow IAT Std, Fwd Header Length
        """)
        
        if feature_names:
            st.write(f"**Expected {len(feature_names)} features:**")
            st.write(", ".join(feature_names[:10]) + "..." if len(feature_names) > 10 else ", ".join(feature_names))

# Footer
st.markdown("---")
st.markdown("**Hybrid ML+DL Intrusion Detection System** | Research Project | Advanced AI for Cybersecurity")
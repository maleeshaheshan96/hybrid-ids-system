# Hybrid AI-Powered Intrusion Detection System

MSc Research Project: AI-Powered Intrusion Detection for IoT over 5G Networks

## Features
- Random Forest + Neural Network Hybrid Ensemble
- 99.99% accuracy on CICIDS 2018 dataset
- Real-time threat detection
- SHAP-based explainability
- Interactive web dashboard

## Live Demo
🔗 [View Live Dashboard](https://your-app-name.streamlit.app)

## Research
This system is part of MSc research in Cybersecurity and AI.

**Performance:**
- Random Forest: 99.98%
- Neural Network: 99.98%
- Hybrid Ensemble: 99.99%

## Author
T.A. Maleesha Heshan Perera
```

---

### **Step 2: Organize Your Project Structure**

Your folder should look like this:
```
your-project/
├── Dashboard.py                 # Your main dashboard file
├── requirements.txt             # New - dependencies
├── README.md                    # New - project description  
├── .gitignore                   # New - what to exclude
├── trained_models_cicids/       # Your trained models folder
│   ├── rf_model.pkl
│   ├── nn_model.keras
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── feature_names.pkl
│   ├── hybrid_ensemble.pkl
│   └── results_summary.pkl
└── (other files...)
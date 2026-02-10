import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Machine Learning Assignment 2 - Cancer Risk Prediction Dashboard", layout="wide")

st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #ff0000;   /* Red color */
        font-size: 3.5em;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #444;
        font-size: 1.5em;
        font-style: italic;   /* Italic subtitle */
        margin-bottom: 25px;
    }
.info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .section-header {
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Cancer Risk Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Compare multiple ML models for cancer risk assessment</div>', unsafe_allow_html=True)

st.markdown("""
<style>
    .info-banner {
        background: linear-gradient(135deg, #ffe6e6, #fff5f5);
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #ff0000;
        font-size: 1.05em;
        color: #333;
        margin-top: 10px;
        margin-bottom: 25px;
        box-shadow: 0 3px 6px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
Hi, Welcome! Upload test CSV, choose a model and view predictions with evaluation metrics.
This dashboard demonstrates multiple classification algorithms trained on cancer risk factors.
</div>
""", unsafe_allow_html=True)

# Upload Test Data
st.markdown('<h2 class="section-header">Step 1: Upload Test Dataset</h2>', unsafe_allow_html=True)

# Download button for test dataset
@st.cache_data
def fetch_test_csv():
    import requests
    url = "https://github.com/2025AA05660/ML_Assignment_2/blob/main/cancer-risk-test.csv"
    response = requests.get(url)
    response.raise_for_status()
    return response.content

try:
    test_csv_data = fetch_test_csv()
    st.markdown("You can download the test dataset here:")
    st.download_button(
        label="Download cancer-risk-test.csv",
        data=test_csv_data,
        file_name="cancer-risk-test.csv",
        mime="text/csv"
    )
except Exception:
    st.info("Could not fetch the test dataset from GitHub. Please download it manually from: https://github.com/2025AB05088/ML_Assignment_2/blob/main/steel_faults_test.csv")

MODELS = {
    "Logistic Regression": "model/logistic.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])

try:
    data = pd.read_csv(uploaded_file)
except:
    try:
        data = pd.read_csv(uploaded_file, sep=';')
    except:
        try:
            data = pd.read_csv(uploaded_file, encoding='latin1')
        except Exception as e:
            st.error("Unable to read the uploaded CSV. Please upload a valid file.")
            st.stop()
    TARGET_COLUMN = "Cancer_Type"

    columns_to_drop = [TARGET_COLUMN]
    if 'Patient_ID' in data.columns:
        columns_to_drop.append('Patient_ID')
    if 'Risk_Level' in data.columns:
        columns_to_drop.append('Risk_Level')
    
    X = data.drop(columns=columns_to_drop, errors='ignore') 
    y_true_original_dtype = data[TARGET_COLUMN].dtype 
    y_true = data[TARGET_COLUMN]

    if y_true_original_dtype == "object":
        y_true = y_true.astype("category").cat.codes 

    st.subheader("Step 2: Select Model to Evaluate") 
    
    model_name = st.selectbox("Select Model", list(MODELS.keys()))
    model = joblib.load(MODELS[model_name])

    if model_name in ["Logistic Regression", "KNN"]:
        scaler = StandardScaler()
        X_input = scaler.fit_transform(X)
    else:
        X_input = X.values

    y_pred = model.predict(X_input)

    try:
        y_prob = model.predict_proba(X_input)
        if y_prob.shape[1] > 2:
            y_prob_for_auc = y_prob
        else:
            y_prob_for_auc = y_prob[:, 1]
    except AttributeError:
        st.warning("Model does not support predict_proba, AUC will not be calculated.")
        y_prob_for_auc = None

    st.subheader("Evaluation Metrics")
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted'),
        "Recall": recall_score(y_true, y_pred, average='weighted'),
        "F1": f1_score(y_true, y_pred, average='weighted'),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }
    if y_prob_for_auc is not None:
        if y_prob_for_auc.ndim == 2 and y_prob_for_auc.shape[1] > 2:
            metrics["AUC"] = roc_auc_score(y_true, y_prob_for_auc, multi_class='ovr', average='weighted')
        elif y_prob_for_auc.ndim == 1 or (y_prob_for_auc.ndim == 2 and y_prob_for_auc.shape[1] == 2):
            metrics["AUC"] = roc_auc_score(y_true, y_prob_for_auc)
        else:
            metrics["AUC"] = "N/A"
    else:
        metrics["AUC"] = "N/A"

    cols = st.columns(len(metrics))
    for i, (metric_name, value) in enumerate(metrics.items()):
        with cols[i]:
            if isinstance(value, (int, float)):
                st.markdown(f"**:red[{metric_name}]**"+f"<h1 style='font-size:40px; color:blue'>{value:.4f}</h1>", unsafe_allow_html=True)
            else:
                st.markdown(f"**:green[{metric_name}]**"+f"<h1 style='font-size:40px; color:yellow'>{value}</h1>", unsafe_allow_html=True)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)



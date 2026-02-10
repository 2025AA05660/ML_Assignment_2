# ======================================
# ML Assignment 2
# Dataset: cancer-risk-factors
# ======================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

# ======================================
# Load dataset
# ======================================
df = pd.read_csv("cancer-risk-factors.csv")
df.index = df.index + 1

print("Dataset shape:", df.shape)
print(df.head())

TARGET_COLUMN = "Cancer_Type"  
X = df.drop(columns=[TARGET_COLUMN, "Patient_ID", "Risk_Level"])
y = df[TARGET_COLUMN]

if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# ======================================
# Train–test split
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================================
# Feature scaling
# ======================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================================
# Metric evaluation function
# ======================================

def evaluate(model, X, y):
    y_pred = model.predict(X)
    
    if hasattr(model, "predict_proba") and len(np.unique(y)) > 2:
        y_prob = model.predict_proba(X)
        
        auc_score = roc_auc_score(y, y_prob, multi_class='ovr', average='weighted')
    elif hasattr(model, "predict_proba") and len(np.unique(y)) == 2:
        y_prob = model.predict_proba(X)[:, 1]
        auc_score = roc_auc_score(y, y_prob)
    else:
        auc_score = np.nan

    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc_score,
        "Precision": precision_score(y, y_pred, average='weighted'),
        "Recall": recall_score(y, y_pred, average='weighted'),
        "F1": f1_score(y, y_pred, average='weighted'),
        "MCC": matthews_corrcoef(y, y_pred)
    }

results = {}

# ======================================
# Logistic Regression
# ======================================

lr_params = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["lbfgs"]
}

lr_grid = GridSearchCV(
    LogisticRegression(max_iter=2000),
    lr_params,
    cv=5,
    scoring="f1_weighted"
)

lr_grid.fit(X_train_scaled, y_train)
best_lr = lr_grid.best_estimator_

results["Logistic Regression"] = evaluate(best_lr, X_test_scaled, y_test)

# ======================================
# Decision Tree
# ======================================

dt_params = {
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10]
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params,
    cv=5,
    scoring="f1_weighted"
)

dt_grid.fit(X_train, y_train)
best_dt = dt_grid.best_estimator_

results["Decision Tree"] = evaluate(best_dt, X_test, y_test)

# ======================================
# KNN
# ======================================

knn_params = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"]
}

knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    knn_params,
    cv=5,
    scoring="f1_weighted"
)

knn_grid.fit(X_train_scaled, y_train)
best_knn = knn_grid.best_estimator_

results["KNN"] = evaluate(best_knn, X_test_scaled, y_test)

# ======================================
# Naive Bayes
# ======================================

nb = GaussianNB()
nb.fit(X_train, y_train)

results["Naive Bayes"] = evaluate(nb, X_test, y_test)

# ======================================
# Random Forest
# ======================================

rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

results["Random Forest"] = evaluate(best_rf, X_test, y_test)

# ======================================
# XGBoost
# ======================================

xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0]
}

xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

xgb_grid = GridSearchCV(
    xgb_model,
    xgb_params,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1
)

xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_

results["XGBoost"] = evaluate(best_xgb, X_test, y_test)

# ======================================
# Display results table
# ======================================

results_df = pd.DataFrame(results).T.reset_index()
results_df.index = results_df.index + 1
results_df.rename(columns={"index": "Model"}, inplace=True)
print("\nMODEL COMPARISON TABLE\n")
print(results_df)

# ======================================
# Save models
# ======================================

os.makedirs("model", exist_ok=True)

joblib.dump(best_lr, "model/logistic.pkl")
joblib.dump(best_dt, "model/decision_tree.pkl")
joblib.dump(best_knn, "model/knn.pkl")
joblib.dump(nb, "model/naive_bayes.pkl")
joblib.dump(best_rf, "model/random_forest.pkl")
joblib.dump(best_xgb, "model/xgboost.pkl")

print("\nAll models saved successfully in /model folder")

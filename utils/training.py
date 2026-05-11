from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import streamlit as st

@st.cache_resource
def train_model(X, y, sensitive_features=None, model_type='Logistic Regression'):
    if sensitive_features is not None:
        X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(X, y, sensitive_features, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        sf_train, sf_test = (None, None)
    if model_type == 'Random Forest':
        model = RandomForestClassifier(random_state=42, n_estimators=100)
    else:
        model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    unique_classes = len(np.unique(y_train))
    avg_str = 'binary' if unique_classes <= 2 else 'macro'
    metrics = {'Accuracy': accuracy_score(y_test, y_pred), 'Precision': precision_score(y_test, y_pred, average=avg_str, zero_division=0), 'Recall': recall_score(y_test, y_pred, average=avg_str, zero_division=0), 'F1 Score': f1_score(y_test, y_pred, average=avg_str, zero_division=0)}
    return (model, metrics, X_test, y_test, y_pred, X_train, y_train, sf_train, sf_test)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    unique_classes = len(np.unique(y_test))
    avg_str = 'binary' if unique_classes <= 2 else 'macro'
    return {'Accuracy': accuracy_score(y_test, y_pred), 'Precision': precision_score(y_test, y_pred, average=avg_str, zero_division=0), 'Recall': recall_score(y_test, y_pred, average=avg_str, zero_division=0), 'F1 Score': f1_score(y_test, y_pred, average=avg_str, zero_division=0)}

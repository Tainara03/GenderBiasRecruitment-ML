import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from fairness_metrics import compute_fairness_metrics, print_group_fairness, group_confusion_stats

def train_logistic_regression(X_train, y_train):
    """
    Treina um modelo de Regressão Logística.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=100):
    """
    Treina um modelo de Random Forest.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Calcula métricas de avaliação clássicas: accuracy, precision, recall e f1-score.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    return y_pred, metrics

def evaluate_model_per_gender(model, X_test, y_test, sensitive_col):
    """
    Avalia métricas de fairness por grupo sensível (ex.: gênero).
    Usa compute_fairness_metrics(model, X, y_true, sensitive_series).
    """
    sensitive_series = X_test[sensitive_col]

    df_group, summary = compute_fairness_metrics(
        model=model,
        X=X_test,
        y_true=y_test,
        sensitive_series=sensitive_series
    )

    print_group_fairness(df_group)

    return df_group, summary

def train_and_evaluate(X_train, X_test, y_train, y_test, model_type="logistic", sensitive_col=None):
    """
    Função integrada: treina, avalia métricas gerais e fairness.
    """
    if model_type == "logistic":
        model = train_logistic_regression(X_train, y_train)
    elif model_type == "rf":
        model = train_random_forest(X_train, y_train)
    else:
        raise ValueError("model_type deve ser 'logistic' ou 'rf'")

    y_pred, metrics = evaluate_model(model, X_test, y_test)
    print("===== MODEL METRICS =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
    print("=========================\n")

    if sensitive_col:
        print(f"===== FAIRNESS EVALUATION (Sensitive: {sensitive_col}) =====")
        evaluate_model_per_gender(model, X_test, y_test, sensitive_col)

    return model, y_pred, metrics

def save_object(obj, filename):
    """Salva um objeto em arquivo .pkl"""
    with open(os.path.join("../outputs", filename), "wb") as f:
        pickle.dump(obj, f)

def load_object(filename):
    """Carrega um objeto .pkl salvo"""
    with open(os.path.join("../outputs", filename), "rb") as f:
        return pickle.load(f)

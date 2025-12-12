import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def demographic_parity_rates(y_pred, sensitive_attr):
    """
    Retorna a taxa de previsão positiva por grupo sensível.
    y_pred: array-like (0/1)
    sensitive_attr: array-like (group labels)
    """
    groups = np.unique(sensitive_attr)
    rates = {}
    for g in groups:
        mask = (sensitive_attr == g)
        rates[g] = np.mean(y_pred[mask])
    return rates

def equal_opportunity_tpr(y_true, y_pred, sensitive_attr):
    """
    Retorna TPR (True Positive Rate) por grupo sensível.
    """
    groups = np.unique(sensitive_attr)
    tpr = {}
    for g in groups:
        mask = (sensitive_attr == g)
        if mask.sum() == 0:
            tpr[g] = np.nan
            continue
        tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask], labels=[0,1]).ravel()
        tpr[g] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    return tpr

def group_confusion_stats(y_true, y_pred, sensitive_attr):
    """
    Para cada grupo, retorna TP, FP, TN, FN (inteiro) e derived rates.
    Retorna um DataFrame com índices = grupos e colunas: TP,FP,TN,FN,TPR,FPR,FNR,positive_rate
    """
    groups = np.unique(sensitive_attr)
    rows = []
    for g in groups:
        mask = (sensitive_attr == g)
        if mask.sum() == 0:
            rows.append({
                "group": g, "TP": 0, "FP": 0, "TN": 0, "FN": 0,
                "TPR": np.nan, "FPR": np.nan, "FNR": np.nan, "positive_rate": np.nan,
                "support": 0
            })
            continue
        y_t = np.asarray(y_true)[mask]
        y_p = np.asarray(y_pred)[mask]
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0,1]).ravel()
        support = len(y_t)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        positive_rate = np.mean(y_p)
        rows.append({
            "group": g, "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "TPR": tpr, "FPR": fpr, "FNR": fnr, "positive_rate": positive_rate,
            "support": int(support)
        })
    return pd.DataFrame(rows).set_index("group")

def remove_sensitive_attributes(df: pd.DataFrame, sensitive_cols):
    """
    Remove colunas sensíveis do DataFrame e retorna uma cópia sem elas.
    - df: DataFrame original (pode ser codificado)
    - sensitive_cols: lista de nomes de colunas a remover (strings)
    Retorna: DataFrame (cópia) com colunas removidas.
    """
    if sensitive_cols is None:
        return df.copy()
    missing = [c for c in sensitive_cols if c not in df.columns]
    if missing:
        print(f"Warning: as seguintes colunas sensíveis não foram encontradas e serão ignoradas: {missing}")
    cols_to_drop = [c for c in sensitive_cols if c in df.columns]
    return df.drop(columns=cols_to_drop).copy()


def compute_fairness_metrics(model, X, y_true, sensitive_series):
    """
    Calcula métricas de fairness por grupo sensível e retorna dois objetos:
    - df_group: DataFrame com TP,FP,TN,FN,TPR,FPR,FNR,positive_rate,support por grupo
    - summary: dicionário agregando diferenças importantes (demographic parity diff, TPR diff, disparate impact)
    
    Parâmetros:
    - model: objeto treinado com .predict(X) (compatível sklearn)
    - X: features (DataFrame ou array) que serão passadas para model.predict
    - y_true: verdadeiros rótulos (pd.Series / array)
    - sensitive_series: série/array com o atributo sensível alinhado com X/y_true (mesmo índice)

    Retorna:
    - df_group (DataFrame)
    - summary (dict)
    """
    y_pred = model.predict(X)

    sensitive_arr = np.asarray(sensitive_series)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    df_group = group_confusion_stats(y_true_arr, y_pred_arr, sensitive_arr)

    positive_rates = df_group["positive_rate"].to_dict()
    tpr_rates = df_group["TPR"].to_dict()

    dp_diff = float(df_group["positive_rate"].max() - df_group["positive_rate"].min())

    tpr_diff = float(np.nanmax(df_group["TPR"].values) - np.nanmin(df_group["TPR"].values))

    pr_vals = df_group["positive_rate"].values
    if np.nanmax(pr_vals) > 0:
        di_ratio = float(np.nanmin(pr_vals) / np.nanmax(pr_vals))
    else:
        di_ratio = np.nan

    summary = {
        "positive_rates": positive_rates,
        "dp_diff": dp_diff,
        "tpr_rates": tpr_rates,
        "tpr_diff": tpr_diff,
        "disparate_impact_ratio": di_ratio,
        "global_accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "global_precision": precision_score(y_true_arr, y_pred_arr, zero_division=0),
        "global_recall": recall_score(y_true_arr, y_pred_arr, zero_division=0)
    }

    return df_group, summary

def print_group_fairness(df_group):
    """
    Print formatado das métricas por grupo.
    """
    print("===== Fairness por grupo =====")
    print(df_group[["support", "positive_rate", "TPR", "FPR", "FNR"]])
    print("==============================")

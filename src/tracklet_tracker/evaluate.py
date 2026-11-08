import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


def compute_confusion_matrix(
    y_true: pd.Series, y_pred_proba: pd.Series, threshold: float = 0.90
):
    y_pred_strict = (y_pred_proba >= threshold).astype(int)
    print(f"Confusion Matrix (Threshold = {threshold}) -------")
    print(confusion_matrix(y_true, y_pred_strict))


def compute_evaluation_metrics(
    y_true: pd.Series, y_pred_proba: pd.Series, threshold: float = 0.90
):
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)

    auc_roc = roc_auc_score(y_true, y_pred_proba)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    total_events = np.sum(y_true)
    event_rate = total_events / len(y_true)

    df = pd.DataFrame({"truth": y_true, "proba": y_pred_proba})
    df = df.sort_values(by="proba", ascending=False).reset_index(drop=True)

    capture_rates = {}
    for percent in [10, 25, 40, 50]:
        cutoff_idx = int(len(df) * (percent / 100.0))

        if total_events == 0:
            capture_rates[f"capture_rate_{percent}pct"] = 0.0
        elif cutoff_idx == 0:
            capture_rates[f"capture_rate_{percent}pct"] = 0.0
        else:
            captured_events = df.loc[: cutoff_idx - 1, "truth"].sum()
            capture_rates[f"capture_rate_{percent}pct"] = captured_events / total_events

    metrics_dict = {
        "auc_roc": [auc_roc],
        "gini": [2 * auc_roc - 1.0],
        "precision": [precision],
        "recall": [recall],
        "accuracy": [accuracy],
        "total_events": [total_events],
        "event_rate": [event_rate],
        "capture_rate_10": [capture_rates["capture_rate_10pct"]],
        "capture_rate_25": [capture_rates["capture_rate_25pct"]],
        "capture_rate_40": [capture_rates["capture_rate_40pct"]],
        "capture_rate_50": [capture_rates["capture_rate_50pct"]],
    }

    result = pd.DataFrame(metrics_dict)
    print(result)

    return result


def print_feature_importance(model: XGBClassifier, X_train: pd.DataFrame):
    importance = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    print("Top 5 Most Important Features: -------------")
    print(importance.head(5))

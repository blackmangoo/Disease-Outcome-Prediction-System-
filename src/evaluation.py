import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test):
    """Evaluate a fitted model. Returns dict with accuracy, precision, recall, f1, roc_auc."""
    y_pred = model.predict(X_test)
    n_classes = len(np.unique(y_test))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    try:
        if n_classes == 2:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                y_score = model.decision_function(X_test)
            roc_auc = roc_auc_score(y_test, y_score)
        else:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)
            else:
                y_score = model.decision_function(X_test)
            roc_auc = roc_auc_score(y_test, y_score, multi_class="ovr", average="weighted")
    except Exception:
        roc_auc = float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def compare_models(models_dict, X_test, y_test):
    """Evaluate multiple models. Returns list of dicts with model name + metrics."""
    results = []
    for name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics["model"] = name
        results.append(metrics)
    return results

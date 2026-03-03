import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.data_loader import load_all_datasets
from src.feature_selection import select_kbest, select_rfe
from src.models import get_classifiers, train_model
from src.evaluation import evaluate_model
from src.pipeline import DiseasePipeline


def test_data_loader_returns_three_diseases():
    datasets = load_all_datasets()
    assert set(datasets.keys()) == {"breast_cancer", "diabetes", "heart_disease"}
    for name, (X, y, feature_names) in datasets.items():
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        assert len(feature_names) == X.shape[1]


def test_kbest_feature_selection():
    datasets = load_all_datasets()
    X, y, feature_names = datasets["breast_cancer"]
    k = 10
    X_sel, sel_names = select_kbest(X, y, feature_names, k=k)
    assert X_sel.shape[1] == k
    assert len(sel_names) == k


def test_rfe_feature_selection():
    datasets = load_all_datasets()
    X, y, feature_names = datasets["diabetes"]
    estimator = LogisticRegression(max_iter=1000, random_state=42)
    n = 5
    X_sel, sel_names = select_rfe(X, y, estimator, feature_names, n_features=n)
    assert X_sel.shape[1] == n
    assert len(sel_names) == n


def test_classifiers_train_and_predict():
    datasets = load_all_datasets()
    X, y, _ = datasets["breast_cancer"]
    classifiers = get_classifiers()
    assert len(classifiers) == 4
    for name, clf in classifiers.items():
        fitted = train_model(clf, X, y)
        preds = fitted.predict(X)
        assert preds.shape == y.shape


def test_evaluate_model_metrics():
    datasets = load_all_datasets()
    X, y, _ = datasets["breast_cancer"]
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)
    metrics = evaluate_model(clf, X, y)
    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert expected_keys == set(metrics.keys())
    for key in expected_keys:
        assert isinstance(metrics[key], float)


def test_pipeline_run_breast_cancer():
    pipeline = DiseasePipeline("breast_cancer")
    results = pipeline.run()
    assert results["disease_name"] == "breast_cancer"
    assert len(results["selected_features"]) == 10
    assert len(results["results"]) == 4
    for r in results["results"]:
        assert "accuracy" in r
        assert 0.0 <= r["accuracy"] <= 1.0


def test_pipeline_run_diabetes():
    pipeline = DiseasePipeline("diabetes")
    results = pipeline.run()
    assert results["disease_name"] == "diabetes"
    assert len(results["results"]) == 4


def test_pipeline_predict():
    pipeline = DiseasePipeline("breast_cancer")
    pipeline.run()
    # Use zero vector as dummy input
    n_features = len(pipeline.selected_features)
    result = pipeline.predict([0.0] * n_features)
    assert "prediction" in result
    assert result["prediction"] in (0, 1)
    assert "probabilities" in result
    assert "model_used" in result

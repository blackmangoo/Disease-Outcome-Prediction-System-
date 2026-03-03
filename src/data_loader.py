import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, make_classification


def load_all_datasets():
    """Load all three disease datasets. Returns dict: name -> (X, y, feature_names)."""
    datasets = {}

    # Breast Cancer (binary: 0=malignant, 1=benign)
    bc = load_breast_cancer()
    datasets["breast_cancer"] = (bc.data, bc.target, list(bc.feature_names))

    # Diabetes: binarize target by median
    db = load_diabetes()
    median_val = np.median(db.target)
    y_binary = (db.target > median_val).astype(int)
    datasets["diabetes"] = (db.data, y_binary, list(db.feature_names))

    # Heart Disease: simulated via make_classification
    X_heart, y_heart = make_classification(
        n_samples=303,
        n_features=13,
        n_informative=10,
        n_redundant=3,
        random_state=42,
        class_sep=1.5,
    )
    heart_feature_names = [f"synthetic_feature_{i}" for i in range(13)]
    datasets["heart_disease"] = (X_heart, y_heart, heart_feature_names)

    return datasets

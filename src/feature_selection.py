import numpy as np
from sklearn.feature_selection import SelectKBest, RFE, f_classif


def select_kbest(X, y, feature_names, k=10):
    """Select top-k features using ANOVA F-value. Returns (X_selected, selected_feature_names)."""
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    mask = selector.get_support()
    selected_names = [name for name, selected in zip(feature_names, mask) if selected]
    return X_selected, selected_names


def select_rfe(X, y, estimator, feature_names, n_features=10):
    """Select features using Recursive Feature Elimination. Returns (X_selected, selected_feature_names)."""
    n_features = min(n_features, X.shape[1])
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    X_selected = rfe.fit_transform(X, y)
    mask = rfe.support_
    selected_names = [name for name, selected in zip(feature_names, mask) if selected]
    return X_selected, selected_names

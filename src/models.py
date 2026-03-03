from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_classifiers():
    """Return dict of classifier name -> classifier instance."""
    return {
        "SVM": SVC(kernel="rbf", probability=True),
        "GaussianNB": GaussianNB(),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    }


def train_model(clf, X_train, y_train):
    """Fit classifier on training data and return fitted model."""
    clf.fit(X_train, y_train)
    return clf

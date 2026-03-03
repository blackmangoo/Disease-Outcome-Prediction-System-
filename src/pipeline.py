import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.data_loader import load_all_datasets
from src.feature_selection import select_kbest, select_rfe
from src.models import get_classifiers, train_model
from src.evaluation import compare_models, evaluate_model


class DiseasePipeline:
    def __init__(self, disease_name):
        self.disease_name = disease_name
        self.X = None
        self.y = None
        self.feature_names = None
        self.X_selected = None
        self.selected_features = None
        self.scaler = None
        self.models = {}
        self.best_model_name = None
        self.X_test_scaled = None
        self.y_test = None

    def load_data(self):
        """Load dataset for this disease."""
        datasets = load_all_datasets()
        if self.disease_name not in datasets:
            raise ValueError(f"Unknown disease: {self.disease_name}. "
                             f"Choose from: {list(datasets.keys())}")
        self.X, self.y, self.feature_names = datasets[self.disease_name]

    def select_features(self, method="kbest", n_features=10):
        """Run feature selection and store results."""
        if self.X is None:
            self.load_data()
        n_features = min(n_features, self.X.shape[1])
        if method == "kbest":
            self.X_selected, self.selected_features = select_kbest(
                self.X, self.y, self.feature_names, k=n_features
            )
        elif method == "rfe":
            estimator = LogisticRegression(max_iter=1000, random_state=42)
            self.X_selected, self.selected_features = select_rfe(
                self.X, self.y, estimator, self.feature_names, n_features=n_features
            )
        else:
            raise ValueError(f"Unknown feature selection method: {method}")

    def train_all(self, test_size=0.2, random_state=42):
        """Split data, scale, and train all classifiers."""
        if self.X_selected is None:
            self.select_features()

        X_train, X_test, y_train, y_test = train_test_split(
            self.X_selected, self.y, test_size=test_size,
            random_state=random_state, stratify=self.y
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.X_test_scaled = X_test_scaled
        self.y_test = y_test

        classifiers = get_classifiers()
        self.models = {}
        for name, clf in classifiers.items():
            self.models[name] = train_model(clf, X_train_scaled, y_train)

    def evaluate_all(self):
        """Evaluate all trained models and return comparison results."""
        if not self.models:
            self.train_all()
        results = compare_models(self.models, self.X_test_scaled, self.y_test)
        # Determine best model by accuracy
        best = max(results, key=lambda r: r["accuracy"])
        self.best_model_name = best["model"]
        return results

    def run(self, feature_method="kbest", n_features=10):
        """Orchestrate all pipeline steps. Returns results dict."""
        self.load_data()
        self.select_features(method=feature_method, n_features=n_features)
        self.train_all()
        results = self.evaluate_all()
        return {
            "disease_name": self.disease_name,
            "feature_method": feature_method,
            "selected_features": self.selected_features,
            "results": results,
        }

    def predict(self, features):
        """Predict using best model. features is array-like of selected feature values."""
        if not self.models or self.best_model_name is None:
            raise RuntimeError("Pipeline not trained. Call run() first.")
        features_arr = np.array(features, dtype=float).reshape(1, -1)
        features_scaled = self.scaler.transform(features_arr)
        model = self.models[self.best_model_name]
        prediction = int(model.predict(features_scaled)[0])
        probabilities = {}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_scaled)[0]
            for i, p in enumerate(proba):
                probabilities[f"class_{i}"] = round(float(p), 4)
        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "model_used": self.best_model_name,
        }


if __name__ == "__main__":
    for disease in ["breast_cancer", "diabetes", "heart_disease"]:
        print(f"\n=== {disease} ===")
        pipeline = DiseasePipeline(disease)
        results = pipeline.run()
        print(f"Selected features: {results['selected_features']}")
        for r in results["results"]:
            print(f"  {r['model']:20s} acc={r['accuracy']:.4f} auc={r['roc_auc']:.4f}")

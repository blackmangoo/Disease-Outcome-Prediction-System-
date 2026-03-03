# Disease Outcome Prediction System

A machine learning web application that predicts disease outcomes for **Breast Cancer**, **Diabetes**, and **Heart Disease** using multiple classifiers with scikit-learn and Flask.

## Features

- **Three disease datasets**: Breast Cancer, Diabetes (binarized), and Heart Disease (simulated)
- **Feature selection**: SelectKBest (ANOVA F-value) and Recursive Feature Elimination (RFE)
- **Four classifiers**: SVM (RBF kernel), Gaussian Naïve Bayes, Random Forest, Logistic Regression
- **Model comparison**: side-by-side metrics table (Accuracy, Precision, Recall, F1, ROC-AUC)
- **Prediction form**: enter feature values and receive a Positive/Negative prediction with probabilities
- **Lazy-loaded pipelines**: models are trained on first request; no pre-training required
- **Responsive UI**: Bootstrap 5 with custom CSS gradients and card hover effects

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the web application

```bash
python app.py
```

Then open <http://127.0.0.1:5000> in your browser.

### Run the CLI pipeline

```bash
python -m src.pipeline
```

This trains and evaluates all three diseases and prints accuracy/AUC for each model.

### Run tests

```bash
pytest tests/
```

## Architecture

```
Disease-Outcome-Prediction-System/
├── app.py                  # Flask web application
├── requirements.txt
├── src/
│   ├── data_loader.py      # Load breast cancer, diabetes, heart disease datasets
│   ├── feature_selection.py# SelectKBest and RFE wrappers
│   ├── models.py           # Classifier definitions and training
│   ├── evaluation.py       # Metrics computation and model comparison
│   └── pipeline.py         # DiseasePipeline orchestration class
├── templates/
│   ├── base.html           # Bootstrap 5 base layout
│   ├── index.html          # Disease selection cards
│   ├── results.html        # Model comparison table
│   ├── predict.html        # Feature input form
│   └── result.html         # Prediction result display
├── static/
│   └── style.css           # Custom CSS
└── tests/
    └── test_pipeline.py    # pytest unit tests
```

## Available Diseases & Models

| Disease | Features | Samples |
|---------|----------|---------|
| Breast Cancer | 30 (sklearn built-in) | 569 |
| Diabetes | 10 (sklearn built-in, binarized by median) | 442 |
| Heart Disease | 13 synthetic features | 303 |

| Model | Algorithm |
|-------|-----------|
| SVM | Support Vector Classifier (RBF kernel) |
| GaussianNB | Gaussian Naïve Bayes |
| RandomForest | Random Forest (100 trees) |
| LogisticRegression | Logistic Regression |
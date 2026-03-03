from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from src.pipeline import DiseasePipeline
from src.data_loader import load_all_datasets

app = Flask(__name__)

# Lazy-loaded trained pipelines
pipelines = {}

DISEASE_DISPLAY = {
    "breast_cancer": "Breast Cancer",
    "diabetes": "Diabetes",
    "heart_disease": "Heart Disease",
}


def get_pipeline(disease):
    """Get or create and train pipeline for a disease."""
    if disease not in pipelines:
        p = DiseasePipeline(disease)
        p.run()
        pipelines[disease] = p
    return pipelines[disease]


def get_feature_names(disease):
    """Return feature names for the selected features of a trained pipeline."""
    pipeline = get_pipeline(disease)
    return pipeline.selected_features


@app.route("/")
def index():
    diseases = list(DISEASE_DISPLAY.items())
    return render_template("index.html", diseases=diseases)


@app.route("/results", methods=["POST"])
def results():
    disease = request.form.get("disease")
    if disease not in DISEASE_DISPLAY:
        return redirect(url_for("index"))
    pipeline = get_pipeline(disease)
    # Re-run to get fresh results dict
    result_data = {
        "disease_name": pipeline.disease_name,
        "feature_method": "kbest",
        "selected_features": pipeline.selected_features,
        "results": pipeline.evaluate_all(),
    }
    return render_template(
        "results.html",
        disease_display=DISEASE_DISPLAY[disease],
        data=result_data,
    )


@app.route("/predict/<disease>", methods=["GET"])
def predict_form(disease):
    if disease not in DISEASE_DISPLAY:
        return redirect(url_for("index"))
    feature_names = get_feature_names(disease)
    return render_template(
        "predict.html",
        disease=disease,
        disease_display=DISEASE_DISPLAY[disease],
        feature_names=feature_names,
    )


@app.route("/predict/<disease>", methods=["POST"])
def predict_submit(disease):
    if disease not in DISEASE_DISPLAY:
        return redirect(url_for("index"))
    feature_names = get_feature_names(disease)
    try:
        features = [float(request.form.get(name, 0)) for name in feature_names]
    except (TypeError, ValueError):
        features = [0.0] * len(feature_names)

    pipeline = get_pipeline(disease)
    prediction_result = pipeline.predict(features)

    label = "Positive" if prediction_result["prediction"] == 1 else "Negative"
    return render_template(
        "result.html",
        disease=disease,
        disease_display=DISEASE_DISPLAY[disease],
        label=label,
        prediction=prediction_result,
    )


if __name__ == "__main__":
    import os
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug)

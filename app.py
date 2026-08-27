"""Flask web application for phishing website detection.

The app exposes a simple user interface where a user enters a URL and receives
an ML-driven phishing risk assessment. The prediction is calculated using the
same feature extraction logic as the training script.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from feature_extraction import FEATURE_COLUMNS, extract_features

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "model" / "phishing_model.pkl"
METADATA_PATH = ROOT_DIR / "model" / "model_metadata.json"

app = Flask(__name__)


@app.after_request
def no_cache(response):
    """Disable browser caching so the latest analysis results appear immediately."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def index():
    """Render the main homepage."""
    return render_template("index.html")


@app.route("/api/model-metrics")
def get_model_metrics():
    """Return the saved model metadata for the dashboard."""
    if not METADATA_PATH.exists():
        return jsonify({"success": False, "message": "Model metadata is not available yet."}), 404

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    return jsonify({
        "success": True,
        "best_model": metadata.get("best_model", "Unknown"),
        "best_accuracy": metadata.get("best_accuracy", 0),
        "metrics": metadata.get("metrics", {}),
    })


@app.route("/api/analyze", methods=["POST"])
def analyze_url():
    """Analyze a submitted URL and return prediction results."""
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()

    if not url:
        return jsonify({"success": False, "message": "Please enter a valid URL."}), 400

    try:
        feature_values = extract_features(url)
        feature_df = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)

        if not MODEL_PATH.exists():
            return jsonify({
                "success": False,
                "message": "The machine learning model has not been trained yet. Please run train_model.py first.",
            }), 500

        model = joblib.load(MODEL_PATH)
        prediction = int(model.predict(feature_df)[0])
        probabilities = model.predict_proba(feature_df)[0]

        legitimate_probability = round(float(probabilities[0] * 100), 2)
        phishing_probability = round(float(probabilities[1] * 100), 2)
        confidence = round(float(probabilities[prediction] * 100), 2)

        if prediction == 1:
            status = "PHISHING DETECTED"
            label = "phishing"
        else:
            status = "LEGITIMATE"
            label = "legitimate"

        metadata = {}
        if METADATA_PATH.exists():
            with METADATA_PATH.open("r", encoding="utf-8") as file:
                metadata = json.load(file)

        response = {
            "success": True,
            "status": status,
            "label": label,
            "confidence": confidence,
            "legitimate_probability": legitimate_probability,
            "phishing_probability": phishing_probability,
            "features": feature_values,
            "model": metadata.get("best_model", "Unknown"),
            "model_accuracy": metadata.get("best_accuracy", 0),
            "note": "This is an ML-based risk assessment and not a guarantee of website safety.",
        }
        return jsonify(response)

    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive fallback
        return jsonify({"success": False, "message": f"Unable to analyze URL: {exc}"}), 500


@app.route("/health")
def health_check():
    """Simple health endpoint for verifying the backend is running."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

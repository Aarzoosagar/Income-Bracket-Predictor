"""
app.py
Flask API for Income Bracket Predictor
"""

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Optional if frontend is separate

# -------------------------------
# Paths to saved files
PREPROCESSOR_PATH = "preprocessor.joblib"
MODEL_PATH = "saved_model/best_model(1).h5"
# -------------------------------

# Expected feature names (must match training)
FEATURES = [
    "age", "workclass", "education", "marital-status",
    "occupation", "relationship", "gender", "capital-gain",
    "capital-loss", "hours-per-week", "native-country"
]

app = Flask(__name__)
CORS(app)  # optional

# Load preprocessor and model
print("Loading preprocessor and model...")
preprocessor = joblib.load(PREPROCESSOR_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("Loaded successfully!")

# -------------------------------
# Serve frontend
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------
# Health check endpoint
# -------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# -------------------------------
# Predict endpoint
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400

    # If single record, make it a list
    if isinstance(data, dict):
        data = [data]

    df = pd.DataFrame(data)

    # Ensure all features exist
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    # Strip whitespace for object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Preprocess
    try:
        X_t = preprocessor.transform(df[FEATURES])
    except Exception as e:
        return jsonify({"error": "Preprocessing failed", "details": str(e)}), 500

    # Predict probabilities
    probs = model.predict(X_t).ravel()
    preds = (probs >= 0.5).astype(int)

    # Map numeric prediction to labels
    label_map = {0: "<=50K", 1: ">50K"}

    results = []
    for i in range(len(preds)):
        results.append({
            "prediction": int(preds[i]),
            "label": label_map[int(preds[i])],
            "probability": float(probs[i])
        })

    return jsonify(results)


if __name__ == "__main__":
    # Run Flask locally
    app.run(host="0.0.0.0", port=5000, debug=True)

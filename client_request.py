"""
client_request.py
Test the Income Bracket Predictor Flask API
"""

import requests

# -------------------------------
# URL of your API
# For local: "http://127.0.0.1:5000/predict"
# For Codespaces forwarded URL, replace below
URL = "https://spidery-mummy-jj45xjqqrr5rhrrx-5000.app.github.dev/predict"
# -------------------------------

# Example inputs (multiple people)
sample_inputs = [
    {
        "age": 37,
        "workclass": "Private",
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Divorced",
        "occupation": "Handlers-cleaners",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    },
    {
        "age": 28,
        "workclass": "Private",
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Tech-support",
        "relationship": "Not-in-family",
        "race": "Asian-Pac-Islander",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }
]

# Send POST request
try:
    response = requests.post(URL, json=sample_inputs)
    response.raise_for_status()
    predictions = response.json()

    # Print nicely
    for i, pred in enumerate(predictions):
        print(f"Sample {i+1}:")
        print(f"  Prediction: {pred['prediction']} ({pred['label']})")
        print(f"  Probability: {pred['probability']:.4f}\n")

except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

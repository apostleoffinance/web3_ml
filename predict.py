import joblib
from flask import Flask, request, jsonify
import numpy as np

# Load the scaler and model
scaler, model = joblib.load('logistic_regression_model.bin')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        data = request.json
        if not isinstance(data, list):
            return jsonify({"error": "Input data must be a list of dictionaries"}), 400
        
        # Convert input data into a 2D array
        features = [
            [
                entry["base_cumulative_return"],
                entry["portfolio_return"],
                entry["daily_sharpe_ratio"],
                entry["number_of_trades"],
                entry["unique_tokens_traded"]
            ]
            for entry in data
        ]

        # Preprocess with scaler
        scaled_features = scaler.transform(features)

        # Make predictions
        predictions = model.predict(scaled_features)
        #probabilities = model.predict_proba(scaled_features).tolist()

        # Return predictions as JSON
        return jsonify({
            "predictions": predictions.tolist()
            #"probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
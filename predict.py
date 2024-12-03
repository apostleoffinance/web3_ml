import joblib
from flask import Flask, request, jsonify
import numpy as np

# Load the scaler and model
model_file = 'logistic_regression_model.bin'
scaler, model = joblib.load(model_file)

# Initialize Flask app
#app = Flask(__name__)
app = Flask('trader_class')

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
    app.run(debug=True, host='0.0.0.0', port=5000)
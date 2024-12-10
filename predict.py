import joblib
from flask import Flask, request, jsonify
import numpy as np

# Load the scaler and model
model_file = 'logistic_regression_model.bin'
scaler, model = joblib.load(model_file)

# Initialize Flask app
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

        # Categorize each trader based on prediction values
        categories = []
        for prediction in predictions:
            if prediction == 2:
                trader_category = 'Good Trader'
            elif prediction == 1:
                trader_category = 'Bad Trader'
            else:
                trader_category = 'Average Trader'
            categories.append(trader_category)

        # Return predictions and categories as JSON
        return jsonify({
            "predictions": predictions.tolist(),
            "categories": categories
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

import pickle
import pandas as pd
from flask import Flask, request, jsonify

model_file = 'logistic_regression_model.bin'

# Load the model and scaler from the .bin file
with open(model_file, 'rb') as f_in:
    scaler, model = pickle.load(f_in)

app = Flask('classifier')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        trader = request.get_json()

        # Ensure input is a dictionary and contains the expected keys
        required_features = ['base_cumulative_return', 'portfolio_return', 'daily_sharpe_ratio', 'number_of_trades', 'unique_tokens_traded']  # Replace with actual feature names
        if not isinstance(trader, list) or not all(key in trader for key in required_features):
            return jsonify({'error': 'Invalid input. Missing required features.'}), 400
        
        input_data = pd.DataFrame(trader)
        input_data = input_data.reindex(columns=required_features,fill_value=0)

        # Extract feature values in the correct order
        input_data = [trader[key] for key in required_features]

        # Validate and transform the input data
        if not all(isinstance(value, (int, float)) for value in input_data):
            return jsonify({'error': 'All feature values must be numeric.'}), 400

        X = scaler.transform([input_data])  # Transform input data using the loaded scaler

        # Get predicted class and probabilities
        predicted_class = int(model.predict(X)[0])  # Get the predicted class
        probabilities = model.predict_proba(X)[0]  # Get class probabilities

        # Prepare response
        response = {
            'predicted_class': predicted_class,
            'probabilities': probabilities.tolist()
        }

        return jsonify(response), 200

    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

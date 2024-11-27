import pickle
from sklearn.preprocessing import StandardScaler
from flask import Flask
from flask import request, jsonify

model_file = 'logistic_regression_model.bin'

# Load the model from the .bin file
with open(model_file, 'rb') as f_in:
    scaler, model = pickle.load(f_in)

app = Flask('classifier')

@app.route('/predict', methods=['POST'])
def predict():
    # json = Python dictionary
    trader = request.get_json()

    # Ensure the input data is in the correct format for the scaler and model
    X = scaler.transform([trader])

    # Get the probability of each class
    probabilities = model.predict_proba(X)[0]

     # Get the predicted class
    y_pred = model.predict_proba(X)[0]

    # Format the response with probabilities and predicted class
    response = {
        'predicted_class': int(y_pred),  # Cast to int for JSON serialization
        'probabilities': probabilities.tolist()  # Convert NumPy array to list for JSON serialization
    }

    return jsonify(response)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
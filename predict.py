import pickle
from sklearn.preprocessing import StandardScaler
from flask import Flask
from flask import request

model_file = 'logistic_regression_model.bin'

# Load the model from the .bin file
with open(model_file, 'rb') as f_in:
    scaler, model = pickle.load(f_in)

app = Flask('classifier')

@app.route('/predict', methods=['POST'])
def predict():
    # json = Python dictionary
    trader = request.get_json()

    X = scaler.transform([trader])
    model.predict_proba(X)
    y_pred = model.predict_proba(X)


# # Create a list of values in the correct order





def predict(trader):
   

    result = {
        'Prediction Probabilities': y_pred

    }
    return y_pred




def ping():
    return "PONG"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
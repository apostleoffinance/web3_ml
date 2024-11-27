import pickle
from sklearn.preprocessing import StandardScaler
from flask import Flask
from flask import request

model_file = 'logistic_regression_model.bin'

# Load the model from the .bin file
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('classifier')

# Convert trader dictionary to a 2D array with the same feature order as during training
features = ['base_cumulative_return', 'portfolio_return', 
            'daily_sharpe_ratio', 'number_of_trades', 'unique_tokens_traded']

# Create a list of values in the correct order
trader = request.get_json()
trader_values = [[trader[feature] for feature in features]]
print(trader_values)

#Normalize the Features
scaler = StandardScaler()
X = scaler.fit_transform(trader_values)

@app.route('/predict', methods=['POST'])

def predict(trader):
    trader_scaled = scaler.transform(X)
    y_pred = model.predict_proba(trader_scaled)
    return y_pred




def ping():
    return "PONG"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
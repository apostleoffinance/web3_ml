import pickle
from sklearn.preprocessing import StandardScaler

model_file = 'logistic_regression_model.bin'

# Load the model from the .bin file
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

print("Model loaded successfully!")

trader = { 
    'user_address': '0x09dc02dfb7de2b150fe9a2d2ab92cf5767d423f9', 
    'base_cumulative_return': -0.952871878, 
    'portfolio_return': 19.799527694, 
    'daily_sharpe_ratio': 1.300632733, 
    'number_of_trades': 36.0, 
    'unique_tokens_traded': 6.0, 
    'trader_class_numeric': 1
    }

# Convert trader dictionary to a 2D array with the same feature order as during training
features = ['base_cumulative_return', 'portfolio_return', 
            'daily_sharpe_ratio', 'number_of_trades', 'unique_tokens_traded']



# Create a list of values in the correct order
trader_values = [[trader[feature] for feature in features]]
print(trader_values)

#Normalize the Features
scaler = StandardScaler()
X = scaler.fit_transform(trader_values)

# Apply scaling
trader_scaled = scaler.transform(X)

# Predict the probabilities for each class
prediction_prob = model.predict_proba(trader_scaled)

print('input', trader)
print(f"Prediction Probabilities: {prediction_prob}")
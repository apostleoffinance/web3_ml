import pickle

# Load the model from the pickle file
with open('logistic_regression_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Use the loaded model for predictions
y_pred = loaded_model.predict(X_test)

print("Predictions made successfully!")

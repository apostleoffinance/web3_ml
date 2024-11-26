import pickle

model_file = 'logistic_regression_model.bin'

# Load the model from the .bin file
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

print("Model loaded successfully!")

# Verify by predicting on the test set
y_pred_loaded = model.predict(X_test)

# Compare predictions from the original and loaded models
assert (y_pred == y_pred_loaded).all(), "Mismatch between original and loaded model predictions!"

print("Loaded model predictions match the original!")
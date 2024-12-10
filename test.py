import requests

# Define the API URL
url = 'http://127.0.0.1:8000/predict'

# Define the trader data
trader = [{ 
    "base_cumulative_return" : -0.952871878, 
    "portfolio_return" : 19.799527694, 
    "daily_sharpe_ratio" : 1.300632733, 
    "number_of_trades" : 36.0, 
    "unique_tokens_traded" : 6.0 
}]

# Define the headers
headers = {'Content-Type': 'application/json'}

# Make the POST request
response = requests.post(url, json=trader, headers=headers)

# Parse the JSON response
if response.status_code == 200:
    data = response.json()

    # Retrieve predictions and categories
    predictions = data.get('predictions', [])
    categories = data.get('categories', [])

    # Check the prediction and category for each trader
    if predictions:
        for i, prediction in enumerate(predictions):
            print(f"Prediction: {prediction} -> Category: {categories[i]}")
else:
    print(f"Error: {response.status_code}, {response.text}")

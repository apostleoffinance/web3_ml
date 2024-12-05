
import requests

url = 'http://127.0.0.1:8000/predict'



trader = [{ 
    "base_cumulative_return" : -0.952871878, 
    "portfolio_return" : 19.799527694, 
    "daily_sharpe_ratio" : 1.300632733, 
    "number_of_trades" : 36.0, 
    "unique_tokens_traded" : 6.0 
    }]

headers = {'Content-Type': 'application/json'}

response = requests.post(url, json=trader, headers=headers)

prediction = response.json().get('predictions')[0]

# Categorize the trader based on the prediction value
if prediction == 2:
    trader_category = 'Good Trader'
elif prediction == 1:
    trader_category = 'Bad Trader'
else:
    trader_category = 'Average Trader'

print(f"Trader Category: {trader_category}")





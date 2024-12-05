
import requests

url = 'http://127.0.0.1:5001/predict'



trader = [{ 
    "base_cumulative_return" : -0.952871878, 
    "portfolio_return" : 19.799527694, 
    "daily_sharpe_ratio" : 1.300632733, 
    "number_of_trades" : 36.0, 
    "unique_tokens_traded" : 6.0 
    }]

headers = {'Content-Type': 'application/json'}

response = requests.post(url, json=trader, headers=headers)

print(response.json())





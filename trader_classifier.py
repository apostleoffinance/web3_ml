import requests
import json

url = "https://api-v2.flipsidecrypto.xyz/json-rpc"

payload = json.dumps({
  "jsonrpc": "2.0",
  "method": "createQueryRun",
  "params": [
    {
      "resultTTLHours": 1,
      "maxAgeMinutes": 0,
      "sql": "SELECT date_trunc('hour', block_timestamp) as hourly_datetime, count(distinct tx_hash) as tx_count from ethereum.core.fact_transactions where block_timestamp >= getdate() - interval'1 month' group by 1 order by 1 desc",
      "tags": {
        "source": "postman-demo",
        "env": "test"
      },
      "dataSource": "snowflake-default",
      "dataProvider": "flipside"
    }
  ],
  "id": 1
})
headers = {
  'Content-Type': 'application/json',
  'x-api-key': '{{api_key}}'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)

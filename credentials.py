from flipside import Flipside

# Initialize `Flipside` with your API Key and API Url
flipside = Flipside("<YOUR_API_KEY>", "https://api-v2.flipsidecrypto.xyz")

sql = """
SELECT 
  date_trunc('hour', block_timestamp) as hour,
  count(distinct tx_hash) as tx_count
FROM ethereum.core.fact_transactions 
WHERE block_timestamp >= GETDATE() - interval'7 days'
GROUP BY 1
"""

# Run the query against Flipside's query engine and await the results
query_result_set = flipside.query(sql)
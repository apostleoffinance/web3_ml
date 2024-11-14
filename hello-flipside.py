from flipside import Flipside
import numpy as np
import pandas as pd 
import plotly.graph_objects as go

""" This repo GITIGNORES api_key.txt, use this convention to store your key safely"""

with open('api_key.txt', 'r') as file:
    # Read the first line
    api_key = file.readline().strip()  # .strip() removes newline characters

"""Initialize Flipside with your API Key / API Url"""
flipside = Flipside(api_key, "https://api-v2.flipsidecrypto.xyz")

""" Example Query: ETH-USD Volume Weighted Average Price """
sql = """
with uniswap_ETH_stable_swaps AS ( SELECT block_number, block_timestamp, amount0_adjusted, amount1_adjusted, token0_symbol, token1_symbol 
FROM ethereum.uniswapv3.ez_swaps 
WHERE POOL_ADDRESS IN ( '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8', '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640', 
    '0x4e68ccd3e89f51c3074ca5072bbac773960dfa36', '0x11b815efb8f581194ae79006d24e0d814b7697f6', '0x7bea39867e4169dbe237d55c8242a8f2fcdcc387',
    '0xc2e9f25be6257c210d7adf0d4cd6e3e881ba25f8' ) 
    AND block_timestamp >= DATEADD('day', -30, current_date) 
    ), 
    
    eth_stable_price AS ( SELECT BLOCK_NUMBER, BLOCK_TIMESTAMP, 
        IFF(TOKEN1_SYMBOL = 'WETH', ABS(DIV0(AMOUNT0_ADJUSTED, AMOUNT1_ADJUSTED)), ABS(DIV0(AMOUNT1_ADJUSTED, AMOUNT0_ADJUSTED)) ) as eth_stable_trade_price, 
        IFF(TOKEN1_SYMBOL = 'WETH', ABS(AMOUNT1_ADJUSTED), ABS(AMOUNT0_ADJUSTED)) as eth_volume,
        IFF(TOKEN1_SYMBOL = 'WETH', TOKEN0_SYMBOL, TOKEN1_SYMBOL) as stable 
        FROM uniswap_eth_stable_swaps 
        WHERE ABS(AMOUNT0_ADJUSTED) > 1e-8 AND ABS(AMOUNT1_ADJUSTED) > 1e-8 ), 
        
    eth_block_price AS ( 
    SELECT BLOCK_NUMBER, BLOCK_TIMESTAMP, 
    div0(SUM(eth_stable_trade_price * eth_volume),sum(eth_volume)) as eth_vwap, 
    SUM(eth_volume) as eth_volume, 
    COUNT(*) as num_swaps 
    FROM eth_stable_price 
    GROUP BY BLOCK_NUMBER, BLOCK_TIMESTAMP order by block_number asc 
    ),
    
    hourly1 AS ( 
    SELECT date_trunc('hour', block_timestamp) as hour_, 
    min(eth_vwap) as low, median(eth_vwap) as med_price,
    avg(eth_vwap) as avg_price, max(eth_vwap) as high, 
    sum(eth_volume) as volume, sum(num_swaps) as swaps 
    FROM eth_block_price group by hour_ ORDER BY hour_ asc ),
    
    hourly2 AS ( 
    SELECT distinct date_trunc('hour', block_timestamp) as hour_,
    first_value(eth_vwap) over (partition by hour_ order by block_number asc) as open,
    last_value(eth_vwap) over (partition by hour_ order by block_number asc) as close
    FROM eth_block_price order by hour_ asc )
    
    select hour_, open, low, high, close, close/open as co_ratio, med_price, avg_price, volume, swaps 
    from hourly1 natural join hourly2 order by hour_ asc"""

"""Run the query against Flipside's query engine and await the results"""
query_result_set = flipside.query(sql)

"""This function will be added to Flipside package after testing, just copy/paste as needed for now"""
def auto_paginate_result(query_result_set, page_size=10000):
    """
    This function auto-paginates a query result to get all the data. It assumes 10,000 rows per page.
    In case of an error, reduce the page size. Uses numpy.
    """
    num_rows = query_result_set.page.totalRows
    page_count = np.ceil(num_rows / page_size).astype(int)
    all_rows = []
    current_page = 1
    while current_page <= page_count:
        results = flipside.get_query_results(
            query_result_set.query_id,
            page_number=current_page,
            page_size=page_size
        )

        if results.records:
            all_rows.extend(results.records)  # Use extend() to add list elements

        current_page += 1  # Increment the current page number

    return all_rows  # Return all_rows in JSON format

""" Get your data as a pandas data frame"""

eth_vwap = auto_paginate_result(query_result_set)
eth_vwap = pd.DataFrame(eth_vwap)

""" Plot Candles Data with Plotly """

fig = go.Figure(data=[go.Candlestick(x=eth_vwap['hour_'],
                open=eth_vwap['open'],
                high=eth_vwap['high'],
                low=eth_vwap['low'],
                close=eth_vwap['close'])])

# Customize the layout
fig.update_layout(title='Candlestick Chart',
                xaxis_title='Date and Time',
                yaxis_title='Price',
                xaxis_rangeslider_visible=True)

# Save chart as html you can open in browser 
fig.write_html('candlestick_chart.html')


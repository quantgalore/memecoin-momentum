# -*- coding: utf-8 -*-
"""
Created in 2025

@author: Quant Galore
"""

import requests
import pandas as pd
import numpy as np
import pytz

from datetime import datetime, timedelta

# New assets every day at 8 pm EST (using data cutoff from 7:59:59)
rebalancing_hour = 19

# =============================================================================
# Logic for only delivering output within specified time threshold.
# =============================================================================

yesterday = (datetime.now(tz=pytz.timezone("America/New_York")) - timedelta(days=1)).strftime("%Y-%m-%d")
today = datetime.now(tz=pytz.timezone("America/New_York")).strftime("%Y-%m-%d")

prior_rebalancing_time = (pd.to_datetime(yesterday) + timedelta(hours = rebalancing_hour)).tz_localize("America/New_York")
todays_rebalancing_time = (pd.to_datetime(today) + timedelta(hours = rebalancing_hour)).tz_localize("America/New_York")

right_now = datetime.now(tz=pytz.timezone("America/New_York"))

if right_now <= todays_rebalancing_time:
    # Fetch data as of the last rebalancing
    rebalancing_time = prior_rebalancing_time
else:
    # Fetch the new basket for today
    rebalancing_time = todays_rebalancing_time
    
lookback_period = 7 # historical days used for calcs.
timeframe = "3600s" # hourly data

times = []

try:
    
    # =============================================================================
    # Iteration through valid eligible tokens via ox.fun
    # =============================================================================
    
    ticker_request = pd.json_normalize(requests.get("https://api.ox.fun/v3/tickers?").json()["data"])
    
    ticker_data = ticker_request.copy().apply(pd.to_numeric, errors='ignore').sort_values("currencyVolume24h", ascending=False)
    
    # Heuristic of having at least $10k volume in the past 24 hours.
    ticker_data = ticker_data[ticker_data["volume24h"] > 10000].copy()
    
    market_codes = ticker_data["marketCode"].values
    date = today
    
    dates = pd.date_range(start = (pd.to_datetime(date) - timedelta(days=7)) + timedelta(hours=rebalancing_hour), end = pd.to_datetime(date) + timedelta(hours=rebalancing_hour)).tz_localize("America/New_York")
    
    start_timestamp = dates[0].value // 10**6
    end_timestamp = dates[-1].value // 10**6

    benchmark_candles_request = requests.get(f"https://api.ox.fun/v3/candles?marketCode=BTC-USD-SWAP-LIN&timeframe={timeframe}&startTime={start_timestamp}&endTime={end_timestamp}&limit=500").json()#["data"]
    
    benchmark_data = pd.json_normalize(benchmark_candles_request["data"]).apply(pd.to_numeric, errors='ignore').sort_values("openedAt", ascending=True)
    benchmark_data["timestamp"] = pd.to_datetime(benchmark_data["openedAt"], unit = "ms", utc = True).dt.tz_convert("America/New_York")
    
    data_list = []
    
    # add time component
    
    # market_code = "SOL-USD-SWAP-LIN" # market_codes[0]
    for market_code in market_codes:    
        
        try:
            
            start_time = datetime.now()
    
            candles_request = requests.get(f"https://api.ox.fun/v3/candles?marketCode={market_code}&timeframe={timeframe}&startTime={start_timestamp}&endTime={end_timestamp}&limit=500").json()#["data"]
            
            candles_data = pd.json_normalize(candles_request["data"]).apply(pd.to_numeric, errors='ignore').sort_values("openedAt", ascending=True)
            candles_data["timestamp"] = pd.to_datetime(candles_data["openedAt"], unit = "ms", utc = True).dt.tz_convert("America/New_York")
            
            candles_data["ticker"] = market_code
            
            # data 7 days ago up to the rebalancing time
            historical_data = candles_data[(candles_data["timestamp"] <= rebalancing_time)].copy().sort_values(by="timestamp", ascending=True)
            
            historical_data["returns"] = round(((historical_data["close"] - historical_data["close"].iloc[0]) / historical_data["close"].iloc[0]) * 100, 2)
            historical_data["pct_change"] = round(historical_data["close"].pct_change() *100 , 2)
            
            ticker_return_over_period = historical_data["returns"].iloc[-1]
            std_of_returns = historical_data["pct_change"].std() * np.sqrt(lookback_period)
            
            sharpe = ticker_return_over_period / std_of_returns
            
            # how the asset performed relative to bitcoin (the chosen benchmark)
            combined_data = pd.merge(left = benchmark_data[["close", "timestamp"]], right = historical_data[["close", "timestamp"]], on = "timestamp")
            combined_data["benchmark_pct_change"] = round(combined_data["close_x"].pct_change() * 100, 2).fillna(0)
            combined_data["ticker_pct_change"] = round(combined_data["close_y"].pct_change() * 100, 2).fillna(0)
        
            covariance_matrix = np.cov(combined_data["ticker_pct_change"], combined_data["benchmark_pct_change"])
            covariance_ticker_benchmark = covariance_matrix[0, 1]
            variance_benchmark = np.var(combined_data["benchmark_pct_change"])
            beta = covariance_ticker_benchmark / variance_benchmark
            
            # the metric used for sorting into deciles
            mom_score = beta * sharpe
            
            # data from the rebalancing time onwards (if available)
            forward_data = candles_data[candles_data["timestamp"] >= rebalancing_time].copy()
            
            if len(forward_data) < 1:
                
                forward_return = np.nan
                
            else:
                forward_return = round(((forward_data["close"].iloc[-1] - forward_data["close"].iloc[0]) / forward_data["close"].iloc[0])*100, 2)
            
            return_datapoint = pd.DataFrame([{"date": rebalancing_time.strftime("%Y-%m-%d"), "hist_return": ticker_return_over_period, "sharpe": sharpe,
                                            "beta": beta, "mom_score": mom_score,
                                            "return_since_added": forward_return, 
                                            "ticker": market_code, "t": candles_data["timestamp"].iloc[-1]}])
            
            
            data_list.append(return_datapoint)
            
            end_time = datetime.now()
            elapsed_time = end_time
            seconds_to_complete = (end_time - start_time).total_seconds()
            times.append(seconds_to_complete)
            iteration = round((np.where(market_codes==market_code)[0][0]/len(market_codes))*100,2)
            iterations_remaining = len(market_codes) - np.where(market_codes==market_code)[0][0]
            average_time_to_complete = np.mean(times)
            estimated_completion_time = (datetime.now() + timedelta(seconds = int(average_time_to_complete*iterations_remaining)))
            time_remaining = estimated_completion_time - datetime.now()
            print(f"{iteration}% complete, {time_remaining} left, ETA: {estimated_completion_time}")
            
        except Exception as error:
            print(error)
            continue
        
    full_dataset = pd.concat(data_list).sort_values(by="mom_score", ascending=False)
    
    top_decile = full_dataset.head(3)
    
    print(f"\n\nTop 3 Highest Momentum Cryptocurrencies (ox.fun) – Last Rebalancing Date: {rebalancing_time.strftime('%Y-%m-%d')}")    
    print(f"{top_decile['ticker'].iloc[0].split('-')[0]} - Return Since Added: {top_decile['return_since_added'].iloc[0]}%")
    print(f"{top_decile['ticker'].iloc[1].split('-')[0]} - Return Since Added: {top_decile['return_since_added'].iloc[1]}%")
    print(f"{top_decile['ticker'].iloc[2].split('-')[0]} - Return Since Added: {top_decile['return_since_added'].iloc[2]}%")

except Exception as error:
    print(error)
import pandas as pd
import datetime as dt
import numpy as np
import yfinance as yf
import pandas_ta as ta

def getStockData(ticker):
    # Set time frame for training data
    end = dt.datetime.now()
    start = end - dt.timedelta(days=3650) # Days can be adjusted

    # Get data from Yahoo Finance using inputed ticker symbol and put data into data frame -- 'df'
    df = yf.Ticker(ticker)
    df = df.history(period="max")
    
    return df

def convertData(df):
    df['RSI'] = ta.rsi(df.Close, length=15)
    df['EMFA'] = ta.ema(df.Close, length=20)
    df['EMFM'] = ta.ema(df.Close, length=100)
    df['EMFS'] = ta.ema(df.Close, length=150)

    df['Tomorrow'] = df['Close'].shift(-1)

    df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)
    

    df.dropna(inplace = True)
    df.reset_index(inplace = True)

    return df

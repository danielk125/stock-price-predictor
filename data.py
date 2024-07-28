import pandas as pd
import datetime as dt
import yfinance as yf

def getStockData(ticker):
    # Set time frame for training data
    end = dt.datetime.now()
    start = end - dt.timedelta(days=365)

    # Get data from Yahoo Finance using inputed ticker symbol and put data into data frame -- 'df'
    df = yf.download(ticker, start=start, end=end)
    
    return df
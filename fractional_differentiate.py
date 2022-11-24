#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pmdarima as pm
import math
import plotly.graph_objs as go
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.stattools import adfuller
plt.rcParams.update({"figure.figsize": (10, 6), "figure.dpi": 120})


# SNIPPET 5.1: page 79
def getWeights(d, size):
    # thres > 0 drops insignificant weights
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d-k+1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w



# SNIPPET 5.2: page 82
def fracDiff(series, d, thres=.01):
    """
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0, 1].
    """
    #1) Compute weights for the longest series
    w = getWeights(d, series.shape[0])
    #2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_/= w_[-1]
    skip = w_[w_ > thres].shape[0]
    #3) Apply weights to values
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            if not np.isfinite(series.loc[loc, name]): continue # exclude NAs
            # figure out this line 
            df_[loc] = np.dot(w[-(iloc+1):, :].T, seriesF.loc[:loc])[0, 0]
        df[name]=df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


# SNIPPET5.3: page 83
def getWeights_FFD(d, thres):
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k*(d-k+1)
        if abs(w_) < thres: break
        w.append(w_); k+=1
    return np.array(w[::-1]).reshape(-1, 1)

def fracDiff_FFD(series, d, thres=1e-4):
    # Constant width window (new solution)
    series = series[~series.index.duplicated(keep='first')]
    w = getWeights_FFD(d, thres)
    width = len(w)-1
    df = {}
    for i, name in enumerate(series.columns):
        seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series(dtype='float64')
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1-width], seriesF.index[iloc1]
            if not np.isfinite(series.iloc[iloc1, i]): continue # exclude NAs
            try:
                df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
            except:
                df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1][1:])[0, 0]
                    
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df

def plot_frac_actual_price(actual_price: pd.DataFrame(), frac_price: pd.DataFrame(), plot_size: int, column: str):
    """
    Plot actual price and fractional differentiated price. 
    
    Args:
        actual_price:(pandas DataFrame) full dataset of actual price. bar data
        frac_price: (pandas DataFrame) full data of fractional differentiated price data
        plot_size: length of plot you wish to show. Max is length of frac_price
        column: (str) price you wish to see ('Open', 'High', 'Low', 'Close')
    """
    plt.rcParams.update({'figure.figsize': (10, 6), 'figure.dpi': 120})
    column = column
    plot_size = plot_size
    d1 = actual_price[column].values
    d2 = frac_price[column].values

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Time (Year)')
    ax1.set_ylabel('Price', color=color)
    ax1.plot(range(len(d1)-plot_size, len(d1)), d1[-plot_size:], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Frac Diff', color=color)
    ax2.plot(range(len(d1)-plot_size, len(d1)), d2[-plot_size:], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.tight_layout()
    plt.show()
    
    
# SNIPPET 5.4: page 85
def plotMinFrac(data):
    out = pd.DataFrame(columns=["adfstat", 'P-val', 'Lags', 'n0bs', '95% conf', 'corr'])
    df0 = data
    for d in np.linspace(0.1, 0.5, 41):
        df1 = df0[["Close"]].resample('1D').last() # downcast to daily obs
        df2 = fracDiff(df1, d, thres=.01)
        corr = np.corrcoef(df1.loc[df2.index, 'Close'], df2["Close"])[0, 1]
        df2 = adfuller(df2["Close"], maxlag=1, regression='c', autolag=None)
        out.loc[d] = list(df2[:4]) + [df2[4]['5%']] + [corr] # with critical value
    out[["adfstat", "corr"]].plot(secondary_y='adfstat')
    plt.axhline(out['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')
    plt.show()
    print(out)
    return 


def plotMinFracFFD(data, thres):
    out = pd.DataFrame(columns=["adfstat", 'P-val', 'Lags', 'n0bs', '95% conf', 'corr'])
    df0 = data
    for d in np.linspace(0, 1, 11):
        df1 = df0[["Close"]].resample('1D').last() # downcast to daily obs
        df2 = fracDiff_FFD(df1, d, thres=thres)
        #print(f"Weight vector size (d={d}): {weight.shape}")
        corr = np.corrcoef(df1.loc[df2.index, 'Close'], df2["Close"])[0, 1]
        df2 = adfuller(df2["Close"], maxlag=1, regression='c', autolag=None)
        out.loc[d] = list(df2[:4]) + [df2[4]['5%']] + [corr] # with critical value
    out[["adfstat", "corr"]].plot(secondary_y='adfstat')
    plt.axhline(out['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')
    plt.show()
    print(out)
    return 








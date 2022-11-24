#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from fractional_differentiate import fracDiff_FFD


# In[ ]:


def kpss_test(series, **kw):
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print(f'Critical Values:')
    for key, value in critical_values.items():
        print(f'   {key}: {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')
    


def adf_test(series):
    result = adfuller(series, maxlag=1, regression='c', autolag=None)
    adf = result[0]
    p = result[1]
    usedlag = result[2]
    nobs = result[3]
    cri_val_1 = result[4]["1%"]
    cri_val_5 = result[4]["5%"]
    cri_val_10 = result[4]["10%"]
    #icbest = result[5]
    print(f"Test Statistic: {adf}\np-value: {p}\n#Lags Used: {usedlag}\nNumber of Observations: {nobs}\nCritical Value (1%): {cri_val_1}\nCritical Value (5%): {cri_val_5}\nCritical Value (10%): {cri_val_10}\n")
    print(f'Result: The series is {"not " if p > 0.05 else ""}stationary')


# In[ ]:


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
    print(out)
    return 


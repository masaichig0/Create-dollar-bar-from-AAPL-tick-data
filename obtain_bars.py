#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from create_bars import GetBars
from fractional_differentiate import fracDiff_FFD

def download_dfs_get_bars(bar_type: str, size: int, data_dir: str, symbol: str, file_name: str) ->list:
    """
    This function load the data from DATA directory then create tick, volume or dollar bars. 
    
    Aregs:
        bar_type: 'tick', 'volume' or 'dollar'
        size: Size of the bar. 
        data_dir: DATA directory
        symbol: symbol of the company
        file_name: name of the file 
        
    Return:
        List of Bar data. 
    """
    
    df = GetBars(data_dir, file_name, symbol, keep_off_time=False)
    if bar_type == 'tick':
        data = df.tick_bars(size)
    elif bar_type == 'volume':
        data = df.volume_bars(size)
    elif bar_type == 'dollar':
        data = df.dollar_bars(size)
    
    print(f'{file_name} processed')
    return data




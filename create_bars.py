#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime as dt


# In[2]:


class PrepBars_:
    
    def __init__(self, raw_data, symbol):
        self.df = raw_data
        self.symbol = symbol
        
    def set_datetime_index(self, timestamp=False):
        self.df["Datetime"] = self.df[['Date', 'Time']].agg(lambda x: ' '.join(x.values), axis=1)
        self.df["Datetime"] = pd.to_datetime(self.df["Datetime"])
        if timestamp:
            self.df['Datetime'] = self.df['Datetime'].view('int64')
        return self.df.set_index('Datetime')
        
    def get_raw_data(self):
        # Trade data
        trade_data = self.df
        columns = trade_data.columns
        # Change column names
        trade_data = trade_data.rename(columns={columns[0]: 'Date', columns[1]: 'Time', columns[2]: 'Price', 
                        columns[3]: 'Volume', columns[4]: 'Exchange Code', columns[5]: 'Sales Condition', 
                        columns[6]: 'Correction Indicator', columns[7]: 'Sequence Number', columns[8]: 'Trade Stop Indicator', 
                        columns[9]: 'Source of Trade', columns[10]: 'MDS 127 / TRF', columns[11]: 'Exclude Record Flag', 
                        columns[12]: 'Filtered Price'})
    
        # Exclude bars that Exclude record flag is 'X'
        exclude_bar = trade_data[trade_data['Exclude Record Flag'] == 'X']
        print(f'The number of exclude bars: {len(exclude_bar)}')
        print(f'Original data length: {len(trade_data)}')
        for idx in exclude_bar.index:
            trade_data.drop(index=idx, inplace=True)
        print(f'Data length after drop rows: {len(trade_data)}')
    
        # Combine Date and time together and change the data type to datetime, then set as index
        trade_data["Datetime"] = trade_data[['Date', 'Time']].agg(lambda x: ' '.join(x.values), axis=1)
        trade_data["Datetime"] = pd.to_datetime(trade_data["Datetime"])
        #trade_data["Datetime_"] = trade_data['Datetime']
        #trade_data.set_index('Datetime', inplace=True)
    
        # Add symbol to DataFrame and get columns I need
        trade_data["Symbol"] = self.symbol
        return trade_data[['Datetime', 'Date', 'Symbol', 'Price', 'Volume']]
    
    def add_tick_info(self):
        data = self.get_raw_data()
        data['Dollar Traded'] = data['Price'] * data['Volume']
        data['Tick'] = np.arange(len(data))
        data['Num of Tick'] = 1
        return data[['Datetime', 'Date', 'Tick', 'Symbol', 'Num of Tick', 'Price', 'Volume', 'Dollar Traded']]
    
    def tick_rule(self):
        trade_data = self.add_tick_info()
        trade_data['Price Change'] = trade_data['Price'].diff()
        difference = trade_data['Price Change'].dropna()
        sequences = [1]
        for diff in difference:
            if diff > 0:
                seq = 1
                sequences.append(seq)
            elif diff < 0:
                seq = -1
                sequences.append(seq)
            elif diff == 0:
                seq = sequences[-1]
                sequences.append(seq)

        sequences = np.array(sequences)
        trade_data['Tick Imbalance'] = sequences
        return trade_data[['Datetime', 'Date', 'Tick', 'Symbol', 'Num of Tick', 'Price', 'Volume', 'Dollar Traded', 
                              'Price Change', 'Tick Imbalance']]
        
    def buy_volume(self):
        trade_data = self.tick_rule()
        # Same logic to create buy volume if sequence is 1
        buy_volume = []
        for vol, seq in trade_data.loc[:, [ 'Volume', 'Tick Imbalance']].values:
            if seq == -1:
                buy_vol = 0
                buy_volume.append(buy_vol)
            elif seq == 1:
                buy_vol = vol
                buy_volume.append(buy_vol)
        trade_data['Buy Volume'] = buy_volume
        return trade_data[['Datetime', 'Date', 'Tick', 'Symbol', 'Num of Tick', 'Price', 'Volume', 'Dollar Traded', 
                              'Price Change', 'Tick Imbalance', 'Buy Volume']]
        
    def cumlative_imbalances(self):
        trade_data = self.buy_volume()
        trade_data['Cum Tick Imbalance'] = trade_data['Tick Imbalance'].cumsum()
        trade_data['Volume Imbalance'] = trade_data['Tick Imbalance'] * trade_data['Volume']
        trade_data['Cum Volume Imbalance'] = trade_data['Volume Imbalance'].cumsum()
        trade_data['Dollar Imbalance'] = trade_data['Tick Imbalance'] * trade_data['Volume'] * trade_data['Price']
        trade_data['Cum Dollar Imbalance'] = trade_data['Dollar Imbalance'].cumsum()
        return trade_data[['Datetime', 'Date', 'Tick', 'Symbol', 'Num of Tick', 'Price', 'Price Change','Volume', 'Buy Volume', 
                           'Dollar Traded', 'Tick Imbalance', 'Volume Imbalance', 'Dollar Imbalance', 
                          'Cum Tick Imbalance', 'Cum Volume Imbalance', 'Cum Dollar Imbalance']]
    
class GetBars_(PrepBars_):
    
    def __init__(self, raw_data, symbol):
        super(PrepBars).__init__()
        self.df = raw_data
        self.symbol = symbol
    
    def drop_off_time_info(self, df):
        return df.between_time(start_time='9:30', end_time='16:00', inclusive='both')
    
    def prep_create_bar(self):
        raw_data = self.cumlative_imbalances()
        raw_data.set_index('Datetime', inplace=True)
        raw_data = self.drop_off_time_info(raw_data)
        return raw_data
    
    def get_cumlatives(self, bars):
        bars.reset_index(inplace=True)
        bars['Cum_ticks'] = bars.groupby(bars['Datetime'].dt.day)['Num of Tick'].cumsum()
        bars['Cum_Volume'] = bars.groupby(bars['Datetime'].dt.day)['Volume'].cumsum()
        bars['Cum_Dollar_Traded'] = bars.groupby(bars['Datetime'].dt.day)['Dollar Traded'].cumsum()
        return bars
    
    def time_bars(self, freq='5min'):
        raw_data = self.prep_create_bar()
        time_bars = raw_data.groupby(pd.Grouper(freq=freq)).agg({'Symbol': 'last', 'Tick': 'last', "Buy Volume": 'sum',
                                            'Num of Tick': 'sum', 'Price': 'ohlc', 'Volume': 'sum', 'Dollar Traded': 'sum'})
    
        symbol = time_bars.Symbol
        tick = time_bars['Tick']
        tick_num = time_bars['Num of Tick']
        volume = time_bars.Volume
        buy_vol = time_bars['Buy Volume']
        prices = time_bars.Price
        prices = prices.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
        dollar_traded = time_bars['Dollar Traded']
        time_bars =  pd.concat((tick, symbol, tick_num, prices, volume, buy_vol, dollar_traded), 
                               axis=1, join='inner')
        time_bars = self.get_cumlatives(time_bars)
        print(f'Generate {freq} Time bars')
        return time_bars
    
    def bar(self, x, y):
        return np.int64(x/y) * y
    
    def tick_bars(self, tick_size):
        raw_data = self.prep_create_bar()
        raw_data.reset_index(inplace=True)
        tick_bars = raw_data.groupby(self.bar(raw_data['Tick'], tick_size)).agg({'Datetime': 'last', 'Date': 'first',
                                                                        'Symbol': 'last', 'Tick': 'last', "Buy Volume": 'sum',
                                            'Num of Tick': 'sum', 'Price': 'ohlc', 'Volume': 'sum', 'Dollar Traded': 'sum'})
        datetime = tick_bars.Datetime 
        date = tick_bars.Date
        symbol = tick_bars.Symbol
        tick = tick_bars.Tick
        tick_num = tick_bars['Num of Tick']
        volume = tick_bars.Volume
        buy_vol = tick_bars['Buy Volume']
        prices = tick_bars.Price
        prices = prices.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
        dollar_traded = tick_bars['Dollar Traded']
        tick_bars = pd.concat((date, datetime, tick, symbol, tick_num, prices, volume, buy_vol, dollar_traded), 
                              axis=1, join='inner')
        tick_bars = self.get_cumlatives(tick_bars)
        print(f'Generate Tick bars, Num of Tick: {tick_size}')
        return tick_bars.drop('index', axis=1)
    
    def volume_bars(self, threshold):
        raw_data = self.prep_create_bar()
        raw_data.reset_index(inplace=True)
        cum_volume = np.cumsum(raw_data['Volume'])
        volume_bars = raw_data.groupby(self.bar(cum_volume, threshold)).agg({'Datetime': 'last', 'Date': 'first', 
                                                                        'Symbol': 'last', 'Tick': 'last', 'Buy Volume': 'sum',
                                            'Num of Tick': 'sum', 'Price': 'ohlc', 'Volume': 'sum', 'Dollar Traded': 'sum'})
        datetime = volume_bars.Datetime 
        date = volume_bars.Date
        symbol = volume_bars.Symbol
        tick = volume_bars.Tick
        tick_num = volume_bars['Num of Tick']
        volume = volume_bars.Volume
        buy_vol = volume_bars['Buy Volume']
        prices = volume_bars.Price
        prices = prices.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
        dollar_traded = volume_bars['Dollar Traded']
        volume_bars = pd.concat((date, datetime, tick, symbol, tick_num, prices, volume, buy_vol, dollar_traded), 
                                axis=1, join='inner')
        volume_bars = self.get_cumlatives(volume_bars)
        print(f'Generate Volume bar, Volume: {threshold}')
        return volume_bars.drop('index', axis=1)
    
    def dollar_bars(self, threshold):
        raw_data = self.prep_create_bar()
        raw_data.reset_index(inplace=True)
        cum_dollar_val = np.cumsum(raw_data['Dollar Traded'])
        dollar_bars = raw_data.groupby(self.bar(cum_dollar_val, threshold)).agg({'Date': 'first', 'Datetime': 'last', 
                                            'Symbol': 'last', 'Tick': 'last', 'Buy Volume': 'sum',
                                            'Num of Tick': 'sum', 'Price': 'ohlc', 'Volume': 'sum', 'Dollar Traded': 'sum'})
        datetime = dollar_bars.Datetime 
        date = dollar_bars.Date
        symbol = dollar_bars.Symbol
        tick = dollar_bars.Tick
        tick_num = dollar_bars['Num of Tick']
        volume = dollar_bars.Volume
        buy_vol = dollar_bars["Buy Volume"]
        prices = dollar_bars.Price
        prices = prices.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
        dollar_traded = dollar_bars['Dollar Traded']
        dollar_bars = pd.concat((date, datetime, tick, symbol, tick_num, prices, volume, buy_vol, dollar_traded), 
                                axis=1, join='inner')
        dollar_bars = self.get_cumlatives(dollar_bars)
        print(f'Generate Dollar bar, Dollar Traded: {threshold}')
        return dollar_bars.drop('index', axis=1)


# In[3]:
class PrepBars:
    """
    This class for the data from http://stocktickdata.co/
    """
    
    def __init__(self, data_dir, data_name, symbol):
        self.data_dir = data_dir
        self.data_name = data_name
        self.symbol = symbol
        self.first_split = '2014-06-09'
        self.second_split = '2020-08-31'
        self.first_split_size = 28
        self.second_split_size = 4
        
    # I still need to work on this part. 
    def split_data(self):
        dfs = []
        for dfchunk in pd.read_csv(self.data_dir + self.data_name, chunksize=self.chunksize):
            day_df = create_time_bar(dfchunk)
            dfs.append(day_df)
        return pd.concat(dfs, ignore_index=True)
        
    def data_extraction(self):
        header = ["Datetime", "Price", "Volume", "Exchange"]
        df = pd.read_csv(self.data_dir + self.data_name, sep=",", header=None, names=header)
        df['Datetime'] = pd.to_datetime(df["Datetime"], format="%Y-%m-%d %H:%M:%S:%f")
        df["Date"] = [d.date() for d in df.Datetime]
        df.set_index('Datetime', inplace=True)
        return df.drop("Exchange", axis=1)
    
    def adjust_split(self):
        df = self.data_extraction()
        
        # If first split date in the data
        if self.first_split in df.Date:
            before_split = df.loc[df.index < self.first_split]
            index_before = before_split.index
            after_split = df.drop(before_split.index)
            index_after = after_split.index
            before_split_price = before_split.Price.to_numpy() / self.first_split_size
            before_split_vol = before_split.Volume.to_numpy() * self.first_split_size
            after_split_price = after_split.Price.to_numpy() / self.second_split_size
            after_split_vol = after_split.Volume.to_numpy() * self.second_split_size
            before_df = pd.DataFrame({'Price': before_split_price, 'Volume': before_split_vol}, index=index_before)
            after_df = pd.DataFrame({'Price': after_split_price, 'Volume': after_split_vol}, index=index_after)
            df = pd.concat((before_df, after_df))
        # if the data has second split date
        elif self.second_split in df.Date:
            before_split = df.loc[df.index < self.second_split]
            index_before = before_split.index
            after_split = df.drop(before_split.index)
            index_after = after_split.index
            before_split_price = before_split.Price.to_numpy() / self.second_split_size
            before_split_vol = before_split.Volume.to_numpy() * self.second_split_size
            after_df = after_split.drop('Date', axis=1)
            df = pd.concat((before_df, after_df))
        # data before first split
        elif df.Date[-1] < dt.datetime.strptime(self.first_split, '%Y-%m-%d').date():
            df['Price'] = df["Price"] / self.first_split_size
            df['Volume'] = df['Volume'] * self.first_split_size
        # Data between first split and second split
        elif df.Date[0] >= dt.datetime.strptime(self.first_split, '%Y-%m-%d').date() and df.Date[-1] < dt.datetime.strptime(self.second_split, '%Y-%m-%d').date():
            df['Price'] = df["Price"] / self.second_split_size
            df['Volume'] = df['Volume'] * self.second_split_size
        # after second split
        else:
            df = df.drop('Date', axis=1)
        return df.reset_index()
        
    def get_raw_data(self):
        # Trade data
        trade_data = self.adjust_split()
    
        # Add symbol to DataFrame and get columns I need
        trade_data["Symbol"] = self.symbol
        return trade_data[['Datetime', 'Symbol', 'Price', 'Volume']]
    
    def add_tick_info(self):
        data = self.get_raw_data()
        data['Dollar Traded'] = data['Price'] * data['Volume']
        data['Tick'] = np.arange(len(data))
        data['Num of Tick'] = 1
        return data[['Datetime', 'Tick', 'Symbol', 'Num of Tick', 'Price', 'Volume', 'Dollar Traded']]
    
    def tick_rule(self):
        trade_data = self.add_tick_info()
        trade_data['Price Change'] = trade_data['Price'].diff()
        difference = trade_data['Price Change'].dropna()
        sequences = [1]
        for diff in difference:
            if diff > 0:
                seq = 1
                sequences.append(seq)
            elif diff < 0:
                seq = -1
                sequences.append(seq)
            elif diff == 0:
                seq = sequences[-1]
                sequences.append(seq)

        sequences = np.array(sequences)
        trade_data['Tick Imbalance'] = sequences
        return trade_data[['Datetime', 'Tick', 'Symbol', 'Num of Tick', 'Price', 'Volume', 'Dollar Traded', 
                              'Price Change', 'Tick Imbalance']]
        
    def buy_volume(self):
        trade_data = self.tick_rule()
        # Same logic to create buy volume if sequence is 1
        buy_volume = []
        for vol, seq in trade_data.loc[:, [ 'Volume', 'Tick Imbalance']].values:
            if seq == -1:
                buy_vol = 0
                buy_volume.append(buy_vol)
            elif seq == 1:
                buy_vol = vol
                buy_volume.append(buy_vol)
        trade_data['Buy Volume'] = buy_volume
        return trade_data[['Datetime', 'Tick', 'Symbol', 'Num of Tick', 'Price', 'Volume', 'Dollar Traded', 
                              'Price Change', 'Tick Imbalance', 'Buy Volume']]
        
    def cumlative_imbalances(self):
        trade_data = self.buy_volume()
        trade_data['Cum Tick Imbalance'] = trade_data['Tick Imbalance'].cumsum()
        trade_data['Volume Imbalance'] = trade_data['Tick Imbalance'] * trade_data['Volume']
        trade_data['Cum Volume Imbalance'] = trade_data['Volume Imbalance'].cumsum()
        trade_data['Dollar Imbalance'] = trade_data['Tick Imbalance'] * trade_data['Volume'] * trade_data['Price']
        trade_data['Cum Dollar Imbalance'] = trade_data['Dollar Imbalance'].cumsum()
        return trade_data[['Datetime', 'Tick', 'Symbol', 'Num of Tick', 'Price', 'Price Change','Volume', 'Buy Volume', 
                           'Dollar Traded', 'Tick Imbalance', 'Volume Imbalance', 'Dollar Imbalance', 
                          'Cum Tick Imbalance', 'Cum Volume Imbalance', 'Cum Dollar Imbalance']]
    
class GetBars(PrepBars):
    
    def __init__(self, data_dir, data_name, symbol, keep_off_time=True):
        super(PrepBars).__init__()
        self.data_dir = data_dir
        self.data_name = data_name
        self.symbol = symbol
        self.first_split = '2014-06-09'
        self.second_split = '2020-08-31'
        self.first_split_size = 28
        self.second_split_size = 4
        self.keep_off_time=keep_off_time
    
    def drop_off_time_info(self, df):
        return df.between_time(start_time='9:30', end_time='16:00', inclusive='both')
    
    def prep_create_bar(self):
        raw_data = self.cumlative_imbalances()
        raw_data.set_index('Datetime', inplace=True)
        if not self.keep_off_time:
            raw_data = self.drop_off_time_info(raw_data)
        return raw_data
    
    def get_cumlatives(self, bars):
        bars.reset_index(inplace=True)
        bars['Cum_ticks'] = bars.groupby(bars['Datetime'].dt.day)['Num of Tick'].cumsum()
        bars['Cum_Volume'] = bars.groupby(bars['Datetime'].dt.day)['Volume'].cumsum()
        bars['Cum_Buy_Volume'] = bars.groupby(bars['Datetime'].dt.day)['Buy Volume'].cumsum()
        bars['Cum_Dollar_Traded'] = bars.groupby(bars['Datetime'].dt.day)['Dollar Traded'].cumsum()
        return bars
    
    def time_bars(self, freq='5min'):
        raw_data = self.prep_create_bar()
        time_bars = raw_data.groupby(pd.Grouper(freq=freq)).agg({'Symbol': 'last', 'Tick': 'last', "Buy Volume": 'sum',
                                            'Num of Tick': 'sum', 'Price': 'ohlc', 'Volume': 'sum', 'Dollar Traded': 'sum'})
    
        symbol = time_bars.Symbol
        tick = time_bars['Tick']
        tick_num = time_bars['Num of Tick']
        volume = time_bars.Volume
        buy_vol = time_bars['Buy Volume']
        prices = time_bars.Price
        prices = prices.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
        dollar_traded = time_bars['Dollar Traded']
        time_bars =  pd.concat((tick, symbol, tick_num, prices, volume, buy_vol, dollar_traded), 
                               axis=1, join='inner')
        time_bars = self.get_cumlatives(time_bars)
        print(f'Generate {freq} Time bars')
        return time_bars
    
    def bar(self, x, y):
        return np.int64(x/y) * y
    
    def tick_bars(self, tick_size):
        raw_data = self.prep_create_bar()
        raw_data.reset_index(inplace=True)
        tick_bars = raw_data.groupby(self.bar(raw_data['Tick'], tick_size)).agg({'Datetime': 'last', 
                                                                        'Symbol': 'last', 'Tick': 'last', "Buy Volume": 'sum',
                                            'Num of Tick': 'sum', 'Price': 'ohlc', 'Volume': 'sum', 'Dollar Traded': 'sum'})
        datetime = tick_bars.Datetime 
        symbol = tick_bars.Symbol
        tick = tick_bars.Tick
        tick_num = tick_bars['Num of Tick']
        volume = tick_bars.Volume
        buy_vol = tick_bars['Buy Volume']
        prices = tick_bars.Price
        prices = prices.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
        dollar_traded = tick_bars['Dollar Traded']
        tick_bars = pd.concat((datetime, tick, symbol, tick_num, prices, volume, buy_vol, dollar_traded), 
                              axis=1, join='inner')
        tick_bars = self.get_cumlatives(tick_bars)
        print(f'Generate Tick bars, Num of Tick: {tick_size}')
        return tick_bars.drop('index', axis=1)
    
    def volume_bars(self, threshold):
        raw_data = self.prep_create_bar()
        raw_data.reset_index(inplace=True)
        cum_volume = np.cumsum(raw_data['Volume'])
        volume_bars = raw_data.groupby(self.bar(cum_volume, threshold)).agg({'Datetime': 'last', 
                                                                        'Symbol': 'last', 'Tick': 'last', 'Buy Volume': 'sum',
                                            'Num of Tick': 'sum', 'Price': 'ohlc', 'Volume': 'sum', 'Dollar Traded': 'sum'})
        datetime = volume_bars.Datetime 
        symbol = volume_bars.Symbol
        tick = volume_bars.Tick
        tick_num = volume_bars['Num of Tick']
        volume = volume_bars.Volume
        buy_vol = volume_bars['Buy Volume']
        prices = volume_bars.Price
        prices = prices.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
        dollar_traded = volume_bars['Dollar Traded']
        volume_bars = pd.concat((datetime, tick, symbol, tick_num, prices, volume, buy_vol, dollar_traded), 
                                axis=1, join='inner')
        volume_bars = self.get_cumlatives(volume_bars)
        print(f'Generate Volume bar, Volume: {threshold}')
        return volume_bars.drop('index', axis=1)
    
    def dollar_bars(self, threshold):
        raw_data = self.prep_create_bar()
        raw_data.reset_index(inplace=True)
        cum_dollar_val = np.cumsum(raw_data['Dollar Traded'])
        dollar_bars = raw_data.groupby(self.bar(cum_dollar_val, threshold)).agg({'Datetime': 'last',
                                            'Symbol': 'last', 'Tick': 'last', 'Buy Volume': 'sum',
                                            'Num of Tick': 'sum', 'Price': 'ohlc', 'Volume': 'sum', 'Dollar Traded': 'sum'})
        datetime = dollar_bars.Datetime 
        symbol = dollar_bars.Symbol
        tick = dollar_bars.Tick
        tick_num = dollar_bars['Num of Tick']
        volume = dollar_bars.Volume
        buy_vol = dollar_bars["Buy Volume"]
        prices = dollar_bars.Price
        prices = prices.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
        dollar_traded = dollar_bars['Dollar Traded']
        dollar_bars = pd.concat((datetime, tick, symbol, tick_num, prices, volume, buy_vol, dollar_traded), 
                                axis=1, join='inner')
        dollar_bars = self.get_cumlatives(dollar_bars)
        print(f'Generate Dollar bar, Dollar Traded: {threshold}')
        return dollar_bars.drop('index', axis=1)
    
    

class GetImbalanceBars(PrepBars):
    
    def __init__(self, raw_data, symbol):
        super(PrepBars).__init__()
        self.df = raw_data
        self.symbol = symbol
        
class CusumFilter(PrepBars):
    
    def __init__(self, raw_data, symbol):
        super(PrepBars).__init__()
        self.df = raw_data
        self.symbol = symbol
    
    def get_t_events(self, h=0.05):
        """
        h: Threshold value default=0.05
        """
        data = self.cumlative_imbalances()
        price = data['Price']
        t_events, s_pos, s_neg = [], 0, 0
        diff = price.diff()
        for i in range(len(diff[1:])):
            s_pos, s_neg = max(0, s_pos + diff.iloc[i]), min(0, s_neg + diff.iloc[i])
            if s_neg < -h:
                s_neg = 0
                t_events.append(i)
            elif s_pos > h:
                s_pos = 0
                t_events.append(i)
        #print(t_events)
        filterd_index = data.iloc[t_events].index
        return pd.DatetimeIndex(filterd_index) 



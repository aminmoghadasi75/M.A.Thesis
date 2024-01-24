import pandas as pd
import numpy as np
import os
import warnings
import talib
import ta


warnings.filterwarnings('ignore')

def adding_Tech_Fund(df_path):
    # Get Data Frame
    df = pd.DataFrame(pd.read_excel(df_path))

    #  Classify the input data
    c_price = df['پایانی*سهم*']
    h_price = df['بالاترین*سهم*']
    l_price = df['پایین\u200cترین*سهم*']
    volume = df['حجم*سهم*']
    pct_close = c_price.pct_change()
    if 'شاخص بورس*بورس*' in df.columns:
        shakhes_kol = df['شاخص بورس*بورس*']
    if 'شاخص فرابورس*فرابورس*' in df.columns:
        shakhes_kol = df['شاخص فرابورس*فرابورس*']


    Market_Returns = shakhes_kol.pct_change()
    Stock_Returns = c_price.pct_change()

    pe_ratio = df['P/E-ttm*سهم*']

    #  World Bank DATA
    expected_growth_rates = {
        2013: 0.273, 2014: 0.367,
        2015: 0.166, 2016: 0.127,
        2017: 0.072, 2018: 0.08,
        2019: 0.18, 2020: 0.399,
        2021: 0.306, 2022: 0.436,
        2023: 0.435,
        }
    risk_free = df['نرخ بدون ریسک*کلان*']





    # *****TECHNICAL PARAMETERS***** #

    # Moving Average
    df['MA26'] = talib.MA(c_price, timeperiod= 26)
    df['MA50'] = talib.MA(c_price, timeperiod= 50)
    df['SMA26'] = talib.SMA(c_price, timeperiod= 26)
    df['SMA50'] = talib.SMA(c_price, timeperiod= 50)

    df['EMA26'] = talib.EMA(c_price, timeperiod= 26)
    df['EMA50'] = talib.EMA(c_price, timeperiod= 50)

    ### MACD
    macd, signal, _ = talib.MACD(c_price, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_signal'] = signal

    ### Bollinger Boands
    upper_band, middle_band, lower_band = talib.BBANDS(c_price, timeperiod=30, nbdevup=2, nbdevdn=2)
    df['Upper_Bollinger_band'] = upper_band
    df['Middle_Bollinger_band'] = middle_band
    df['Lower_Bollinger_band'] = lower_band

    ### RSI
    rsi = talib.RSI(c_price, timeperiod=30)
    df['RSI'] = rsi

    ### MFI
    mfi = talib.MFI(h_price, l_price, c_price, volume, timeperiod=14)
    df['MFI'] = mfi
    ### Momentum
    momentum12 = talib.MOM(c_price, timeperiod=12)
    df['Momentum_12'] = momentum12

    ### Stochastic Oscillator
    slowk, slowd = talib.STOCH(h_price, l_price, c_price, fastk_period=5 , slowk_period=3, slowd_period=3)
    df['Stochastic Oscillator_slowk'] = slowk
    df['Stochastic Oscillator_slowd'] = slowd

    ### On-Balance Volume
    obv = talib.OBV(c_price, volume)
    df['On-Balance Volume'] = obv


    # ****** Fandamental Parameters ****** #

    ### beta
    merged_data = pd.concat([Stock_Returns, Market_Returns], axis=1).dropna()
    merged_data.rename(columns={'پایانی*سهم*': 'Stock_Returns'}, inplace=True)
    if 'شاخص بورس*بورس*' in df.columns:
        merged_data.rename(columns={'شاخص بورس*بورس*': 'Market_Returns'}, inplace=True)
    if 'شاخص فرابورس*فرابورس*' in df.columns:
        merged_data.rename(columns={'شاخص فرابورس*فرابورس*': 'Market_Returns'}, inplace=True)
    rolling_beta = merged_data['Stock_Returns'].rolling(window=30).cov(merged_data['Market_Returns']) / merged_data['Market_Returns'].rolling(window=30).var()
    df['Beta30'] = rolling_beta





    # Emotional Factors
    ### VOL20
    vol20 = volume.rolling(window=20).mean()
    df['VOL20'] = vol20
    ## daily average volume 20 (DAVOL20)
    davol20 = (volume - vol20) / vol20
    df['DAVOL20'] = davol20

    ### Volume Oscillator (VOSC)
    # Calculate the short-term and long-term volume moving averages
    short_Vol_MA = talib.SMA(volume, timeperiod=12)
    long_Vol_MA = talib.SMA(volume, timeperiod=26)

    # Calculate Volume Oscillator (VOSC)
    vosc = short_Vol_MA - long_Vol_MA
    df['VOSC'] = vosc

    ### Volume moving average
    volume_MA26 = talib.SMA(volume, timeperiod=26)
    df['Volume_MA26'] = volume_MA26
    volume_MA50 = talib.SMA(volume, timeperiod=50)
    df['Volume_MA50'] = volume_MA50


    ### VMACD
    # Calculate the short-term and long-term volume moving averages
    short_Vol_MA1 = volume.rolling(window=12).mean()
    long_Vol_MA1 = volume.rolling(window=26).mean()

    # Calculate Volume MACD (VMACD)
    Vmacd = short_Vol_MA1 - long_Vol_MA1
    df['VMACD'] = Vmacd


    ### ART14
    # Calculate True Range (TR)
    tr14 = talib.TRANGE(h_price, l_price, c_price)

    # Calculate Average True Range (ATR)
    Atr14 = talib.ATR(h_price, l_price, c_price, timeperiod=14)

    # Calculate TOOL14-ATR14 (True Range over 14 periods divided by ATR over 14 periods)
    tr14_Atr14 = tr14 / Atr14
    df['Tr14_Atr14'] = tr14_Atr14



    ## PEG ratio - Growth rate is equals to inflation rate
    pegs = [ ]
    for index in range(len(df)):
        year = df.iloc[index]['Date'].year
        if year in expected_growth_rates.keys():
            pegs.append(pe_ratio[index]/expected_growth_rates[year])

    df['PEG'] = pegs

    ## Individual purchase per capita
    individual_per_cap = df['سرانه خرید حقیقی*سهم*'] / df['ارزش بازار*سهم*']
    df['Individual Per Capita'] = individual_per_cap



    # Risk Factors
    ### Variance20
    variance_20d = c_price.rolling(window=20).var()
    df['Variance20'] = variance_20d

    ### Sharp Ratio
    rolling_std = pct_close.rolling(window=20).std()
    rolling_mean = pct_close.rolling(window=20).mean()
    sharpe_ratio_20 = (rolling_mean - risk_free) / rolling_std
    df['Sharp Ratio 20'] = sharpe_ratio_20

    ### Sharp Ratio 60
    rolling_std60 = pct_close.rolling(window=60).std()
    rolling_mean60 = pct_close.rolling(window=60).mean()
    sharpe_ratio_60 = (rolling_mean60 - risk_free) / rolling_std60
    df['Sharp Ratio 60'] = sharpe_ratio_60

    ### kurtosis
    rolling_kurtosis = pct_close.rolling(window=20).kurt()
    df['Kurtosis'] = rolling_kurtosis

    ### skewness
    rolling_skewness = pct_close.rolling(window=20).skew()
    df['Skewness'] = rolling_skewness

    ### Rate of Change - ROC20
    # Calculate ROC20 (Rate of Change over a 20-day period)
    # Period for ROC calculation

    Close_Shifted = c_price.shift(20)
    Roc20 = ((c_price - Close_Shifted) / Close_Shifted) * 100
    df['ROC20'] = Roc20

    ### Volume momentum 20
    Volume_Shifted = volume.shift(20)
    Volume_Momentum_1M = ((volume - Volume_Shifted) / Volume_Shifted) * 100
    df['Volume Momentum 20'] = Volume_Momentum_1M
    ### Thriple exponential moving average
    EMA1 = ta.trend.ema_indicator(close=c_price, window=20)
    EMA2 = ta.trend.ema_indicator(close=EMA1, window=20)
    TEMA20 = (3 * EMA1) - (3 * EMA2) + ta.trend.ema_indicator(close=EMA2, window=20)
    df['TEMA20'] = TEMA20
    ### Price Momentum 1 month

    Price_Momentum_1M = (c_price - Close_Shifted / Close_Shifted) * 100
    df['Price Momentum 1M'] = Price_Momentum_1M

    ### Price Linear Regression Curve 12
    def linear_regression(df , period = 12):
        x = pd.Series(range(1, len(df) + 1))
        y = df.values
        x_mean = x.rolling(window=period).mean()
        y_mean = df.rolling(window=period).mean()
        covariance = ((x - x_mean) * (y - y_mean)).rolling(window=period).sum()
        variance = ((x - x_mean) ** 2).rolling(window=period).sum()
        slope = covariance / variance
        intercept = y_mean - (x_mean * slope)
        return slope * period + intercept

    Price_Linear_Regression = linear_regression(c_price)
    df['PLRC12'] = Price_Linear_Regression



    # *** Volatility parameters *** #

    # Calculate logarithmic returns
    Log_Returns = np.log(c_price / c_price.shift(1))

    # Calculate Historical Volatility as the standard deviation of logarithmic returns
    historical_volatility = Log_Returns.rolling(window=20).std() * np.sqrt(20)
    df['Historical Volatility'] = historical_volatility
    
    # Realized_Volatility RV

    realized_volatility = Stock_Returns.rolling(window=20).std() * np.sqrt(20)
    df['Realized_Volatility'] = realized_volatility


    # Calculate Average True Range (ATR)
    data = pd.DataFrame()
    period = 14  # Adjust as needed

    data['High_Low'] = abs(h_price - l_price)
    data['High_PreviousClose'] = abs(h_price - c_price.shift(1))
    data['Low_PreviousClose'] = abs(l_price - c_price.shift(1))

    data['True_Range'] = data[['High_Low', 'High_PreviousClose', 'Low_PreviousClose']].max(axis=1)
    ATR = data['True_Range'].rolling(window=period).mean()
    df['ATR14'] = ATR


    ### ***** Clearing New Data ***** ###
    def clean_data(df):
        # Replace gaps forward from the previous valid value in: 'قیمت به درآمد-بورس*بورس*', 'ارزش بازار (دلار آزاد)-بورس*بورس*' and 76 other columns
        df = df.fillna({'MA26': df['MA26'].ffill(), 'MA50': df['MA50'].ffill(), 'SMA26': df['SMA26'].ffill(), 'SMA50': df['SMA50'].ffill(), 'EMA26': df['EMA26'].ffill(), 'EMA50': df['EMA50'].ffill(), 'MACD': df['MACD'].ffill(), 'MACD_signal': df['MACD_signal'].ffill(), 'Upper_Bollinger_band': df['Upper_Bollinger_band'].ffill(), 'Middle_Bollinger_band': df['Middle_Bollinger_band'].ffill(), 'Lower_Bollinger_band': df['Lower_Bollinger_band'].ffill(), 'RSI': df['RSI'].ffill(), 'MFI': df['MFI'].ffill(), 'Momentum_12': df['Momentum_12'].ffill(), 'Stochastic Oscillator_slowk': df['Stochastic Oscillator_slowk'].ffill(), 'Stochastic Oscillator_slowd': df['Stochastic Oscillator_slowd'].ffill(), 'On-Balance Volume': df['On-Balance Volume'].ffill(), 'Beta30': df['Beta30'].ffill(), 'VOL20': df['VOL20'].ffill(), 'DAVOL20': df['DAVOL20'].ffill(), 'VOSC': df['VOSC'].ffill(), 'Volume_MA26': df['Volume_MA26'].ffill(), 'Volume_MA50': df['Volume_MA50'].ffill(), 'VMACD': df['VMACD'].ffill(), 'Tr14_Atr14': df['Tr14_Atr14'].ffill(), 'PEG': df['PEG'].ffill(), 'Individual Per Capita': df['Individual Per Capita'].ffill(), 'Variance20': df['Variance20'].ffill(), 'Sharp Ratio 20': df['Sharp Ratio 20'].ffill(), 'Sharp Ratio 60': df['Sharp Ratio 60'].ffill(), 'Kurtosis': df['Kurtosis'].ffill(), 'Skewness': df['Skewness'].ffill(), 'ROC20': df['ROC20'].ffill(), 'Volume Momentum 20': df['Volume Momentum 20'].ffill(), 'TEMA20': df['TEMA20'].ffill(), 'Price Momentum 1M': df['Price Momentum 1M'].ffill(), 'PLRC12': df['PLRC12'].ffill(), 'Historical Volatility': df['Historical Volatility'].ffill(), 'Realized_Volatility': df['Realized_Volatility'].ffill(), 'ATR14': df['ATR14'].ffill()})
        # Replace missing values with 0 in columns: 'قیمت به درآمد-بورس*بورس*', 'ارزش بازار (دلار آزاد)-بورس*بورس*' and 76 other columns
        df = df.fillna({'MA26': 0, 'MA50': 0, 'SMA26': 0, 'SMA50': 0, 'EMA26': 0, 'EMA50': 0, 'MACD': 0, 'MACD_signal': 0, 'Upper_Bollinger_band': 0, 'Middle_Bollinger_band': 0, 'Lower_Bollinger_band': 0, 'RSI': 0, 'MFI': 0, 'Momentum_12': 0, 'Stochastic Oscillator_slowk': 0, 'Stochastic Oscillator_slowd': 0, 'On-Balance Volume': 0, 'Beta30': 0, 'VOL20': 0, 'DAVOL20': 0, 'VOSC': 0, 'Volume_MA26': 0, 'Volume_MA50': 0, 'VMACD': 0, 'Tr14_Atr14': 0, 'PEG': 0, 'Individual Per Capita': 0, 'Variance20': 0, 'Sharp Ratio 20': 0, 'Sharp Ratio 60': 0, 'Kurtosis': 0, 'Skewness': 0, 'ROC20': 0, 'Volume Momentum 20': 0, 'TEMA20': 0, 'Price Momentum 1M': 0, 'PLRC12': 0, 'Historical Volatility': 0, 'Realized_Volatility': 0, 'ATR14': 0})
        # Change column type to float64 for columns: 'ATR14', 'Realized_Volatility' and 35 other columns
        df = df.astype({'ATR14': 'float64', 'Realized_Volatility': 'float64', 'Historical Volatility': 'float64', 'PLRC12': 'float64', 'TEMA20': 'float64', 'Volume Momentum 20': 'float64', 'ROC20': 'float64', 'Skewness': 'float64', 'Kurtosis': 'float64', 'Sharp Ratio 60': 'float64', 'Sharp Ratio 20': 'float64', 'Variance20': 'float64', 'Individual Per Capita': 'float64', 'PEG': 'float64', 'Tr14_Atr14': 'float64', 'VMACD': 'float64', 'Volume_MA50': 'float64', 'Volume_MA26': 'float64', 'VOSC': 'float64', 'DAVOL20': 'float64', 'VOL20': 'float64', 'Beta30': 'float64', 'Stochastic Oscillator_slowd': 'float64', 'Stochastic Oscillator_slowk': 'float64', 'MFI': 'float64', 'RSI': 'float64', 'Lower_Bollinger_band': 'float64', 'Middle_Bollinger_band': 'float64', 'Upper_Bollinger_band': 'float64', 'MACD_signal': 'float64', 'MACD': 'float64', 'EMA50': 'float64', 'EMA26': 'float64', 'SMA50': 'float64', 'SMA26': 'float64', 'MA50': 'float64', 'MA26': 'float64'})
        # Change column type to int64 for columns: 'Price Momentum 1M', 'On-Balance Volume', 'Momentum_12'
        df = df.astype({'Price Momentum 1M': 'int64', 'On-Balance Volume': 'int64', 'Momentum_12': 'int64'})
        return df

    df_clean = clean_data(df.copy())




    return df_clean


folder_path = './boursview_final_data'
file_list = os.listdir(folder_path)
for file in file_list:
    df_path = f'{folder_path}/{file}'
    finnal_df = adding_Tech_Fund(df_path)
    finnal_df.to_excel(f'./FINNAL_DATA/{file}', index=False)

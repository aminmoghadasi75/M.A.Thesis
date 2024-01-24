import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# create an empty dataframe
date_range = pd.date_range(start='2013-09-01', end='2023-08-31', name='Date')
df = pd.DataFrame(date_range)

folder_path = './raw_data'
data_list = os.listdir(folder_path)

for data in data_list :
    df2 = pd.read_excel(f'{folder_path}/{data}')
    df2.rename(columns={'تاریخ میلادی' : 'Date'}, inplace=True)
    df2['Date'] = pd.to_datetime(df2['Date'])
    start_date = pd.to_datetime('2013-09-01')
    end_date = pd.to_datetime('2023-08-31')
    df2 = df2[(df2['Date'] >= start_date) & (df2['Date'] <= end_date)]
    if 'تاریخ شمسی' not in df.columns :
        df = pd.concat([df.set_index('Date'), df2.set_index('Date')], axis=1, join='outer')
    else :
        df2 = df2.drop(columns=['تاریخ شمسی'])
        df = pd.concat([df.set_index('Date'), df2.set_index('Date')], axis=1, join='outer')


df.reset_index(inplace=True)

df.to_excel(f'./merged_hyper_eco.xlsx' , index=False)

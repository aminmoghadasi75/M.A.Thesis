import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')








folder_path = './datas'
data_list = os.listdir(folder_path)
for data in data_list:

    file_path = f'{folder_path}/{data}'
    stock_path = os.listdir(file_path)





    df = pd.DataFrame(pd.read_excel('./تاریخ.xlsx'))
    df['Date'] = pd.to_datetime(df['Date'])


    for file in stock_path:
        df1 = pd.DataFrame(pd.read_excel(f'{file_path}/{file}'))
        if 'Unnamed: 0' in df1.columns.to_list():
            df1 = df1.drop(columns=[ 'Unnamed: 0'])
        if 'تاریخ شمسی' in df1.columns.to_list():
            df1 = df1.drop(columns=['تاریخ شمسی'])
        df = pd.merge(df , df1, on='Date', how='outer')


    df.to_excel(f'./boursview_final_data/{data}.xlsx', index=False)















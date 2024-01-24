import pandas as pd
import numpy as np
import os



def clean_data(file_path: str) -> pd.DataFrame:
    """
    Clean the data.

    Args:
        df (pd.DataFrame): The data to clean.

    Returns:
        pd.DataFrame: The cleaned data.
    """

    start_date = '2013-09-01'
    end_date = '2023-08-31'
    


    his_price = pd.read_excel(file_path, skiprows=7)
    his_price = pd.DataFrame(his_price)
    his_price = his_price.drop(columns=['Unnamed: 0'])


    his_price.rename(columns={'تاریخ میلادی': 'Date'}, inplace=True)
    his_price['Date'] = pd.to_datetime(his_price['Date'])

    
    his_price = his_price[(his_price['Date'] >= pd.to_datetime(start_date)) & (his_price['Date'] <= pd.to_datetime(end_date))]

    date_range = pd.date_range(start=start_date, end=end_date)
    df = pd.DataFrame(date_range)
    df.rename(columns={0 : 'Date'}, inplace=True)



    df = pd.merge(df , his_price, on='Date', how='outer')

    df.replace(to_replace='-', value=np.nan, inplace=True)

    def clean_data(df):
        # Replace gaps forward from the previous valid value in: 'بازگشایی', 'بالاترین' and 19 other columns
        df = df.fillna({'بازگشایی': df['بازگشایی'].ffill(), 'بالاترین': df['بالاترین'].ffill(), 'پایین\u200cترین': df['پایین\u200cترین'].ffill(), 'آخرین معامله': df['آخرین معامله'].ffill(), 'قیمت تعدیلی': df['قیمت تعدیلی'].ffill(), 'پایانی': df['پایانی'].ffill(), 'میزان تغییر': df['میزان تغییر'].ffill(), 'درصد تغییر': df['درصد تغییر'].ffill(), 'ارزش بازار': df['ارزش بازار'].ffill(), 'تعداد معاملات': df['تعداد معاملات'].ffill(), 'حجم': df['حجم'].ffill(), 'ارزش معاملات': df['ارزش معاملات'].ffill(), 'سهام شناور': df['سهام شناور'].ffill(), 'P/E-ttm': df['P/E-ttm'].ffill(), 'P/B': df['P/B'].ffill(), 'P/S': df['P/S'].ffill(), 'قدرت خرید حقیقی': df['قدرت خرید حقیقی'].ffill(), 'درصد خالص حقیقی': df['درصد خالص حقیقی'].ffill(), 'خالص حقیقی': df['خالص حقیقی'].ffill(), 'سرانه خرید حقیقی': df['سرانه خرید حقیقی'].ffill(), 'سرانه فروش حقیقی': df['سرانه فروش حقیقی'].ffill()})
        # Replace missing values with 0 in columns: 'بازگشایی', 'بالاترین' and 19 other columns
        df = df.fillna({'بازگشایی': 0, 'بالاترین': 0, 'پایین\u200cترین': 0, 'آخرین معامله': 0, 'قیمت تعدیلی': 0, 'پایانی': 0, 'میزان تغییر': 0, 'درصد تغییر': 0, 'ارزش بازار': 0, 'تعداد معاملات': 0, 'حجم': 0, 'ارزش معاملات': 0, 'سهام شناور': 0, 'P/E-ttm': 0, 'P/B': 0, 'P/S': 0, 'قدرت خرید حقیقی': 0, 'درصد خالص حقیقی': 0, 'خالص حقیقی': 0, 'سرانه خرید حقیقی': 0, 'سرانه فروش حقیقی': 0})
        # Change column type to datetime64[ns] for column: 'Date'
        df = df.astype({'Date': 'datetime64[ns]'})
        # Change column type to string for column: 'تاریخ شمسی'
        df = df.astype({'تاریخ شمسی': 'string'})
        # Change column type to int64 for columns: 'بازگشایی', 'بالاترین' and 11 other columns
        df = df.astype({'بازگشایی': 'int64', 'بالاترین': 'int64', 'پایین\u200cترین': 'int64', 'آخرین معامله': 'int64', 'قیمت تعدیلی': 'int64', 'پایانی': 'int64', 'ارزش بازار': 'int64', 'تعداد معاملات': 'int64', 'حجم': 'int64', 'ارزش معاملات': 'int64', 'خالص حقیقی': 'int64', 'سرانه خرید حقیقی': 'int64', 'سرانه فروش حقیقی': 'int64'})
        # Change column type to int64 for column: 'میزان تغییر'
        df = df.astype({'میزان تغییر': 'int64'})
        # Change column type to float64 for columns: 'درصد تغییر', 'سهام شناور' and 5 other columns
        df = df.astype({'درصد تغییر': 'float64', 'سهام شناور': 'float64', 'P/E-ttm': 'float64', 'P/B': 'float64', 'P/S': 'float64', 'قدرت خرید حقیقی': 'float64', 'درصد خالص حقیقی': 'float64'})
        return df

    df_clean = clean_data(df.copy())
    df_clean.head()



    dates_path = './تاریخ.xlsx'
    dates = pd.DataFrame(pd.read_excel(dates_path))
    df_clean['تاریخ شمسی'] = dates['تاریخ شمسی']

    columns_to_exclude = ['Date', 'تاریخ شمسی']
    selected_columns = df_clean.columns.difference(columns_to_exclude).tolist()
    for column in selected_columns:
        df_clean.rename(columns={column: f'{column}*سهم*'}, inplace=True)

    return df_clean




folder_path = './historical_price'

file_name_list = os.listdir(folder_path)

for file_name in file_name_list :
    df_clean = clean_data(f'{folder_path}/{file_name}')
    df_clean.to_excel(f'./stock_clean/{file_name}', index=True)
import pandas as pd
import numpy as np
import os

def clean_data(df):
    # Replace missing values with 0 in columns: 'آخرین معامله*صنعت*', 'میزان تغییر*صنعت*' and 16 other columns
    df = df.fillna({ 'آخرین معامله*صنعت*': 0,'میزان تغییر*صنعت*': 0, 'درصد تغییر*صنعت*': 0, 'حجم*صنعت*': 0, 'ارزش معاملات*صنعت*': 0, 'ارزش بازار*صنعت*': 0, 'خالص حقوقی-حجم(تعداد سهام)-تجمعی*صنعت*': 0, 'خالص حقیقی-حجم(تعداد سهام)-تجمعی*صنعت*': 0, 'خالص حقوقی-ارزش(ریال)-تجمعی*صنعت*': 0, 'خالص حقیقی-ارزش(ریال)-تجمعی*صنعت*': 0, 'خرید حقوقی-حجم(تعداد سهام)*صنعت*': 0, 'فروش حقوقی-حجم(تعداد سهام)*صنعت*': 0, 'خرید حقیقی-حجم(تعداد سهام)*صنعت*': 0, 'فروش حقیقی-حجم(تعداد سهام)*صنعت*': 0, 'خرید حقوقی-ارزش(ریال)*صنعت*': 0, 'فروش حقوقی-ارزش(ریال)*صنعت*': 0, 'خرید حقیقی-ارزش(ریال)*صنعت*': 0, 'فروش حقیقی-ارزش(ریال)*صنعت*': 0})
    # Replace gaps back from the next valid value in: 'سهام شناور*صنعت*', 'P/E-ttm*صنعت*' and 3 other columns
    df = df.fillna({'سهام شناور*صنعت*': df['سهام شناور*صنعت*'].ffill(), 'P/E-ttm*صنعت*': df['P/E-ttm*صنعت*'].ffill(), 'P/B*صنعت*': df['P/B*صنعت*'].ffill(), 'P/S*صنعت*': df['P/S*صنعت*'].ffill(), 'قیمت*صنعت*': df['قیمت*صنعت*'].ffill()})



    # Define the range for random float numbers
    mean1 = df['سهام شناور*صنعت*'].mean()
    std1 = df['سهام شناور*صنعت*'].std()
    min_value = mean1 - (0.7 * std1)    # Minimum value for random float
    max_value = mean1 + (0.7 * std1)    # Maximum value for random float

    # Replace missing values with random float numbers within the specified range
    def fill_missing_with_random(column, min_val, max_val):
        missing_mask = column.isnull()  # Mask for missing values in the column
        num_missing = missing_mask.sum()  # Count of missing values in the column

        if num_missing > 0:
            # Generate random float numbers for missing values in the specified range
            random_values = np.random.uniform(min_val, max_val, size=num_missing)
            column.loc[missing_mask] = random_values

        return column
    # check is df['آخرین معامله*صنعت*'] doesnt change , replace the missing value ffill esle use the function
    missing_value_inx = df[df['سهام شناور*صنعت*'].isnull()].index
    for idx in missing_value_inx:
        if idx - 1 >= 0 and df.loc[idx, 'آخرین معامله*صنعت*'] == df.loc[idx - 1, 'آخرین معامله*صنعت*']:
            df.loc[idx, 'سهام شناور*صنعت*'] = df.loc[idx - 1, 'سهام شناور*صنعت*']
        else:
            column_name = 'سهام شناور*صنعت*'
            df[column_name] = fill_missing_with_random(df[column_name], min_value, max_value)





    # Change column type to datetime64[ns] for column: 'Date'
    df = df.astype({'Date': 'datetime64[ns]'})
    # Change column type to string for column: 'تاریخ شمسی'
    df = df.astype({'تاریخ شمسی': 'string'})
    # Change column type to int64 for columns: 'آخرین معامله*صنعت*', 'حجم*صنعت*' and 15 other columns
    df = df.astype({'آخرین معامله*صنعت*': 'int64', 'حجم*صنعت*': 'int64', 'ارزش معاملات*صنعت*': 'int64', 'ارزش بازار*صنعت*': 'int64', 'قیمت*صنعت*': 'int64', 'خالص حقوقی-حجم(تعداد سهام)-تجمعی*صنعت*': 'int64', 'خالص حقیقی-حجم(تعداد سهام)-تجمعی*صنعت*': 'int64', 'خالص حقوقی-ارزش(ریال)-تجمعی*صنعت*': 'int64', 'خالص حقیقی-ارزش(ریال)-تجمعی*صنعت*': 'int64', 'خرید حقوقی-حجم(تعداد سهام)*صنعت*': 'int64', 'فروش حقوقی-حجم(تعداد سهام)*صنعت*': 'int64', 'خرید حقیقی-حجم(تعداد سهام)*صنعت*': 'int64', 'فروش حقیقی-حجم(تعداد سهام)*صنعت*': 'int64', 'خرید حقوقی-ارزش(ریال)*صنعت*': 'int64', 'فروش حقوقی-ارزش(ریال)*صنعت*': 'int64', 'خرید حقیقی-ارزش(ریال)*صنعت*': 'int64', 'فروش حقیقی-ارزش(ریال)*صنعت*': 'int64'})
    # Change column type to float64 for columns: 'درصد تغییر*صنعت*', 'سهام شناور*صنعت*' and 3 other columns
    df = df.astype({'درصد تغییر*صنعت*': 'float64', 'سهام شناور*صنعت*': 'float64', 'P/E-ttm*صنعت*': 'float64', 'P/B*صنعت*': 'float64', 'P/S*صنعت*': 'float64'})
    return df




folder_path = './merged_data'
data_list = os.listdir(folder_path)
for data in data_list :
    df = pd.read_excel(f'{folder_path}/{data}')
    df_clean = clean_data(df.copy())
    df_clean.to_excel(f'./sanat_cleaned/{data}', index=True)
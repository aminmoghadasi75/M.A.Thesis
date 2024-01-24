import pandas as pd
import numpy as np
from typing import List
import os

def preparing_sanat_data(his_price_address: str, hogh_hagh_address: str) -> pd.DataFrame:
    """
    Read two Excel files and merge them based on the 'تاریخ شمسی' column.
    Rearrange the columns and reset the index.
    Convert column types to the appropriate data types.
    Return the merged dataframe.

    Parameters:
        his_price_address (str): The file path of the first Excel file.
        hogh_hagh_address (str): The file path of the second Excel file.

    Returns:
        pd.DataFrame: The merged dataframe.
    """
    # Read historical price data from excel file
    price_historical_df = pd.read_excel(his_price_address)

    # Read hoghoghi haghighi data from excel file
    hoghoghi_haghighi_df = pd.read_excel(hogh_hagh_address)

    # Merge the two dataframes based on the 'تاریخ شمسی' column
    merged_df = pd.merge(price_historical_df, hoghoghi_haghighi_df, on='تاریخ شمسی', how='outer')
    merged_df.rename(columns={'تاریخ میلادی_x': 'Date'}, inplace=True)
    merged_df.drop(columns=['تاریخ میلادی_y'], inplace=True)

    # Convert 'Date' column to datetime format
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    # Filter data within the specified date range
    start_date = pd.to_datetime('2013-09-01')
    end_date = pd.to_datetime('2023-08-31')
    merged_df = merged_df[(merged_df['Date'] >= start_date) & (merged_df['Date'] <= end_date)]

    # Create a date range DataFrame from '2013-09-01' to '2023-08-31'
    date_range = pd.date_range(start='2013-09-01', end='2023-08-31', name='Date')
    df_fake = pd.DataFrame({'Date': date_range})

    # Convert 'Date' column to datetime format
    df_fake['Date'] = pd.to_datetime(df_fake['Date'])

    # Merge the date range DataFrame with the merged DataFrame to fill in missing dates
    df = pd.merge(df_fake, merged_df, on='Date', how='outer')

    # Replace '-' with NaN values
    df.replace(to_replace='-', value=np.nan, inplace=True)

    #Labeling the features
    for col in df.columns :
        df.rename(columns={col: col + '*صنعت*'}, inplace=True)

    df.reset_index(inplace=True)



    return df


historical_price_folder_path = './historical_price'
hogh_hagh_folder_path = './haghighi_hoghoghi'

hp_list = os.listdir(historical_price_folder_path)
hh_list = os.listdir(hogh_hagh_folder_path)


for hp, hh in zip(hp_list, hh_list):
    df = preparing_sanat_data(
        his_price_address=f'{historical_price_folder_path}/{hp}',
        hogh_hagh_address=f'{hogh_hagh_folder_path}/{hh}'
    )
    df.to_excel(f'./merged_data/{hp}', index=False)

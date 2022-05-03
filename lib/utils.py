#!/usr/bin/env python3.10.4
import datetime
import numpy as np
import pandas as pd
from typing import Tuple


def int_from_str(string: str) -> int:
    """Exctract an integer number from a string.

    Args:
        string (str): string containing a number.

    Returns:
        int: integer number inside the string.
    """
    num = int(''.join(filter(str.isdigit, string)))
    return num


def read_data_from_yahoo(symbol: str, start: datetime.date,
                         end: datetime.date) -> pd.DataFrame:
    """Read data from yahoo site and get it as dataframe object.

    Args:
        symbol (str): Stok index
        start (datetime.date): start date,
        end (datetime.date): end date.

    Returns:
        pd.DataFrame: Dataframe with data in the choosen time range.
    """
    import pandas_datareader as pdr # Work only on ubuntu
    df = pdr.get_data_yahoo(symbols=f'{symbol}', start=start, end=end)
    df.reset_index(inplace=True, drop=False)
    # Remove last duplicated row
    df = df[~df.Date.duplicated()]
    return df


def read_gold_data(file_path: str, format: str = "%d/%m/%Y",
                   save: bool = False) -> pd.DataFrame:
    """Read data of Daily Treasury Par Yield Curve Rates in the date
       format wanted. Source of the data: 
       https://home.treasury.gov/resource-center/data-chart-center/
       interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr
       _date_value=2022.

    Args:
        file_path (str): file path where is stored the file,
        format (str, optional): date format wanted. Defaults to "%d/%m/%Y".
        save (bool, optional): bool var to save data. Defaults to False.

    Returns:
        pd.DataFrame: dataframe containg the data well formatted.
    """

    df = pd.read_csv(file_path)
    # reverse the rows to get first the older date
    df = df.iloc[::-1].reset_index(drop=True)

    df['Date'] = pd.to_datetime(df.Date, format="%m/%d/%Y").dt.strftime(format)
    df['Date'] = pd.to_datetime(df.Date, format=format)
    
    if save:
        # Save data with the wanted date format
        year = int_from_str(file_path)
        df.to_csv(f'../Data/Tresure_gold_data_{year}.csv', index=False)
    else:
        return df


def fancy_legend(legend) -> None:
    for lh in legend.legendHandles: 
        lh.set_alpha(1)
        lh.set_linewidth(1)


def fancy_binwidth(df: pd.DataFrame, col: str = 'y_plr',
                   x_breaks_num: int = 15,
                   y_breaks_num: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    x = df.index
    x_breaks_low = min(x)
    x_breaks_up = max(x)
    x_binwidth = np.ceil((x_breaks_up - x_breaks_low) / x_breaks_num)
    x_breaks = np.arange(x_breaks_low, x_breaks_up, x_binwidth, dtype='int')

    y = df[col]
    y_binwidth = round((max(y) - min(y)) / y_breaks_num, 3)
    y_breaks_low = np.floor((min(y) / y_binwidth)) * y_binwidth
    y_breaks_up = np.ceil((max(y) / y_binwidth)) * y_binwidth
    y_breaks = np.round(np.arange(y_breaks_low, y_breaks_up, y_binwidth), 3)
    return x_breaks, y_breaks

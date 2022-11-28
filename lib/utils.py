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
    return int(''.join(filter(str.isdigit, string)))


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


def fancy_binwidth(df: pd.DataFrame, X: str, Y: str,
                   x_breaks_num: int = 10, x_prec: int = 4,
                   y_breaks_num: int = 10,
                   y_prec: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    x_breaks = get_variable_bins(df, X, x_breaks_num, x_prec)
    y_breaks = get_variable_bins(df, Y, y_breaks_num, y_prec)
    return x_breaks[1:], y_breaks[1:]


def get_variable_bins(df, X, Y, prec):
    var = df[X]
    var_binwidth = round((max(var) - min(var)) / Y, prec)
    var_breaks_low = np.floor((min(var) / var_binwidth)) * var_binwidth
    var_breaks_up = np.ceil((max(var) / var_binwidth)) * var_binwidth
    return np.round(np.arange(var_breaks_low, var_breaks_up, var_binwidth), prec)

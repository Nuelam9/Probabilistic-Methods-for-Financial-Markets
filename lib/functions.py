#!/usr/bin/env python3.8.10
import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
import pandas_datareader as pdr # Work only on ubuntu
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf


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
    df = pdr.get_data_yahoo(symbols=f'{symbol}', start=start, end=end)
    df.reset_index(inplace=True, drop=False)
    # Remove last duplicated row
    df = df[~df.Date.duplicated()]
    return df


def log_return(df: pd.DataFrame, column : str = 'Adj Close') -> None:
    """Compute the logarithm return.

    Args:
        df (pd.DataFrame): data,
        column (str, optional): adjusted value for which compute the 
                                logarithm return. Defaults to 'Adj Close'.
    """
    # Compute the logarithm return
    y_log = np.log(df[column])
    df['y_lr'] = y_log.diff(periods=1)
    # Compute the percentage logarithm return
    df['y_plr'] = df['y_lr'] * 100.
    # Remove all the rows containing nan 
    mask = ~df.isna().any(axis=1)
    df = df[mask].reset_index(drop=True)


def fancy_legend(leg):
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
        lh.set_linewidth(1)



def autocorrelogram_plot(df: pd.DataFrame, symbol: str, link: str,
                         column: str = 'y_plr', lag_method: str = 'Hyndman') -> None:
    """Autocorrelogram plot of a given variable inside column.

    Args:
        df (pd.DataFrame): _description_
        column (str, optional): _description_. Defaults to 'y_plr'.
        lag_method (str, optional): _description_. Defaults to 'Hyndman'.
    """
    z = df[column].to_numpy()
    n = len(z)
    
    if lag_method == 'Default':
        maxlag = np.ceil(10. * np.log10(n))
    elif lag_method == 'Box-Jenkins':
        maxlag = np.ceil(np.sqrt(n) + 45)
    elif lag_method == 'Hyndman':
        maxlag = np.min((n / 4., 10.))

    # fft = False to avoid warning
    Aut_Fun_z = acf(z, nlags=maxlag, fft=False)

    fig, ax = plt.subplots(figsize=(15, 8))
    plt.grid()
    for i, y in enumerate(Aut_Fun_z):
        if y > 0:
            plt.vlines(x=i, ymax=y, ymin=0, colors='k')
        elif y < 0:
            plt.vlines(x=i, ymax=0, ymin=y, colors='k')
    
    for ci, color in zip(['0.90', '0.95', '0.99'], ['r', 'b', 'g']):
        CI = norm.ppf((1. + float(ci)) / 2.) / np.sqrt(n)
        text = f"ci {int(float(ci) * 100.)}%"  
        plt.plot([-1, maxlag+1], [CI]*2, color +'.-.', alpha=0.6, label=text)
        plt.plot([-1, maxlag+1], [-CI]*2, color +'.-.', alpha=0.6)

    first_day = df.loc[0, 'Date']
    last_day = df.loc[n - 1, 'Date']
    leg = plt.legend()
    fancy_legend(leg)
    plt.xlabel('Lag')
    plt.ylabel('Acf value')   
    plt.suptitle("University of Roma \"Tor Vergata\" - Corso di Metodi"
               + " Probabilistici e Statistici per i Mercati Finanziari \n"
               + " Autocorrelogram of S&P 500 Percentage Logarithm Returns"
               + f" from {first_day} to {last_day}")
    plt.title(f"Path length {n} sample points. Data from Yahoo Finance " 
              + f"{symbol} - {link}")
    ax.set_xticks(range(11))
    ax.set_yticks(np.arange(0, 1.25, 0.25))
    plt.xlim((-0.5, 10.5))

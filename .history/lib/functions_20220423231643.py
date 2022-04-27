#!/usr/bin/env python3.8.10
import datetime
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.stats import norm
import statsmodels.api as sm
import pandas_datareader as pdr # Work only on ubuntu
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from matplotlib.ticker import FormatStrFormatter
from sklearn.linear_model import LinearRegression


def extract_


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


def save_gold_data(file_path: str, format: str = "%d/%m/%Y") -> None:
    """Save data of Daily Treasury Par Yield Curve Rates in the date
       format wanted. Data is taken from the site: 
       https://home.treasury.gov/resource-center/data-chart-center/
       interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr
       _date_value=2022.

    Args:
        file_path (str): file path at which is stored the file
        format (str, optional): date format wanted.
                                Defaults to "%d/%m/%Y". 
    """
    df = pd.read_csv(file_path)

    df['Date'] = pd.to_datetime(df.Date, format="%m/%d/%Y").dt.strftime(format)
    df['Date'] = pd.to_datetime(df.Date, format=format)

    df = df.sort_values(by='Date').reset_index(drop=True)

    df['Date'] = pd.to_datetime(df.Date).dt.strftime("%d/%m/%Y")
    # Save data with the wanted date format
    df.to_csv('../Data/Tresure_gold_data.csv', index=False)


def log_return(df: pd.DataFrame, column : str = 'Adj Close') -> pd.DataFrame:
    """Compute the logarithm return.

    Args:
        df (pd.DataFrame): data,
        column (str, optional): adjusted value for which compute the 
                                logarithm return. Defaults to 'Adj Close'.
    Returns:
        pd.DataFrame: Dataframe with logarithm return and the percentage
        logarithm return, without nan.
    """
    # Compute the logarithm return
    y_log = np.log(df[column])
    df['y_lr'] = y_log.diff(periods=1)
    # Compute the percentage logarithm return
    df['y_plr'] = df['y_lr'] * 100.
    # Remove all the rows containing nan 
    mask = ~df.isna().any(axis=1)
    df = df[mask].reset_index(drop=True)
    return df


def fancy_legend(leg):
    for lh in leg.legendHandles: 
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


def data_visualization(df: pd.DataFrame, kind: str, symbol: str,
                       link: str, column : str = 'y_plr') -> None:
    """Scatter plot of column variable with the regression line and 
       the LOESS curve.

    Args:
        df (pd.DataFrame): data,
        symbol (str): Stock index, 
        link (str): link at with the data are taken,
        column (str, optional): df's column to plot. Defaults to 'y_plr'.
    """
    if kind == 'scatter':
        marker = '.'
        ms = 3
    elif kind == 'line':
        marker = '-'
        ms = 1
    
    if column == 'Adj Close':
        ylabel = 'Adjusted Close'
        title = ylabel
    elif column == 'y_plr':
        ylabel = 'percentage logarithm returns'
        title = 'Percentage Logarithm Returns'

    x = df.index.to_numpy()
    y = df[column].to_numpy()

    lowess_sm = sm.nonparametric.lowess
    yest_sm = lowess_sm(y, x, frac=1./3., it=3, return_sorted = False)

    X = x.reshape((-1, 1))

    reg = LinearRegression().fit(X, y)
    print(f'Intercept: {reg.intercept_}, Index: {reg.coef_[0]}')

    n = len(y)
    fig, ax = plt.subplots(figsize=(15, 9), constrained_layout=True)
    dates = df['Date']
    plt.plot(dates, y, 'b' + marker, markersize=ms, label=f'S&P 500 {ylabel}')
    plt.plot(dates, X*reg.coef_[0] + reg.intercept_, 'lime', label='Regression Line')
    plt.plot(dates, yest_sm, 'r--', label='LOESS Curve')
    first_day = df.loc[0, 'Date']
    last_day = df.loc[n - 1, 'Date']
    leg = plt.legend()
    fancy_legend(leg)
    x_breaks, y_breaks = fancy_binwidth(df, column)
    plt.grid()
    plt.xlabel('Dates')
    plt.ylabel(f'S&P 500 {ylabel}')
    # One tick for month 
    plt.xticks(dates[x_breaks], rotation=45)
    plt.yticks(y_breaks)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.suptitle("University of Roma \"Tor Vergata\" - Corso di Metodi"
                + " Probabilistici e Statistici per i Mercati Finanziari \n"
                + f" {kind.capitalize()} plot of S&P 500" 
                + f" {title} from {first_day} to {last_day}")
    plt.title(f"Path length {n} sample points. Data from Yahoo Finance " 
                + f"{symbol} - {link}")


def autocorrelogram(df: pd.DataFrame, symbol: str, link: str,
                    partial: bool = False, squared: bool = False, 
                    column: str = 'y_plr',
                    lag_method: str = 'Hyndman') -> None:
    """Autocorrelogram/Partial Autocorrelogram plot of a given variable
       inside column.

    Args:
        df (pd.DataFrame): data,
        symbol (str): Stock index, 
        link (str): link at with the data are taken,
        partial (bool, optional): to choose between the autocorrelogram
                                  and the partial autocorrelogram.
                                  Defaults to False,
        column (str, optional): df's column to plot. Defaults to 'y_plr',
        lag_method (str, optional): method to compute the maxlag.
                                    Defaults to 'Hyndman'.
    """
    z = df[column].to_numpy()
    n = len(z)
    
    if lag_method == 'Default':
        maxlag = np.ceil(10. * np.log10(n))
    elif lag_method == 'Box-Jenkins':
        maxlag = np.ceil(np.sqrt(n) + 45)
    elif lag_method == 'Hyndman':
        maxlag = np.min((n / 4., 10.))

    string = ' '
    if partial:
        kind = 'Partial Autocorrelogram'
        ylabel = 'Pacf value'
        Aut_Fun_z = pacf(z, nlags=maxlag)[1:]
        start = 1
        yticks = np.arange(-0.1, 0.1, 0.1)
    else:
        kind = 'Autocorrelogram'
        ylabel = 'Acf value'

        if squared:
            z = df[column].to_numpy() ** 2.
            string = ' Squared'            

        # fft = False to avoid warning
        Aut_Fun_z = acf(z, nlags=maxlag, fft=False)
        start = 0
        yticks = np.arange(0, 1.25, 0.25)

    fig, ax = plt.subplots(figsize=(15, 8))
    plt.grid()
    for i, y in enumerate(Aut_Fun_z, start=start):
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
    #fig.subplots_adjust(bottom=0.25)
    leg = plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    fancy_legend(leg)
    plt.xlabel('Lag')
    plt.ylabel(ylabel)   
    plt.suptitle("University of Roma \"Tor Vergata\" - Corso di Metodi"
               + " Probabilistici e Statistici per i Mercati Finanziari \n"
               + f" {kind} of S&P 500{string}Percentage Logarithm Returns"
               + f" from {first_day} to {last_day}")
    plt.title(f"Path length {n} sample points. Data from Yahoo Finance " 
              + f"{symbol} - {link}")
    ax.set_xticks(range(start, int(maxlag) + 1))
    ax.set_yticks(yticks)
    plt.xlim((-0.5 + start, maxlag + 0.5))


def linear_regression(x: np.ndarray, y: np.ndarray,
                        offset: int) -> np.ndarray:
    """Compute the linear regression given x and y variables.

    Args:
        x (np.ndarray): x data,
        y (np.ndarray): y data,
        offset (int): offset to evaluate linear regeression.

    Returns:
        np.ndarray: linear regression.
    """
    X = x.reshape((-1, 1))
    reg = LinearRegression().fit(X, y)
    print(f'Intercept: {reg.intercept_}, Index: {reg.coef_[0]}')
    return reg.intercept_ + reg.coef_[0] * range(offset + 1)

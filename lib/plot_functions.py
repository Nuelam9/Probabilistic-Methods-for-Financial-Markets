#!/usr/bin/env python3.10.4
import numpy as np
from utils import *
import pandas as pd

import matplotlib.pyplot as plt


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
    import statsmodels.api as sm
    from matplotlib.ticker import FormatStrFormatter
    from sklearn.linear_model import LinearRegression

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

    # reshape x to use LinearRegression
    X = x.reshape((-1, 1))

    reg = LinearRegression().fit(X, y)
    print(f'Intercept: {reg.intercept_}, Index: {reg.coef_[0]}')

    n = len(y)
    fig, ax = plt.subplots(figsize=(18, 9), constrained_layout=True)
    dates = df['Date']
    plt.plot(dates, y, 'b' + marker, markersize=ms, label=f'S&P 500 {ylabel}')
    plt.plot(dates, X*reg.coef_[0] + reg.intercept_, 'lime', label='Regression Line')
    plt.plot(dates, yest_sm, 'r--', label='LOESS Curve')
    first_day = df.loc[0, 'Date'].strftime("%Y-%m-%d")
    last_day = df.loc[n - 1, 'Date'].strftime("%Y-%m-%d")
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
    from scipy.stats import norm
    from statsmodels.tsa.stattools import acf, pacf

    df.reset_index(drop=True, inplace=True)
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
        
        if squared:
            z = df[column].to_numpy() ** 2.
            string += 'Squared '            

        # fft = False to avoid warning
        Aut_Fun_z = pacf(z, nlags=maxlag    )[1:]
        start = 0
        yticks = np.arange(0, 1.25, 0.25)        
        
    else:
        kind = 'Autocorrelogram'
        ylabel = 'Acf value'

        if squared:
            z = df[column].to_numpy() ** 2.
            string += 'Squared '            

        # fft = False to avoid warning
        Aut_Fun_z = acf(z, nlags=maxlag, fft=False)
        start = 0
        yticks = np.arange(0, 1.25, 0.25)

    fig, ax = plt.subplots(figsize=(18, 8))
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

    first_day = df.loc[0, 'Date'].strftime("%Y-%m-%d")
    last_day = df.loc[n - 1, 'Date'].strftime("%Y-%m-%d")
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


def plot_yield_rates(df: pd.DataFrame, start_day: str, end_day: str) -> None:
    """Plot of the Daily Treasury Par Yield Curve Rates.

    Args:
        df (pd.DataFrame): data,
        start_day (str): starting day (format "%Y-%m-%d"),
        end_day (str): ending day (format "%Y-%m-%d").
    """
    mask = (df.Date >= start_day) & (df.Date <= end_day)
    tmp = df[mask].transpose()
    tmp.columns = tmp.loc['Date']
    tmp.drop(index='Date', inplace=True)
    dates = tmp.columns.strftime('%Y-%m-%d')
    n = len(dates)

    link = "https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics"

    fig, ax = plt.subplots(figsize=(18, 9))
    tmp.plot(ax=ax, grid=True)
    plt.suptitle("University of Roma \"Tor Vergata\" - Corso di Metodi"
                + " Probabilistici e Statistici per i Mercati Finanziari \n"
                + f" Line plots U.S. Treasury Yield Curve Rates " 
                + f"(busines days from {start_day} to {end_day})")
    plt.title(f"Path length {n} sample points. Data from U.S. Department of the " 
                + f"Treasure - {link}")
    plt.legend(labels=dates)
    plt.xlabel('Time to maturity')
    plt.ylabel('Yield rates')

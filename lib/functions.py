#!/usr/bin/env python3.10.4
import numpy as np
import pandas as pd
from typing import List


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


def bootstrap(x: np.ndarray, statistics: List[str], N: int = 1000) -> np.ndarray:
    """Compute the bootstrap simulation.

    Args:
        x (np.ndarray): samples data,
        statistics (List[str]): statistics used in the simulation,
        N (int, optional): number of simulations. Defaults to 1000.

    Returns:
        np.ndarray: the simulated samples in which is applyed the
                    statistics wanted.
    """
    n = len(x)
    # bootstrap for all statistics
    boot = []
    for _ in range(N):
        bootsample = np.random.choice(x, size=n, replace=True)
        boot.append([stat(bootsample) for stat in statistics])
        
    # convert the list results in an array
    return np.array(boot)


def bootstrap_summary(x: np.ndarray, N: int = 1000,
                      ci: float = 0.90) -> pd.DataFrame:
    """Bootstrap simulations summary.

    Args:
        x (np.ndarray): samples data,
        N (int, optional): number of simulations. Defaults to 1000.
        ci (float, optional): confidence interval. Defaults to 0.90.

    Returns:
        pd.DataFrame: statistics results summary.
    """
    from scipy.stats import skew, kurtosis
    # define the wanted statistic methods
    statistics = [np.mean, skew, kurtosis]
    boot = bootstrap(x, statistics, N)
    
    # simulated mean of all statistics
    bootmeans = np.mean(boot, axis=0)

    # simulated standard deviation of all statistics
    bootmean_stds = np.std(boot, axis=0)

    sup = (1. + ci) / 2. 
    inf = (1. - ci) / 2.

    lower = np.quantile(boot, inf, axis=0)
    upper = np.quantile(boot, sup, axis=0)
    
    index = [f'{stat.__name__}' for stat in statistics]
    columns = ['mean', 'std', f'P{inf*100:.1f}', f'P{sup*100:.1f}']
    data = np.column_stack((bootmeans, bootmean_stds, lower, upper))
    return pd.DataFrame(data, columns=columns, index=index)

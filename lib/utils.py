from matplotlib.pyplot import legend
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib


legend_type = matplotlib.legend.Legend
def fancy_legend(leg: legend_type) -> None:
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

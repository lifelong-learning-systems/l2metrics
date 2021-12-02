"""
Copyright © 2021 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def smooth(x: np.ndarray, window_len: int = None, window: str = 'flat') -> np.ndarray:
    """Smooths the data using a window with requested size.

    Code from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    example:
        t = linspace(-2,2,0.1)
        x = sin(t)+randn(len(t))*0.1
        y = smooth(x)
    
    See also:
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    NOTE: length(output) != length(input).
    To correct this, return y[(window_len/2-1):-(window_len/2)] instead of just y.

    Args:
        x (np.ndarray): The input signal.
        window_len (int, optional): The dimension of the smoothing window. Defaults to None.
        window (str, optional): The type of window from 'flat', 'hanning', 'hamming', 'bartlett',
            'blackman'. Flat window will produce a moving average smoothing. Defaults to 'flat'.

    Raises:
        ValueError: If input signal has more than one dimension.
        ValueError: If window type is not supported.

    Returns:
        np.ndarray: The smoothed signal.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if window_len is None or x.size < window_len:
        # raise(ValueError, "Input vector needs to be bigger than window size.")
        window_len = min(int(x.size * 0.2), 100)

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    # Perform reflections at both ends of signal
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    s = s[~np.isnan(s)]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    # Changed to return output of same length as input
    start_ind = int(np.floor(window_len/2-1))
    end_ind = -int(np.ceil(window_len/2))
    return y[start_ind:end_ind]


def get_block_saturation_perf(data: Union[pd.DataFrame, List], col_to_use: str = None,
                              prev_sat_val: float = None, window_len: int = None) -> Tuple[float, int, int]:
    """Calculates the saturation value, experiences to saturation, and experiences to recovery.

    Args:
        data (Union[pd.DataFrame, List]): The input data.
        col_to_use (str): The column name of the metric to use for calculations. Defaults to None.
        prev_sat_val (float, optional): Previous saturation value for calculating recovery time.
            Defaults to None.
        window_len (int, optional): The window length for smoothing the data. Defaults to None.

    Returns:
        Tuple[float, int, int]: Saturation value, experiences to saturation, and experiences to recovery.
    """

    # Aggregate multiple reward values for the same experience
    if isinstance(data, pd.DataFrame):
        mean_reward_per_experience = data.loc[:, ['exp_num', col_to_use]].groupby('exp_num').mean()
        mean_data = np.ravel(mean_reward_per_experience.to_numpy())
    else:
        mean_data = np.array(data)

    # Take the moving average of the mean of the per experience reward
    smoothed_data = smooth(mean_data, window_len=window_len)
    smoothed_data = smoothed_data[~np.isnan(smoothed_data)]

    if len(smoothed_data):
        sat_val = np.max(smoothed_data)

        # Calculate the number of experiences to "saturation", which we define as the max of the moving average
        indices = np.where(smoothed_data == sat_val)
        exp_to_sat = int(indices[0][0])
        exp_to_rec = len(data) + 1

        if prev_sat_val:
            indices = np.where(smoothed_data >= prev_sat_val)
            if len(indices[0]):
                exp_to_rec = indices[0][0]
    else:
        sat_val = np.nan
        exp_to_sat = np.nan
        exp_to_rec = np.nan

    return sat_val, exp_to_sat, exp_to_rec


def get_terminal_perf(data: pd.DataFrame, col_to_use: str, prev_val: float = None,
                      term_window_ratio: float = 0.1) -> Tuple[float, int, int]:
    """Calculates the terminal performance, experiences to terminal performance, and experiences to recovery.

    Args:
        data (pd.DataFrame): The input data.
        col_to_use (str): The column name of the metric to use for calculations.
        prev_val (float, optional): Previous saturation value for calculating recovery time.
            Defaults to None.
        term_window_ratio (float, optional): The ratio of terminal data points used to compute the
            terminal performance. Defaults to 0.1 for training blocks and 1.0 for evaluation blocks.

    Returns:
        Tuple[float, int, int]: Terminal performance, experiences to terminal performance,
            and experiences to recovery.
    """

    # Aggregate multiple reward values for the same experience
    if data.shape[0] > 1:
        mean_reward_per_experience = data.loc[:, ['exp_num', col_to_use]].groupby('exp_num').mean()
        mean_data = np.ravel(mean_reward_per_experience.to_numpy())
    else:
        mean_data = np.ravel(data[col_to_use].to_numpy(dtype=float))

    mean_data = mean_data[~np.isnan(mean_data)]

    if len(mean_data):
        # Average all data points in test block for terminal performance
        if data.iloc[0]['block_type'] == 'test':
            term_window_ratio = 1

        terminal_value = np.mean(mean_data[int((1-term_window_ratio)*mean_data.size):])

        # Calculate the number of experiences to terminal performance
        experiences_to_terminal_perf = int((1-(term_window_ratio/2))*mean_data.size)

        # Initialize recovery time to one more than number of learning experiences in the data
        experiences_to_recovery = len(data) + 1

        if prev_val is not None:
            indices = np.where(mean_data >= prev_val)
            if len(indices[0]):
                experiences_to_recovery = indices[0][0]
    else:
        terminal_value = np.nan
        experiences_to_terminal_perf = np.nan
        experiences_to_recovery = np.nan

    return terminal_value, experiences_to_terminal_perf, experiences_to_recovery


def fill_metrics_df(data: dict, metric_string_name: str, metrics_df: pd.DataFrame, dict_key: str = None) -> pd.DataFrame:
    """Fills the metrics DataFrame with additional data.

    Args:
        data (dict): The new metric data to insert into the metrics DataFrame.
        metric_string_name (str): The name of the new metric to add.
        metrics_df (pd.DataFrame): The metrics DataFrame to insert the new data into.
        dict_key (str, optional): The dictionary key of the metrics DataFrame to insert the new
            metric into, allows a higher level of insertion if value is passed. Defaults to None.

    Returns:
        pd.DataFrame: The updated metrics DataFrame.
    """

    if not dict_key:
        metrics_df[metric_string_name] = np.full_like(metrics_df['regime_num'], np.nan, dtype=np.object)
        for idx in data.keys():
            metrics_df.at[idx, metric_string_name] = data[idx]
    else:
        metrics_df[dict_key][metric_string_name] = np.full_like(metrics_df[dict_key]['regime_num'], np.nan, dtype=np.double)
        for idx in data.keys():
            metrics_df[dict_key].loc[idx, metric_string_name] = data[idx]

    return metrics_df


def get_simple_rl_task_names(task_names: list) -> list:
    """Simplifies the task name.

    For each task name in the provided list, this function splits the names using an underscore as
    delimiter, then returns the last element in the split list as the simplified name.

    Args:
        task_names (list): The list of task names to simplify.

    Returns:
        list: The list of simplified task names.
    """

    simple_names = []

    for t in task_names:
        splits = str.split(t, '_')
        simple_names.append(splits[-1].lower())

    return simple_names

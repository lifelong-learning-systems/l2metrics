# (c) 2019 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).
# All Rights Reserved. This material may be only be used, modified, or reproduced
# by or for the U.S. Government pursuant to the license rights granted under the
# clauses at DFARS 252.227-7013/7014 or FAR 52.227-14. For any other permission,
# please contact the Office of Technology Transfer at JHU/APL.

# NO WARRANTY, NO LIABILITY. THIS MATERIAL IS PROVIDED “AS IS.” JHU/APL MAKES NO
# REPRESENTATION OR WARRANTY WITH RESPECT TO THE PERFORMANCE OF THE MATERIALS,
# INCLUDING THEIR SAFETY, EFFECTIVENESS, OR COMMERCIAL VIABILITY, AND DISCLAIMS
# ALL WARRANTIES IN THE MATERIAL, WHETHER EXPRESS OR IMPLIED, INCLUDING (BUT NOT
# LIMITED TO) ANY AND ALL IMPLIED WARRANTIES OF PERFORMANCE, MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF INTELLECTUAL PROPERTY
# OR OTHER THIRD PARTY RIGHTS. ANY USER OF THE MATERIAL ASSUMES THE ENTIRE RISK
# AND LIABILITY FOR USING THE MATERIAL. IN NO EVENT SHALL JHU/APL BE LIABLE TO ANY
# USER OF THE MATERIAL FOR ANY ACTUAL, INDIRECT, CONSEQUENTIAL, SPECIAL OR OTHER
# DAMAGES ARISING FROM THE USE OF, OR INABILITY TO USE, THE MATERIAL, INCLUDING,
# BUT NOT LIMITED TO, ANY DAMAGES FOR LOST PROFITS.

import glob
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
from l2logger.util import *

from . import _localutil


def get_ste_data_names():
    # This function will return a list of the Single-Task-Expert data files names from all of
    # available single task baselines that have been stored in $L2DATA/taskinfo/
    return [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(get_l2root_base_dirs('taskinfo') + "\\*.pkl")]


def load_ste_data(task_name):
    # This function will return a dataframe of the specified task's Single-Task-Expert data that has
    # been stored in $L2DATA/taskinfo/

    if task_name in get_ste_data_names():
        data_file_name = get_l2root_base_dirs('taskinfo', task_name + '.pkl')
        dataframe = pd.read_pickle(data_file_name)
        return dataframe
    else:
        return None


def save_ste_data(log_dir, perf_measure):
    # Load data from ste logs
    ste_data = read_log_data(log_dir, [perf_measure])
    ste_data = ste_data.sort_values(by=['regime_num', 'exp_num']).set_index("regime_num", drop=False)
    ste_data = ste_data[ste_data['block_type'] == 'train']

    # Get task name
    task_name = ste_data.task_name.unique()

    # Check for number of tasks in scenario
    if task_name.size != 1:
        raise Exception('Scenario contains more than one task')

    # Create task info directory if it doesn't exist
    if not os.path.exists(get_l2root_base_dirs('taskinfo')):
        os.makedirs(get_l2root_base_dirs('taskinfo'))

    # Get base directory to store ste data
    filename = get_l2root_base_dirs('taskinfo', task_name[0] + '.pkl')

    # Store ste data in task info directory
    ste_data.to_pickle(filename)

    print(f'Stored STE data for {task_name[0]}')


def plot_performance(dataframe, block_info, do_smoothing=False, col_to_plot='reward',
                     x_axis_col='exp_num', input_title=None, do_save_fig=True, plot_filename=None,
                     input_xlabel='Episodes', input_ylabel='Performance', show_block_boundary=True,
                     shade_test_blocks=True, max_smoothing_window=100):
    # This function takes a dataframe and plots the desired columns. Has an option to save the figure in the current
    # directory and/or customize the title, axes labeling, filename, etc. Color is supported for agent tasks only.

    unique_tasks = dataframe.loc[:, 'task_name'].unique()
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    color_selection = ['blue', 'green', 'red', 'black', 'magenta', 'cyan', 'orange', 'purple']
    if len(unique_tasks) < len(color_selection):
        task_colors = color_selection[:len(unique_tasks)]
    else:
        task_colors = [color_selection[i % len(color_selection)] for i in range(unique_tasks)]

    for c, t in zip(task_colors, unique_tasks):
        for regime in block_info['regime_num']:
            if block_info.loc[regime, :]['task_name'] == t:
                x = dataframe.loc[(dataframe['task_name'] == t) & (
                    dataframe['regime_num'] == regime), x_axis_col].values
                y = dataframe.loc[(dataframe['task_name'] == t) & (
                    dataframe['regime_num'] == regime), col_to_plot].values

                if do_smoothing:
                    window = int(len(y) * 0.2)
                    custom_window = min(window, max_smoothing_window)
                    y = _localutil.smooth(y, window_len=custom_window)

                ax.scatter(x, y, color=c, marker='*', linestyle='None', label=t)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    if show_block_boundary:
        unique_blocks = dataframe.loc[:, 'regime_num'].unique()
        df2 = dataframe.set_index("exp_num", drop=False)
        for b in unique_blocks:
            idx = df2[df2['regime_num'] == b].index[0]
            ax.axes.axvline(idx, linewidth=1, linestyle=':')

    if shade_test_blocks:
        blocks = dataframe.loc[:, ['block_num', 'block_type']].drop_duplicates()
        df2 = dataframe.set_index("exp_num", drop=False)

        for _, block in blocks.iterrows():
            if block['block_type'] == 'test':
                df3 = df2[(df2['block_num'] == block['block_num']) &
                          (df2['block_type'] == block['block_type'])]
                x1 = df3.index[0]
                x2 = df3.index[-1]
                ax.axvspan(x1, x2, alpha=0.1, color='black')

    if os.path.dirname(input_title) != "":
        _, plot_filename = os.path.split(input_title)
        input_title = plot_filename
    else:
        plot_filename = input_title

    # Want the saved figured to have a grid so do this before saving
    ax.set(xlabel=input_xlabel, ylabel=input_ylabel, title=input_title)
    ax.grid()

    if do_save_fig:
        if not plot_filename:
            if not input_title:
                plot_filename = 'plot.png'

        print(f'Saving figure with name: {plot_filename}')

        fig.savefig(plot_filename)
    else:
        plt.show()

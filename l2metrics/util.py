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
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

from learnkit.data_util.utils import get_l2data_root

from . import _localutil


def get_l2root_base_dirs(directory_to_append, sub_to_get=None):
    # This function uses a learnkit utility function to get the base $L2DATA path and goes one level down, with the
    # option to return the path string for the directory or the file underneath:
    # e.g. $L2DATA/logs/some_log_directory
    # or   $L2DATA/taskinfo/info.json
    file_info_to_return = os.path.join(get_l2data_root(), directory_to_append)

    if sub_to_get:
        base_dir = file_info_to_return
        file_info_to_return = os.path.join(base_dir, sub_to_get)

    return file_info_to_return


def load_chance_data():
    # This function will return a dictionary of "chance" definitions for all of the available classification tasks
    # stored in this JSON file, located at $L2DATA/taskinfo/chance.json
    json_file = get_l2root_base_dirs('taskinfo', 'chance.json')

    # Load the defaults from the json file, return them as a dictionary
    with open(json_file) as f:
        chance_dict = json.load(f)

    return chance_dict


def load_default_ste_data():
    # This function will return a dictionary of the Single-Task-Expert data from all of the available single task
    # baselines that have been stored in this JSON file, located at $L2DATA/taskinfo/info.json
    json_file = get_l2root_base_dirs('taskinfo', 'info.json')

    # Load the defaults from the json file, return them as a dictionary
    with open(json_file) as f:
        ste_dict = json.load(f)

    return ste_dict


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


def load_default_random_agent_data():
    # This function will return a dictionary of the Random Agent data from all of the available baselines that have \
    # been stored in this JSON file, located at $L2DATA/taskinfo/random_agent.json
    json_file = get_l2root_base_dirs('taskinfo', 'random_agent.json')

    # Load the defaults from the json file, return them as a dictionary
    with open(json_file) as f:
        random_agent_dict = json.load(f)

    return random_agent_dict


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

                ax.scatter(x, y, color=c, marker='*', linestyle='None')

    ax.legend(unique_tasks)

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
                df3 = df2[(df2['block_num'] == block['block_num']) & (df2['block_type'] == block['block_type'])]
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


def get_fully_qualified_name(log_dir):
    if os.path.dirname(log_dir) == '':
        return get_l2root_base_dirs('logs', log_dir)
    else:
        if os.path.isdir(log_dir):
            return log_dir
        else:
            raise NotADirectoryError


def read_log_data(input_dir, analysis_variables=None):
    # This function scrapes the TSV files containing syllabus metadata and system performance log data and returns a
    # pandas dataframe with the merged data
    logs = None
    blocks = None

    fully_qualified_dir = get_fully_qualified_name(input_dir)

    for root, _, files in os.walk(fully_qualified_dir):
        for file in files:
            if file == 'data-log.tsv':
                if analysis_variables is not None:
                    df = pd.read_csv(os.path.join(root, file), sep='\t')[
                        ['timestamp', 'block_num', 'regime_num', 'exp_num'] + analysis_variables]
                else:
                    df = pd.read_csv(os.path.join(root, file), sep='\t')
                if logs is None:
                    logs = df
                else:
                    logs = pd.concat([logs, df])
            if file == 'block-report.tsv':
                df = pd.read_csv(os.path.join(root, file), sep='\t')
                if blocks is None:
                    blocks = df
                else:
                    blocks = pd.concat([blocks, df])

    return logs.merge(blocks, on=['block_num', 'regime_num'])


def read_column_info(input_dir):
    # This function reads the column info JSON file in the input directory returns the contents

    fully_qualified_dir = get_fully_qualified_name(input_dir)

    with open(fully_qualified_dir + '/column_metric_list.json') as json_file:
        return json.load(json_file)

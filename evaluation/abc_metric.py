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

"""
This is meant to be a Python module containing function definitions for 
computing the Area Between Curve (ABC) transfer measure of a lifelong 
learning experiment.  

"""

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from prettytable import PrettyTable


def make_n_divs(perf1: np.ndarray, perf2: np.ndarray, divs: int = 4):
    """
    Given two performance curves, divide them into 'divs' equal divisions, and return the areas under
    each section of each curve. Note that each area is normalized to fit in the interval [0, 1], which is the same as
    normalizing by the number of LXs.

    :param perf1 (ndarray) Performance curve 1. Representing the lifelong learning performance per LX over a lifetime.
    :param perf2 (ndarray) Performance curve 2.
    :param divs (int) How many equal divisions to divide the performance curve into. Note that the size of each division
    might not be perfectly equal if 'divs' doesn't perfectly divide the full size of the curve.
    :return (ndarray, ndarray) Areas under the curves of each division for each performance curve.
    """
    n_lxs = min(len(perf1), len(perf2))
    q1 = n_lxs // divs
    n_divs1 = dict(
        [('d' + str(i + 1), np.trapz(perf1[i * q1:(i + 1) * q1], dx=1 / perf1[i * q1:(i + 1) * q1].shape[0])) for i in
         range(divs)])
    n_divs2 = dict(
        [('d' + str(i + 1), np.trapz(perf2[i * q1:(i + 1) * q1], dx=1 / perf2[i * q1:(i + 1) * q1].shape[0])) for i in
         range(divs)])
    return n_divs1, n_divs2


def get_n_divs_matrix(perf1: np.ndarray, perf2: np.ndarray, divs: int = 4):
    """
    Given two performance curves, divide them into 'divs' equal divisions, and return the areas under
    each section of each curve. Then compute the differences in area between 'perf2' divisions and the 'perf1' divisions
    and compile the differences into a matrix. Note that each area is normalized to fit in the interval [0, 1], which is
    the same as normalizing by the number of LXs.

    :param perf1 (ndarray) Performance curve 1. Representing the lifelong learning performance per LX over a lifetime.
    :param perf2 (ndarray) Performance curve 2.
    :param divs (int) How many equal divisions to divide the performance curve into. Note that the size of each division
    might not be perfectly equal if 'divs' doesn't perfectly divide the full size of the curve.
    :return (ndarray) <divs> by <divs> size matrix of area under the curve differences between performance curve 2 and
    performance curve 1.
    """
    t1_n_divs, t2_n_divs = make_n_divs(perf1, perf2, divs=divs)

    end = divs + 1
    div_matrix = np.array(
        [[t2_n_divs['d' + str(i)] - t1_n_divs['d' + str(j)] for i in range(1, end)] for j in range(1, end)]
    )
    return div_matrix


def collect_task_data(feather_data: pd.DataFrame, perf_key: str, run_ids: list = None, num_divs: int = 4):
    """
    Collect performance data for all 1st and 2nd training task pairs for a given feather file and the desired set of
     run IDs. Construct the mean base performance for each 1st task, and compute the areas between the 2nd task curve
     and the computed (mean) base performance for each scenario. Also compute the 'n_div_matrix' of area between curve
     value differences. Return lists of dictionaries containing:
        - Performance vectors for each 1st, 2nd task of a scenario
        - Area between 2nd task performance and computed base performance
        - n-division area between curve values
        and
        - Computed base performance vector (1st task)

    :param feather_data: (DataFrame) SG data from feather file
    :param perf_key: (str) Performance value used for this SG to extract performance values from previously mentioned
        DataFrames, for example: 'performance_normalized'.
    :param run_ids: (list of strings) Run IDs to pull from the feather data to use in computing ABC measure. If None,
        get all run IDs from the feather file
    :param num_divs: (int) How many equal divisions to divide the performance curve into. Note that the size of each
        division might not be perfectly equal if 'divs' doesn't perfectly divide the full size of the curve.
    :return: (Tuple(dict, dict)) samples, base performance curves
    """

    all_samples = []
    for syl, df in feather_data.groupby('run_id'):
        if run_ids is None or syl in run_ids:
            # Get the block numbers for all the training blocks
            train_blocks = df[(df['block_type'] == 'train')]['block_num'].unique()

            # get the block numbers (and then block data) of the first 2 training blocks
            train_block1 = train_blocks[0]
            train_block2 = train_blocks[1]
            block1 = df[(df['block_num'] == train_block1)]
            block2 = df[(df['block_num'] == train_block2)]
            task1 = block1['task_name'].iloc[0]
            task2 = block2['task_name'].iloc[0]
            perf1 = block1[perf_key].to_numpy()
            perf2 = block2[perf_key].to_numpy()
            data_point = {"task1_id": task1, "task1_perf": perf1,
                          "task2_id": task2, 'task2_perf': perf2}

            all_samples.append(data_point)

    # get mean of performance curves for all pre-tasks as baseline comparisons
    base_perfs = {}
    for s in all_samples:
        t_id = s['task1_id']
        if t_id not in base_perfs.keys():
            base_perfs[t_id] = {'perfs': [], 'n_lxs': []}
        base_perfs[t_id]['perfs'].append(s['task1_perf'])
        base_perfs[t_id]['n_lxs'].append(len(s['task1_perf']))

    # Make all of the base performance samples the same length, and get the mean base performance between all
    for ts in base_perfs.keys():
        nlx = np.min(base_perfs[ts]['n_lxs'])
        base_perfs[ts] = np.mean([perf[:nlx] for perf in base_perfs[ts]['perfs']], axis=0)
        assert len(base_perfs[ts]) == nlx

    # add metric measures to all samples
    for s in all_samples:
        base_perf = base_perfs[s['task2_id']]
        perf2 = s['task2_perf']
        common_lxs = min(len(perf2), len(base_perf))  # makes sure you are only getting AOE between common LXs
        base_perf = base_perf[:common_lxs]
        perf2 = perf2[:common_lxs]
        s['task2_perf'] = perf2  # store with new len
        s["base_perf"] = base_perf  # store with new len
        s["area_between_curves"] = np.trapz(perf2 - base_perf, dx=1 / common_lxs)  # normalize by num lxs in integral
        s["n_div_matrix"] = get_n_divs_matrix(base_perf, perf2, divs=num_divs)
        assert len(base_perf) == len(perf2)

    return all_samples, base_perfs


def mean_task_results(task_data_samples: list, base_perfs: dict):
    """
    Compute the mean area-between-curve (AreaBC) and n-division AreaBC differences between runs with
    repeated (1st, 2nd) task pairs. Return that as well as the numbers of samples for each.
    :param task_data_samples: (list) A list of dictionaries, where each contains the data for one ABC value.
    :param base_perfs: (dict) A dictionary of performance curves for each task trained in the first block. The
        performance curve for each task is the mean performance over all of the performance curves available for that
        task.
    :return (dict, dict) A dictionary of mean performance data per task-tuple pair, and another with just the number
        of samples per pair.
    """
    all_task_tuples = {}
    num_samples = {}
    for sample in task_data_samples:
        tup = (sample['task1_id'], sample['task2_id'])
        if tup not in all_task_tuples.keys():
            all_task_tuples[tup] = {'task2_perf': [], "area_between_curves": [], "n_div_matrix": []}
        if tup not in num_samples.keys():
            num_samples[tup] = 0
        num_samples[tup] += 1
        all_task_tuples[tup]['task2_perf'].append(sample['task2_perf'])
        all_task_tuples[tup]['area_between_curves'].append(np.array(sample['area_between_curves']))
        all_task_tuples[tup]['n_div_matrix'].append(sample['n_div_matrix'])

    for tup, data_dict in all_task_tuples.items():
        # get common number of lxs across samples for task2_perf
        nlx = np.min([len(l) for l in data_dict['task2_perf']])
        data_dict['task2_perf'] = [perf[:nlx] for perf in data_dict['task2_perf']]

        # get mean values for each set of samples: 'task2_perf', "area_between_curves", "n_div_matrix"
        for k in data_dict.keys():
            data_dict[k] = np.mean(data_dict[k], axis=0)

            # Add the base performance with the same number of LXs as the task performance
        task2_n_lxs = data_dict['task2_perf'].shape[0]
        data_dict['base_perf'] = base_perfs[tup[1]][:task2_n_lxs]

    return all_task_tuples, num_samples


def pretty_table(div_matrix: np.ndarray):
    """
    Create human readable table of the ABC difference values matrix.
    :param div_matrix: (ndarray) Square matrix of division ABC differences.
    :return (PrettyTable) A PrettyTable object that is easily printed in human readable format.
    """
    tab = PrettyTable()
    num_rows_cols = div_matrix.shape[0]
    tab.field_names = ['T1\\T2 Division Diffs'] + ['T2.D' + str(i) for i in range(1, num_rows_cols + 1)]
    for i in range(num_rows_cols):
        tab.add_row(['T1.D' + str(i + 1)] + [div_matrix[i, j] for j in range(num_rows_cols)])
    return tab


def get_task_transfer_quarters(feather_data, run_ids, perf_key='reward_normalized'):
    """
    Get the mean ABC values and n_div_matrices. Return those and the number of samples per task pair used to compute
    the mean. Really just a convenience function.

    :param feather_data: (DataFrame) SG data from feather file
    :param run_ids: (list of strings) Run IDs to pull from the feather data to use in computing ABC measure.
    :param perf_key: (str) Performance value used for this SG to extract performance values from previously mentioned
        DataFrames, for example: 'performance_normalized'.
    :return (dict, dict) A dictionary of mean performance data per task-tuple pair, and another with just the number
        of samples per pair.
    """
    task_information, base_perfs = collect_task_data(feather_data, run_ids, perf_key, num_divs=4)
    means, counts = mean_task_results(task_information, base_perfs)
    return means, counts


def show_task_transfer_info(means, counts, print_table=False, plots=False):
    """
    Given the means and numbers of samples of task-pair performances, print useful information, normalized
    by the number of LXs in each curve, and plot performance curves and n_div_matrix.

    :param means: (dict) A dictionary of mean performance data per task-tuple pair.
    :param counts: (dict) A dictionary with the number of samples per pair used to compute the mean performance data.
    :param print_table: (bool) If true, print the table of division data as part of the output.
    :param plots: (bool) If true, plot the n_div_matrix as a heatmap and the curve mean performance curves.
    :return None
    """
    for m in means.keys():
        pre_train_task = m[0]
        task2 = m[1]
        num_samples = counts[m]
        abcs = means[m]['area_between_curves']
        base_perf = means[m]['base_perf']
        perf_with_transfer = means[m]['task2_perf']
        n_div_matrix = means[m]['n_div_matrix']
        print()
        print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        print('Pre-train task (Task1):', pre_train_task, '-- Learning task (Task2):', task2)
        print('Number of samples for this task pair (averaged over):', num_samples)
        print("Area between curves (normalized by num LXs):", abcs)
        print("Division matrix diagonal values +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        n_diags = n_div_matrix.shape[0]
        out_strs = [f"{i}:  {n_div_matrix[i, i]}," for i in range(n_diags)]
        print(' '.join(out_strs)[:-1])
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        if print_table:
            print(pretty_table(n_div_matrix))
        if plots:
            sns.heatmap(n_div_matrix)
            plt.show()
            if base_perf.shape[0] != perf_with_transfer.shape[0]:
                print(pre_train_task, task2, base_perf.shape, perf_with_transfer.shape)
            plt.figure()
            plt.plot(base_perf, label='base perf')
            plt.plot(perf_with_transfer, label='with ' + pre_train_task + ' pre-training')
            plt.fill_between(range(len(base_perf)), base_perf, perf_with_transfer,
                             where=(perf_with_transfer >= base_perf),
                             color='green', alpha=0.3)
            plt.fill_between(range(len(base_perf)), base_perf, perf_with_transfer,
                             where=(perf_with_transfer < base_perf),
                             color='red', alpha=0.3)
            plt.title(task2)
            plt.legend()
            plt.show()
        print("----------------------------------------------------------------------------------------\n\n")


def task_transfer_quarters(feather_data, run_ids, perf_key='reward_normalized', print_table=True, plots=True):
    """
    A convenience function. Compute and show mean area-between-curve transfer information for a given set of feather
    data.
    """
    means, counts = get_task_transfer_quarters(feather_data, run_ids, perf_key=perf_key)
    show_task_transfer_info(means, counts, print_table=print_table, plots=plots)

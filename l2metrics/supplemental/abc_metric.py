"""
Copyright © 2021-2022 The Johns Hopkins University Applied Physics Laboratory LLC

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
learning experiment.  ABC transfer is an alternate transfer metric that
attempts to account for the differences in training an L2 agent without
experience with another task and with experience with another task. This
is done by computing the area between the performance curves for the 
two aforementioned cases. It is hoped that this measure will better
reflect the overall benefits of pre-training, including:
- Accelerated learning rate
- Initial jumpstart transfer
- Difference in final performance

We define **ABC Transfer** as follows:


    ABC(T_1, T_2) = integral{ (P_{T_2 | T_1}(l) - B_{T_2}(l))dl } 
                  = integral{ P_{T_2 | T_1}(l)dl } - integral{ B_{T_2}(l)dl }


where T_1 is the task pre-trained on, and T_2 is the task trying to be learned. P_{T_i| T_j}(l) is the performance
curve of task i given that it was pre-trained on task j, B_{T_i} is some baseline performance of a learning 
algorithm on task i, and l in R denotes the l-th learning experience (LX) in the training block. 

This Implementation:

- Expected data format and baseline performance
This implementation assumes that there will be a set number of tasks for the agent to learn, and that the agent will 
experience all of them in a single lifetime. Then multiple lifetimes will be ran with different permutations of the 
tasks. This implementation consumes data from a set of lifetimes (runs), computes a baseline performance for each task 
by averaging the performance curves of each task ran in the first training block. For example, if there are two 
lifetimes in which task $T_b$ is trained on first, the baseline performance curve for task $T_b$ will be the average of 
the two curves, and so on for each task. In cases where the lengths of the curves are different, we cut off the end of 
the longer curve so they have the same number of LXs. Naturally, *that means there may be tasks for which ABC will not 
be computed if it was never ran first in the training sequence*. 

- Normalization
In order to facilitate comparability between different environments, tasks, and systems, we normalize the ABC values by 
the number of LXs in the performance curve (or performance curve segments if examining ABC of a section of the curve). 
This is accomplished by making the substitution x = l/n, where n is the number of LXs in the performance and baseline 
curves (if their lengths are different, we drop the end of the longer curve to make them the same). The result is

    ABC(T_1, T_2) = integral |(0, n) { (P_{T_2 | T_1}(l) - B_{T_2}(l))dl }
    = integral |(0, n) { P_{T_2 | T_1}(l)dl } - integral |(0, n) { B_{T_2}(l)dl }
    = integral |(0, 1) { (P_{T_2 | T_1}(nx) - B_{T_2}(nx))dx }
    = integral |(0, 1) { P_{T_2 | T_1}(nx)dx } - integral |(0, 1) { B_{T_2}(nx)dx } 

Computation:

The computation is accomplished via the Trapezoid Rule using `numpy`'s `trapz` method. 

"""

import os
from typing import Union, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from prettytable import PrettyTable


def _make_n_divs(
    perf1: np.ndarray, perf2: np.ndarray, divs: int = 4
) -> (np.ndarray, np.ndarray):
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
        [
            (
                "d" + str(i + 1),
                np.trapz(
                    perf1[i * q1 : (i + 1) * q1],
                    dx=1 / perf1[i * q1 : (i + 1) * q1].shape[0],
                ),
            )
            for i in range(divs)
        ]
    )
    n_divs2 = dict(
        [
            (
                "d" + str(i + 1),
                np.trapz(
                    perf2[i * q1 : (i + 1) * q1],
                    dx=1 / perf2[i * q1 : (i + 1) * q1].shape[0],
                ),
            )
            for i in range(divs)
        ]
    )
    return n_divs1, n_divs2


def _get_n_divs_matrix(
    perf1: np.ndarray, perf2: np.ndarray, divs: int = 4
) -> np.ndarray:
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
    performance curve 1. The following is an example of the data fit into a table:
    +----------------------+--------+-------+-------+-------+
    | T1\\T2 Division Diffs | T2.D1 | T2.D2 | T2.D3 | T2.D4 |
    +----------------------+-------+-------+-------+-------+
    |        T1.D1         | 41.73 | 53.07 | 62.18 | 66.32 |
    |        T1.D2         | 28.00 | 39.34 | 48.45 | 52.58 |
    |        T1.D3         | 13.06 | 24.40 | 33.51 | 37.64 |
    |        T1.D4         | 00.70 | 12.04 | 21.15 | 25.29 |
    +----------------------+-------+-------+-------+-------+
    """
    t1_n_divs, t2_n_divs = _make_n_divs(perf1, perf2, divs=divs)

    end = divs + 1
    div_matrix = np.array(
        [
            [t2_n_divs["d" + str(i)] - t1_n_divs["d" + str(j)] for i in range(1, end)]
            for j in range(1, end)
        ]
    )
    return div_matrix


def calculate_abc(
    feather_data: Union[pd.DataFrame, str],
    perf_key: str,
    run_ids: Union[list, None] = None,
    num_divs: int = 4,
) -> Tuple[list, dict]:
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

    :param feather_data: (DataFrame or path) Agent lifetime performance data included in the feather file generated from
        running L2Metrics (see README.md for L2Metrics). For example, if running L2Metrics on a set of lifetimes were to
        produce the file `/Users/user/data/agent1_experiment.feather`, then

        ```
        feather_data = pd.read_feather("/Users/user/data/agent1_experiment.feather")
        ```

        would be an appropriate designation for this parameter. If a path-like input is given, the function will attempt
        to create a Pandas DataFrame using the given path and assuming it is in feather format
        (https://pandas.pydata.org/docs/reference/api/pandas.read_feather.html). See the L2Metrics README for more
        information on feather files.
    :param perf_key: (str) Name of the performance column in the feather file to use. This is likely the name of the
        application-specific performance and the preprocessing applied separated by an underscore, e.g.
        "performance_normalized". See the L2Metrics README for more information about the columns of the feather file.
    :param run_ids: (list of strings, or None) Run IDs (designating each lifetime) to pull from the feather data to use
        in computing ABC measure. These are under the "run_id" column of the feather file. If None, use all run IDs
        included in the given feather data.
    :param num_divs: (int) How many equal divisions to divide the performance curve into. Note that the size of each
        division might not be perfectly equal if 'divs' doesn't perfectly divide the full size of the curve.
    :return: (Tuple(list, dict)) List of individual ABC measure samples, and base performance curves. Each ABC sample
        is of the form:

            {
            "task1_id": <str>,               # ID for the 1st training task, i.e. task being transferred from
            "task1_perf": <numpy.ndarray>,   # Performance curve for task 1
            "task2_id": <str>,               # ID for the 2nd training task, i.e. task being transferred to
            "task2_perf": <numpy.ndarray>,   # Performance curve for task 2
            "area_between_curves": <float>,  # ABC value from task 1 to task 2
            "base_perf": <numpy.ndarray>,    # Baseline (mean) performance for this agent on task 2, from feather data
            "n_div_matrix": <numpy.ndarray>  # Matrix of ABC values between each subsection of task 2 and task 1
            }

        The base performance curve dictionary is simply a separate dictionary containing the baseline performance
        curves for all tasks, e.g.

            {
            <task1_id>: <numpy.ndarray>,
            ...
            <taskN_id>: <numpy.ndarray>
            }

        and is returned in this format for convenience.
    """

    if not isinstance(feather_data, pd.DataFrame):
        if not os.path.isfile(feather_data):
            raise TypeError(
                "Argument 'feather_data' must be a Pandas DataFrame object or path to a feather file "
                "readable via `pandas.read_feather`. Instead got type {} with value "
                "{}".format(type(feather_data), feather_data)
            )
        else:
            feather_data = pd.read_feather(feather_data)

    all_samples = []
    for syl, df in feather_data.groupby("run_id"):
        if run_ids is None or syl in run_ids:
            # Get the block numbers for all the training blocks
            train_blocks = df[(df["block_type"] == "train")]["block_num"].unique()

            # get the block numbers (and then block data) of the first 2 training blocks
            train_block1 = train_blocks[0]
            train_block2 = train_blocks[1]
            block1 = df[(df["block_num"] == train_block1)]
            block2 = df[(df["block_num"] == train_block2)]
            task1 = block1["task_name"].iloc[0]
            task2 = block2["task_name"].iloc[0]
            perf1 = block1[perf_key].to_numpy()
            perf2 = block2[perf_key].to_numpy()
            data_point = {
                "task1_id": task1,
                "task1_perf": perf1,
                "task2_id": task2,
                "task2_perf": perf2,
            }

            all_samples.append(data_point)

    # get mean of performance curves for all pre-tasks as baseline comparisons
    base_perfs = {}
    for s in all_samples:
        t_id = s["task1_id"]
        if t_id not in base_perfs.keys():
            base_perfs[t_id] = {"perfs": [], "n_lxs": []}
        base_perfs[t_id]["perfs"].append(s["task1_perf"])
        base_perfs[t_id]["n_lxs"].append(len(s["task1_perf"]))

    # Make all of the base performance samples the same length, and get the mean base performance between all
    for ts in base_perfs.keys():
        nlx = np.min(base_perfs[ts]["n_lxs"])
        base_perfs[ts] = np.mean(
            [perf[:nlx] for perf in base_perfs[ts]["perfs"]], axis=0
        )
        assert len(base_perfs[ts]) == nlx

    # add metric measures to all samples
    for s in all_samples:
        base_perf = base_perfs[s["task2_id"]]
        perf2 = s["task2_perf"]
        common_lxs = min(
            len(perf2), len(base_perf)
        )  # makes sure you are only getting AOE between common LXs
        base_perf = base_perf[:common_lxs]
        perf2 = perf2[:common_lxs]
        s["task2_perf"] = perf2  # store with new len
        s["base_perf"] = base_perf  # store with new len
        s["area_between_curves"] = np.trapz(
            perf2 - base_perf, dx=1 / common_lxs
        )  # normalize by num lxs in integral
        s["n_div_matrix"] = _get_n_divs_matrix(base_perf, perf2, divs=num_divs)
        assert len(base_perf) == len(perf2)

    return all_samples, base_perfs


def mean_task_results(task_data_samples: list, base_perfs: dict) -> Tuple[dict, dict]:
    """
    Compute the mean area-between-curve (ABC) and n-division ABC differences between all repeated task pairs.
    Return that as well as the numbers of samples for each.  For example, if `task_data_samples` contains two samples of
    (taskA, taskB) data (i.e. two dictionaries with task1_id="taskA" and task2_id="taskB"), then average their ABC
    values, n_div_matrix values, and their performance curves together to get just one "sample" per task pair.

    :param task_data_samples: (list) A list of dictionaries, where each contains the data for one ABC value. See
        specification of first parameter returned from the <calculate_abc> function.
    :param base_perfs: (dict) A dictionary of performance curves for each task trained in the first block. The
        performance curve for each task is the mean performance over all of the performance curves available for that
        task. See specification of second parameter returned from the <calculate_abc> function.
    :return (dict, dict) (means, num_samples) A dictionary of mean performance data per task-tuple pair, and a
        dictionary with the number of samples per task pair. For example, means would be of the form:

        {
        '(task1_id, task2_id)': {
                                   {
                                    'task2_perf': <np.ndarray>,
                                    "area_between_curves": float,
                                    "n_div_matrix": <np.ndarray>,
                                    "base_perf": <np.ndarray>
                                   }
                                },
       '(task1_id, task3_id)': <dict>,
       ...
       '(task(N-1)_id, taskN_id): <dict>'
       }

       and num_samples would be of the form:

       '(task1_id, task2_id)': <int>,
       '(task1_id, task3_id)': <int>,
       ...
       '(task(N-1)_id, taskN_id): <int>'
       }
    """
    all_task_tuples = {}
    num_samples = {}
    for sample in task_data_samples:
        tup = (sample["task1_id"], sample["task2_id"])
        if tup not in all_task_tuples.keys():
            all_task_tuples[tup] = {
                "task2_perf": [],
                "area_between_curves": [],
                "n_div_matrix": [],
            }
        if tup not in num_samples.keys():
            num_samples[tup] = 0
        num_samples[tup] += 1
        all_task_tuples[tup]["task2_perf"].append(sample["task2_perf"])
        all_task_tuples[tup]["area_between_curves"].append(
            np.array(sample["area_between_curves"])
        )
        all_task_tuples[tup]["n_div_matrix"].append(sample["n_div_matrix"])

    for tup, data_dict in all_task_tuples.items():
        # get common number of lxs across samples for task2_perf
        nlx = np.min([len(l) for l in data_dict["task2_perf"]])
        data_dict["task2_perf"] = [perf[:nlx] for perf in data_dict["task2_perf"]]

        # get mean values for each set of samples: 'task2_perf', "area_between_curves", "n_div_matrix"
        for k in data_dict.keys():
            data_dict[k] = np.mean(data_dict[k], axis=0)

            # Add the base performance with the same number of LXs as the task performance
        task2_n_lxs = data_dict["task2_perf"].shape[0]
        data_dict["base_perf"] = base_perfs[tup[1]][:task2_n_lxs]

    return all_task_tuples, num_samples


def _pretty_table(div_matrix: np.ndarray) -> PrettyTable:
    """
    Create human readable table of the ABC difference values matrix.

    :param div_matrix: (ndarray) Square matrix of division ABC differences.
    :return (PrettyTable) A PrettyTable object that is easily printed in human readable format.
    """
    tab = PrettyTable()
    num_rows_cols = div_matrix.shape[0]
    tab.field_names = ["T1\\T2 Division Diffs"] + [
        "T2.D" + str(i) for i in range(1, num_rows_cols + 1)
    ]
    for i in range(num_rows_cols):
        tab.add_row(
            ["T1.D" + str(i + 1)] + [div_matrix[i, j] for j in range(num_rows_cols)]
        )
    return tab


def get_task_transfer_quarters(
    feather_data: Union[pd.DataFrame, str],
    run_ids: Union[list, None] = None,
    perf_key: str = "reward_normalized",
) -> Tuple[dict, dict]:
    """
    Convenience function, combining `calculate_abc` and `mean_task_results`. Get the mean ABC values and n_div_matrices.
    Return those and the number of samples per task pair used to compute the mean.

    :param feather_data: (DataFrame or path) See definition for the same variable in `calculate_abc`.
    :param run_ids: (list of strings, or None) See definition for the same variable in `calculate_abc`.
    :param perf_key: (str) See definition for the same variable in `calculate_abc`.
    :return (dict, dict) See definition for the return value of `mean_task_results`.
    """
    task_information, base_perfs = calculate_abc(
        feather_data, perf_key, run_ids, num_divs=4
    )
    means, counts = mean_task_results(task_information, base_perfs)
    return means, counts


def show_task_transfer_info(
    means: dict, counts: dict, print_table: bool = False, plots: bool = False
) -> None:
    """
    Given the means and numbers of samples of task-pair performances (i.e. output of `mean_task_results`), print the
    following values for each task pair:
        - task 1 name ("pre-train", or "transfer-from" task)
        - task 2 name ("learning", or "transfer-to" task)
        - Number of samples for the given task pair
        - Average ABC for the given task pair
        - Diagonal values of the n-div-matrix; corresponds to the ABC values for each division with itself

    Optionally, print a human readable table of the n_div_matrix results if 'print_table' is set to True. Also
    optionally, show a plot of the n_div_matrix as a heatmap, and show another plot of mean performance curve and the
    baseline performance curve the area between the curves filled green for positive ABC, and filled red for negative
    ABC.

    :param means: (dict) See definition for the first return value of `mean_task_results`.
    :param counts: (dict) See definition for the second return value of `mean_task_results`.
    :param print_table: (bool) If true, print the table of division data as part of the output.
    :param plots: (bool) If true, plot the n_div_matrix as a heatmap and the mean performance curve and baseline
        performance with the area between those curves filled green for positive ABC and red for negative ABC.
    :return None
    """
    for m in means.keys():
        pre_train_task = m[0]
        task2 = m[1]
        num_samples = counts[m]
        abcs = means[m]["area_between_curves"]
        base_perf = means[m]["base_perf"]
        perf_with_transfer = means[m]["task2_perf"]
        n_div_matrix = means[m]["n_div_matrix"]
        print()
        print(
            "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv"
        )
        print(
            "Pre-train task (Task1):",
            pre_train_task,
            "-- Learning task (Task2):",
            task2,
        )
        print("Number of samples for this task pair (averaged over):", num_samples)
        print("Area between curves (normalized by num LXs):", abcs)
        print(
            "Division matrix diagonal values +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        )
        n_diags = n_div_matrix.shape[0]
        out_strs = [f"{i}:  {n_div_matrix[i, i]}," for i in range(n_diags)]
        print(" ".join(out_strs)[:-1])
        print(
            "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        )
        if print_table:
            print(_pretty_table(n_div_matrix))
        if plots:
            sns.heatmap(n_div_matrix)
            plt.show()
            if base_perf.shape[0] != perf_with_transfer.shape[0]:
                print(pre_train_task, task2, base_perf.shape, perf_with_transfer.shape)
            plt.figure()
            plt.plot(base_perf, label="base perf")
            plt.plot(
                perf_with_transfer, label="with " + pre_train_task + " pre-training"
            )
            plt.fill_between(
                range(len(base_perf)),
                base_perf,
                perf_with_transfer,
                where=(perf_with_transfer >= base_perf),
                color="green",
                alpha=0.3,
            )
            plt.fill_between(
                range(len(base_perf)),
                base_perf,
                perf_with_transfer,
                where=(perf_with_transfer < base_perf),
                color="red",
                alpha=0.3,
            )
            plt.title(task2)
            plt.legend()
            plt.show()
        print(
            "----------------------------------------------------------------------------------------\n\n"
        )


def calculate_abc_quarters(
    feather_data: Union[pd.DataFrame, str],
    run_ids: Union[list, None] = None,
    perf_key: str = "reward_normalized",
    print_table: bool = True,
    plots: bool = True,
) -> None:
    """
    A top-level convenience function combining "get_task_transfer_quarters", and "show_task_transfer_info". Effectively
    reduces all other functions in the module into one function call to compute and then show mean
    area-between-curve transfer information for the given feather data.

    :param feather_data: (DataFrame or path) See definition for the same variable in `calculate_abc`.
    :param run_ids: (list of strings or None) See definition for the same variable in `calculate_abc`.
    :param perf_key: (str) See definition for the same variable in `calculate_abc`.
    :param print_table: (bool) If true, print the table of division data as part of the output.
    :param plots: (bool) If true, plot the n_div_matrix as a heatmap and the curve mean performance curves.
    :return None
    """
    means, counts = get_task_transfer_quarters(feather_data, run_ids, perf_key=perf_key)
    show_task_transfer_info(means, counts, print_table=print_table, plots=plots)

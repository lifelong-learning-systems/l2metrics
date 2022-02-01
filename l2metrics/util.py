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

import logging
import os
import pickle
from collections import OrderedDict
from math import ceil, sqrt
from pathlib import Path
from typing import List

import l2logger.util as l2l
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Get default color cycler
color_cycler = plt.rcParams["axes.prop_cycle"]

logger = logging.getLogger(__name__)


def get_ste_data_names() -> list:
    """Gets the names of the stored STE data in $L2DATA/taskinfo/.

    Returns:
        list: The STE task names.
    """

    ste_files = list(Path(l2l.get_l2root_base_dirs("taskinfo")).glob("*.pickle"))

    if ste_files:
        return np.char.lower([f.stem for f in ste_files])
    else:
        return []


def load_ste_data(task_name: str) -> List[pd.DataFrame]:
    """Loads the STE data corresponding to the given task name.

    Args:
        task_name (str): The name of the STE data file.

    Returns:
        List[pd.DataFrame]: The STE data if found, else empty list.
    """

    # Variant-aware STE task names
    ste_task_variant_names = get_ste_data_names()

    # Variant-agnostic STE task names
    ste_task_base_names = set(
        [task_name.split("_")[0] for task_name in ste_task_variant_names]
    )

    if task_name in ste_task_variant_names:
        # Load variant-aware STE data
        ste_file_name = l2l.get_l2root_base_dirs("taskinfo", task_name + ".pickle")
        with open(ste_file_name, "rb") as ste_file:
            ste_data = pickle.load(ste_file)
            return ste_data
    elif task_name in ste_task_base_names:
        ste_data = []
        # Load variant-agnostic STE data
        for ste_variant_file in l2l.get_l2root_base_dirs("taskinfo").glob(
            task_name + "*.pickle"
        ):
            with open(ste_variant_file, "rb") as ste_file:
                ste_data.extend(pickle.load(ste_file))

        # Remove variant label from task names
        for idx, ste_data_df in enumerate(ste_data):
            ste_data[idx]["task_name"] = ste_data_df["task_name"].apply(
                lambda x: x.split("_")[0]
            )

        return ste_data
    else:
        return []


def store_ste_data(log_dir: Path, mode: str = "w") -> None:
    """Stores the STE data in the given log directory as a serialized DataFrame.

    Args:
        log_dir (Path): The log directory of the STE data.
        mode (str, optional): The mode for saving STE data. Defaults to 'w'.
            'w' - Write - Opens a file for writing, overwrites data if the file exists.
            'a' - Append - Opens a file for writing, appends data if the file exists.

    Raises:
        ValueError: If scenario does not only contain one task for training.
    """

    # Load data from ste logs
    ste_data_df = l2l.read_log_data(log_dir)

    # Get metric fields
    logger_info = l2l.read_logger_info(log_dir)

    # Validate data format
    l2l.validate_log(ste_data_df, logger_info["metrics_columns"])

    # Filter data by completed experiences
    ste_data_df = ste_data_df[ste_data_df["exp_status"] == "complete"]

    # Fill in regime number and sort
    ste_data_df = l2l.fill_regime_num(ste_data_df)
    ste_data_df = ste_data_df.sort_values(by=["regime_num", "exp_num"])

    # Get training task name
    task_name = list(
        ste_data_df[ste_data_df["block_type"] == "train"].task_name.unique()
    )

    # Check for number of tasks in scenario
    if len(task_name) != 1:
        raise ValueError(
            f"Expected 1 trained task in {log_dir.name} but found {len(task_name)}"
        )

    # Add STE dataframe to list
    ste_data = [ste_data_df]

    # Create task info directory if it doesn't exist
    task_info_dir = l2l.get_l2root_base_dirs("taskinfo")
    if not task_info_dir.exists():
        task_info_dir.mkdir(parents=True, exist_ok=True)

    # Get base directory to store ste data
    filename = task_info_dir / (task_name[0] + ".pickle")

    # Store ste data in task info directory
    if mode == "a":
        # Load existing STE data and append
        if filename.exists():
            with open(filename, "rb") as ste_file:
                stored_ste_data = pickle.load(ste_file)
                ste_data.extend(stored_ste_data)

    # Write/Overwrite STE data to file
    with open(filename, "wb") as ste_file:
        pickle.dump(ste_data, ste_file)

    logger.info(f"Stored STE data for {task_name[0]} in {log_dir.name}")


def plot_blocks(
    dataframe: pd.DataFrame,
    reward: str,
    unique_tasks: list,
    task_colors: dict = {},
    input_title: str = "",
    output_dir: str = "",
    do_save_fig: bool = False,
    plot_filename: str = "block_plot",
):
    """Plot learning performance curves and evaluation blocks as separate plots.

    Args:
        dataframe (pd.DataFrame): The performance data to plot.
        reward (str): The column name of the metric to plot.
        unique_tasks (list): List of unique tasks in scenario.
        task_colors (dict): Dict of task names and colors for plotting. Defaults to {}.
        input_title (str, optional): The plot title. Defaults to ''.
        output_dir (str, optional): Output directory of results. Defaults to ''.
        do_save_fig (bool, optional): Flag for enabling saving figure. Defaults to False.
        plot_filename (str, optional): The filename to use for saving. Defaults to 'block_plot'.
    """

    reward_col = reward + "_raw"

    if reward + "_smoothed" in dataframe.columns:
        reward_col_smooth = reward + "_smoothed"
    else:
        reward_col_smooth = None

    # Use sleep evaluation blocks if they exist and filter out wake evaluation
    if "sleep" in dataframe["block_subtype"].to_numpy():
        dataframe = dataframe[
            ~(
                dataframe.block_type.isin(["test"])
                & dataframe.block_subtype.isin(["wake"])
            )
        ]

    df_test = dataframe[dataframe.block_type == "test"]
    df_train = dataframe[dataframe.block_type == "train"]

    fig, axes = plt.subplots(
        len(unique_tasks) + 1,
        1,
        figsize=(12, max(8, 2 * len(unique_tasks))),
        sharex=True,
        constrained_layout=True,
    )
    ax0 = axes[0]
    ax0.set_xlabel("Experiences")
    ax0.set_ylabel(reward_col + " (LX)")
    ax0.grid()

    task_idx = 1
    xv_max = None  # Workaround for inconsistent # of samples

    # Assign colors for each task
    if not task_colors:
        task_colors = {
            task: c["color"] for c, task in zip(color_cycler(), unique_tasks)
        }

    # Plot training and test data
    for task_name in unique_tasks:
        x = df_train[df_train["task_name"] == task_name].exp_num
        y = df_train[df_train["task_name"] == task_name][reward_col]
        ax0.plot(x, y, ".", label=task_name, color=task_colors[task_name], markersize=4)

        ax = axes[task_idx]
        x = df_test[df_test["task_name"] == task_name].exp_num
        y = df_test[df_test["task_name"] == task_name][reward_col]
        ax.plot(x, y, ".", label=task_name, color=task_colors[task_name], markersize=4)
        task_ex_block_data = df_test[df_test["task_name"] == task_name].groupby(
            "block_num"
        )
        xv = task_ex_block_data.exp_num.median()
        if xv_max is None or len(xv) > len(xv_max):
            xv_max = xv
        ex_median = task_ex_block_data[reward_col].median()
        ex_iqr_lower = task_ex_block_data[reward_col].quantile(0.25)
        ex_iqr_upper = task_ex_block_data[reward_col].quantile(0.75)
        ax.fill_between(
            xv, ex_iqr_lower, ex_iqr_upper, color=task_colors[task_name], alpha=0.5
        )
        ax.plot(xv, ex_median, color=task_colors[task_name])
        ax.set_ylabel(task_name + " (EX)")
        ax.grid()
        task_idx += 1

    # Plot smoothed training data
    if reward_col_smooth is not None:
        for _, group in df_train.groupby("block_num"):
            ax0.plot(group.exp_num, group[reward_col_smooth], "k")

    # Set plot title
    if Path(input_title).parent != Path("."):
        _, input_title = os.path.split(input_title)

    ax0.set_title(input_title)

    # Enable plot legend
    ax0.legend(loc="center left", bbox_to_anchor=(1, 0.5), markerscale=2)

    # Set y-axis limits
    if not df_test.empty:
        plt.setp(
            fig.axes,
            ylim=(np.nanmin(df_test[reward_col]), np.nanmax(df_test[reward_col])),
        )

    if do_save_fig:
        logger.info(f'Saving block plot with name: {plot_filename.replace(" ", "_")}')
        fig.savefig(Path(output_dir) / (plot_filename.replace(" ", "_") + ".png"))


def plot_performance(
    dataframe: pd.DataFrame,
    block_info: pd.DataFrame,
    unique_tasks: list,
    task_colors: dict = {},
    x_axis_col: str = "exp_num",
    y_axis_col: str = "reward",
    input_title: str = "",
    input_xlabel: str = "Experiences",
    input_ylabel: str = "Performance",
    show_eval_lines: bool = True,
    show_block_boundary: bool = False,
    shade_test_blocks: bool = True,
    output_dir: str = "",
    do_save_fig: bool = False,
    plot_filename: str = "performance_plot",
) -> None:
    """Plots the performance curves for the given DataFrame.

    Args:
        dataframe (pd.DataFrame): The performance data to plot.
        block_info (pd.DataFrame): The block info of the DataFrame.
        unique_tasks (list): List of unique tasks in scenario.
        task_colors (dict): Dict of task names and colors for plotting. Defaults to {}.
        x_axis_col (str, optional): The column name of the x-axis data. Defaults to 'exp_num'.
        y_axis_col (str, optional): The column name of the metric to plot. Defaults to 'reward'.
        input_title (str, optional): The plot title. Defaults to ''.
        input_xlabel (str, optional): The x-axis label. Defaults to 'Experiences'.
        input_ylabel (str, optional): The y-axis label. Defaults to 'Performance'.
        show_eval_lines (bool, optional): Flag for enabling lines between evaluation blocks to show
            changing slope of evaluation performance. Defaults to True.
        show_block_boundary (bool, optional): Flag for enabling block boundaries. Defaults to True.
        shade_test_blocks (bool, optional): Flag for enabling block shading. Defaults to True.
        output_dir (str, optional): Output directory of results. Defaults to ''.
        do_save_fig (bool, optional): Flag for enabling saving figure. Defaults to False.
        plot_filename (str, optional): The filename to use for saving. Defaults to 'performance_plot'.
    """

    # Initialize figure
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(111)

    # Use sleep evaluation blocks if they exist and filter out wake evaluation
    if "sleep" in block_info["block_subtype"].to_numpy():
        block_info = block_info[
            ~(
                block_info.block_type.isin(["test"])
                & block_info.block_subtype.isin(["wake"])
            )
        ]
        dataframe = dataframe[
            ~(
                dataframe.block_type.isin(["test"])
                & dataframe.block_subtype.isin(["wake"])
            )
        ]

    # Assign colors for each task
    if not task_colors:
        task_colors = {
            task: c["color"] for c, task in zip(color_cycler(), unique_tasks)
        }

    # Loop through tasks and plot their performance curves
    for task_name in unique_tasks:
        if show_eval_lines:
            eval_x_data = []
            eval_y_data = []
            (eval_line,) = ax.plot(
                eval_x_data,
                eval_y_data,
                color=task_colors[task_name],
                linestyle="--",
                alpha=0.2,
            )

        for _, row in block_info[block_info["task_name"] == task_name].iterrows():
            regime_num = row["regime_num"]
            block_type = row["block_type"]

            # Get data for current regime
            x = dataframe.loc[
                dataframe["regime_num"] == regime_num, x_axis_col
            ].to_numpy()
            y = dataframe.loc[
                dataframe["regime_num"] == regime_num, y_axis_col
            ].to_numpy()

            if show_block_boundary:
                ax.axes.axvline(x[0], linewidth=1, linestyle=":")

            if shade_test_blocks and block_type == "test":
                ax.axvspan(x[0], x[-1] + 1, alpha=0.1, facecolor="black")

            if block_type == "test":
                if show_eval_lines:
                    x = list(range(x[0], x[0] + len(y)))
                    eval_x_data.extend(x)
                    eval_y_data.extend(np.nanmean(y) * np.ones(len(x)))
                    eval_line.set_data(eval_x_data, eval_y_data)
                    plt.draw()

            # Match smoothed x and y length if data had NaNs
            if len(x) != len(y):
                x = list(range(x[0], x[0] + len(y)))

            ax.scatter(
                x, y, color=task_colors[task_name], marker="*", s=8, label=task_name
            )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        markerscale=2,
    )

    if Path(input_title).parent != Path("."):
        _, input_title = os.path.split(input_title)

    # Want the saved figured to have a grid so do this before saving
    ax.set(xlabel=input_xlabel, ylabel=input_ylabel, title=input_title)
    ax.grid()

    if do_save_fig:
        logger.info(
            f'Saving performance plot with name: {plot_filename.replace(" ", "_")}'
        )
        fig.savefig(Path(output_dir) / (plot_filename.replace(" ", "_") + ".png"))


def plot_ste_data(
    dataframe: pd.DataFrame,
    ste_data: dict,
    block_info: pd.DataFrame,
    unique_tasks: list,
    task_colors: dict = {},
    perf_measure: str = "reward",
    ste_averaging_method: str = "metrics",
    input_title: str = "",
    input_xlabel: str = "Experiences",
    input_ylabel: str = "Performance",
    output_dir: str = "",
    do_save: bool = False,
    plot_filename: str = "ste_plot",
) -> None:
    """Plots the relative performance of tasks compared to Single-Task Experts.

    Args:
        dataframe (pd.DataFrame): The performance data to plot.
        ste_data (dict): STE data.
        block_info (pd.DataFrame): The block info of the DataFrame.
        unique_tasks (list): List of unique tasks in scenario.
        task_colors (dict): Dict of task names and colors for plotting. Defaults to {}.
        perf_measure (str, optional): The column name of the metric to plot. Defaults to 'reward'.
        ste_averaging_method (str, optional): Method for handling STE metric averaging. Defaults to 'metrics'.
        input_title (str, optional): Plot title. Defaults to ''.
        input_xlabel (str, optional): The x-axis label. Defaults to 'Experiences'.
        input_ylabel (str, optional): The y-axis label. Defaults to 'Performance'.
        output_dir (str, optional): Output directory of results. Defaults to ''.
        do_save (bool, optional): Flag for enabling saving figure. Defaults to False.
        plot_filename (str, optional): The filename to use for saving. Defaults to 'ste_plot'.
    """

    # Calculate subplot dimensions
    cols = ceil(sqrt(len(unique_tasks)))
    rows = ceil(len(unique_tasks) / cols)

    # Initialize figure
    fig = plt.figure(
        figsize=(min(18, 6 * cols), max(6, len(unique_tasks) // 2)),
        constrained_layout=True,
    )
    fig.suptitle(input_title)

    # Initialize axis limits
    x_limit = 0
    y_limit = (np.nan, np.nan)

    # Assign colors for each task
    if not task_colors:
        task_colors = {
            task: c["color"] for c, task in zip(color_cycler(), unique_tasks)
        }

    for index, task_name in enumerate(unique_tasks):
        # Get block info for task during training
        task_blocks = block_info[
            (block_info["task_name"] == task_name)
            & (block_info["block_type"] == "train")
            & (block_info["block_subtype"] == "wake")
        ]

        # Get concatenated data for task
        task_data = dataframe[
            dataframe["regime_num"].isin(task_blocks["regime_num"])
        ].reset_index(drop=True)

        if len(task_data):
            # Create subplot
            ax = fig.add_subplot(rows, cols, index + 1)

            plt.scatter(
                [],
                [],
                label=task_name,
                color=task_colors[task_name],
                marker="*",
                s=8,
            )
            plt.scatter([], [], label="STE", color="orange", marker="*", s=8)

            # Plot LL data
            y_ll = task_data[perf_measure].to_numpy()
            x_ll = list(range(0, len(y_ll)))
            ax.scatter(
                x_ll, y_ll, color=task_colors[task_name], marker="*", s=8, zorder=3
            )

            if ste_data.get(task_name):
                # Get STE data
                y_ste = [
                    ste_data_df[ste_data_df["block_type"] == "train"][
                        perf_measure
                    ].to_numpy()
                    for ste_data_df in ste_data.get(task_name)
                ]

                if ste_averaging_method == "time":
                    # Average all the STE data together after truncating to same length
                    y_ste = np.array([x[: min(map(len, y_ste))] for x in y_ste]).mean(0)
                    x_ste = list(range(0, len(y_ste)))
                    ax.scatter(x_ste, y_ste, color="orange", marker="*", s=8)

                    x_limit = max(x_limit, len(y_ste), len(y_ll))
                    y_limit = (
                        np.nanmin([y_limit[0], np.nanmin(y_ste), np.nanmin(y_ll)]),
                        np.nanmax([y_limit[1], np.nanmax(y_ste), np.nanmax(y_ll)]),
                    )
                else:
                    # Plot runs of STE data
                    for y in y_ste:
                        x = list(range(0, len(y)))
                        ax.scatter(x, y, color="orange", marker="*", s=8)
                        x_limit = max(x_limit, len(y), len(y_ll))
                        y_limit = (
                            np.nanmin([y_limit[0], np.nanmin(y), np.nanmin(y_ll)]),
                            np.nanmax([y_limit[1], np.nanmax(y), np.nanmax(y_ll)]),
                        )
            else:
                x_limit = max(x_limit, len(y_ll))
                y_limit = (
                    np.nanmin([y_limit[0], np.nanmin(y_ll)]),
                    np.nanmax([y_limit[1], np.nanmax(y_ll)]),
                )

                logger.warning(f"STE data for task cannot be found: {task_name}")

            # Draw line at block boundaries of task data
            for x_val in task_data[task_data.regime_num.diff() != 0].index.tolist():
                ax.axes.axvline(x=x_val, color="black", linestyle="--")

            ax.set(xlabel=input_xlabel, ylabel=input_ylabel)
            ax.grid()
            ax.legend(loc="lower right", markerscale=2)
        else:
            logger.warning(
                f"Scenario does not contain training data for task: {task_name}"
            )

    plt.setp(fig.axes, xlim=(0, x_limit), ylim=y_limit)

    if do_save:
        logger.info(f'Saving STE plot with name: {plot_filename.replace(" ", "_")}')
        fig.savefig(Path(output_dir) / (plot_filename.replace(" ", "_") + ".png"))

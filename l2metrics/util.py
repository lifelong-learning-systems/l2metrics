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
from math import ceil, floor, sqrt
from pathlib import Path
from typing import List

import l2logger.util as l2l
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler

# Create color cycler
color_cycler = cycler(
    color=[
        "#1f77b4",
        "#aec7e8",
        "#ff7f0e",
        "#ffbb78",
        "#2ca02c",
        "#98df8a",
        "#d62728",
        "#ff9896",
        "#9467bd",
        "#c5b0d5",
        "#8c564b",
        "#c49c94",
        "#e377c2",
        "#f7b6d2",
        "#7f7f7f",
        "#c7c7c7",
        "#bcbd22",
        "#dbdb8d",
        "#17becf",
        "#9edae5",
    ]
)

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


def plot_raw(
    dataframe: pd.DataFrame,
    unique_tasks: list,
    task_colors: dict = {},
    x_axis_col: str = "exp_num",
    y_axis_col: str = "reward",
    input_title: str = "",
    output_dir: str = "",
    do_save_fig: bool = False,
    plot_filename: str = "raw_plot",
):
    """Plot raw learning performance curves with smoothed curve overlaid.

    Args:
        dataframe (pd.DataFrame): The performance data to plot.
        reward (str): The column name of the metric to plot.
        unique_tasks (list): List of unique tasks in scenario.
        task_colors (dict): Dict of task names and colors for plotting. Defaults to {}.
        x_axis_col (str, optional): The column name of the x-axis data. Defaults to 'exp_num'.
        y_axis_col (str, optional): The column name of the metric to plot. Defaults to 'reward'.
        input_title (str, optional): The plot title. Defaults to ''.
        output_dir (str, optional): Output directory of results. Defaults to ''.
        do_save_fig (bool, optional): Flag for enabling saving figure. Defaults to False.
        plot_filename (str, optional): The filename to use for saving. Defaults to 'raw_plot'.
    """

    reward_col_raw = y_axis_col + "_raw"

    if y_axis_col + "_smoothed" in dataframe.columns:
        reward_col_smooth = y_axis_col + "_smoothed"
    else:
        reward_col_smooth = None

    df_train = dataframe[dataframe.block_type == "train"]

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(111)

    # Assign colors for each task
    if not task_colors:
        task_colors = {
            task: color["color"] for color, task in zip(color_cycler, unique_tasks)
        }

    # Plot raw training data
    for task_name in unique_tasks:
        x = df_train[df_train["task_name"] == task_name][x_axis_col]
        y = df_train[df_train["task_name"] == task_name][reward_col_raw]
        ax.plot(x, y, ".", label=task_name, color=task_colors[task_name], markersize=4)

    # Plot smoothed training data
    if reward_col_smooth is not None:
        for _, group in df_train.groupby("block_num"):
            ax.plot(group[x_axis_col], group[reward_col_smooth], "k")

    # Set plot title
    if Path(input_title).parent != Path("."):
        _, input_title = os.path.split(input_title)

    ax.set_title(input_title)
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    ax.set_xlabel("Experiences")
    ax.set_ylabel("Raw Performance")
    ax.grid()

    # Enable plot legend
    # TODO: Figure out why all tasks are showing in legend
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), markerscale=2)

    if do_save_fig:
        logger.info(f'Saving raw plot with name: {plot_filename.replace(" ", "_")}')
        fig.savefig(Path(output_dir) / (plot_filename.replace(" ", "_") + ".png"))


def plot_evaluation_blocks(
    dataframe: pd.DataFrame,
    unique_tasks: list,
    task_colors: dict = {},
    x_axis_col: str = "exp_num",
    y_axis_col: str = "reward",
    input_title: str = "",
    input_xlabel: str = "Experiences",
    input_ylabel: str = "Performance",
    output_dir: str = "",
    do_save_fig: bool = False,
    plot_filename: str = "eval_plot",
) -> None:
    """Plots the evaluation block data for the given DataFrame.

    Args:
        dataframe (pd.DataFrame): The performance data to plot.
        unique_tasks (list): List of unique tasks in scenario.
        task_colors (dict): Dict of task names and colors for plotting. Defaults to {}.
        x_axis_col (str, optional): The column name of the x-axis data. Defaults to 'exp_num'.
        y_axis_col (str, optional): The column name of the metric to plot. Defaults to 'reward'.
        input_title (str, optional): The plot title. Defaults to ''.
        input_xlabel (str, optional): The x-axis label. Defaults to 'Experiences'.
        input_ylabel (str, optional): The y-axis label. Defaults to 'Performance'.
        output_dir (str, optional): Output directory of results. Defaults to ''.
        do_save_fig (bool, optional): Flag for enabling saving figure. Defaults to False.
        plot_filename (str, optional): The filename to use for saving. Defaults to 'performance_plot'.
    """
    # Use sleep evaluation blocks if they exist and filter out wake evaluation
    if "sleep" in dataframe["block_subtype"].to_numpy():
        dataframe = dataframe[
            ~(
                dataframe.block_type.isin(["test"])
                & dataframe.block_subtype.isin(["wake"])
            )
        ]

    df_test = dataframe[dataframe.block_type == "test"]

    task_clusters = np.unique([task_name.split("_")[0] for task_name in unique_tasks])

    # Calculate subplot dimensions
    cols = ceil(sqrt(len(task_clusters)))
    if cols == 0:
        return
    rows = ceil(len(task_clusters) / cols)

    # Initialize figure
    fig = plt.figure(
        figsize=(min(18, max(12, 6 * cols)), max(6, len(task_clusters) // 2)),
        constrained_layout=True,
    )
    fig.suptitle(input_title)

    # Assign colors for each task
    if not task_colors:
        task_colors = {
            task: color["color"] for color, task in zip(color_cycler, unique_tasks)
        }

    for idx, task_cluster in enumerate(task_clusters):
        # Create subplot
        ax = fig.add_subplot(rows, cols, idx + 1)

        cluster_eval_data = df_test[df_test["task_name"].str.contains(task_cluster)]

        if not cluster_eval_data.empty:
            sns.pointplot(
                x="block_num",
                y=y_axis_col,
                hue="task_name",
                palette=task_colors,
                data=cluster_eval_data,
            )

            ax.grid()
            ax.legend(loc="lower right")

    # Set common y-axis limits if data is normalized
    if any("normalized" in col for col in df_test.columns):
        plt.setp(fig.axes, ylim=(0, 101))

    if do_save_fig:
        logger.info(
            f'Saving evaluation plot with name: {plot_filename.replace(" ", "_")}'
        )
        fig.savefig(Path(output_dir) / (plot_filename.replace(" ", "_") + ".png"))


def plot_learning_blocks(
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
    plot_filename: str = "learning_plot",
    fig=None,
) -> None:
    """Plots the learning block performance curves for the given DataFrame.

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
        show_block_boundary (bool, optional): Flag for enabling block boundaries. Defaults to False.
        shade_test_blocks (bool, optional): Flag for enabling block shading. Defaults to True.
        output_dir (str, optional): Output directory of results. Defaults to ''.
        do_save_fig (bool, optional): Flag for enabling saving figure. Defaults to False.
        plot_filename (str, optional): The filename to use for saving. Defaults to 'performance_plot'.
    """

    # Initialize figure
    if fig is None:
        fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    else:
        plt.clf()
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
            task: color["color"] for color, task in zip(color_cycler, unique_tasks)
        }

    if show_eval_lines:
        eval_x_data = {}
        eval_y_data = {}
        eval_lines = {}

        for task_name in unique_tasks:
            eval_x_data[task_name] = []
            eval_y_data[task_name] = []
            (eval_lines[task_name],) = ax.plot(
                [],
                [],
                color=task_colors[task_name],
                linestyle="--",
                alpha=0.2,
            )

    # Initialize exp indices
    lx_idx = 0
    ex_idx = 0

    # Loop DataFrame and plot performance curves
    for _, row in block_info.iterrows():
        regime_num = row["regime_num"]
        block_type = row["block_type"]
        task_name = row["task_name"]

        # Get data for current regime
        x = dataframe.loc[dataframe["regime_num"] == regime_num, x_axis_col].to_numpy()
        y = dataframe.loc[dataframe["regime_num"] == regime_num, y_axis_col].to_numpy()

        if show_block_boundary:
            ax.axes.axvline(
                x[0] - ex_idx, color="black", linewidth=0.5, linestyle="--", alpha=0.2
            )

        if block_type == "test":
            # if shade_test_blocks:
            #     ax.axvspan(x[0], x[-1] + 1, alpha=0.1, facecolor="black")

            if show_eval_lines:
                eval_x_data[task_name].extend([lx_idx])
                eval_y_data[task_name].extend([np.nanmean(y)])
                eval_lines[task_name].set_data(
                    eval_x_data[task_name], eval_y_data[task_name]
                )
                plt.draw()

            ex_idx += x[-1] - x[0] + 1
        else:
            ax.scatter(
                x - ex_idx,
                y,
                color=task_colors[task_name],
                marker="*",
                s=8,
                label=task_name,
            )

            lx_idx += x[-1] - x[0] + 1

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
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    ax.grid()

    if do_save_fig:
        logger.info(
            f'Saving learning plot with name: {plot_filename.replace(" ", "_")}'
        )
        fig.savefig(Path(output_dir) / (plot_filename.replace(" ", "_") + ".png"))


def plot_ste(
    dataframe: pd.DataFrame,
    ste_data: dict,
    block_info: pd.DataFrame,
    unique_tasks: list,
    x_axis_col: str = "exp_num",
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
        x_axis_col (str, optional): The column name of the x-axis data. Defaults to 'exp_num'.
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
    if cols == 0:
        return
    rows = ceil(len(unique_tasks) / cols)

    # Initialize figure
    fig = plt.figure(
        figsize=(min(18, max(12, 6 * cols)), max(6, len(unique_tasks) // 2)),
        constrained_layout=True,
    )
    fig.suptitle(input_title)

    # Initialize axis limits
    x_limit = 0
    y_limit = (np.nan, np.nan)

    # Assign colors for each task
    if not task_colors:
        task_colors = {
            task: color["color"] for color, task in zip(color_cycler, unique_tasks)
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

            x_ll = []
            last_exp = 0
            mean_exp_diff = 0
            for reg_idx, regime in enumerate(task_data["regime_num"].unique()):
                x = task_data.loc[
                    task_data["regime_num"] == regime, x_axis_col
                ].to_numpy()
                if reg_idx == 0:
                    x_ll.extend(x - x[0])
                else:
                    # Draw line at block boundaries of task data
                    ax.axes.axvline(
                        x=x_ll[-1] + mean_exp_diff, color="black", linestyle="--"
                    )

                    x_ll.extend(x - (x[0] - last_exp) + mean_exp_diff)
                last_exp = x_ll[-1]
                mean_exp_diff = np.mean(np.diff(x))

            ax.scatter(
                x_ll, y_ll, color=task_colors[task_name], marker="*", s=8, zorder=3
            )

            if ste_data.get(task_name):
                x_ste = []
                y_ste = []

                # Get STE data
                for ste_data_df in ste_data.get(task_name):
                    x = ste_data_df[ste_data_df["block_type"] == "train"][
                        x_axis_col
                    ].to_numpy()
                    x_ste.append(x - x[0])
                    y_ste.append(
                        ste_data_df[ste_data_df["block_type"] == "train"][
                            perf_measure
                        ].to_numpy()
                    )

                if ste_averaging_method == "time":
                    # Average all the STE data together after truncating to same length
                    y_ste = np.array([x[: min(map(len, y_ste))] for x in y_ste]).mean(0)
                    x_ste = np.array([x[: min(map(len, x_ste))] for x in x_ste]).mean(0)
                    ax.scatter(x_ste, y_ste, color="orange", marker="*", s=8)

                    x_limit = max(x_limit, np.nanmax(x_ste), np.nanmax(x_ll))
                    y_limit = (
                        np.nanmin([y_limit[0], np.nanmin(y_ste), np.nanmin(y_ll)]),
                        np.nanmax([y_limit[1], np.nanmax(y_ste), np.nanmax(y_ll)]),
                    )
                    logger.warning("Time STE averaging method is deprecated")
                else:
                    # Plot runs of STE data
                    for x, y in zip(x_ste, y_ste):
                        ax.scatter(x, y, color="orange", marker="*", s=4)
                        x_limit = max(x_limit, np.nanmax(x), np.nanmax(x_ll))
                        y_limit = (
                            np.nanmin([y_limit[0], np.nanmin(y), np.nanmin(y_ll)]),
                            np.nanmax([y_limit[1], np.nanmax(y), np.nanmax(y_ll)]),
                        )
            else:
                x_limit = max(x_limit, np.nanmax(x_ll))
                y_limit = (
                    np.nanmin([y_limit[0], np.nanmin(y_ll)]),
                    np.nanmax([y_limit[1], np.nanmax(y_ll)]),
                )

                logger.warning(f"STE data for task cannot be found: {task_name}")

            ax.set(xlabel=input_xlabel, ylabel=input_ylabel)
            ax.xaxis.set_major_formatter(ticker.EngFormatter())
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

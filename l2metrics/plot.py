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

import argparse
import json
import logging
from datetime import datetime as dt
from pathlib import Path
from time import sleep
from typing import List, Tuple, Union

from tqdm import tqdm
import l2logger.util as l2l
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from ._localutil import smooth
from .normalizer import Normalizer
from .util import (
    load_ste_data,
    plot_evaluation_blocks,
    plot_learning_blocks,
    plot_raw,
    plot_ste,
)

logging.captureWarnings(True)
logger = logging.getLogger("l2metrics.plot")


def build_plot_parser():
    parser = argparse.ArgumentParser(
        description="Produce L2Metrics plots from the command line"
    )

    # Log directories can be absolute paths, relative paths, or paths found in $L2DATA/logs
    parser.add_argument(
        "-l",
        "--log-dir",
        default=None,
        type=str,
        help="Log directory of scenario. Defaults to None.",
    )

    # Flag for enabling live plotting
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live plotting of the specified log directory. Defaults to False.",
    )

    # Flag for enabling live plotting
    parser.add_argument(
        "-i",
        "--interval",
        default=30,
        type=int,
        help="Update interval, in seconds. Defaults to 30.",
    )

    # Plot types to generate
    parser.add_argument(
        "--plot-types",
        default="lb",
        type=str,
        nargs="+",
        choices=["all", "raw", "eb", "lb", "ste"],
        help="Specify which plot types to generate. Defaults to all.",
    )

    # Method for handling task variants
    parser.add_argument(
        "-r",
        "--variant-mode",
        default="aware",
        type=str,
        choices=["aware", "agnostic"],
        help="Mode for computing metrics with respect to task variants. Defaults to aware.",
    )

    # Flag for recursively calculating metrics on valid subdirectories within log directory
    parser.add_argument(
        "-R",
        "--recursive",
        action="store_true",
        help="Recursively compute metrics on logs found in specified directory. \
                            Defaults to false.",
    )

    # Choose application measure to use as performance column
    parser.add_argument(
        "-p",
        "--perf-measure",
        default="reward",
        type=str,
        help="Name of column to use for metrics calculations. Defaults to reward.",
    )

    # Horizontal axis unit
    parser.add_argument(
        "-u",
        "--unit",
        default="exp_num",
        type=str,
        choices=["exp_num", "steps"],
        help="Unit for plotting data. Defaults to exp_num.",
    )

    # Method for normalization
    parser.add_argument(
        "-n",
        "--normalization-method",
        default="task",
        type=str,
        choices=["task", "run", "none"],
        help="Method for normalizing data. Defaults to task.",
    )

    # Method for smoothing
    parser.add_argument(
        "-g",
        "--smoothing-method",
        default="flat",
        type=str,
        choices=["flat", "hanning", "hamming", "bartlett", "blackman", "none"],
        help="Method for smoothing data, window type. Defaults to flat.",
    )

    # Flag for smoothing evaluation block data
    parser.add_argument(
        "-G",
        "--smooth-eval-data",
        dest="do_smooth_eval_data",
        default=False,
        action="store_true",
        help="Smooth evaluation block data. Defaults to false.",
    )

    # Window length for smoothing
    parser.add_argument(
        "-w",
        "--window-length",
        default=None,
        type=int,
        help="Window length for smoothing data. Defaults to None.",
    )

    # Flag for removing outliers
    parser.add_argument(
        "-x",
        "--clamp-outliers",
        action="store_true",
        help="Remove outliers in data for metrics by clamping to quantiles. Defaults to false.",
    )

    # Data range file for normalization
    parser.add_argument(
        "-d",
        "--data-range-file",
        default=None,
        type=str,
        help="JSON file containing task performance ranges for normalization. Defaults to None.",
    )

    # Flag for showing evaluation block lines
    parser.add_argument(
        "-e",
        "--show-eval-lines",
        action="store_true",
        help="Show lines between evaluation blocks. Defaults to false.",
    )

    # Flag for enabling/disabling save
    parser.add_argument(
        "-S",
        "--save",
        dest="do_save",
        default=False,
        action="store_true",
        help="Save plot. Defaults to false.",
    )
    parser.add_argument(
        "--no-save",
        dest="do_save",
        action="store_false",
        help="Do not save plot",
    )

    return parser


def plot(log_dir, data_range, fig, args) -> None:
    # Get metric fields
    try:
        logger_info = l2l.read_logger_info(log_dir)
    except FileNotFoundError as e:
        print(f"{log_dir.name}: Logger info file not found!")
        return

    # # Do a check to make sure the performance measure exists in logger info
    # if args.perf_measure not in logger_info["metrics_columns"]:
    #     raise KeyError(
    #         f"Performance measure ({args.perf_measure}) not found in valid metrics columns: "
    #         f"{logger_info['metrics_columns']}"
    #     )

    # Gets all data from the relevant log files
    log_data = l2l.read_log_data(log_dir)

    # Do a check to make sure the performance measure is logged
    # if args.perf_measure not in log_data.columns:
    #     raise KeyError(
    #         f"Performance measure ({args.perf_measure}) not found in the log data"
    #     )

    # Validate data format
    # l2l.validate_log(log_data, logger_info["metrics_columns"])

    # Filter data by completed experiences
    log_data = log_data[log_data["exp_status"] == "complete"]

    # Drop all rows with NaN values
    log_data = log_data[log_data[args.perf_measure].notna()]

    # Modify logs for variant-aware or variant-agnostic calculations
    if args.variant_mode == "agnostic":
        # Remove variant label from task names
        log_data.task_name = log_data.task_name.apply(lambda x: x.split("_")[0])

    # Check for log data after filtering
    if log_data.empty:
        raise ValueError(f"Logs do not contain any valid data for: {args.perf_measure}")

    # Fill in regime number and sort
    log_data = l2l.fill_regime_num(log_data)
    log_data = log_data.sort_values(by=["regime_num", "exp_num"]).set_index(
        "regime_num", drop=False
    )

    # Save raw data as separate column
    log_data[args.perf_measure + "_raw"] = log_data[args.perf_measure].to_numpy()

    # Get block summary
    block_info = l2l.parse_blocks(
        log_data, include_task_params=args.variant_mode == "aware"
    )

    # Store unique task names by order of training
    unique_tasks = list(
        block_info.sort_values(["block_type", "block_num"], ascending=[False, True])[
            "task_name"
        ].unique()
    )

    # Load all STE data for tasks in log data
    ste_data = {}

    for task in unique_tasks:
        temp_ste_data = load_ste_data(task)

        # Drop all rows with NaN values
        for idx, ste_data_df in enumerate(temp_ste_data):
            temp_ste_data[idx] = ste_data_df[ste_data_df[args.perf_measure].notna()]

        ste_data[task] = temp_ste_data

    # Smooth LL and STE data
    if args.smoothing_method != "none":
        # Smooth LX data
        for regime_num in block_info["regime_num"].unique():
            if (
                block_info.iloc[regime_num].block_type == "train"
                or args.do_smooth_eval_data
            ):
                x = log_data[log_data["regime_num"] == regime_num][
                    args.perf_measure
                ].to_numpy()
                log_data.loc[
                    log_data["regime_num"] == regime_num, args.perf_measure
                ] = smooth(
                    x, window_len=args.window_length, window=args.smoothing_method
                )

        # Save smoothed data as separate column
        log_data[args.perf_measure + "_smoothed"] = log_data[
            args.perf_measure
        ].to_numpy()

        # Smooth STE data
        for task, ste_data_list in ste_data.items():
            if ste_data_list is not None:
                for idx, ste_data_df in enumerate(ste_data_list):
                    for regime_num in ste_data_df["regime_num"].unique():
                        x = ste_data_df[ste_data_df["regime_num"] == regime_num][
                            args.perf_measure
                        ].to_numpy()
                        ste_data[task][idx].loc[
                            ste_data_df["regime_num"] == regime_num, args.perf_measure
                        ] = smooth(
                            x,
                            window_len=args.window_length,
                            window=args.smoothing_method,
                        )

    # Remove outliers
    if args.clamp_outliers:
        quantiles = (0.1, 0.9)
        # Filter outliers per-task
        for task in unique_tasks:
            # Get task data from dataframe
            x = log_data[log_data["task_name"] == task][args.perf_measure].to_numpy()

            # Initialize bounds
            lower_bound = 0
            upper_bound = 100

            if data_range:
                lower_bound = data_range[task]["min"]
                upper_bound = data_range[task]["max"]
            else:
                if ste_data.get(task):
                    x_ste = np.concatenate(
                        [
                            ste_data_df[ste_data_df["block_type"] == "train"][
                                args.perf_measure
                            ].to_numpy()
                            for ste_data_df in ste_data.get(task)
                        ]
                    )
                    x_comb = np.append(x, x_ste)
                    lower_bound, upper_bound = np.quantile(x_comb, quantiles)
                else:
                    lower_bound, upper_bound = np.quantile(x, quantiles)

            # Filter LL data
            log_data.loc[log_data["task_name"] == task, args.perf_measure] = x.clip(
                lower_bound, upper_bound
            )

            # Filter STE data
            for idx, ste_data_df in enumerate(ste_data.get(task, [])):
                x = ste_data_df[ste_data_df["task_name"] == task][
                    args.perf_measure
                ].to_numpy()
                ste_data[task][idx].loc[
                    ste_data_df["task_name"] == task, args.perf_measure
                ] = x.clip(lower_bound, upper_bound)

        # Save filtered data as separate column
        log_data[args.perf_measure + "_filtered"] = log_data[
            args.perf_measure
        ].to_numpy()

    # Normalize LL and STE data
    if args.normalization_method != "none":
        # Instantiate normalizer
        normalizer = Normalizer(
            perf_measure=args.perf_measure,
            data=log_data[["task_name", args.perf_measure]].set_index("task_name"),
            ste_data=ste_data,
            data_range=data_range,
            method=args.normalization_method,
        )

        # Normalize LL data
        log_data = normalizer.normalize(log_data)

        # Save normalized data as separate column
        log_data[args.perf_measure + "_normalized"] = log_data[
            args.perf_measure
        ].to_numpy()

        # Normalize STE data
        for task, temp_ste_data in ste_data.items():
            if temp_ste_data is not None:
                for idx, ste_data_df in enumerate(temp_ste_data):
                    ste_data[task][idx] = normalizer.normalize(ste_data_df)
    else:
        normalizer = None

    log_dir_name = log_dir.name

    # Check for plotting units
    if args.unit == "steps":
        # Add steps column to log data
        if "episode_step_count" in log_data.columns:
            log_data["steps"] = log_data["episode_step_count"].cumsum()
        else:
            raise KeyError("Step information not available in log")

    if any(plot_type in args.plot_types for plot_type in ["all", "lb"]):
        plot_learning_blocks(
            log_data,
            block_info,
            unique_tasks=unique_tasks,
            show_eval_lines=args.show_eval_lines,
            x_axis_col=args.unit,
            y_axis_col=args.perf_measure,
            input_title="Learning Performance\n" + log_dir_name,
            do_save_fig=args.do_save,
            plot_filename=log_dir_name + "_learning",
            fig=fig,
        )

    if any(plot_type in args.plot_types for plot_type in ["all", "raw"]):
        plot_raw(
            log_data,
            unique_tasks,
            x_axis_col=args.unit,
            y_axis_col=args.perf_measure,
            input_title="Raw and Smoothed Performance\n" + log_dir_name,
            do_save_fig=args.do_save,
            plot_filename=log_dir_name + "_raw",
        )

    if any(plot_type in args.plot_types for plot_type in ["all", "eb"]):
        plot_evaluation_blocks(
            log_data,
            unique_tasks=unique_tasks,
            x_axis_col=args.unit,
            y_axis_col=args.perf_measure,
            input_title="Evaluation Performance\n" + log_dir_name,
            do_save_fig=args.do_save,
            plot_filename=log_dir_name + "_evaluation",
        )

    if any(plot_type in args.plot_types for plot_type in ["all", "ste"]):
        # Only send list of unique tasks with training data
        unique_tasks = block_info[
            block_info["block_type"] == "train"
        ].task_name.unique()

        if args.unit == "steps":
            # Add steps column to STE data
            for task, temp_ste_data in ste_data.items():
                if temp_ste_data is not None:
                    for idx, ste_data_df in enumerate(temp_ste_data):
                        if "episode_step_count" in ste_data_df.columns:
                            ste_data[task][idx]["steps"] = ste_data_df[
                                "episode_step_count"
                            ].cumsum()
                        else:
                            raise KeyError("Step information not available in STE logs")

        plot_ste(
            log_data,
            ste_data,
            block_info,
            unique_tasks,
            x_axis_col=args.unit,
            perf_measure=args.perf_measure,
            input_title="Performance Relative to STE\n" + log_dir_name,
            do_save=args.do_save,
            plot_filename=log_dir_name + "_ste",
        )


def main() -> None:
    # Initialize parser
    parser = build_plot_parser()

    # Parse arguments
    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    if args.data_range_file:
        with open(args.data_range_file) as data_range_file:
            data_range = json.load(data_range_file)
            data_range = {key.lower(): val for key, val in data_range.items()}
    else:
        data_range = None

    # Modify data range based on variant mode
    if args.variant_mode == "agnostic" and args.data_range is not None:
        temp_data_range = {}
        for task_name in set(
            [variant_name.split("_")[0] for variant_name in args.data_range.keys()]
        ):
            task_ranges = [
                data_range
                for variant_name, data_range in args.data_range.items()
                if task_name in variant_name
            ]
            temp_data_range[task_name] = {
                "min": np.min([d["min"] for d in task_ranges]),
                "max": np.max([d["max"] for d in task_ranges]),
            }
        data_range = temp_data_range

    # fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    # fig.show()
    fig = None

    if args.live:
        while True:
            logger.info("Updating plot...")
            plot(log_dir, data_range, fig, args)
            plt.pause(0.0001)
            sleep(args.interval)
    else:
        if args.recursive:
            dirs = [p for p in log_dir.glob("*") if p.is_dir()]
            for dir in tqdm(dirs):
                plt.close("all")
                plot(dir, data_range, fig, args)
        else:
            plot(log_dir, data_range, fig, args)
            plt.show()


if __name__ == "__main__":
    # Configure logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    handler.setStream(tqdm)
    handler.terminator = ""

    logging.basicConfig(level=logging.INFO, handlers=[handler])

    try:
        main()
    except (KeyboardInterrupt, KeyError, ValueError) as e:
        logger.exception(e)

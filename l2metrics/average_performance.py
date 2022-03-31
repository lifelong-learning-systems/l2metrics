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

import numpy as np
import pandas as pd

from ._localutil import fill_metrics_df
from .core import Metric

logger = logging.getLogger(__name__)


class AvgPerf(Metric):
    name = "Average Performance"
    capability = "adapt_to_new_tasks"
    requires = {"syllabus_type": "agent"}
    description = (
        "Calculates the average performance over all blocks,"
        "effectively computing an area under the performance curve for training blocks"
    )

    def __init__(self, perf_measure: str) -> None:
        super().__init__()
        self.perf_measure = perf_measure

    def validate(self, block_info: pd.DataFrame) -> None:
        # Initialize variables for checking block type format
        last_block_num = -1
        last_block_type = ""

        # Ensure alternating block types
        for _, regime in block_info.iterrows():
            if regime["block_num"] != last_block_num:
                last_block_num = regime["block_num"]

                if regime["block_type"] == "test":
                    if last_block_type == "test":
                        logger.warning("Block types are not alternating")
                        break
                    last_block_type = "test"
                elif regime["block_type"] == "train":
                    if last_block_type == "train":
                        logger.warning("Block types are not alternating")
                        break
                    last_block_type = "train"

    def calculate(
        self,
        dataframe: pd.DataFrame,
        block_info: pd.DataFrame,
        metrics_df: pd.DataFrame,
    ) -> pd.DataFrame:
        # Validate block structure
        self.validate(block_info)

        # Use sleep evaluation blocks if they exist
        if "sleep" in block_info["block_subtype"].to_numpy():
            block_info_df = block_info[
                ~(
                    block_info.block_type.isin(["test"])
                    & block_info.block_subtype.isin(["wake"])
                )
            ]
        else:
            block_info_df = block_info.copy()

        # Initialize metric columns
        avg_perf = {}

        # Get unique tasks in scenario
        unique_tasks = block_info_df.task_name.unique()

        # Iterate over tasks
        for task in unique_tasks:
            # Get training regimes
            training_regs = block_info_df[
                (block_info_df["task_name"] == task)
                & (block_info_df["block_type"] == "train")
            ]["regime_num"].to_numpy()

            test_regs = block_info_df[
                (block_info_df["task_name"] == task)
                & (block_info_df["block_type"] == "test")
            ]["regime_num"].to_numpy()

            # Iterate over training regimes
            for training_regime in training_regs:
                # Get performance of current training regime
                training_perf = dataframe[self.perf_measure][
                    dataframe["regime_num"] == training_regime
                ].mean()
                avg_perf[training_regime] = training_perf

            # Iterate over evaluation regimes
            for test_regime in test_regs:
                # Get performance of current test regime
                test_perf = dataframe[self.perf_measure][
                    dataframe["regime_num"] == test_regime
                ].mean()
                avg_perf[test_regime] = test_perf

        # Note: creating separate columns for train and eval
        # that contain the same data - this is a patch
        metrics_df = fill_metrics_df(avg_perf, "avg_train_perf", metrics_df)
        metrics_df = fill_metrics_df(avg_perf, "avg_eval_perf", metrics_df)

        return metrics_df

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
    description = "Calculates the average performance within each block"

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
        avg_train_perf = {}
        avg_eval_perf = {}

        # Iterate over all regimes
        for regime_num in block_info_df["regime_num"].unique():
            # Get block type of current regime
            block_type = block_info_df.loc[
                block_info_df["regime_num"] == regime_num, "block_type"
            ].to_numpy()[0]

            # Get performance of current regime
            regime_data = dataframe.loc[
                dataframe["regime_num"] == regime_num, self.perf_measure
            ]

            if block_type == "train":
                avg_train_perf[regime_num] = np.nanmean(regime_data)
            elif block_type == "test":
                avg_eval_perf[regime_num] = np.nanmean(regime_data)
            else:
                logger.warning(f"Invalid block type: {block_type}")

        metrics_df = fill_metrics_df(avg_train_perf, "avg_train_perf", metrics_df)
        metrics_df = fill_metrics_df(avg_eval_perf, "avg_eval_perf", metrics_df)

        return metrics_df

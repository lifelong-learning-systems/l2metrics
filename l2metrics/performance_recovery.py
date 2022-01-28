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
from scipy import stats

from ._localutil import fill_metrics_df
from .core import Metric

logger = logging.getLogger(__name__)


class PerformanceRecovery(Metric):
    name = "Performance Recovery"
    capability = "adapt_to_new_tasks"
    requires = {"syllabus_type": "agent"}
    description = "Calculates the performance recovery value corresponding to a change of task or parameters"

    def __init__(self, perf_measure: str) -> None:
        super().__init__()
        self.perf_measure = perf_measure

    def validate(self, metrics_df: pd.DataFrame) -> None:
        # Get number of recovery times
        if "recovery_time" in metrics_df.columns:
            r = metrics_df["recovery_time"]
            r = r[r.notna()]
            r_count = r.count()

            if r_count <= 1:
                raise ValueError(
                    "Not enough recovery times to assess performance recovery"
                )
        else:
            raise ValueError("No recovery times")

    def calculate(
        self,
        dataframe: pd.DataFrame,
        block_info: pd.DataFrame,
        metrics_df: pd.DataFrame,
    ) -> pd.DataFrame:
        try:
            # Get the places where we should calculate recovery time
            self.validate(metrics_df)

            # Initialize metric dictionary
            pr_values = {}

            # Calculate performance recovery for each task
            for task in block_info.loc[:, "task_name"].unique():
                r = metrics_df[metrics_df["task_name"] == task]["recovery_time"]
                r = r[r.notna()]

                # Get Theil-Sen slope
                y = np.array(r)

                if len(y) > 1:
                    slope, _, _, _ = stats.theilslopes(y)

                    # Set performance recovery value as negative slope (greater is better)
                    idx = block_info[block_info["task_name"] == task][
                        "regime_num"
                    ].max()
                    pr_values[idx] = -slope
                else:
                    logger.warning(
                        f"Cannot compute {self.name} for task {task} - Not enough recovery times"
                    )

            return fill_metrics_df(pr_values, "perf_recovery", metrics_df)
        except ValueError as e:
            logger.warning(f"Cannot compute {self.name} - {e}")
            return metrics_df

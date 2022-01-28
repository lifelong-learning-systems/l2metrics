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

import pandas as pd

from ._localutil import fill_metrics_df, get_terminal_perf
from .core import Metric


class TerminalPerformance(Metric):
    name = "Terminal Performance"
    capability = "continual_learning"
    requires = {"syllabus_type": "agent"}
    description = "Calculates the terminal performance within each block"

    def __init__(self, perf_measure: str) -> None:
        super().__init__()
        self.perf_measure = perf_measure

    def validate(self, block_info) -> None:
        pass

    def calculate(
        self,
        dataframe: pd.DataFrame,
        block_info: pd.DataFrame,
        metrics_df: pd.DataFrame,
    ) -> pd.DataFrame:
        # Initialize metric dictionaries
        term_perf_values = {}
        exp_to_terminal_perf = {}

        # Iterate over all of the blocks and compute the within block performance
        for idx in range(max(block_info["regime_num"].to_numpy()) + 1):
            # Need to get the part of the data corresponding to the block
            block_data = dataframe.loc[dataframe["regime_num"] == idx]

            # Make within block calculations
            term_perf_values[idx], exp_to_terminal_perf[idx], _ = get_terminal_perf(
                block_data, col_to_use=self.perf_measure
            )

        metrics_df = fill_metrics_df(term_perf_values, "term_perf", metrics_df)
        return fill_metrics_df(exp_to_terminal_perf, "exp_to_term_perf", metrics_df)

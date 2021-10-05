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

import warnings

import numpy as np
import pandas as pd

from ._localutil import fill_metrics_df
from .core import Metric


class STERelativePerf(Metric):
    name = "Performance relative to STE"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the performance of each task relative to it's corresponding single-task expert"

    def __init__(self, perf_measure: str, ste_data: dict, ste_averaging_method: str = 'metrics') -> None:

        super().__init__()
        self.perf_measure = perf_measure
        self.ste_data = ste_data
        if ste_averaging_method not in ['time', 'metrics']:
            raise Exception(f'Invalid STE averaging method: {ste_averaging_method}')
        else:
            self.ste_averaging_method = ste_averaging_method

    def validate(self, block_info: pd.DataFrame) -> None:
        # Check if there is STE data for each task in the scenario
        self.unique_tasks = block_info.loc[:, 'task_name'].unique()
        ste_names = tuple(self.ste_data.keys())

        # Raise exception if none of the tasks have STE data
        if ~np.any(np.isin(self.unique_tasks, ste_names)):
            raise Exception('No STE data available for any task')

        # Make sure STE baselines are available for all tasks, else send warning
        if ~np.all(np.isin(self.unique_tasks, ste_names)):
            warnings.warn('STE data not available for all tasks')

    def calculate(self, dataframe: pd.DataFrame, block_info: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Validate the STE
            self.validate(block_info)

            # Initialize metric dictionaries
            ste_rel_perf = {}

            for task in self.unique_tasks:
                # Get block info for task during training
                task_blocks = block_info[(block_info['task_name'] == task) & (
                    block_info['block_type'] == 'train') & (block_info['block_subtype'] == 'wake')]

                # Get concatenated data for task
                task_data = dataframe[dataframe.regime_num.isin(task_blocks['regime_num'])]

                if len(task_data):
                    # Get STE data
                    ste_data = self.ste_data.get(task)

                    if ste_data:
                        if self.ste_averaging_method == 'time':
                            # Average all the STE data together after truncating to same length
                            x_ste = [ste_data_df[ste_data_df['block_type'] == 'train']
                                     [self.perf_measure].to_numpy() for ste_data_df in ste_data]
                            min_ste_exp = min(map(len, x_ste))
                            x_ste = np.array([x[:min_ste_exp] for x in x_ste]).mean(0)

                            # Compute relative performance
                            min_exp = min(task_data.shape[0], len(x_ste))
                            task_perf = np.nansum(task_data.head(
                                min_exp)[self.perf_measure].to_numpy())
                            ste_perf = np.nansum(x_ste[:min_exp])
                            rel_perf = task_perf / ste_perf
                            ste_rel_perf[task_data['regime_num'].iloc[-1]] = [rel_perf]
                        elif self.ste_averaging_method == 'metrics':
                            rel_perf_vals = []

                            for ste_data_df in ste_data:
                                ste_data_df = ste_data_df[ste_data_df['block_type'] == 'train']

                                # Compute relative performance
                                min_exp = np.min([task_data.shape[0], ste_data_df.shape[0]])
                                task_perf = np.nansum(task_data.head(
                                    min_exp)[self.perf_measure].to_numpy())
                                ste_perf = np.nansum(ste_data_df.head(
                                    min_exp)[self.perf_measure].to_numpy())
                                rel_perf_vals.append(task_perf / ste_perf)

                            ste_rel_perf[task_data['regime_num'].iloc[-1]] = rel_perf_vals
                    else:
                        print(f"Cannot compute {self.name} for task {task} - No STE data available")

            return fill_metrics_df(ste_rel_perf, 'ste_rel_perf', metrics_df)
        except Exception as e:
            print(f"Cannot compute {self.name} - {e}")
            return metrics_df

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

import logging

import numpy as np
import pandas as pd

from ._localutil import fill_metrics_df, get_block_saturation_perf
from .core import Metric

logger = logging.getLogger(__name__)


class SampleEfficiency(Metric):
    name = "Sample Efficiency"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the sample efficiency relative to the single-task expert"

    def __init__(self, perf_measure: str, ste_data: dict, ste_averaging_method: str = 'metrics') -> None:

        super().__init__()
        self.perf_measure = perf_measure
        self.ste_data = ste_data
        if ste_averaging_method not in ['time', 'metrics']:
            raise KeyError(f'Invalid STE averaging method: {ste_averaging_method}')
        else:
            self.ste_averaging_method = ste_averaging_method

    def validate(self, block_info: pd.DataFrame) -> None:
        # Check if there is STE data for each task in the scenario
        self.unique_tasks = block_info.loc[:, 'task_name'].unique()
        ste_names = tuple(self.ste_data.keys())

        # Raise value error if none of the tasks have STE data
        if ~np.any(np.isin(self.unique_tasks, ste_names)):
            raise ValueError('No STE data available for any task')

        # Make sure STE baselines are available for all tasks, else send warning
        if ~np.all(np.isin(self.unique_tasks, ste_names)):
            logger.warning('STE data not available for all tasks')

    def calculate(self, dataframe: pd.DataFrame, block_info: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Validate the STE
            self.validate(block_info)

            # Initialize metric dictionaries
            se_task_saturation = {}
            se_task_exp_to_sat = {}
            se_ste_saturation = {}
            se_ste_exp_to_sat = {}
            se_saturation = {}
            se_exp_to_sat = {}
            sample_efficiency = {}

            for task in self.unique_tasks:
                # Get block info for task during training
                task_blocks = block_info[(block_info['task_name'] == task) & (
                    block_info['block_type'] == 'train') & (block_info['block_subtype'] == 'wake')]

                # Get data concatenated data for task
                task_data = dataframe[dataframe['regime_num'].isin(task_blocks['regime_num'])]

                if len(task_data):
                    # Get STE data
                    ste_data = self.ste_data.get(task)

                    if ste_data:
                        # Get task saturation value and experiences to saturation
                        task_saturation, task_exp_to_sat, _ = get_block_saturation_perf(
                            task_data, col_to_use=self.perf_measure)

                        # Check for valid performance
                        if task_exp_to_sat == 0:
                            logger.warning(
                                f"Cannot compute {self.name} for task {task} - Saturation not achieved")
                            continue
                        
                        # Store task saturation value and experiences to saturation
                        se_task_saturation[task_data['regime_num'].iloc[-1]] = task_saturation
                        se_task_exp_to_sat[task_data['regime_num'].iloc[-1]] = task_exp_to_sat

                        if self.ste_averaging_method == 'time':
                            # Average all the STE data together after truncating to same length
                            x_ste = [ste_data_df[ste_data_df['block_type'] == 'train']
                                     [self.perf_measure].to_numpy() for ste_data_df in ste_data]
                            min_ste_exp = min(map(len, x_ste))
                            x_ste = np.array([x[:min_ste_exp] for x in x_ste]).mean(0)

                            # Get STE saturation value and experiences to saturation
                            ste_saturation, ste_exp_to_sat, _ = get_block_saturation_perf(x_ste)

                            # Check for valid performance
                            if ste_exp_to_sat == 0:
                                logger.warning(
                                    f"Cannot compute {self.name} for task {task} - Saturation not achieved")
                                continue
                            
                            # Store STE saturation value and experiences to saturation
                            se_ste_saturation[task_data['regime_num'].iloc[-1]] = [ste_saturation]
                            se_ste_exp_to_sat[task_data['regime_num'].iloc[-1]] = [ste_exp_to_sat]

                            # Compute sample efficiency
                            se_saturation[task_data['regime_num'].iloc[-1]] = [task_saturation / ste_saturation]
                            se_exp_to_sat[task_data['regime_num'].iloc[-1]] = [ste_exp_to_sat / task_exp_to_sat]
                            sample_efficiency[task_data['regime_num'].iloc[-1]] = \
                                [(task_saturation / ste_saturation) * (ste_exp_to_sat / task_exp_to_sat)]
                        elif self.ste_averaging_method == 'metrics':
                            se_ste_saturation_vals = []
                            se_ste_exp_to_sat_vals = []
                            se_saturation_vals = []
                            se_exp_to_sat_vals = []
                            sample_efficiency_vals = []
                            
                            for ste_data_df in ste_data:
                                ste_data_df = ste_data_df[ste_data_df['block_type'] == 'train']

                                # Get STE saturation value and experiences to saturation
                                ste_saturation, ste_exp_to_sat, _ = get_block_saturation_perf(
                                    ste_data_df, col_to_use=self.perf_measure)

                                # Check for valid performance
                                if ste_exp_to_sat == 0:
                                    logger.warning(
                                        f"Cannot compute {self.name} for task {task} - Saturation not achieved")
                                    continue

                                # Compute sample efficiency
                                se_ste_saturation_vals.append(ste_saturation)
                                se_ste_exp_to_sat_vals.append(ste_exp_to_sat)
                                se_saturation_vals.append(task_saturation / ste_saturation)
                                se_exp_to_sat_vals.append(ste_exp_to_sat / task_exp_to_sat)
                                sample_efficiency_vals.append(
                                    (task_saturation / ste_saturation) * (ste_exp_to_sat / task_exp_to_sat))

                            se_ste_saturation[task_data['regime_num'].iloc[-1]] = se_ste_saturation_vals
                            se_ste_exp_to_sat[task_data['regime_num'].iloc[-1]] = se_ste_exp_to_sat_vals
                            se_saturation[task_data['regime_num'].iloc[-1]] = se_saturation_vals
                            se_exp_to_sat[task_data['regime_num'].iloc[-1]] = se_exp_to_sat_vals
                            sample_efficiency[task_data['regime_num'].iloc[-1]] = sample_efficiency_vals
                    else:
                        logger.warning(
                            f"Cannot compute {self.name} for task {task} - No STE data available")

            metrics_df = fill_metrics_df(se_task_saturation, 'se_task_saturation', metrics_df)
            metrics_df = fill_metrics_df(se_task_exp_to_sat, 'se_task_exp_to_sat', metrics_df)
            metrics_df = fill_metrics_df(se_ste_saturation, 'se_ste_saturation', metrics_df)
            metrics_df = fill_metrics_df(se_ste_exp_to_sat, 'se_ste_exp_to_sat', metrics_df)
            metrics_df = fill_metrics_df(se_saturation, 'se_saturation', metrics_df)
            metrics_df = fill_metrics_df(se_exp_to_sat, 'se_exp_to_sat', metrics_df)
            return fill_metrics_df(sample_efficiency, 'sample_efficiency', metrics_df)
        except ValueError as e:
            logger.warning(f"Cannot compute {self.name} - {e}")
            return metrics_df

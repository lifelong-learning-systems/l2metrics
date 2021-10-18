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


class PerformanceMaintenance(Metric):
    name = "Performance Maintenance"
    capability = "adapting_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the average difference between the most recent" \
        "terminal learning performance of a task and each evaluation performance"

    def __init__(self, perf_measure: str, method: str = 'mrlep') -> None:
        super().__init__()
        self.perf_measure = perf_measure

        # Check for valid method
        if method not in ['mrtlp', 'mrlep', 'both']:
            raise Exception(f'Invalid performance maintenance method: {method}')
        else:
            self.do_mrtlp = method in ['mrtlp', 'both']
            self.do_mrlep = method in ['mrlep', 'both']

    def validate(self, block_info: pd.DataFrame) -> None:
        # Initialize variables for checking block type format
        last_block_num = -1
        last_block_type = ''

        # Ensure alternating block types
        for _, regime in block_info.iterrows():
            if regime['block_num'] != last_block_num:
                last_block_num = regime['block_num']

                if regime['block_type'] == 'test':
                    if last_block_type == 'test':
                        warnings.warn('Block types are not alternating')
                        break
                    last_block_type = 'test'
                elif regime['block_type'] == 'train':
                    if last_block_type == 'train':
                        warnings.warn('Block types are not alternating')
                        break
                    last_block_type = 'train'

    def calculate(self, dataframe: pd.DataFrame, block_info: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Validate block structure
            self.validate(block_info)

            # Use sleep evaluation blocks if they exist
            if 'sleep' in block_info['block_subtype'].to_numpy():
                block_info_df = block_info[~(block_info.block_type.isin(['test']) & block_info.block_subtype.isin(['wake']))]
            else:
                block_info_df = block_info.copy()

            # Initialize metric columns
            maintenance_values_mrtlp = {}
            maintenance_values_mrlep = {}
            pm_values_mrtlp = {}
            pm_values_mrlep = {}

            # Get unique tasks in scenario
            unique_tasks = block_info_df.task_name.unique()

            # Iterate over tasks
            for task in unique_tasks:
                # Get training and test regimes
                training_regs = block_info_df[(block_info_df['task_name'] == task) &
                                              (block_info_df['block_type'] == 'train')]['regime_num'].to_numpy()

                test_regs = block_info_df[(block_info_df['task_name'] == task) &
                                          (block_info_df['block_type'] == 'test')]['regime_num'].to_numpy()

                # Get reference test regimes
                ref_test_regs = np.array([test_regs[np.argmax(
                    test_regs > x)] for x in training_regs if np.any(test_regs > x)])

                # Iterate over test regimes
                for test_regime in test_regs:
                    # Check that current test block occurred after last reference test
                    if np.any(test_regime > ref_test_regs) and test_regime not in ref_test_regs:
                        # Get performance of current test regime
                        test_perf = metrics_df['term_perf'][test_regime]

                        if self.do_mrtlp:
                            ref_regime = training_regs[test_regime > training_regs][-1]
                            mrtlp = metrics_df['term_perf'][ref_regime]
                            maintenance_values_mrtlp[test_regime] = test_perf - mrtlp
                        if self.do_mrlep:
                            ref_regime = ref_test_regs[test_regime > ref_test_regs][-1]
                            mrlep = metrics_df['term_perf'][ref_regime]
                            maintenance_values_mrlep[test_regime] = test_perf - mrlep

            if self.do_mrtlp and maintenance_values_mrtlp:
                # Fill metrics dataframe with most recent terminal learning performance differences
                metrics_df = fill_metrics_df(maintenance_values_mrtlp, 'maintenance_val_mrtlp', metrics_df)

                # Iterate over task performance differences for performance maintenance
                for task in unique_tasks:
                    # Get the task maintenance values
                    m = metrics_df[metrics_df['task_name'] == task]['maintenance_val_mrtlp'].to_numpy(dtype=float)

                    # Remove NaNs
                    m = m[~np.isnan(m)]

                    # Calculate performance maintenance value
                    if m.size:
                        pm_values_mrtlp[block_info_df.index[block_info_df['task_name'] == task][-1]] = np.mean(m)

                metrics_df = fill_metrics_df(pm_values_mrtlp, 'perf_maintenance_mrtlp', metrics_df)
            
            if self.do_mrlep and maintenance_values_mrlep:
                # Fill metrics dataframe with most recent terminal learning performance differences
                metrics_df = fill_metrics_df(maintenance_values_mrlep, 'maintenance_val_mrlep', metrics_df)

                # Iterate over task performance differences for performance maintenance
                for task in unique_tasks:
                    # Get the task maintenance values
                    m = metrics_df[metrics_df['task_name'] == task]['maintenance_val_mrlep'].to_numpy(dtype=float)

                    # Remove NaNs
                    m = m[~np.isnan(m)]

                    # Calculate performance maintenance value
                    if m.size:
                        pm_values_mrlep[block_info_df.index[block_info_df['task_name'] == task][-1]] = np.mean(m)

                metrics_df = fill_metrics_df(pm_values_mrlep, 'perf_maintenance_mrlep', metrics_df)

            return metrics_df
        except Exception as e:
            print(f"Cannot compute {self.name} - {e}")
            return metrics_df

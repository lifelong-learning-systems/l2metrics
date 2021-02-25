# (c) 2019 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).
# All Rights Reserved. This material may be only be used, modified, or reproduced
# by or for the U.S. Government pursuant to the license rights granted under the
# clauses at DFARS 252.227-7013/7014 or FAR 52.227-14. For any other permission,
# please contact the Office of Technology Transfer at JHU/APL.

# NO WARRANTY, NO LIABILITY. THIS MATERIAL IS PROVIDED “AS IS.” JHU/APL MAKES NO
# REPRESENTATION OR WARRANTY WITH RESPECT TO THE PERFORMANCE OF THE MATERIALS,
# INCLUDING THEIR SAFETY, EFFECTIVENESS, OR COMMERCIAL VIABILITY, AND DISCLAIMS
# ALL WARRANTIES IN THE MATERIAL, WHETHER EXPRESS OR IMPLIED, INCLUDING (BUT NOT
# LIMITED TO) ANY AND ALL IMPLIED WARRANTIES OF PERFORMANCE, MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF INTELLECTUAL PROPERTY
# OR OTHER THIRD PARTY RIGHTS. ANY USER OF THE MATERIAL ASSUMES THE ENTIRE RISK
# AND LIABILITY FOR USING THE MATERIAL. IN NO EVENT SHALL JHU/APL BE LIABLE TO ANY
# USER OF THE MATERIAL FOR ANY ACTUAL, INDIRECT, CONSEQUENTIAL, SPECIAL OR OTHER
# DAMAGES ARISING FROM THE USE OF, OR INABILITY TO USE, THE MATERIAL, INCLUDING,
# BUT NOT LIMITED TO, ANY DAMAGES FOR LOST PROFITS.

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

    def __init__(self, perf_measure: str) -> None:
        super().__init__()
        self.perf_measure = perf_measure

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

            # Initialize metric columns
            maintenance_values = {}
            pm_values = {}

            # Iterate over the regimes
            for _, regime in block_info.iterrows():

                # Check for evaluation or test block
                if regime['block_type'] == 'test':

                    # Get the most recent terminal learning performance of the current task
                    training_tasks = block_info[(block_info['task_name'] == regime['task_name']) &
                                                (block_info['block_type'] == 'train') &
                                                (block_info['block_num'] < regime['block_num'])]

                    # Check to make sure the task has been trained on
                    if len(training_tasks) > 0:

                        # Check that current train block occurred after last training, but not
                        # immediately after
                        if training_tasks.iloc[-1]['block_num'] < regime['block_num'] - 1:
                            mrtp = metrics_df['term_perf'][training_tasks.iloc[-1]['regime_num']]
                            test_perf = metrics_df['term_perf'][regime['regime_num']]
                            maintenance_values[regime['regime_num']] = test_perf - mrtp

            if len(maintenance_values):
                # Fill metrics dataframe with performance differences
                metrics_df = fill_metrics_df(maintenance_values, 'maintenance_val', metrics_df)

                # Iterate over task performance differences for performance maintenance
                for task in block_info.loc[:, 'task_name'].unique():

                    # Get the task maintenance values
                    m = metrics_df[metrics_df['task_name'] == task]['maintenance_val'].values

                    # Remove NaNs
                    m = m[~np.isnan(m)]

                    # Calculate performance maintenance value
                    if m.size:
                        pm_values[block_info.index[block_info['task_name'] == task][-1]] = np.mean(m)

                return fill_metrics_df(pm_values, 'perf_maintenance', metrics_df)
            else:
                return metrics_df
        except Exception as e:
            print(f"Cannot compute {self.name} - {e}")
            return metrics_df

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


class BackwardTransfer(Metric):
    name = "Backward Transfer"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the backward transfer for valid task pairs"

    def __init__(self, perf_measure: str = 'reward', transfer_method: str = 'contrast') -> None:
        super().__init__()
        self.perf_measure = perf_measure
        self.transfer_method = transfer_method

    def validate(self, block_info: pd.DataFrame) -> dict:
        # Check for valid transfer method
        if self.transfer_method not in ['contrast', 'ratio', 'both']:
            raise Exception(f'Invalid transfer method: {self.transfer_method}')

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

        # Find eligible tasks for backward transfer
        unique_tasks = block_info.loc[:, 'task_name'].unique()

        # Initialize list of tasks for transfer matrix
        tasks_for_bt = {}

        # Iterate over all regimes in scenario
        for _, regime in block_info.iterrows():
            # Get regime info
            block_type = regime['block_type']
            task_name = regime['task_name']
            regime_num = regime['regime_num']

            # Check for valid backward transfer pair
            if block_type == 'train':
                # Compare with other tasks
                other_tasks = np.delete(unique_tasks, np.where(unique_tasks == task_name))

                for other_task in other_tasks:
                    other_blocks = block_info[block_info['task_name'] == other_task]
                    other_test_regs = other_blocks[other_blocks['block_type'] == 'test']['regime_num'].values
                    other_train_regs = other_blocks[other_blocks['block_type'] == 'train']['regime_num'].values

                    # BT - Other task must have been trained and tested before current regime, then
                    # tested again after
                    if np.any(other_train_regs < regime_num) and len(other_test_regs) >= 2:
                        for idx in range(len(other_test_regs) - 1):
                            if other_test_regs[idx] < regime['regime_num'] < other_test_regs[idx + 1]:
                                key = (task_name, other_task)
                                if key not in tasks_for_bt.keys():
                                    tasks_for_bt[key] = [(other_test_regs[idx], other_test_regs[idx + 1])]
                                else:
                                    tasks_for_bt[key].append((other_test_regs[idx], other_test_regs[idx + 1]))

        if not tasks_for_bt:
            raise Exception('No valid task pairs for backward transfer')

        return tasks_for_bt

    def calculate(self, dataframe: pd.DataFrame, block_info: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Validate data and get pairs eligible for backward transfer
            tasks_for_bt = self.validate(block_info)

            # Initialize flags for transfer methods
            do_contrast = False
            do_ratio = False

            if self.transfer_method in ['contrast', 'both']:
                do_contrast = True
            if self.transfer_method in ['ratio', 'both']:
                do_ratio = True

            # Initialize metric dictionaries
            backward_transfer_contrast = {}
            backward_transfer_ratio = {}

            # Calculate backward transfer for valid task pairs
            for task_pair, regime_pairs in tasks_for_bt.items():
                for regime_pair in regime_pairs:
                    tp_1 = metrics_df[(metrics_df['regime_num'] == regime_pair[0])]['term_perf'].values[0]
                    tp_2 = metrics_df[(metrics_df['regime_num'] == regime_pair[1])]['term_perf'].values[0]
                    idx = regime_pair[1]

                    if do_contrast:
                        backward_transfer_contrast[idx] = [{task_pair[0]: (tp_2 - tp_1) / (tp_1 + tp_2)}]
                    if do_ratio:
                        backward_transfer_ratio[idx] = [{task_pair[0]: tp_2 / tp_1}]

            if do_contrast:
                metrics_df = fill_metrics_df(
                    backward_transfer_contrast, 'backward_transfer_contrast', metrics_df)
            if do_ratio:
                metrics_df = fill_metrics_df(
                    backward_transfer_ratio, 'backward_transfer_ratio', metrics_df)
            return metrics_df
        except Exception as e:
            print(f"Cannot compute {self.name} - {e}")
            return metrics_df

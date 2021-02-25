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
from collections import defaultdict
from itertools import permutations

import numpy as np
import pandas as pd

from ._localutil import fill_metrics_df
from .core import Metric


class ForwardTransfer(Metric):
    name = "Forward Transfer"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the forward transfer for valid task pairs"

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

        # Find eligible tasks for forward transfer
        unique_tasks = block_info.loc[:, 'task_name'].unique()

        # Initialize list of tasks for transfer matrix
        tasks_for_ft = defaultdict(dict)

        # Determine valid transfer pairs
        for task_pair in permutations(unique_tasks, 2):
            # Get testing and training indices for task pair
            training_blocks = block_info[(block_info['task_name'] == task_pair[0]) & (
                block_info['block_type'] == 'train')]['block_num'].values

            other_blocks = block_info[block_info['task_name'] == task_pair[1]]
            other_test_blocks = other_blocks[other_blocks['block_type'] == 'test']['block_num'].values
            other_training_blocks = other_blocks[other_blocks['block_type'] == 'train']['block_num'].values

            # FT - Must have tested task 2 before task 2 then tested task 2 again
            if len(training_blocks):
                # Get valid training blocks for first task
                valid_training_blocks = training_blocks[training_blocks < np.min(
                    other_training_blocks)] if len(other_training_blocks) else training_blocks

                if len(valid_training_blocks) and len(other_test_blocks) >= 2:
                    for idx in range(len(other_test_blocks) - 1):
                        for t_idx in valid_training_blocks:
                            if other_test_blocks[idx] < t_idx < other_test_blocks[idx + 1]:
                                tasks_for_ft[task_pair[0]][task_pair[1]] = (
                                    other_test_blocks[idx], other_test_blocks[idx + 1])
                                break   # Only compute one value per task pair
                        else:
                            continue
                        break

        if np.sum([len(value) for _, value in tasks_for_ft.items()]) == 0:
            raise Exception('No valid task pairs for forward transfer')

        return tasks_for_ft

    def calculate(self, dataframe: pd.DataFrame, block_info: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Validate data and get pairs eligible for forward transfer
            tasks_for_ft = self.validate(block_info)

            # Initialize flags for transfer methods
            do_contrast = False
            do_ratio = False

            if self.transfer_method in ['contrast', 'both']:
                do_contrast = True
            if self.transfer_method in ['ratio', 'both']:
                do_ratio = True

            # Initialize metric dictionaries
            forward_transfer_contrast = {}
            forward_transfer_ratio = {}

            # Calculate forward transfer for valid task pairs
            for task, value in tasks_for_ft.items():
                for trans_task, trans_blocks in value.items():
                    tp_1 = metrics_df[(metrics_df['task_name'] == trans_task) & (
                        metrics_df['block_num'] == trans_blocks[0])]['term_perf'].values[0]
                    tp_2 = metrics_df[(metrics_df['task_name'] == trans_task) & (
                        metrics_df['block_num'] == trans_blocks[1])]['term_perf'].values[0]
                    idx = block_info[(block_info['task_name'] == trans_task) & (
                        block_info['block_num'] == trans_blocks[1])]['regime_num'].values[0]

                    if do_contrast:
                        forward_transfer_contrast[idx] = [{task: (tp_2 - tp_1) / (tp_1 + tp_2)}]
                    if do_ratio:
                        forward_transfer_ratio[idx] = [{task: tp_2 / tp_1}]

            if do_contrast:
                metrics_df = fill_metrics_df(
                    forward_transfer_contrast, 'forward_transfer_contrast', metrics_df)
            if do_ratio:
                metrics_df = fill_metrics_df(
                    forward_transfer_ratio, 'forward_transfer_ratio', metrics_df)
            return metrics_df
        except Exception as e:
            print(f"Cannot compute {self.name} - {e}")
            return metrics_df

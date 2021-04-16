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
from itertools import permutations

import numpy as np
import pandas as pd

from ._localutil import fill_metrics_df
from .core import Metric


class Transfer(Metric):
    name = "Transfer"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the forward and backward transfer for valid task pairs"

    def __init__(self, perf_measure: str, method: str = 'contrast') -> None:
        super().__init__()
        self.perf_measure = perf_measure

        # Check for valid transfer method
        if method not in ['contrast', 'ratio', 'both']:
            raise Exception(f'Invalid transfer method: {method}')
        else:
            self.do_contrast = method in ['contrast', 'both']
            self.do_ratio = method in ['ratio', 'both']

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
            # Validate data
            self.validate(block_info)

            # Initialize metric dictionaries
            forward_transfer = {'ratio': {}, 'contrast': {}}
            backward_transfer = {'ratio': {}, 'contrast': {}}

            # Find eligible tasks for forward transfer
            unique_tasks = block_info.loc[:, 'task_name'].unique()

            # Determine valid transfer pairs
            for task, other_task in permutations(unique_tasks, 2):
                # Get testing and training indices for task pair
                training_regs = block_info[(block_info['task_name'] == task) & (
                    block_info['block_type'] == 'train')]['regime_num'].to_numpy()

                other_blocks = block_info[block_info['task_name'] == other_task]
                other_test_regs = other_blocks[other_blocks['block_type'] == 'test']['regime_num'].to_numpy()
                other_training_regs = other_blocks[other_blocks['block_type'] == 'train']['regime_num'].to_numpy()

                # FT - Must have tested task 2 before training task 1 then tested task 2 again
                if len(training_regs):
                    # Get valid training regimes for first task forward transfer
                    valid_ft_training_regs = training_regs[training_regs < np.min(
                        other_training_regs)] if len(other_training_regs) else training_regs

                    # Get valid training regimes for first task backward transfer
                    valid_bt_training_regs = training_regs[training_regs > np.min(
                        other_training_regs)] if len(other_training_regs) else []

                    # Calculate forward transfer
                    for training_regime in valid_ft_training_regs:
                        for test_regime_1, test_regime_2 in zip(other_test_regs, other_test_regs[1:]):
                            if test_regime_1 < training_regime < test_regime_2:
                                tp_1 = metrics_df[metrics_df['regime_num'] == test_regime_1]['term_perf'].to_numpy()[0]
                                tp_2 = metrics_df[metrics_df['regime_num'] == test_regime_2]['term_perf'].to_numpy()[0]

                                if self.do_contrast:
                                    forward_transfer['contrast'][test_regime_2] = [{task: (tp_2 - tp_1) / (tp_1 + tp_2)}]
                                if self.do_ratio:
                                    if tp_1 == 0:
                                        continue
                                    else:
                                        forward_transfer['ratio'][test_regime_2] = [{task: tp_2 / tp_1}]

                    # Calculate backward transfer
                    for training_regime in valid_bt_training_regs:
                        for test_regime_1, test_regime_2 in zip(other_test_regs, other_test_regs[1:]):
                            if test_regime_1 < training_regime < test_regime_2:                                    
                                tp_1 = metrics_df[(metrics_df['regime_num'] == test_regime_1)]['term_perf'].to_numpy()[0]
                                tp_2 = metrics_df[(metrics_df['regime_num'] == test_regime_2)]['term_perf'].to_numpy()[0]

                                if self.do_contrast:
                                    backward_transfer['contrast'][test_regime_2] = [{task: (tp_2 - tp_1) / (tp_1 + tp_2)}]
                                if self.do_ratio:
                                    if tp_1 == 0:
                                        continue
                                    else:
                                        backward_transfer['ratio'][test_regime_2] = [{task: tp_2 / tp_1}]

            if not (forward_transfer['contrast'] or forward_transfer['ratio']):
                print('No valid task pairs for forward transfer')
            else:
                if self.do_contrast:
                    metrics_df = fill_metrics_df(
                        forward_transfer['contrast'], 'forward_transfer_contrast', metrics_df)
                if self.do_ratio:
                    metrics_df = fill_metrics_df(
                        forward_transfer['ratio'], 'forward_transfer_ratio', metrics_df)

            if not (backward_transfer['contrast'] or backward_transfer['ratio']):
                print('No valid task pairs for backward transfer')
            else:
                if self.do_contrast:
                    metrics_df = fill_metrics_df(
                        backward_transfer['contrast'], 'backward_transfer_contrast', metrics_df)
                if self.do_ratio:
                    metrics_df = fill_metrics_df(
                        backward_transfer['ratio'], 'backward_transfer_ratio', metrics_df)

            return metrics_df
        except Exception as e:
            print(f"Cannot compute {self.name} - {e}")
            return metrics_df

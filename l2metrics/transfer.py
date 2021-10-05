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
        if method not in ['ratio', 'contrast', 'both']:
            raise Exception(f'Invalid transfer method: {method}')
        else:
            self.do_ratio = method in ['ratio', 'both']
            self.do_contrast = method in ['contrast', 'both']

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

            # Use sleep evaluation blocks if they exist
            if 'sleep' in block_info['block_subtype'].to_numpy():
                block_info_df = block_info[~(block_info.block_type.isin(['test']) & block_info.block_subtype.isin(['wake']))]
            else:
                block_info_df = block_info.copy()

            # Initialize metric dictionaries
            forward_transfer = {'ratio': {}, 'contrast': {}}
            backward_transfer = {'ratio': {}, 'contrast': {}}

            # Find eligible tasks for forward transfer
            unique_tasks = block_info_df.task_name.unique()

            # Determine valid transfer pairs
            for task, other_task in permutations(unique_tasks, 2):
                # Get testing and training indices for task pair
                training_regs = block_info_df[(block_info_df['task_name'] == task) & (
                    block_info_df['block_type'] == 'train')]['regime_num'].to_numpy()

                other_blocks = block_info_df[block_info_df['task_name'] == other_task]
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

                                if self.do_ratio:
                                    if tp_1 != 0:
                                        forward_transfer['ratio'][test_regime_2] = {task: tp_2 / tp_1}
                                if self.do_contrast:
                                    if (tp_1 + tp_2) != 0:
                                        forward_transfer['contrast'][test_regime_2] = {task: (tp_2 - tp_1) / (tp_1 + tp_2)}

                    # Calculate backward transfer
                    for training_regime in valid_bt_training_regs:
                        for test_regime_1, test_regime_2 in zip(other_test_regs, other_test_regs[1:]):
                            if test_regime_1 < training_regime < test_regime_2:                                    
                                tp_1 = metrics_df[(metrics_df['regime_num'] == test_regime_1)]['term_perf'].to_numpy()[0]
                                tp_2 = metrics_df[(metrics_df['regime_num'] == test_regime_2)]['term_perf'].to_numpy()[0]

                                if self.do_ratio:
                                    if tp_1 != 0:
                                        backward_transfer['ratio'][test_regime_2] = {task: tp_2 / tp_1}
                                if self.do_contrast:
                                    if (tp_1 + tp_2) != 0:
                                        backward_transfer['contrast'][test_regime_2] = {task: (tp_2 - tp_1) / (tp_1 + tp_2)}

            if not (forward_transfer['ratio'] or forward_transfer['contrast']):
                print('No valid task pairs for forward transfer')
            else:
                if self.do_ratio:
                    metrics_df = fill_metrics_df(
                        forward_transfer['ratio'], 'forward_transfer_ratio', metrics_df)
                if self.do_contrast:
                    metrics_df = fill_metrics_df(
                        forward_transfer['contrast'], 'forward_transfer_contrast', metrics_df)

            if not (backward_transfer['ratio'] or backward_transfer['contrast']):
                print('No valid task pairs for backward transfer')
            else:
                if self.do_ratio:
                    metrics_df = fill_metrics_df(
                        backward_transfer['ratio'], 'backward_transfer_ratio', metrics_df)
                if self.do_contrast:
                    metrics_df = fill_metrics_df(
                        backward_transfer['contrast'], 'backward_transfer_contrast', metrics_df)

            return metrics_df
        except Exception as e:
            print(f"Cannot compute {self.name} - {e}")
            return metrics_df

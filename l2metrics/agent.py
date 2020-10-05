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

import os
from abc import ABC
from itertools import permutations

import numpy as np
from scipy import stats
from tabulate import tabulate

from . import _localutil, core, util


"""
Standard metrics for Agent Learning (RL tasks)
"""

class AgentMetric(core.Metric, ABC):
    """
    A single metric for an Agent (aka. Reinforcement Learning) learner
    """

    max_window_size = 100

    def __init__(self):
        pass
        # self.validate()

    def plot(self, result):
        pass

    def validate(self, block_info):
        # TODO: Add structure validation of block_info
        pass


class WithinBlockSaturation(AgentMetric):
    name = "Average Within Block Saturation Calculation"
    capability = "continual_learning"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the max performance within each block"

    def __init__(self, perf_measure):
        super().__init__()
        self.perf_measure = perf_measure

    def validate(self, block_info):
        pass

    def calculate(self, dataframe, block_info, metrics_df):
        metrics_df['saturation'] = np.full_like(metrics_df['regime_num'], np.nan, dtype=np.double)
        metrics_df['eps_to_sat'] = np.full_like(metrics_df['regime_num'], np.nan, dtype=np.double)
        saturation_values = {}
        eps_to_saturation = {}

        # Iterate over all of the blocks and compute the within block performance
        for idx in range(block_info.loc[:, 'regime_num'].max() + 1):
            # Need to get the part of the data corresponding to the block
            block_data = dataframe.loc[dataframe['regime_num'] == idx]

            # Get block window size for smoothing
            window = int(len(block_data) * 0.2)
            custom_window = min(window, self.max_window_size)

            # Make within block calculations
            sat_value, eps_to_sat, _ = _localutil.get_block_saturation_perf(
                block_data, col_to_use=self.perf_measure, window_len=custom_window)

            # Record them
            saturation_values[idx] = sat_value
            eps_to_saturation[idx] = eps_to_sat

        metrics_df = _localutil.fill_metrics_df(saturation_values, 'saturation', metrics_df)

        return _localutil.fill_metrics_df(eps_to_saturation, 'eps_to_sat', metrics_df)


class MostRecentTerminalPerformance(AgentMetric):
    name = "Most Recent Terminal Performance"
    capability = "continual_learning"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the terminal performance within each block"

    def __init__(self, perf_measure):
        super().__init__()
        self.perf_measure = perf_measure

    def validate(self, block_info):
        pass

    def calculate(self, dataframe, block_info, metrics_df):
        metrics_df['term_perf'] = np.full_like(metrics_df['regime_num'], np.nan, dtype=np.double)
        metrics_df['eps_to_term_perf'] = np.full_like(metrics_df['regime_num'], np.nan, dtype=np.double)
        terminal_perf_values = {}
        eps_to_terminal_perf = {}

        # Iterate over all of the blocks and compute the within block performance
        for idx in range(block_info.loc[:, 'regime_num'].max() + 1):
            # Need to get the part of the data corresponding to the block
            block_data = dataframe.loc[dataframe['regime_num'] == idx]

            # Get block window size for smoothing
            window = int(len(block_data) * 0.2)
            custom_window = min(window, self.max_window_size)

            # Make within block calculations
            term_perf, eps_to_term_perf, _ = _localutil.get_terminal_perf(
                block_data, col_to_use=self.perf_measure, window_len=custom_window)

            # Record them
            terminal_perf_values[idx] = term_perf
            eps_to_terminal_perf[idx] = eps_to_term_perf

        metrics_df = _localutil.fill_metrics_df(terminal_perf_values, 'term_perf', metrics_df)
        return _localutil.fill_metrics_df(eps_to_terminal_perf, 'eps_to_term_perf', metrics_df)


class RecoveryTime(AgentMetric):
    name = "Recovery Time"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates whether the system recovers after a change of task or parameters and calculate how long it takes if recovery is achieved"

    def __init__(self, perf_measure):
        super().__init__()
        self.perf_measure = perf_measure

    def validate(self, block_info):
        # Determine where we need to assess recovery time
        tr_bl_inds_to_use = []
        tr_bl_inds_to_assess = []

        # Get train blocks, in order of appearance
        tr_bl_info = block_info.sort_index().loc[block_info['block_type'] == 'train', ['regime_num', 'task_name',
                                                                                       'param_set']]
        tb_inds = tr_bl_info.index

        # Blocks are defined as new combinations of task + params, but can repeat, so check for changes across blocks
        first = True
        for idx, block_idx in enumerate(tb_inds):
            if first:
                first = False
                continue
            # Either the task name or the param set must be different
            if tr_bl_info.loc[block_idx, 'task_name'] != tr_bl_info.loc[tb_inds[idx - 1], 'task_name'] or \
                    tr_bl_info.loc[block_idx, 'param_set'] != tr_bl_info.loc[tb_inds[idx - 1], 'param_set']:
                tr_bl_inds_to_assess.append(block_idx)
                tr_bl_inds_to_use.append(tb_inds[idx - 1])

        if tr_bl_inds_to_assess is None or tr_bl_inds_to_use is None:
            raise Exception('No changes across training blocks to assess recovery time')

        return tr_bl_inds_to_use, tr_bl_inds_to_assess

    def calculate(self, dataframe, block_info, metrics_df):
        # Get the places where we should calculate recovery time
        try:
            tr_inds_to_use, tr_inds_to_assess = self.validate(block_info)

            metrics_df['recovery_time'] = np.full_like(metrics_df['regime_num'], np.nan, dtype=np.double)
            recovery_time = {}

            for use_ind, assess_ind in zip(tr_inds_to_use, tr_inds_to_assess):
                prev_val = metrics_df['term_perf'][use_ind]
                block_data = dataframe.loc[assess_ind]

                # Get block window size for smoothing
                window = int(len(block_data) * 0.2)
                custom_window = min(window, self.max_window_size)

                _, _, eps_to_rec = _localutil.get_terminal_perf(block_data,
                                                                col_to_use=self.perf_measure,
                                                                prev_val=prev_val,
                                                                window_len=custom_window)
                recovery_time[assess_ind] = eps_to_rec

            return _localutil.fill_metrics_df(recovery_time, 'recovery_time', metrics_df)
        except Exception as e:
            print(f"Cannot compute {self.name} - {e}")
            return metrics_df


class PerformanceRecovery(AgentMetric):
    name = "Performance Recovery"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the performance recovery value corresponding to a change of task or parameters"

    def __init__(self, perf_measure):
        super().__init__()
        self.perf_measure = perf_measure

    def validate(self, metrics_df):
        # Get number of recovery times
        r = metrics_df['recovery_time']
        r = r[r.notna()]
        r_count = r.count()

        if r_count <= 1:
            raise Exception('Not enough recovery times to assess performance recovery')
        elif r_count != metrics_df["block_type"].value_counts()["train"] - 1:
            raise Exception('There are blocks where the system did not recover')

        return

    def calculate(self, dataframe, block_info, metrics_df):
        # Get the places where we should calculate recovery time
        try:
            self.validate(metrics_df)

            # Get recovery times
            r = metrics_df['recovery_time']
            r = r[r.notna()]

            # Get linear regression
            x = np.array(range(len(r)))
            y = np.array(r)
            slope, _, _, _, _ = stats.linregress(x, y)

            pr_values = {}

            for idx in range(block_info.loc[:, 'regime_num'].max()):
                pr_values[idx] = np.nan

            pr_values[idx + 1] = slope

            return _localutil.fill_metrics_df(pr_values, 'perf_recovery', metrics_df)
        except Exception as e:
            print(f"Cannot compute {self.name} - {e}")
            return metrics_df


class PerformanceMaintenance(AgentMetric):
    name = "Peformance Maintenance"
    capability = "adapting_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the average difference between the most recent" \
        "terminal learning performance of a task and each evaluation performance"

    def __init__(self, perf_measure):
        super().__init__()
        self.perf_measure = perf_measure

    def validate(self, block_info):
        # TODO: Add structure validation of block_info
        # Must ensure that the training block has only one regime or else handle multiple
        pass

    def calculate(self, dataframe, block_info, metrics_df):
        try:
            # Initialize metric columns
            maintenance_values = {}
            pm_values = {}

            # Iterate over the regimes
            for _, regime in block_info.iterrows():

                # Check for evaluation or test block
                if regime['block_type'] == 'test':

                    # Get the most recent terminal learning performance of the current task
                    training_tasks = block_info[(block_info['task_name'] == regime['task_name']) &
                                                (block_info['block_type'] == 'train')]

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
                metrics_df = _localutil.fill_metrics_df(maintenance_values, 'maintenance_val', metrics_df)

                # Iterate over task performance differences for performance maintenance
                for task in block_info.loc[:, 'task_name'].unique():

                    # Get the task maintence values
                    m = metrics_df[metrics_df['task_name'] == _localutil.get_simple_rl_task_names(
                        [task])[0]]['maintenance_val'].values

                    # Remove NaNs
                    m = m[~np.isnan(m)]

                    # Calculate performance maintenance value
                    if m.size:
                        pm_values[block_info.index[block_info['task_name'] == task][-1]] = np.mean(m)

                return _localutil.fill_metrics_df(pm_values, 'perf_maintenance', metrics_df)
            else:
                return metrics_df
        except:
            print("Cannot compute", self.name)
            return metrics_df


class ForwardTransfer(AgentMetric):
    name = "Forward Transfer"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the forward transfer for valid task pairs"

    def __init__(self, perf_measure):
        super().__init__()
        self.perf_measure = perf_measure

    def validate(self, block_info):

        # Initialize variables for checking block type format
        last_block_num = -1
        last_block_type = ''

        # Ensure alternating block types
        for _, regime in block_info.iterrows():
            if regime['block_num'] != last_block_num:
                last_block_num = regime['block_num']

                if regime['block_type'] == 'test':
                    if last_block_type == 'test':
                        raise Exception('Block types must be alternating')
                    last_block_type = 'test'
                elif regime['block_type'] == 'train':
                    if last_block_type == 'train':
                        raise Exception('Block types must be alternating')
                    last_block_type = 'train'

        # Find eligible tasks for forward and backward transfer
        unique_tasks = block_info.loc[:, 'task_name'].unique()

        # Initialize list of tasks for transfer matrix
        tasks_for_ft = dict.fromkeys(unique_tasks, {})

        # Determine valid transfer pairs
        for task_pair in permutations(unique_tasks):
            # Get testing and training indices for task pair
            task_blocks = block_info[block_info['task_name'] == task_pair[0]]
            training_blocks = task_blocks[task_blocks['block_type'] == 'train']['block_num'].values

            other_blocks = block_info[block_info['task_name'] == task_pair[1]]
            other_test_blocks = other_blocks[other_blocks['block_type'] == 'test']['block_num'].values
            other_training_blocks = other_blocks[other_blocks['block_type'] == 'train']['block_num'].values

            # FT - Must have tested task before training another task then more testing
            if np.any(training_blocks < np.min(other_training_blocks)):
                if len(other_test_blocks) >= 2:
                    for idx in range(0, len(other_test_blocks) - 1):
                        for t_idx in training_blocks[training_blocks < np.min(other_training_blocks)]:
                            if other_test_blocks[idx] < t_idx < other_test_blocks[idx + 1]:
                                tasks_for_ft[task_pair[0]] = {task_pair[1] : (
                                    other_test_blocks[idx], other_test_blocks[idx + 1])}

        if np.sum([len(value) for key, value in tasks_for_ft.items()]) == 0:
            raise Exception('No valid task pairs for forward transfer')

        return tasks_for_ft

    def calculate(self, dataframe, block_info, metrics_df):
        try:
            # Validate data and get pairs eligible for forward transfer
            tasks_for_ft = self.validate(block_info)
            
            metrics_df['forward_transfer'] = np.full_like(metrics_df['regime_num'], np.nan, dtype=np.double)
            forward_transfer = {}

            # Calculate forward transfer for valid task pairs
            for task, value in tasks_for_ft.items():
                for trans_task, trans_blocks in value.items():
                    tp_1 = metrics_df[(metrics_df['task_name'] == _localutil.get_simple_rl_task_names(
                        [trans_task])[0]) & (metrics_df['block_num'] == trans_blocks[0])]['term_perf'].values[0]

                    if tp_1:
                        tp_2 = metrics_df[(metrics_df['task_name'] == _localutil.get_simple_rl_task_names(
                            [trans_task])[0]) & (metrics_df['block_num'] == trans_blocks[1])]['term_perf'].values[0]
                        idx = block_info[(block_info['task_name'] == trans_task) & (
                            block_info['block_num'] == trans_blocks[1])]['regime_num'].values[0]
                        forward_transfer[idx] = tp_2 / tp_1

            return _localutil.fill_metrics_df(forward_transfer, 'forward_transfer', metrics_df)
        except Exception as e:
            print(f"Cannot compute {self.name} - {e}")
            return metrics_df



class BackwardTransfer(AgentMetric):
    name = "Backward Transfer"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the backward transfer for valid task pairs"

    def __init__(self, perf_measure):
        super().__init__()
        self.perf_measure = perf_measure

    def validate(self, block_info):

        # Initialize variables for checking block type format
        last_block_num = -1
        last_block_type = ''

        # Ensure alternating block types
        for _, regime in block_info.iterrows():
            if regime['block_num'] != last_block_num:
                last_block_num = regime['block_num']

                if regime['block_type'] == 'test':
                    if last_block_type == 'test':
                        raise Exception('Block types must be alternating')
                    last_block_type = 'test'
                elif regime['block_type'] == 'train':
                    if last_block_type == 'train':
                        raise Exception('Block types must be alternating')
                    last_block_type = 'train'

        # Find eligible tasks for forward and backward transfer
        unique_tasks = block_info.loc[:, 'task_name'].unique()

        # Initialize list of tasks for transfer matrix
        tasks_for_bt = dict.fromkeys(unique_tasks, {})

        # Determine valid transfer pairs
        for task_pair in permutations(unique_tasks):
            # Get testing and training indices for task pair
            task_blocks = block_info[block_info['task_name'] == task_pair[0]]
            training_blocks = task_blocks[task_blocks['block_type'] == 'train']['block_num'].values

            other_blocks = block_info[block_info['task_name'] == task_pair[1]]
            other_test_blocks = other_blocks[other_blocks['block_type'] == 'test']['block_num'].values
            other_training_blocks = other_blocks[other_blocks['block_type'] == 'train']['block_num'].values

            # BT - Must have trained task with testing then training another task then more testing
            if np.any(other_training_blocks < np.min(training_blocks)):
                if len(other_test_blocks) >= 2:
                    for idx in range(0, len(other_test_blocks) - 1):
                        if other_test_blocks[idx] < np.min(training_blocks) < other_test_blocks[idx + 1]:
                            tasks_for_bt[task_pair[0]] = {task_pair[1] : (
                                    other_test_blocks[idx], other_test_blocks[idx + 1])}

        if np.sum([len(value) for key, value in tasks_for_bt.items()]) == 0:
            raise Exception('No valid task pairs for backward transfer')

        return tasks_for_bt

    def calculate(self, dataframe, block_info, metrics_df):
        try:
            # Validate data and get pairs eligible for backward transfer
            tasks_for_bt = self.validate(block_info)
            
            metrics_df['backward_transfer'] = np.full_like(metrics_df['regime_num'], np.nan, dtype=np.double)
            backward_transfer = {}

            # Calculate backward transfer for valid task pairs
            for task, value in tasks_for_bt.items():
                for trans_task, trans_blocks in value.items():
                    tp_1 = metrics_df[(metrics_df['task_name'] == _localutil.get_simple_rl_task_names(
                        [trans_task])[0]) & (metrics_df['block_num'] == trans_blocks[0])]['term_perf'].values[0]
                    tp_2 = metrics_df[(metrics_df['task_name'] == _localutil.get_simple_rl_task_names(
                        [trans_task])[0]) & (metrics_df['block_num'] == trans_blocks[1])]['term_perf'].values[0]
                    idx = block_info[(block_info['task_name'] == trans_task) & (
                        block_info['block_num'] == trans_blocks[1])]['regime_num'].values[0]
                    backward_transfer[idx] = tp_2 / tp_1

            return _localutil.fill_metrics_df(backward_transfer, 'backward_transfer', metrics_df)
        except Exception as e:
            print(f"Cannot compute {self.name} - {e}")
            return metrics_df


class STERelativePerf(AgentMetric):
    name = "Performance relative to STE"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the performance of each task relative to it's corresponding single task expert"

    def __init__(self, perf_measure):
        super().__init__()
        self.perf_measure = perf_measure

    def validate(self, block_info):
        # Check if there is STE data for each task in the scenario
        unique_tasks = block_info.loc[:, 'task_name'].unique()
        ste_names = util.get_ste_data_names()

        # Make sure STE baselines are available for all tasks, else send warning
        if ~np.all(np.isin(unique_tasks, ste_names)):
            Warning('STE data not available for all tasks')

        # Raise exception if none of the tasks have STE data
        if ~np.any(np.isin(unique_tasks, ste_names)):
            raise Exception('No STE data available for any task')

    def calculate(self, dataframe, block_info, metrics_df):
        try:
            # Validate the STE
            self.validate(block_info)

            # Initialize metric column
            metrics_df['ste_rel_perf'] = np.full_like(metrics_df['regime_num'], np.nan, dtype=np.double)
            ste_rel_perf = {}

            # Iterate through unique tasks and STE
            unique_tasks = block_info.loc[:, 'task_name'].unique()

            for task in unique_tasks:
                # Get block info for task during training
                task_phases = block_info[(block_info['task_name'] == task) & (
                    block_info['block_type'] == 'train')]

                # Get data concatenated data for task
                task_data = dataframe[dataframe['regime_num'].isin(task_phases['regime_num'])]

                # Load STE data
                ste_data = util.load_ste_data(task)

                if ste_data is not None:
                    # Compute relative performance with no smoothing on data
                    min_exp = np.min([task_data.shape[0], ste_data.shape[0]])
                    task_perf = task_data.head(min_exp)[self.perf_measure].sum()
                    ste_perf = ste_data.head(min_exp)[self.perf_measure].sum()
                    rel_perf = task_perf / ste_perf
                    ste_rel_perf[task_data['regime_num'].iloc[-1]] = rel_perf

            return _localutil.fill_metrics_df(ste_rel_perf, 'ste_rel_perf', metrics_df)
        except Exception as e:
            print(f"Cannot compute {self.name} - {e}")
            return metrics_df


class SampleEfficiency(AgentMetric):
    name = "Sample Efficiency"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the sample efficiency relative to the single task expert"

    def __init__(self, perf_measure):
        super().__init__()
        self.perf_measure = perf_measure

    def validate(self, block_info):
        # Check if there is STE data for each task in the scenario
        unique_tasks = block_info.loc[:, 'task_name'].unique()
        ste_names = util.get_ste_data_names()

        # Make sure STE baselines are available for all tasks, else send warning
        if ~np.all(np.isin(unique_tasks, ste_names)):
            Warning('STE data not available for all tasks')

        # Raise exception if none of the tasks have STE data
        if ~np.any(np.isin(unique_tasks, ste_names)):
            raise Exception('No STE data available for any task')

    def calculate(self, dataframe, block_info, metrics_df):
        try:
            # Validate the STE
            self.validate(block_info)

            # Initialize metric column
            metrics_df['se_saturation'] = np.full_like(metrics_df['regime_num'], np.nan, dtype=np.double)
            metrics_df['se_eps_to_sat'] = np.full_like(metrics_df['regime_num'], np.nan, dtype=np.double)
            metrics_df['sample_efficiency'] = np.full_like(metrics_df['regime_num'], np.nan, dtype=np.double)
            se_saturation = {}
            se_eps_to_sat = {}
            sample_efficiency = {}

            # Iterate through unique tasks and STE
            unique_tasks = block_info.loc[:, 'task_name'].unique()

            for task in unique_tasks:
                # Get block info for task during training
                task_phases = block_info[(block_info['task_name'] == task) & (
                    block_info['block_type'] == 'train')]

                # Get data concatenated data for task
                task_data = dataframe[dataframe['regime_num'].isin(task_phases['regime_num'])]

                # Load STE data
                ste_data = util.load_ste_data(task)

                if ste_data is not None:
                    # Get task saturation value and episodes to saturation
                    window = int(len(task_data) * 0.2)
                    custom_window = min(window, self.max_window_size)

                    task_saturation, task_eps_to_sat, _ = _localutil.get_block_saturation_perf(
                        task_data, col_to_use=self.perf_measure, window_len=custom_window)

                    # Get STE saturation value and episodes to saturation
                    window = int(len(ste_data) * 0.2)
                    custom_window = min(window, self.max_window_size)

                    ste_saturation, ste_eps_to_sat, _ = _localutil.get_block_saturation_perf(
                        ste_data, col_to_use=self.perf_measure, window_len=custom_window)

                    # Compute sample efficiency
                    se_saturation[task_data['regime_num'].iloc[-1]] = task_saturation / ste_saturation
                    se_eps_to_sat[task_data['regime_num'].iloc[-1]] = ste_eps_to_sat / task_eps_to_sat
                    sample_efficiency[task_data['regime_num'].iloc[-1]] = \
                        (task_saturation / ste_saturation) * (ste_eps_to_sat / task_eps_to_sat)

            metrics_df = _localutil.fill_metrics_df(se_saturation, 'se_saturation', metrics_df)
            metrics_df = _localutil.fill_metrics_df(se_eps_to_sat, 'se_eps_to_sat', metrics_df)
            return _localutil.fill_metrics_df(sample_efficiency, 'sample_efficiency', metrics_df)
        except Exception as e:
            print(f"Cannot compute {self.name} - {e}")
            return metrics_df


class AgentMetricsReport(core.MetricsReport):
    """
    Aggregates a list of metrics for an Agent learner
    """

    def __init__(self, **kwargs):
        # Defines log_dir and initializes the metrics list
        super().__init__(**kwargs)

        if 'perf_measure' in kwargs:
            perf_measure = kwargs['perf_measure']
        else:
            perf_measure = 'reward'

        # Gets all data from the relevant log files
        self._log_data = util.read_log_data(self.log_dir, [perf_measure])
        self._log_data = self._log_data.sort_values(
            by=['regime_num', 'exp_num']).set_index("regime_num", drop=False)
        _, self.block_info = _localutil.parse_blocks(self._log_data)

        # Do a check to make sure the performance measure is logged
        if perf_measure not in self._log_data.columns:
            raise Exception(f'Performance measure ({perf_measure}) not found in the log data')

        # Adds default metrics
        self._add_default_metrics(perf_measure)

        # Initialize a results dictionary that can be returned at the end of the calculation step and an internal
        # dictionary that can be passed around for internal calculations
        block_info_keys_to_include = ['block_num', 'block_type', 'task_name', 'regime_num']
        if len(self.block_info.loc[:, 'param_set'].unique()) > 1:
            block_info_keys_to_include.append('param_set')

        self._metrics_df = self.block_info[block_info_keys_to_include].copy()
        self._metrics_df['task_name'] = _localutil.get_simple_rl_task_names(
            self._metrics_df.loc[:, 'task_name'].values)

    def _add_default_metrics(self, perf_measure):
        # Default metrics no matter the syllabus type
        self.add(WithinBlockSaturation(perf_measure))
        self.add(MostRecentTerminalPerformance(perf_measure))
        self.add(RecoveryTime(perf_measure))
        self.add(PerformanceRecovery(perf_measure))
        self.add(PerformanceMaintenance(perf_measure))
        self.add(ForwardTransfer(perf_measure))
        self.add(BackwardTransfer(perf_measure))
        self.add(STERelativePerf(perf_measure))
        self.add(SampleEfficiency(perf_measure))

    def calculate(self):
        for metric in self._metrics:
            self._metrics_df = metric.calculate(self._log_data, self.block_info, self._metrics_df)

    def report(self, save=False, output=None):
        print(tabulate(self._metrics_df.reset_index(drop=True), headers='keys', tablefmt='psql'))

        if save:
            # Generate filename
            if output is None:
                _, filename = os.path.split(self.log_dir)
            else:
                filename = output.replace(" ", "_")

            # Save metrics to file
            self._metrics_df.reset_index(drop=True).to_csv(filename + '_metrics.tsv', sep='\t')

    def plot(self, save=False, output=None):
        if output is None:
            input_title = os.path.split(self.log_dir)[-1]
        else:
            input_title = output

        print('Plotting a smoothed reward curve')
        util.plot_performance(self._log_data, self.block_info, do_smoothing=True, do_save_fig=save,
                              max_smoothing_window=AgentMetric.max_window_size,
                              input_title=input_title)

    def add(self, metrics_list):
        self._metrics.append(metrics_list)

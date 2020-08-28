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
from abc import ABC
from . import core, util, _localutil
import numpy as np
import os
from tabulate import tabulate

"""
Standard metrics for Agent Learning (RL tasks)
"""


class AgentMetric(core.Metric, ABC):
    """
    A single metric for an Agent (aka. Reinforcement Learning) learner
    """

    max_window_size = 300

    def __init__(self):
        pass
        # self.validate()

    def plot(self, result):
        pass

    def validate(self, phase_info):
        # TODO: Add structure validation of phase_info
        pass


class MeanRewardPerEpisodes(AgentMetric):
    name = "Achieved Reward, Averaged Per Episodes within a Block"
    capability = "continual_learning"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the performance across all tasks and phases"
    metric_title = {}

    def __init__(self):
        super().__init__()
        # self.validate()

    def validate(self, phase_info):
        pass

    def calculate(self, dataframe, phase_info, metrics_df):
        avg_reward = {}

        # Iterate over all of the blocks and compute the within block performance
        for idx in range(phase_info.loc[:, 'block'].max() + 1):
            # Need to get the part of the data corresponding to the block
            block_data = dataframe.loc[dataframe["block"] == idx]

            avg_reward[idx] = block_data['reward'].mean()

        return _localutil.fill_metrics_df(avg_reward, 'avg_achieved_reward', metrics_df)


class WithinBlockSaturation(AgentMetric):
    name = "Average Within Block Saturation Calculation"
    capability = "continual_learning"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the max performance within each block"

    def __init__(self):
        super().__init__()
        # self.validate()

    def validate(self, phase_info):
        pass

    def calculate(self, dataframe, phase_info, metrics_df):
        metrics_df['saturation_value'] = np.full_like(metrics_df['block'], np.nan, dtype=np.double)
        metrics_df['eps_to_saturation'] = np.full_like(metrics_df['block'], np.nan, dtype=np.double)
        saturation_values = {}
        eps_to_saturation = {}

        # Iterate over all of the blocks and compute the within block performance
        for idx in range(phase_info.loc[:, 'block'].max() + 1):
            # Need to get the part of the data corresponding to the block
            block_data = dataframe.loc[dataframe["block"] == idx]

            # Get block window size for smoothing
            window = int(block_data.size * 0.2)
            custom_window = min(window, self.max_window_size)

            # Make within block calculations
            sat_value, eps_to_sat, _ = _localutil.get_block_saturation_perf(
                block_data, column_to_use='reward', window_len=custom_window)

            # Record them
            saturation_values[idx] = sat_value
            eps_to_saturation[idx] = eps_to_sat

        metrics_df = _localutil.fill_metrics_df(saturation_values, 'saturation_value', metrics_df)
        return _localutil.fill_metrics_df(eps_to_saturation, 'eps_to_saturation', metrics_df)


class MostRecentTerminalPerformance(AgentMetric):
    name = "Most Recent Terminal Performance"
    capability = "continual_learning"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the terminal performance within each block"

    def __init__(self):
        super().__init__()

    def validate(self, phase_info):
        pass

    def calculate(self, dataframe, phase_info, metrics_df):
        metrics_df['terminal_performance_value'] = np.full_like(metrics_df['block'], np.nan, dtype=np.double)
        metrics_df['eps_to_terminal_performance'] = np.full_like(metrics_df['block'], np.nan, dtype=np.double)
        terminal_perf_values = {}
        eps_to_terminal_perf = {}

        # Iterate over all of the blocks and compute the within block performance
        for idx in range(phase_info.loc[:, 'block'].max() + 1):
            # Need to get the part of the data corresponding to the block
            block_data = dataframe.loc[dataframe["block"] == idx]

            window = int(block_data.size * 0.2)
            custom_window = min(window, self.max_window_size)

            # Make within block calculations
            terminal_value, eps_to_term_perf, _ = _localutil.get_terminal_perf(
                block_data, column_to_use='reward', window_len=custom_window)

            # Record them
            terminal_perf_values[idx] = terminal_value
            eps_to_terminal_perf[idx] = eps_to_term_perf

        metrics_df = _localutil.fill_metrics_df(terminal_perf_values, 'terminal_performance_value', metrics_df)
        return _localutil.fill_metrics_df(eps_to_terminal_perf, 'eps_to_terminal_performance', metrics_df)


class AverageLearningReward(AgentMetric):
    name = "Average Learning Reward"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the average reward accumulated until the system achieves saturation"

    def __init__(self):
        super().__init__()

    def validate(self, phase_info):
        pass

    def calculate(self, dataframe, phase_info, metrics_df):
        metrics_df['average_learning'] = np.full_like(metrics_df['block'], np.nan, dtype=np.double)
        learning_rate = {}
        train_block_ids = phase_info.loc[phase_info['phase_type'] == 'train', 'block'].values

        for tr_block in train_block_ids:
            num_episodes = metrics_df['eps_to_saturation'][tr_block]

            if num_episodes is np.NaN:
                reward_until_saturation = 0

            else:
                block_data = dataframe.loc[dataframe['block'] == tr_block]
                first_episode_num = block_data.iloc[0]['task']
                saturation_task_num = num_episodes + first_episode_num

                reward_until_saturation = block_data.loc[block_data['task'] <= saturation_task_num, 'reward'].sum()

            learning_rate[tr_block] = reward_until_saturation / num_episodes

        return _localutil.fill_metrics_df(learning_rate, 'average_learning', metrics_df)


class RecoveryTime(AgentMetric):
    name = "Recovery Time"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates whether the system recovers after a change of task or parameters"

    def __init__(self):
        super().__init__()
        # self.validate()

    def validate(self, phase_info):
        # Determine where we need to assess recovery time
        tr_bl_inds_to_use = []
        tr_bl_inds_to_assess = []

        # Get train blocks, in order of appearance
        tr_bl_info = phase_info.sort_index().loc[phase_info['phase_type'] == 'train', ['block', 'task_name',
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

        if tr_bl_inds_to_assess is None:
            raise Exception('No changes across training blocks to assess recovery time!')

        return tr_bl_inds_to_use, tr_bl_inds_to_assess

    def calculate(self, dataframe, phase_info, metrics_df):
        # Get the places where we should calculate recovery time
        tr_inds_to_use, tr_inds_to_assess = self.validate(phase_info)
        if len(tr_inds_to_use) == 0:
            return metrics_df

        metrics_df['recovery_time'] = np.full_like(metrics_df['block'], np.nan, dtype=np.double)
        recovery_time = {}

        window = int(np.floor(len(dataframe) * 0.02))
        custom_window = max(window, self.max_window_size)

        for use_ind, assess_ind in zip(tr_inds_to_use, tr_inds_to_assess):
            prev_val = metrics_df['saturation_value'][use_ind]
            block_data = dataframe.loc[assess_ind]
            _, _, eps_to_rec = _localutil.get_block_saturation_perf(block_data,
                                                                    column_to_use='reward',
                                                                    previous_saturation_value=prev_val,
                                                                    window_len=custom_window)
            recovery_time[assess_ind] = eps_to_rec

        return _localutil.fill_metrics_df(recovery_time, 'recovery_time', metrics_df)


class STERelativePerf(AgentMetric):
    name = "Performance relative to S.T.E"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the performance of each task relative to it's corresponding single task expert"

    def __init__(self):
        super().__init__()
        # self.validate()

    def validate(self, phase_info):
        # Load the single task experts and compare them to the ones in the logs
        ste_dict = util.load_default_ste_data()
        unique_tasks = phase_info.loc[:, 'task_name'].unique()

        # Make sure STE baselines are available for all tasks, else complain
        if unique_tasks.any() not in ste_dict:
            raise Exception('STE baselines not available for all tasks')

        # TODO: Add structure validation of phase_info

        return ste_dict

    def calculate(self, dataframe, phase_info, metrics_df):
        # Validate the STE
        try:
            ste_dict = self.validate(phase_info)
            metrics_df['STE_normalized_saturation'] = np.full_like(metrics_df['block'], np.nan, dtype=np.double)
            ste_normalized_saturation = {}

            for idx in range(phase_info.loc[:, 'block'].max() + 1):
                # Get which task this block is and grab the STE performance for that task
                this_task = phase_info.loc[idx, "task_name"]
                this_ste_comparison = ste_dict[this_task]

                # Compare the saturation value of this block to the STE performance and store it
                ste_normalized_saturation[idx] = metrics_df["saturation_value"][idx] / this_ste_comparison

            return _localutil.fill_metrics_df(ste_normalized_saturation, 'STE_normalized_saturation', metrics_df)
        except:
            print("Data not suitable for", self.name)
            return metrics_df


class PerfDifferenceANT(AgentMetric):
    name = "Performance Difference from Previously Trained Task Performance"
    capability = "adapting_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the difference in performance of each task, in each evaluation block, " \
                  "relative to the previously trained task"

    def __init__(self):
        super().__init__()
        # self.validate()

    def validate(self, phase_info):
        # TODO: Add structure validation of phase_info
        # Must ensure that the training phase has only one block or else handle multiple
        pass

    def calculate(self, dataframe, phase_info, metrics_df):
        # This metric must compute in each evaluation block the performance of the tasks
        # relative to the previously trained ones
        previously_trained_tasks = np.array([])
        previously_trained_task_ids = np.array([])
        saturation_difference = {}
        eps_to_sat_dif = {}

        # Iterate over the phases, just the evaluation portion. We need to do this in order.
        for phase in phase_info.sort_index().loc[:, 'phase_number'].unique():
            # Get the task names that were used for the train portion of the phase
            trained_task = phase_info[(phase_info.phase_type == 'train') &
                                      (phase_info.phase_number == phase)].loc[:, 'task_name'].to_numpy()
            trained_task = trained_task[0]
            trained_task_ids = phase_info[(phase_info.phase_type == 'train') &
                                          (phase_info.phase_number == phase)].loc[:, 'block'].to_numpy()

            # Validation will have ensured that the training phase has exactly one training block
            previously_trained_tasks = np.append(previously_trained_tasks, trained_task)
            previously_trained_task_ids = np.append(previously_trained_task_ids, trained_task_ids)

            this_phase_test_tasks = phase_info[(phase_info.phase_type == 'test') &
                                               (phase_info.phase_number == phase)].loc[:, 'task_name'].to_numpy()
            this_phase_test_task_ids = phase_info[(phase_info.phase_type == 'test') &
                                                  (phase_info.phase_number == phase)].loc[:, 'block'].to_numpy()

            for idx, task in enumerate(this_phase_test_tasks):
                # Skip the evaluation block immediately following a train task
                if task == trained_task:
                    continue
                if task in previously_trained_tasks:
                    # Get the inds in the previously_trained_tasks array to get the saturation values for comparison
                    inds_where_task = np.where(previously_trained_tasks == task)

                    # TODO: Handle multiple comparison points
                    block_ids_for_comparison = previously_trained_task_ids[inds_where_task]
                    previously_trained_sat_value = metrics_df['saturation_value'][block_ids_for_comparison[0]]
                    previously_trained_num_eps_to_sat = metrics_df['eps_to_saturation'][block_ids_for_comparison[0]]

                    new_sat_value = metrics_df['saturation_value'][this_phase_test_task_ids[idx]]
                    new_num_eps_to_sat = metrics_df['eps_to_saturation'][this_phase_test_task_ids[idx]]

                    this_sat_val_comparison = previously_trained_sat_value - new_sat_value
                    this_num_eps_to_sat_comparison = previously_trained_num_eps_to_sat - new_num_eps_to_sat

                    block_id = this_phase_test_task_ids[idx]

                    saturation_difference[block_id] = this_sat_val_comparison
                    eps_to_sat_dif[block_id] = this_num_eps_to_sat_comparison

        metrics_df = _localutil.fill_metrics_df(saturation_difference, 'saturation_difference', metrics_df)

        return _localutil.fill_metrics_df(eps_to_sat_dif, 'eps_to_sat_dif', metrics_df)


class RewardPerStep(AgentMetric):
    name = "Reward per Step"
    capability = "continual_learning"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the reward achieved per steps used to achieve it"

    def __init__(self):
        super().__init__()
        # self.validate()

    def validate(self, phase_info):
        # TODO: Add structure validation of phase_info
        pass

    def calculate(self, dataframe, phase_info, metrics_df):
        metrics_df['reward_per_step'] = np.full_like(metrics_df['block'], np.nan, dtype=np.double)
        reward_per_step = {}

        for idx in range(phase_info.loc[:, 'block'].max() + 1):
            block_data = dataframe.loc[dataframe['block'] == idx]

            reward = np.array(block_data.loc[:, "reward"].values)
            steps = np.array(block_data.loc[:, 'steps'].values)

            summed_reward = np.sum(reward)
            # cumsum_reward = np.cumsum(reward)
            # cumsum_steps = np.cumsum(steps)

            res = summed_reward / np.sum(steps)
            reward_per_step[idx] = res

        return _localutil.fill_metrics_df(reward_per_step, 'reward_per_step', metrics_df)


class TransferMatrix(AgentMetric):
    name = "Transfer Matrix - both forward and reverse transfer"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates a transfer matrix for all trained tasks"

    def __init__(self):
        super().__init__()
        # self.validate()

    def validate(self, phase_info):
        # Load the single task experts and compare them to the ones in the logs
        ste_dict = util.load_default_ste_data()
        unique_tasks = phase_info.loc[:, 'task_name'].unique()

        # Return tasks which have STE baselines
        tasks_with_ste = [t for t in ste_dict.keys() if t in unique_tasks]
        tasks_for_transfer_matrix = {'forward': [], 'reverse': []}

        for task in tasks_with_ste:
            types_per_this_task = phase_info[phase_info['task_name'] == task]
            phase_types = types_per_this_task.loc[:, 'phase_type'].to_numpy()
            phase_nums = types_per_this_task.loc[:, 'phase_number'].to_numpy(dtype=int)
            if 'train' in phase_types and 'test' in phase_types:
                # Check eligibility for both forward and reverse transfer
                train_phase_nums = phase_nums[phase_types == 'train']
                test_phase_nums = phase_nums[phase_types == 'test']
                if len(train_phase_nums) > 1:
                    raise Exception('Too many training instances of task: {:s}'.format(task))
                train_phase_num = train_phase_nums[0]

                if any(test_phase_nums < train_phase_num):
                    phase_nums_to_add = test_phase_nums[np.where(test_phase_nums < train_phase_num)]
                    for num in phase_nums_to_add:
                        tmp = types_per_this_task.loc[types_per_this_task['phase_type'] == 'test']
                        blocks_to_add = tmp.loc[tmp['phase_number'] == str(num), 'block']
                        if len(blocks_to_add) > 1:
                            raise Exception('Too many eval instances of task: {:s}'.format(task))
                        block_to_add = blocks_to_add.values[0]
                        tasks_for_transfer_matrix['forward'].append((task, block_to_add))

                if any(test_phase_nums > train_phase_num):
                    phase_nums_to_add = test_phase_nums[np.where(test_phase_nums > train_phase_num)]
                    for num in phase_nums_to_add:
                        tmp = types_per_this_task.loc[types_per_this_task['phase_type'] == 'test']
                        blocks_to_add = tmp.loc[tmp['phase_number'] == str(num), 'block']
                        if len(blocks_to_add) > 1:
                            raise Exception('Too many eval instances of task: {:s}'.format(task))
                        block_to_add = blocks_to_add.values[0]
                        tasks_for_transfer_matrix['reverse'].append((task, block_to_add))

        return ste_dict, tasks_for_transfer_matrix

    def calculate(self, data, metadata, metrics_df):
        # Make sure to load Single Task Expert performance and figure out where we should calculate transfer
        ste_dict, tasks_to_compute = self.validate(metadata)
        metrics_df['forward_transfer'] = np.full_like(metrics_df['block'], np.nan, dtype=np.double)
        metrics_df['reverse_transfer'] = np.full_like(metrics_df['block'], np.nan, dtype=np.double)
        reverse_transfer = {}
        forward_transfer = {}

        # Calculate, for each task, (task eval saturation / ste saturation)
        for task, block in tasks_to_compute['forward']:
            print('Computing forward transfer for {:s}'.format(task))
            this_transfer_val = metrics_df['saturation_value'][block] / ste_dict[task]
            forward_transfer[block] = this_transfer_val

        for task, block in tasks_to_compute['reverse']:
            print('Computing reverse transfer for {:s}'.format(task))
            this_transfer_val = metrics_df['saturation_value'][block] / ste_dict[task]
            reverse_transfer[block] = this_transfer_val

        metrics_df = _localutil.fill_metrics_df(forward_transfer, 'forward_transfer', metrics_df)
        return _localutil.fill_metrics_df(reverse_transfer, 'reverse_transfer', metrics_df)


class AgentMetricsReport(core.MetricsReport):
    """
    Aggregates a list of metrics for an Agent learner
    """

    def __init__(self, **kwargs):
        # Defines log_dir and initializes the metrics list
        super().__init__(**kwargs)

        # Gets all data from the relevant log files
        self._log_data = util.read_log_data(self.log_dir)
        self._log_data = self._log_data.sort_values(by=['block', 'task']).set_index("block", drop=False)
        _, self.phase_info = _localutil.parse_blocks(self._log_data)

        # Adds default metrics
        self._add_default_metrics()

        # Do an initial check to make sure that reward has been logged
        if 'reward' not in self._log_data.columns:
            raise Exception('Reward column is required in the log data!')

        # Initialize a results dictionary that can be returned at the end of the calculation step and an internal
        # dictionary that can be passed around for internal calculations
        phase_info_keys_to_include = ['phase_number', 'phase_type', 'task_name', 'block']
        if len(self.phase_info.loc[:, 'param_set'].unique()) > 1:
            phase_info_keys_to_include.append('param_set')

        self._metrics_df = self.phase_info[phase_info_keys_to_include].copy()
        self._metrics_df['task_name'] = _localutil.get_simple_rl_task_names(
            self._metrics_df.loc[:, 'task_name'].values)

    def _add_default_metrics(self):
        # Default metrics no matter the syllabus type
        self.add(WithinBlockSaturation())
        self.add(RewardPerStep())
        self.add(AverageLearningReward())
        self.add(MeanRewardPerEpisodes())
        self.add(MostRecentTerminalPerformance())
        self.add(RecoveryTime())
        self.add(STERelativePerf())
        self.add(PerfDifferenceANT())
        self.add(TransferMatrix())

    def calculate(self):
        for metric in self._metrics:
            self._metrics_df = metric.calculate(self._log_data, self.phase_info, self._metrics_df)

    def report(self, save=False):
        print(tabulate(self._metrics_df, headers='keys', tablefmt='psql'))

        if save:
            # Generate filename
            if os.path.dirname(self.log_dir) != "":
                _, filename = os.path.split(self.log_dir)
            else:
                filename = 'agent'

            # Save collated log data to file
            self._log_data.to_csv(filename + '_data.tsv', sep='\t')

            # Save metrics to file
            self._metrics_df.to_csv(filename + '_metrics.tsv', sep='\t')

    def plot(self, save=False):
        print('Plotting a smoothed reward curve')
        util.plot_performance(self._log_data, do_smoothing=True, do_task_colors=True, do_save_fig=save,
                              max_smoothing_window=100, input_title=self.log_dir)

    def add(self, metrics_list):
        self._metrics.append(metrics_list)

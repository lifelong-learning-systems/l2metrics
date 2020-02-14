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
"""
Standard metrics for Agent Learning (RL tasks)
"""


class AgentMetric(core.Metric, ABC):
    """
    A single metric for an Agent (aka. Reinforcement Learning) learner
    """

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
    requires = {'syllabus_type': 'agent', 'syllabus_subtype': 'all'}
    description = "Calculates the performance across all tasks and phases"

    def __init__(self):
        super().__init__()
        # self.validate()

    def validate(self, phase_info):
        pass

    def calculate(self, dataframe, phase_info, metrics_dict):
        avg_reward = {}

        # Iterate over all of the blocks and compute the within block performance
        for idx in range(phase_info.loc[:, 'block'].max() + 1):
            # Need to get the part of the data corresponding to the block
            block_data = dataframe.loc[dataframe["block"] == idx]

            avg_reward[idx] = block_data['reward'].mean()

        metrics_dict["avg_achieved_reward"] =  avg_reward
        metric_to_return = {'global_avg_achieved_reward': np.mean(list(avg_reward.values()))}

        return metric_to_return, metrics_dict


class WithinBlockSaturation(AgentMetric):
    name = "Average Within Block Saturation Calculation"
    capability = "continual_learning"
    requires = {'syllabus_type': 'agent', 'syllabus_subtype': 'CL'}
    description = "Calculates the performance within each block"

    def __init__(self):
        super().__init__()
        # self.validate()

    def validate(self, phase_info):
        pass

    def calculate(self, dataframe, phase_info, metrics_dict):
        saturation_value = {}
        eps_to_saturation = {}
        all_sat_vals = []
        all_eps_to_sat = []

        # Iterate over all of the blocks and compute the within block performance
        for idx in range(phase_info.loc[:, 'block'].max() + 1):
            # Need to get the part of the data corresponding to the block
            block_data = dataframe.loc[dataframe["block"] == idx]
            # Make within block calculations
            sat_value, eps_to_sat, _ = _localutil.get_block_saturation_perf(block_data, column_to_use='reward')

            # Record them
            saturation_value[idx] = sat_value
            all_sat_vals.append(sat_value)
            eps_to_saturation[idx] = eps_to_sat
            all_eps_to_sat.append(eps_to_sat)

        metrics_dict = {"saturation_value": saturation_value, "eps_to_saturation": eps_to_saturation}
        metric_to_return = {'global_within_block_saturation': np.mean(all_sat_vals),
                            'global_num_eps_to_saturation': np.mean(all_eps_to_sat)}

        return metric_to_return, metrics_dict


class RecoveryTime(AgentMetric):
    name = "Recovery Time"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent', 'syllabus_subtype': 'ANT_A'}
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
            if tr_bl_info.loc[block_idx, 'task_name'] != tr_bl_info.loc[tb_inds[idx-1], 'task_name'] or \
                    tr_bl_info.loc[block_idx, 'param_set'] != tr_bl_info.loc[tb_inds[idx-1], 'param_set']:
                tr_bl_inds_to_assess.append(block_idx)
                tr_bl_inds_to_use.append(tb_inds[idx-1])

        if tr_bl_inds_to_assess is None:
            raise Exception('No changes across training blocks to assess recovery time!')

        return tr_bl_inds_to_use, tr_bl_inds_to_assess

    def calculate(self, dataframe, phase_info, metrics_dict):
        # Get the places where we should calculate recovery time
        metrics_dict['recovery_time'] = {}
        tr_inds_to_use, tr_inds_to_assess = self.validate(phase_info)

        for use_ind, assess_ind in zip(tr_inds_to_use, tr_inds_to_assess):
            prev_val = metrics_dict['saturation_value'][use_ind]
            block_data = dataframe.loc[assess_ind]
            _, _, eps_to_rec = _localutil.get_block_saturation_perf(block_data,
                                                                    column_to_use='reward',
                                                                    previous_saturation_value=prev_val)
            metrics_dict['recovery_time'][assess_ind] = eps_to_rec
        return {'global_avg_recovery_time': np.mean(list(metrics_dict['recovery_time'].values()))}, metrics_dict


class STERelativePerf(AgentMetric):
    name = "Performance relative to S.T.E"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent', 'syllabus_subtype': 'ANT_A'}
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
            raise Exception

        # TODO: Add structure validation of phase_info

        return ste_dict

    def calculate(self, dataframe, phase_info, metrics_dict):
        # Validate the STE
        ste_dict = self.validate(phase_info)
        STE_normalized_saturation = {}
        all_STE_normalized_saturations = []

        for idx in range(phase_info.loc[:, 'block'].max()):
            # Get which task this block is and grab the STE performance for that task
            this_task = phase_info.loc[idx, "task_name"]
            this_ste_comparison = ste_dict[this_task]

            # Compare the saturation value of this block to the STE performance and store it

            STE_normalized_saturation[idx] = metrics_dict["saturation_value"][idx] / this_ste_comparison
            all_STE_normalized_saturations.append(STE_normalized_saturation[idx])

        metrics_dict["STE_normalized_saturation"] = STE_normalized_saturation
        metric_to_return = {'global_STE_normalized_saturation': np.mean(all_STE_normalized_saturations)}

        return metric_to_return, metrics_dict


class PerfMaintenanceANT(AgentMetric):
    name = "Performance Maintenance relative to previously trained task"
    capability = "adapting_to_new_tasks"
    requires = {'syllabus_type': 'agent', 'syllabus_subtype': 'ANT'}
    description = "Calculates the performance of each task, in each evaluation block, " \
                  "relative to the previously trained task"

    def __init__(self):
        super().__init__()
        # self.validate()

    def validate(self, phase_info):
        # TODO: Add structure validation of phase_info
        # Must ensure that the training phase has only one block or else handle multiple
        pass

    def calculate(self, dataframe, phase_info, metrics_dict):
        # This metric must compute in each evaluation block the performance of the tasks
        # relative to the previously trained ones
        previously_trained_tasks = np.array([])
        previously_trained_task_ids = np.array([])
        this_metric = {}
        all_sat_diff_vals = []
        all_eps_to_sat_diff_vals = []
        this_sat_val_comparison = np.nan
        this_num_eps_to_sat_comparison = np.nan

        # Iterate over the phases, just the evaluation portion. We need to do this in order.
        for phase in phase_info.sort_index().loc[:, 'phase_number'].unique():
            # Get the task names that were used for the train portion of the phase
            trained_tasks = phase_info[(phase_info.phase_type == 'train') &
                                       (phase_info.phase_number == phase)].loc[:, 'task_name'].to_numpy()
            trained_task_ids = phase_info[(phase_info.phase_type == 'train') &
                                          (phase_info.phase_number == phase)].loc[:, 'block'].to_numpy()

            # Validation will have ensured that the training phase has exactly one training block
            previously_trained_tasks = np.append(previously_trained_tasks, trained_tasks)
            previously_trained_task_ids = np.append(previously_trained_task_ids, trained_task_ids)

            this_phase_test_tasks = phase_info[(phase_info.phase_type == 'test') &
                                               (phase_info.phase_number == phase)].loc[:, 'task_name'].to_numpy()
            this_phase_test_task_ids = phase_info[(phase_info.phase_type == 'test') &
                                                  (phase_info.phase_number == phase)].loc[:, 'block'].to_numpy()

            for idx, task in enumerate(this_phase_test_tasks):
                if task in previously_trained_tasks:
                    # Get the inds in the previously_trained_tasks array to get the saturation values for comparison
                    inds_where_task = np.where(previously_trained_tasks == task)

                    # TODO: Handle multiple comparison points
                    block_ids_for_comparison = previously_trained_task_ids[inds_where_task]
                    previously_trained_sat_values = metrics_dict['saturation_value'][block_ids_for_comparison[0]]
                    previously_trained_num_eps_to_sat = metrics_dict['eps_to_saturation'][block_ids_for_comparison[0]]

                    new_sat_value = metrics_dict['saturation_value'][this_phase_test_task_ids[idx]]
                    new_num_eps_to_sat = metrics_dict['eps_to_saturation'][this_phase_test_task_ids[idx]]

                    this_sat_val_comparison = previously_trained_sat_values - new_sat_value
                    this_num_eps_to_sat_comparison = previously_trained_num_eps_to_sat - new_num_eps_to_sat

                    key_str_1 = task + '_phase_' + str(phase) + '_sat_value_maintenance'
                    key_str_2 = task + '_phase_' + str(phase) + '_num_eps_maintenance'

                    this_metric[key_str_1] = this_sat_val_comparison
                    this_metric[key_str_2] = this_num_eps_to_sat_comparison

                    all_sat_diff_vals.append(this_sat_val_comparison)
                    all_eps_to_sat_diff_vals.append(this_num_eps_to_sat_comparison)

        metric_to_return = {'mean_saturation_value_diff': np.mean(all_sat_diff_vals),
                            'mean_num_eps_to_saturation_diff': np.mean(all_eps_to_sat_diff_vals)}

        metrics_dict['saturation_maintenance'] = this_sat_val_comparison
        metrics_dict['num_eps_maintenance'] = this_num_eps_to_sat_comparison

        return metric_to_return, metrics_dict


class RewardPerStep(AgentMetric):
    name = "Reward per Step"
    capability = "continual_learning"
    requires = {'syllabus_type': 'agent', 'syllabus_subtype': 'all'}
    description = "Calculates the reward achieved per steps used to achieve it"

    def __init__(self):
        super().__init__()
        # self.validate()

    def validate(self, phase_info):
        # TODO: Add structure validation of phase_info
        pass

    def calculate(self, dataframe, phase_info, metrics_dict):
        reward = np.array(dataframe.loc[:, "reward"].values)
        steps = np.array(dataframe.loc[:, 'steps'].values)

        summed_reward = np.sum(reward)
        # cumsum_reward = np.cumsum(reward)
        # cumsum_steps = np.cumsum(steps)

        res = summed_reward / np.sum(steps)
        metric_to_return = {'reward_per_step': res}
        metrics_dict['reward_per_step'] = res

        return metric_to_return, metrics_dict


class TransferMatrix(AgentMetric):
    name = "Transfer Matrix - both forward and reverse transfer"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent', 'syllabus_subtype': 'ANT'}
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

    def calculate(self, data, metadata, metrics_dict):
        # Make sure to load Single Task Expert performance and figure out where we should calculate transfer
        ste_dict, tasks_to_compute = self.validate(metadata)

        reverse_transfer = {}
        reverse_vals = []
        forward_transfer = {}
        forward_vals = []

        # Calculate, for each task, (task eval saturation / ste saturation)
        for task, block in tasks_to_compute['forward']:
            print('Computing forward transfer for {:s}'.format(task))
            this_transfer_val = metrics_dict['saturation_value'][block] / ste_dict[task]
            forward_transfer[(task, block)] = this_transfer_val
            forward_vals.append(this_transfer_val)

        for task, block in tasks_to_compute['reverse']:
            print('Computing reverse transfer for {:s}'.format(task))
            this_transfer_val = metrics_dict['saturation_value'][block] / ste_dict[task]
            reverse_transfer[(task, block)] = this_transfer_val
            reverse_vals.append(this_transfer_val)

        metrics_dict['forward_transfer'] = forward_transfer
        metrics_dict['reverse_transfer'] = reverse_transfer

        metric_to_return = {'forward_transfer': np.mean(forward_vals),
                            'reverse_transfer': np.mean(reverse_vals)}

        return metric_to_return, metrics_dict


class AgentMetricsReport(core.MetricsReport):
    """
    Aggregates a list of metrics for an Agent learner
    """

    def __init__(self, **kwargs):
        # Defines log_dir, syllabus_subtype, and initializes the _metrics list
        super().__init__(**kwargs)

        # Gets all data from the relevant log files
        self._log_data = util.read_log_data(util.get_l2root_base_dirs('logs', self.log_dir))
        self._log_data = self._log_data.sort_values(by=['block', 'task']).set_index("block", drop=False)
        _, self.phase_info = _localutil.parse_blocks(self._log_data)

        # Adds default metrics to list based on passed syllabus subtype
        self._add_default_metrics()

        # Do an initial check to make sure that reward has been logged
        if 'reward' not in self._log_data.columns:
            raise Exception('Reward column is required in the log data!')

        # Initialize a results dictionary that can be returned at the end of the calculation step and an internal
        # dictionary that can be passed around for internal calculations
        self._results = {}
        self._collated_metrics_dict = {}
        self._metrics_dict = {}
        self._phase_info = None

    def _add_default_metrics(self):

        self.add(WithinBlockSaturation())

        if self.syllabus_subtype == "STE":
            self.add(MeanRewardPerEpisodes())
            self.add(RewardPerStep())

        elif self.syllabus_subtype == "CL":
            self.add(WithinBlockSaturation())

        elif self.syllabus_subtype == "ANT_A":
            self.add(RecoveryTime())
            self.add(STERelativePerf())
            self.add(PerfMaintenanceANT())

        elif self.syllabus_subtype == "ANT_B":
            self.add(STERelativePerf())
            self.add(TransferMatrix())  # This metric is under construction

        # This is an unhandled syllabus type as of right now
        elif self.syllabus_subtype == "ANT_C":
            raise Exception('This syllabus type ({:s}) will be handled in the future, but is not yet supported!'
                            .format(self.syllabus_subtype))

        else:
            raise Exception('Unhandled syllabus type {:s}! Supported syllabus types are: CL, ANT_A, and ANT_B'
                            .format(self.syllabus_subtype))

    def calculate(self):
        previously_calculated_metric_keys = []
        this_metrics_dict = {}
        for metric in self._metrics:
            this_result, this_metrics_dict = metric.calculate(self._log_data, self.phase_info, this_metrics_dict)
            self._results[metric.name] = this_result

            this_metrics_dict_subset = {k: this_metrics_dict[k] for k in this_metrics_dict
                                        if k not in previously_calculated_metric_keys}  # Only get the new keys

            self._metrics_dict[metric.name] = this_metrics_dict_subset

            previously_calculated_metric_keys.extend([k for k in this_metrics_dict
                                                      if k not in previously_calculated_metric_keys])

    def report(self):
        # Call a describe method to inform printing
        for r_key in self._results:
            print('\nMetric: {:s}'.format(r_key))
            print('Averaged Value: {:s}'.format(str(self._results[r_key])))
            print('Per Block Values: {:s}'.format(str(self._metrics_dict[r_key])))

    def plot(self):
        print('Plotting a smoothed reward curve:')
        window = int(np.floor(len(self._log_data)*0.02))
        custom_window = min(window, 100)
        save = True

        util.plot_performance(self._log_data, do_smoothing=True, do_task_colors=True, do_save_fig=save,
                              new_smoothing_value=custom_window, input_title=self.log_dir)

    def add(self, metrics_list):
        self._metrics.append(metrics_list)

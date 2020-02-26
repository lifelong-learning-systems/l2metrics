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
Standard metrics for Classification tasks
"""


class ClassificationMetric(core.Metric, ABC):
    def __init__(self):
        pass
        # self.validate()

    def plot(self, result):
        pass

    def validate(self, phase_info):
        pass


class AveragedScorePerBatch(ClassificationMetric):
    name = "Average Score per block"
    capability = "continual_learning"
    requires = {'syllabus_type': 'type2'}
    description = "Calculates the performance across all tasks and phases"

    def __init__(self):
        super().__init__()
        # self.validate()

    def validate(self, phase_info):
        # TODO: validate that the proper columns are present here rather than in calculate?
        # relevant_columns = _localutil.extract_relevant_columns(block_data, keyword='score')
        # Check that len(relevant_columns) >= 1 and return it if so
        pass

    def calculate(self, dataframe, phase_info, metrics_dict):
        average_scores = {}

        # Iterate over all of the blocks and compute the within block performance
        for idx in range(phase_info.loc[:, 'block'].max() + 1):
            # Need to get the part of the data corresponding to the block
            block_data = dataframe.loc[dataframe["block"] == idx]
            # Make within block calculations
            relevant_columns = _localutil.extract_relevant_columns(block_data, keyword='score')

            for col in relevant_columns:
                average_scores[(idx, col)] = block_data.loc[block_data['source'] == 'GET_LABELS', col].mean()

        metrics_dict["average_score_per_block"] = average_scores
        metric_to_return = {'global_average_score_per_block': np.mean(list(average_scores.values()))}

        return metric_to_return, metrics_dict


class WithinBlockSaturation(ClassificationMetric):
    name = "Average Within Block Saturation Calculation"
    capability = "continual_learning"
    requires = {'syllabus_type': 'class', 'syllabus_subtype': 'CL'}
    description = "Calculates the performance within each block"

    def __init__(self):
        super().__init__()
        # self.validate()

    def validate(self, phase_info):
        # TODO: validate that the proper columns are present here rather than in calculate?
        # Since I'm getting relevant columns per block that might not make as much sense.
        pass

    def calculate(self, dataframe, phase_info, metrics_dict):
        saturation_value = {}
        eps_to_saturation = {}
        all_sat_vals = []
        all_eps_to_sat = []

        relevant_columns = _localutil.extract_relevant_columns(dataframe, keyword='score')

        for col in relevant_columns:
            # Iterate over all of the blocks and compute the within block performance
            for idx in range(phase_info.loc[:, 'block'].max() + 1):
                # Get the part of the data corresponding to the relevant block
                block_data = dataframe.loc[dataframe["block"] == idx]
                # Make within block calculations
                sat_value, eps_to_sat, _ = _localutil.get_block_saturation_perf(block_data, column_to_use=col)

                # Record them
                saturation_value[(col, idx)] = sat_value
                eps_to_saturation[(col, idx)] = eps_to_sat

        metrics_dict["saturation_value"] = saturation_value
        metrics_dict["eps_to_saturation"] = eps_to_saturation
        metric_to_return = {'global_within_block_saturation': np.mean(list(saturation_value.values())),
                            'global_num_eps_to_saturation': np.mean(list(eps_to_saturation.values()))}

        return metric_to_return, metrics_dict


class LearningRate(ClassificationMetric):
    name = "Learning Rate"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent', 'syllabus_subtype': 'ANT_A'}
    description = "Calculates the average reward accumulated until the system achieves saturation"

    def __init__(self):
        super().__init__()

    def validate(self, phase_info):
        pass

    def calculate(self, dataframe, phase_info, metrics_dict):
        metrics_dict['learning_rate'] = {}
        train_block_ids = phase_info.loc[phase_info['phase_type'] == 'train', 'block'].values
        relevant_columns = _localutil.extract_relevant_columns(dataframe, keyword='score')

        for col in relevant_columns:
            for tr_block in train_block_ids:
                num_episodes = metrics_dict['eps_to_saturation'][(col, tr_block)]

                if num_episodes is np.NaN:
                    score_until_saturation = 0

                else:
                    block_data = dataframe.loc[dataframe['block'] == tr_block]
                    first_episode_num = block_data.iloc[0]['task']
                    saturation_task_num = num_episodes + first_episode_num

                    score_until_saturation = block_data.loc[block_data['task'] <= saturation_task_num, col].sum()

                metrics_dict['learning_rate'][(col, tr_block)] = score_until_saturation / num_episodes

        return {'global_learning_rate': np.mean(list(metrics_dict['learning_rate'].values()))}, metrics_dict


class RecoveryTime(ClassificationMetric):
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

        # Grab the map of names in logs vs high level task names
        unique_tasks = phase_info.loc[:, 'task_name'].unique()
        task_map, block_list, name_map, type_map = _localutil.simplify_task_names(unique_tasks, phase_info)

        # Blocks are defined as new combinations of task + params, but can repeat, so check for changes across blocks
        first = True
        for idx, block_idx in enumerate(tb_inds):
            if first:
                first = False
                continue
            # Either the task name or the param set must be different
            this_task_name = tr_bl_info.loc[block_idx, 'task_name']
            next_task_name = tr_bl_info.loc[tb_inds[idx - 1], 'task_name']
            this_simple_task_name = name_map['full_name_map'][this_task_name]
            this_simple_next_task_name = name_map['full_name_map'][next_task_name]
            if this_simple_task_name != this_simple_next_task_name or \
                    tr_bl_info.loc[block_idx, 'param_set'] != tr_bl_info.loc[tb_inds[idx - 1], 'param_set']:
                tr_bl_inds_to_assess.append(block_idx)
                tr_bl_inds_to_use.append(tb_inds[idx - 1])

        if tr_bl_inds_to_assess is None:
            raise Exception('No changes across training blocks to assess recovery time!')

        return tr_bl_inds_to_use, tr_bl_inds_to_assess

    def calculate(self, dataframe, phase_info, metrics_dict):
        # Get the places where we should calculate recovery time
        metrics_dict['recovery_time'] = {}
        tr_inds_to_use, tr_inds_to_assess = self.validate(phase_info)
        relevant_columns = _localutil.extract_relevant_columns(dataframe, keyword='score')

        for col in relevant_columns:
            for use_ind, assess_ind in zip(tr_inds_to_use, tr_inds_to_assess):
                prev_val = metrics_dict['saturation_value'][(col, use_ind)]
                block_data = dataframe.loc[dataframe['block'] == assess_ind]
                _, _, eps_to_rec = _localutil.get_block_saturation_perf(block_data,
                                                                        column_to_use=col,
                                                                        previous_saturation_value=prev_val)
                metrics_dict['recovery_time'][(col, assess_ind)] = eps_to_rec
        return {'global_avg_recovery_time': np.mean(list(metrics_dict['recovery_time'].values()))}, metrics_dict


class PerfMaintenanceANT(ClassificationMetric):
    name = "Performance Maintenance on Previously Trained Task"
    capability = "adapting_to_new_tasks"
    requires = {'syllabus_type': 'class', 'syllabus_subtype': 'ANT'}
    description = "Calculates the performance of each task, in each evaluation block, " \
                  "relative to the previously trained task"

    def __init__(self):
        super().__init__()
        # self.validate()

    def validate(self, phase_info):
        # TODO: Add structure validation of phase_info
        # Must ensure that the training phase has only one block or else handle multiple
        pass

    def calculate(self, data, phase_info, metrics_dict):
        # This metric must compute in each evaluation block the performance of the tasks
        # relative to the previously trained ones

        # Initialize some variables
        metrics_dict['saturation_maintenance'] = {}
        metrics_dict['num_eps_maintenance'] = {}
        previously_trained_tasks = np.array([])
        previously_trained_task_ids = np.array([])
        all_sat_diff_vals = []
        all_eps_to_sat_diff_vals = []

        # Get the simplified version of task names in classification logs
        unique_tasks = phase_info.loc[:, 'task_name'].unique()
        task_map, block_list, name_map, type_map = _localutil.simplify_task_names(unique_tasks, phase_info)

        # Get relevant columns for this dataframe
        relevant_columns = _localutil.extract_relevant_columns(data, keyword='score')

        # Iterate over the phases, just the evaluation portion. We need to do this in order.
        for phase in phase_info.sort_index().loc[:, 'phase_number'].unique():
            # Get the task names that were used for the train portion of the phase
            trained_tasks_full_name = phase_info[(phase_info.phase_type == 'train') &
                                                 (phase_info.phase_number == phase)].loc[:, 'task_name'].to_numpy()
            trained_tasks = np.array(name_map['full_name_map'][trained_tasks_full_name[0]])
            trained_task_ids = phase_info[(phase_info.phase_type == 'train') &
                                          (phase_info.phase_number == phase)].loc[:, 'block'].to_numpy()

            # Validation will have ensured that the training phase has exactly one training block
            previously_trained_tasks = np.append(previously_trained_tasks, trained_tasks)
            previously_trained_task_ids = np.append(previously_trained_task_ids, trained_task_ids)

            this_phase_test_task_names = phase_info[(phase_info.phase_type == 'test') &
                                                    (phase_info.phase_number == phase)].loc[:, 'task_name'].to_numpy()
            this_phase_test_tasks = np.array([name_map['full_name_map'][task_name]
                                              for task_name in this_phase_test_task_names])
            this_phase_test_task_ids = phase_info[(phase_info.phase_type == 'test') &
                                                  (phase_info.phase_number == phase)].loc[:, 'block'].to_numpy()

            for col in relevant_columns:
                for idx, task in enumerate(this_phase_test_tasks):
                    if task in previously_trained_tasks:
                        # Get the inds in the previously_trained_tasks array to get the saturation values for comparison
                        inds_where_task = np.where(previously_trained_tasks == task)

                        # TODO: Handle multiple comparison points
                        block_ids_to_compare = previously_trained_task_ids[inds_where_task]
                        previously_trained_sat_values = metrics_dict['saturation_value'][(col, block_ids_to_compare[0])]
                        previously_trained_num_eps_to_sat = metrics_dict['eps_to_saturation'][(col, block_ids_to_compare[0])]

                        new_sat_value = metrics_dict['saturation_value'][(col, this_phase_test_task_ids[idx])]
                        new_num_eps_to_sat = metrics_dict['eps_to_saturation'][(col, this_phase_test_task_ids[idx])]

                        this_sat_val_comparison = previously_trained_sat_values - new_sat_value
                        this_num_eps_to_sat_comparison = previously_trained_num_eps_to_sat - new_num_eps_to_sat

                        key = (task, phase)

                        metrics_dict['saturation_maintenance'][key] = this_sat_val_comparison
                        metrics_dict['num_eps_maintenance'][key] = this_num_eps_to_sat_comparison

                        all_sat_diff_vals.append(this_sat_val_comparison)
                        all_eps_to_sat_diff_vals.append(this_num_eps_to_sat_comparison)

        metric_to_return = {'mean_saturation_value_diff': np.mean(all_sat_diff_vals),
                            'mean_num_eps_to_saturation_diff': np.mean(all_eps_to_sat_diff_vals)}

        metrics_dict['saturation_maintenance'] = this_sat_val_comparison
        metrics_dict['num_eps_maintenance'] = this_num_eps_to_sat_comparison

        return metric_to_return, metrics_dict


class STERelativePerf(ClassificationMetric):
    name = "Performance relative to S.T.E"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'class', 'syllabus_subtype': 'ANT_A'}
    description = "Calculates the performance of each task relative to it's corresponding single task expert"

    def __init__(self):
        super().__init__()
        # self.validate()

    def validate(self, phase_info):
        # Load the single task experts and compare them to the ones in the logs
        ste_dict = util.load_default_ste_data()
        unique_tasks = phase_info.loc[phase_info['phase_type'] == 'train', 'task_name'].unique()

        # Make sure STE baselines are available for all tasks, else complain
        if unique_tasks.any() not in ste_dict:
            Warning('STE Baseline not found for all tasks trained!')

        # TODO: Add structure validation of phase_info

        return ste_dict

    def calculate(self, dataframe, phase_info, metrics_dict):
        # Validate the STE
        ste_dict = self.validate(phase_info)
        STE_normalized_saturation = {}
        all_STE_normalized_saturations = []
        relevant_columns = _localutil.extract_relevant_columns(dataframe, keyword='score')

        for col in relevant_columns:
            for idx in range(phase_info.loc[:, 'block'].max()):
                if phase_info.loc[idx, 'phase_type'] != 'train':
                    continue
                # Get which task this block is and grab the STE performance for that task
                this_task = phase_info.loc[idx, "task_name"]
                this_ste_comparison = ste_dict[this_task]

                # Compare the saturation value of this block to the STE performance and store it

                STE_normalized_saturation[(col, idx)] = metrics_dict["saturation_value"][(col, idx)] / \
                                                        this_ste_comparison
                all_STE_normalized_saturations.append(STE_normalized_saturation[(col, idx)])

            metrics_dict["STE_normalized_saturation"] = STE_normalized_saturation
            metric_to_return = {'global_STE_normalized_saturation': np.mean(all_STE_normalized_saturations)}

        return metric_to_return, metrics_dict


class TransferMatrix(ClassificationMetric):
    name = "Transfer Matrix - both forward and reverse transfer"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'class', 'syllabus_subtype': 'ANT'}
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

        task_map, block_list, name_map, type_map = _localutil.simplify_task_names(unique_tasks, phase_info)

        for task in task_map.keys():
            task_blocks = task_map[task]
            types_per_this_task = [type_map[blk] for blk in task_blocks]

            if 'train' in types_per_this_task and 'test' in types_per_this_task:
                # Check eligibility for both forward and reverse transfer
                train_block_nums = np.array(task_blocks[np.isin(types_per_this_task, 'train')])
                test_block_nums = np.array(task_blocks[np.isin(types_per_this_task, 'test')])

                train_phase_nums = np.array([int(phase_info.loc[b, 'phase_number']) for b in train_block_nums])
                test_phase_nums = np.array([int(phase_info.loc[b, 'phase_number']) for b in test_block_nums])

                # TODO: Are multiple training blocks of the same task ok for transfer matrix calculation?
                if len(train_block_nums) > 1:
                    raise Exception('Too many training instances of task: {:s}'.format(task))
                train_phase_num = train_phase_nums[0]

                if any(test_phase_nums < train_phase_num):
                    block_nums_to_add = test_block_nums[np.where(test_phase_nums < train_phase_num)]
                    [tasks_for_transfer_matrix['forward'].append((task, f)) for f in block_nums_to_add]

                if any(test_phase_nums > train_phase_num):
                    block_nums_to_add = test_block_nums[np.where(test_phase_nums > train_phase_num)]
                    [tasks_for_transfer_matrix['reverse'].append((task, f)) for f in block_nums_to_add]

        return ste_dict, tasks_for_transfer_matrix, name_map

    def calculate(self, data, metadata, metrics_dict):
        # Load Single Task Expert performance, and figure out where we should calculate transfer
        # based on what STEs we have. Can only calculate forward/reverse transfer for tasks which have an STE
        ste_dict, tasks_to_compute, task_name_map = self.validate(metadata)

        reverse_transfer = {}
        reverse_vals = []
        forward_transfer = {}
        forward_vals = []

        # Calculate, for each task, (task eval saturation / ste saturation)
        for task, block in tasks_to_compute['forward']:
            print('Computing forward transfer for {:s}'.format(task))
            this_transfer_val = metrics_dict['saturation_value'][block] / ste_dict[task_name_map[task]]
            forward_transfer[(task, block)] = this_transfer_val
            forward_vals.append(this_transfer_val)

        for task, block in tasks_to_compute['reverse']:
            print('Computing reverse transfer for {:s}'.format(task))
            this_transfer_val = metrics_dict['saturation_value'][block] / ste_dict[task_name_map[task]]
            reverse_transfer[(task, block)] = this_transfer_val
            reverse_vals.append(this_transfer_val)

        metrics_dict['forward_transfer'] = forward_transfer
        metrics_dict['reverse_transfer'] = reverse_transfer

        metric_to_return = {'global_forward_transfer': np.mean(forward_vals),
                            'global_reverse_transfer': np.mean(reverse_vals)}

        return metric_to_return, metrics_dict


class ClassificationMetricsReport(core.MetricsReport):

    def __init__(self, **kwargs):
        # Defines log_dir, syllabus_subtype, and initializes the _metrics list
        super().__init__(**kwargs)

        # Gets all data from the relevant log files
        self._log_data = util.read_log_data(util.get_l2root_base_dirs('logs', self.log_dir))
        _, self.phase_info = _localutil.parse_blocks(self._log_data)

        # Adds default metrics to list based on passed syllabus subtype
        self._add_default_metrics()

        # Initialize a results dictionary that can be returned at the end of the calculation step and an internal
        # dictionary that can be passed around for internal calculations
        self._results = {}
        self._metrics_dict = {}
        self._phase_info = None

    def _add_default_metrics(self):
        self.add(AveragedScorePerBatch())
        self.add(WithinBlockSaturation())
        self.add(LearningRate())

        if self.syllabus_subtype == "CL":
            self.add(RecoveryTime()) # This metric is under construction
            pass

        elif self.syllabus_subtype == "ANT_A":
            self.add(RecoveryTime())
            self.add(PerfMaintenanceANT())
            self.add(STERelativePerf())

        elif self.syllabus_subtype == "ANT_B":
            self.add(RecoveryTime())
            self.add(STERelativePerf())
            self.add(PerfMaintenanceANT())
            self.add(TransferMatrix())

        # This is an unhandled syllabus type as of right now
        elif self.syllabus_subtype == "ANT_C":
            raise Exception('This syllabus type ({:s}) will be handled in the future, but is not yet supported!'
                            .format(self.syllabus_subtype))

        else:
            raise Exception('Unhandled syllabus type {:s}! Supported syllabus types are: CL, ANT_A, and ANT_B, and STE'
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

            print(previously_calculated_metric_keys)

    def plot(self):
        # Ignore the rows indicating that a new batch was requested; only get evaluation rows
        relevant_dataframe = self._log_data[self._log_data['source'] == 'GET_LABELS']
        relevant_columns = _localutil.extract_relevant_columns(relevant_dataframe, keyword='score')
        print('Plotting a performance curve for each score column:')
        for col in relevant_columns:
            util.plot_performance(relevant_dataframe, col_to_plot=col, do_smoothing=True, input_xlabel='Batches',
                                  input_title=self.log_dir)

    def report(self):
        # Call a describe method to inform printing
        for r_key in self._results:
            print('\nMetric: {:s}'.format(r_key))
            # print('Averaged Value: {:s}'.format(str(self._results[r_key])))
            print('Per Block Values: {:s}'.format(str(self._metrics_dict[r_key])))

    def add(self, metrics_lst):
        self._metrics.append(metrics_lst)

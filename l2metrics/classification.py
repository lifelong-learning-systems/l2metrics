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
        pass

    def calculate(self, dataframe, phase_info, metrics_df):
        average_scores = {}

        # Iterate over all of the blocks and compute the within block performance
        for idx in range(phase_info.loc[:, 'block'].max() + 1):
            # Need to get the part of the data corresponding to the block
            block_data = dataframe.loc[dataframe["block"] == idx]
            # Make within block calculations
            relevant_columns = _localutil.extract_relevant_columns(block_data, keyword='score')

            for col in relevant_columns:

                average_scores[idx] = block_data.loc[block_data['source'] == 'GET_LABELS', col].mean()

            metrics_df = _localutil.fill_metrics_df(average_scores, 'average_scores', metrics_df, dict_key=col)

        return metrics_df


class WithinBlockSaturation(ClassificationMetric):
    name = "Average Within Block Saturation Calculation"
    capability = "continual_learning"
    requires = {'syllabus_type': 'class', 'syllabus_subtype': 'CL'}
    description = "Calculates the performance within each block"

    def __init__(self):
        super().__init__()
        # self.validate()

    def validate(self, phase_info):
        pass

    def calculate(self, dataframe, phase_info, metrics_df):
        relevant_columns = _localutil.extract_relevant_columns(dataframe, keyword='score')
        saturation_values = {}
        eps_to_saturation = {}

        for col in relevant_columns:
            metrics_df[col]['saturation_value'] = np.full_like(metrics_df[col]['block'], np.nan, dtype=np.double)
            metrics_df[col]['eps_to_saturation'] = np.full_like(metrics_df[col]['block'], np.nan, dtype=np.double)
            # Iterate over all of the blocks and compute the within block performance
            for idx in range(phase_info.loc[:, 'block'].max() + 1):
                # Get the part of the data corresponding to the relevant block
                block_data = dataframe.loc[dataframe["block"] == idx]
                # Make within block calculations
                sat_value, eps_to_sat, _ = _localutil.get_block_saturation_perf(block_data, column_to_use=col)
                # Record them
                saturation_values[idx] = sat_value
                eps_to_saturation[idx] = eps_to_sat

            metrics_df = _localutil.fill_metrics_df(saturation_values, 'saturation_value', metrics_df, dict_key=col)
            metrics_df = _localutil.fill_metrics_df(eps_to_saturation, 'eps_to_saturation', metrics_df, dict_key=col)

        return metrics_df


class AverageLearning(ClassificationMetric):
    name = "Average Score During Learning"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent', 'syllabus_subtype': 'ANT_A'}
    description = "Calculates the average reward accumulated until the system achieves saturation"

    def __init__(self):
        super().__init__()

    def validate(self, phase_info):
        pass

    def calculate(self, dataframe, phase_info, metrics_df):
        metrics_df['average_learning'] = {}
        train_block_ids = phase_info.loc[phase_info['phase_type'] == 'train', 'block'].values
        relevant_columns = _localutil.extract_relevant_columns(dataframe, keyword='score')
        average_learning = {}

        for col in relevant_columns:
            for tr_block in train_block_ids:
                num_episodes = metrics_df[col]['eps_to_saturation'][tr_block]

                if num_episodes is np.NaN:
                    score_until_saturation = 0

                else:
                    block_data = dataframe.loc[dataframe['block'] == tr_block]
                    first_episode_num = block_data.iloc[0]['task']
                    saturation_task_num = num_episodes + first_episode_num

                    score_until_saturation = block_data.loc[block_data['task'] <= saturation_task_num, col].sum()

                average_learning[tr_block] = score_until_saturation / num_episodes

            metrics_df = _localutil.fill_metrics_df(average_learning, 'average_learning', metrics_df, dict_key=col)

        return metrics_df


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
        task_map, block_list, name_map, type_map = _localutil.simplify_classification_task_names(unique_tasks, phase_info)

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

    def calculate(self, dataframe, phase_info, metrics_df):
        # Get the places where we should calculate recovery time
        recovery_time = {}
        tr_inds_to_use, tr_inds_to_assess = self.validate(phase_info)
        if len(tr_inds_to_assess) == 0:
            return metrics_df
        relevant_columns = _localutil.extract_relevant_columns(dataframe, keyword='score')

        for col in relevant_columns:
            for use_ind, assess_ind in zip(tr_inds_to_use, tr_inds_to_assess):
                prev_val = metrics_df[col]['saturation_value'][use_ind]
                block_data = dataframe.loc[dataframe['block'] == assess_ind]
                _, _, eps_to_rec = _localutil.get_block_saturation_perf(block_data,
                                                                        column_to_use=col,
                                                                        previous_saturation_value=prev_val)

                recovery_time[assess_ind] = eps_to_rec

            metrics_df = _localutil.fill_metrics_df(recovery_time, 'recovery_time', metrics_df, dict_key=col)

        return metrics_df


class PerfDifferenceANT(ClassificationMetric):
    name = "Performance Difference from Previously Trained Task Performance"
    capability = "adapting_to_new_tasks"
    requires = {'syllabus_type': 'class', 'syllabus_subtype': 'ANT'}
    description = "Calculates the performance of each task, in each evaluation block, " \
                  "relative to the previously trained task"

    def __init__(self):
        super().__init__()
        # self.validate()

    def validate(self, phase_info):
        pass

    def calculate(self, data, phase_info, metrics_df):
        # This metric must compute in each evaluation block the performance of the tasks
        # relative to the previously trained ones
        # Initialize some variables
        previously_trained_tasks = np.array([])
        previously_trained_task_ids = np.array([])
        saturation_difference = {}
        eps_to_sat_diff = {}

        # Get the simplified version of task names in classification logs
        unique_tasks = phase_info.loc[:, 'task_name'].unique()
        task_map, block_list, name_map, type_map = _localutil.simplify_classification_task_names(unique_tasks, phase_info)

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

                        # TODO: Should handle multiple comparison points?
                        block_ids_to_compare = previously_trained_task_ids[inds_where_task]
                        previously_trained_sat_values = metrics_df[col]['saturation_value'][block_ids_to_compare[0]]
                        previously_trained_num_eps_to_sat = metrics_df[col]['eps_to_saturation'][block_ids_to_compare[0]]

                        new_sat_value = metrics_df[col]['saturation_value'][this_phase_test_task_ids[idx]]
                        new_num_eps_to_sat = metrics_df[col]['eps_to_saturation'][this_phase_test_task_ids[idx]]

                        this_sat_val_comparison = previously_trained_sat_values - new_sat_value
                        this_num_eps_to_sat_comparison = previously_trained_num_eps_to_sat - new_num_eps_to_sat

                        block_id = this_phase_test_task_ids[idx]

                        saturation_difference[block_id] = this_sat_val_comparison
                        eps_to_sat_diff[block_id] = this_num_eps_to_sat_comparison

                metrics_df = _localutil.fill_metrics_df(saturation_difference, 'saturation_diff', metrics_df, dict_key=col)
                metrics_df = _localutil.fill_metrics_df(eps_to_sat_diff, 'eps_to_sat_diff', metrics_df, dict_key=col)

        return metrics_df


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

        return ste_dict

    def calculate(self, dataframe, phase_info, metrics_df):
        # Validate the STE
        ste_dict = self.validate(phase_info)
        ste_normalized_saturation = {}
        relevant_columns = _localutil.extract_relevant_columns(dataframe, keyword='score')

        for col in relevant_columns:
            for idx in range(phase_info.loc[:, 'block'].max()):
                if phase_info.loc[idx, 'phase_type'] != 'train':
                    continue
                # Get which task this block is and grab the STE performance for that task
                this_task = phase_info.loc[idx, "task_name"]
                this_ste_comparison = ste_dict[this_task]

                # Compare the saturation value of this block to the STE performance and store it
                ste_normalized_saturation[idx] = metrics_df[col]["saturation_value"][idx] / this_ste_comparison

            metrics_df = _localutil.fill_metrics_df(ste_normalized_saturation, 'STE_normalized_saturation', metrics_df, dict_key=col)

        return metrics_df


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

        task_map, block_list, name_map, type_map = _localutil.simplify_classification_task_names(unique_tasks, phase_info)

        for task in task_map.keys():
            task_blocks = task_map[task]
            types_per_this_task = [type_map[blk] for blk in task_blocks]

            if 'train' in types_per_this_task and 'test' in types_per_this_task:
                # Check eligibility for both forward and reverse transfer
                train_block_nums = np.array(task_blocks[np.isin(types_per_this_task, 'train')])
                test_block_nums = np.array(task_blocks[np.isin(types_per_this_task, 'test')])

                train_phase_nums = np.array([int(phase_info.loc[b, 'phase_number']) for b in train_block_nums])
                test_phase_nums = np.array([int(phase_info.loc[b, 'phase_number']) for b in test_block_nums])

                # TODO: Are multiple training blocks of the same task ok for transfer matrix calculation? Right now, no
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

    def calculate(self, dataframe, metadata, metrics_df):
        # Load Single Task Expert performance, and figure out where we should calculate transfer
        # based on what STEs we have. Can only calculate forward/reverse transfer for tasks which have an STE
        ste_dict, tasks_to_compute, task_name_map = self.validate(metadata)
        relevant_columns = _localutil.extract_relevant_columns(dataframe, keyword='score')
        reverse_transfer = {}
        forward_transfer = {}

        # Calculate, for each task, (task eval saturation / ste saturation)
        for col in relevant_columns:
            for task, block in tasks_to_compute['forward']:
                print('Computing forward transfer for {:s}'.format(task))
                this_transfer_val = metrics_df[col]['saturation_value'][block] / ste_dict[task_name_map[task]]
                forward_transfer[block] = this_transfer_val

            for task, block in tasks_to_compute['reverse']:
                print('Computing reverse transfer for {:s}'.format(task))
                this_transfer_val = metrics_df[col]['saturation_value'][block] / ste_dict[task_name_map[task]]
                reverse_transfer[block] = this_transfer_val

        metrics_df = _localutil.fill_metrics_df(forward_transfer, 'forward_transfer', metrics_df, dict_key=col)
        metrics_df = _localutil.fill_metrics_df(reverse_transfer, 'reverse_transfer', metrics_df, dict_key=col)

        return metrics_df


class ClassificationMetricsReport(core.MetricsReport):

    def __init__(self, **kwargs):
        # Defines log_dir, syllabus_subtype, and initializes the _metrics list
        super().__init__(**kwargs)

        # Gets all data from the relevant log files
        self._log_data = util.read_log_data(self.log_dir)
        _, self.phase_info = _localutil.parse_blocks(self._log_data)

        # Adds default metrics to list based on passed syllabus subtype
        self._add_default_metrics()

        # Initialize a results dictionary that can be returned at the end of the calculation step and an internal
        # dictionary that can be passed around for internal calculations
        self._metrics_df = {}
        phase_info_keys_to_include = ['phase_number', 'phase_type', 'task_name', 'block']
        if len(self.phase_info.loc[:, 'param_set'].unique()) > 1:
            phase_info_keys_to_include.append('param_set')

        cols = _localutil.extract_relevant_columns(self._log_data[self._log_data['source'] == 'GET_LABELS'],
                                                   keyword='score')

        for col in cols:
            self._metrics_df[col] = self.phase_info[phase_info_keys_to_include].copy()
            simple_names = _localutil.get_simple_class_task_names(self._metrics_df[col].loc[:, 'task_name'])
            self._metrics_df[col]['task_name'] = simple_names.values()

    def _add_default_metrics(self):
        self.add(AveragedScorePerBatch())
        self.add(WithinBlockSaturation())
        self.add(AverageLearning())

        if self.syllabus_subtype == "CL":
            self.add(RecoveryTime())  # This metric is under construction

        elif self.syllabus_subtype == "ANT_A":
            self.add(RecoveryTime())
            self.add(PerfDifferenceANT())
            self.add(STERelativePerf())

        elif self.syllabus_subtype == "ANT_B":
            self.add(RecoveryTime())
            self.add(STERelativePerf())
            self.add(PerfDifferenceANT())
            self.add(TransferMatrix())

        # This is an unhandled syllabus type as of right now
        elif self.syllabus_subtype == "ANT_C":
            raise Exception('This syllabus type ({:s}) will be handled in the future, but is not yet supported!'
                            .format(self.syllabus_subtype))

        else:
            raise Exception('Unhandled syllabus type {:s}! Supported syllabus types are: CL, ANT_A, and ANT_B, and STE'
                            .format(self.syllabus_subtype))

    def calculate(self):
        for metric in self._metrics:
            self._metrics_df = metric.calculate(self._log_data, self.phase_info, self._metrics_df)

    def report(self, save=False):
        # Print out the metrics per performance column in the dataframe
        for key in self._metrics_df.keys():
            print(tabulate(self._metrics_df[key], headers='keys', tablefmt='psql'))

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
        # Ignore the rows indicating that a new batch was requested; only get evaluation rows
        relevant_dataframe = self._log_data[self._log_data['source'] == 'GET_LABELS']
        relevant_columns = _localutil.extract_relevant_columns(relevant_dataframe, keyword='score')
        print('Plotting a performance curve for each score column:')
        for col in relevant_columns:
            util.plot_performance(relevant_dataframe, col_to_plot=col, do_smoothing=True, input_xlabel='Batches',
                                  do_save_fig=save, input_title=self.log_dir)

    def add(self, metrics_lst):
        self._metrics.append(metrics_lst)

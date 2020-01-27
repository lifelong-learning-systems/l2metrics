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
        # TODO: Add structure validation of phase_info
        pass


class GlobalMean(ClassificationMetric):
    name = "Global Mean Classification Performance"
    capability = "continual_learning"
    requires = {'syllabus_type': 'type2'}
    description = "Calculates the performance across all tasks and phases"

    def __init__(self):
        super().__init__()
        # self.validate()

    def validate(self):
        pass

    def calculate(self, dataframe, phase_info, metrics_dict):
        source_column = "GET_LABELS"

        # This could be moved to the validate method in the future
        relevant_columns, num_cols = _localutil.extract_relevant_columns(dataframe, keyword='score')
        if num_cols != 1:
            raise Exception('Wrong number of performance columns!')

        col = relevant_columns[0]

        data_rows = dataframe.loc[dataframe["source"] == source_column]
        global_perf_cross_blocks = data_rows[col].mean()
        metrics_dict = {"global_perf": global_perf_cross_blocks}

        return {'global_perf': global_perf_cross_blocks}, metrics_dict


class WithinBlockSaturation(ClassificationMetric):
    name = "Average Within Block Saturation Calculation"
    capability = "continual_learning"
    requires = {'syllabus_type': 'class', 'syllabus_subtype': 'CL'}
    description = "Calculates the performance within each block"

    def __init__(self):
        super().__init__()
        # self.validate()

    def validate(self, phase_info):
        # TODO: Add structure validation of phase_info
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
            relevant_columns, num_cols = _localutil.extract_relevant_columns(block_data, keyword='score')
            # TODO: Put in a validation/check of relevant columns
            sat_value, eps_to_sat, _ = _localutil.get_block_saturation_performance(block_data, column_to_use=relevant_columns[0])

            # Record them
            saturation_value[idx] = sat_value
            all_sat_vals.append(sat_value)
            eps_to_saturation[idx] = eps_to_sat
            all_eps_to_sat.append(eps_to_sat)

        metrics_dict = {"saturation_value": saturation_value, "eps_to_saturation": eps_to_saturation}
        metric_to_return = {'global_within_block_saturation': np.mean(all_sat_vals),
                            'global_num_eps_to_saturation': np.mean(all_eps_to_sat)}

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

        for idx in range(phase_info.loc[:, 'block'].max()):
            if phase_info.loc[idx, 'phase_type'] != 'train':
                continue
            # Get which task this block is and grab the STE performance for that task
            this_task = phase_info.loc[idx, "task_name"]
            this_ste_comparison = ste_dict[this_task]

            # Compare the saturation value of this block to the STE performance and store it

            STE_normalized_saturation[idx] = metrics_dict["saturation_value"][idx] / this_ste_comparison
            all_STE_normalized_saturations.append(STE_normalized_saturation[idx])

        metrics_dict["STE_normalized_saturation"] = STE_normalized_saturation
        metric_to_return = {'global_STE_normalized_saturation': np.mean(all_STE_normalized_saturations)}

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
        if self.syllabus_subtype == "CL":
            self.add(GlobalMean())

        elif self.syllabus_subtype == "ANT_A":
            self.add(WithinBlockSaturation())
            self.add(STERelativePerf())

        elif self.syllabus_subtype == "ANT_B":
            self.add(WithinBlockSaturation())
            self.add(STERelativePerf())

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

    def plot(self):
        pass

    def report(self):
        # Call a describe method to inform printing
        for r_key in self._results:
            print('\nMetric: {:s}'.format(r_key))
            print('Value: {:s}'.format(str(self._results[r_key])))

    def add(self, metrics_lst):
        self._metrics.append(metrics_lst)

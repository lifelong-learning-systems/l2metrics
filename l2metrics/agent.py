from . import core, util, _localutil
import numpy as np

"""
Standard metrics for Agent Learning (RL tasks)
"""


class AgentMetric(core.Metric):
    """
    A single metric for an Agent (aka. Reinforcement Learning) learner
    """
    def __init__(self):
        pass
        # self.validate()

    def plot(self, result):
        pass

    def validate(self, phase_info):
        pass

class GlobalMean(AgentMetric):
    name = "Global Mean Performance"
    capability = "continual_learning"
    requires = {'syllabus_type': 'agent', 'syllabus_subtype': 'all'}
    description = "Calculates the performance across all tasks and phases"

    def __init__(self):
        super().__init__()
        self.validate()

    def validate(self, phase_info):
        pass

    def calculate(self, dataframe, phase_info, metrics_dict):
        return {'global_perf': np.mean(dataframe.loc[:, "reward"])}


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
        base_query_str = 'block == '

        # Iterate over all of the blocks and compute the within block performance
        for idx in range(phase_info.loc[:, 'block'].max()):
            # Need to get the part of the data corresponding to the block
            block_data = _localutil.query_dataframe(dataframe, base_query_str + str(idx))

            # Actually make within block calculations
            sat_value, eps_to_sat, _ = _localutil.get_block_saturation_performance(block_data)

            # Record them in the correct row
            metrics_dict["saturation_value"] = sat_value
            metrics_dict["eps_to_saturation"] = eps_to_sat

        metric_to_return = {'global_within_block_saturation': np.mean(metrics_dict["saturation_value"]),
                            'global_num_eps_to_saturation': np.mean(metrics_dict["eps_to_saturation"])}

        return metric_to_return, metrics_dict


class STERelativePerf(AgentMetric):
    name = "Performance relative to S.T.E"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent', 'syllabus_subtype': 'ANT_B'}
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

        for idx in range(phase_info.loc[:, 'block'].max()):
            # Get which task this block is and grab the STE performance for that task
            this_task = phase_info.loc[idx, "task_name"]
            this_ste_comparison = ste_dict[this_task]

            # Compare the saturation value of this block to the STE performance and store it
            metrics_dict["STE_normalized_saturation"] = metrics_dict["saturation_value"] / this_ste_comparison

        metric_to_return = {'global_STE_normalized_saturation': np.mean(metrics_dict["STE_normalized_saturation"])}

        return metric_to_return, metrics_dict


class PerfMaintenance(AgentMetric):
    name = "Performance Maintenance relative to previously trained task"
    capability = "continual_learning"
    requires = {'syllabus_type': 'agent', 'syllabus_subtype': 'CL'}
    description = "Calculates the performance of each task, in each evaluation block, " \
                  "relative to the previously trained task"

    def __init__(self):
        super().__init__()
        self.validate()

    def validate(self, phase_info):
        pass

    def calculate(self, dataframe, phase_info, metrics_dict):
        pass


class AgentMetricsReport(core.MetricsReport):
    """
    Aggregates a list of metrics for an Agent learner
    """
    def __init__(self, **kwargs):
        # Defines log_dir, syllabus_subtype, and initializes the _metrics list
        super().__init__(**kwargs)

        # Gets all data from the relevant log files
        self._log_data = util.read_log_data(util.get_l2root_base_dirs('logs', self.log_dir))
        _, self.phase_info = _localutil.parse_blocks(self._log_data)

        # Adds default metrics to list based on passed syllabus subtype
        # TODO: Pass the phase_info to metrics constructor so that the metric can check what it's looking for
        self._add_default_metrics()

        # Do an initial check to make sure that reward has been logged
        if 'reward' not in self._log_data.columns:
            raise Exception('Reward column is required in the log data!')

        # Initialize a results dictionary that can be returned at the end of the calculation step
        self._results = {}
        self._phase_info = None

    def _add_default_metrics(self):
        # TODO: Add validation in the constructors to make sure syllabus has expected structure
        if self.syllabus_subtype == "CL":
            self.add(WithinBlockSaturation())

        elif self.syllabus_subtype == "ANT_A":
            self.add(WithinBlockSaturation())
            self.add(STERelativePerf())

        elif self.syllabus_subtype == "ANT_B":
            self.add(WithinBlockSaturation())
            self.add(STERelativePerf())

        elif self.syllabus_subtype == "ANT_C":
            self.add(WithinBlockSaturation())
            self.add(STERelativePerf())

        else:
            raise NotImplementedError

    def calculate(self):
        # TODO: Refactor this such that phase info is not modified by each metric and instead modifies a dictionary
        metrics_dict = {}

        for metric in self._metrics:
            this_result, metrics_dict = metric.calculate(self._log_data, self.phase_info, metrics_dict)
            self._results[metric.name] = this_result

    def report(self):
        # Call a describe method to inform printing
        for r_key in self._results:
            print('\nMetric: {:s}'.format(r_key))
            print('Value: {:s}'.format(str(self._results[r_key])))

    def plot(self):
        # TODO: Actually, you know, implement plotting
        pass

    def add(self, metrics_list):
        self._metrics.append(metrics_list)




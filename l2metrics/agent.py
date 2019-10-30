from . import core, util, _localutil
import numpy as np

"""
Standard metrics for Agent Learning (RL tasks)
"""


class AgentMetric(core.Metric):
    """
    A single metric for an Agent (aka. Reinforcement Learning) learner
    """


class AgentMetric1(AgentMetric):
    name = "Global Mean Performance"
    capability = "continual_learning"
    requires = {'syllabus_subtype': 'all'}
    description = "Calculates the performance across all tasks and phases"
    
    def calculate(self, dataframe, info):
        return {'global_perf': np.mean(dataframe["perf"])}


class AgentMetric2(AgentMetric):
    name = "Average Within Block Saturation Calculation"
    capability = "continual_learning"
    requires = {'syllabus_subtype': 'ANT_A'}
    description = "Calculates the performance within each block"

    def calculate(self, dataframe, phase_info):
        # Initialize the columns of the phase info dataframe
        phase_info["saturation_value"] = np.nan
        phase_info["eps_to_saturation"] = np.nan

        max_block_num = dataframe.loc[:, 'block'].max()
        base_query_str = 'block == '

        # Iterate over the blocks and compute the within block performance
        for idx in range(max_block_num):
            # Need to get the index of the phase_info so we can update it
            phase_info_index = phase_info.index[phase_info["block"] == idx]

            # Need to get the part of the data corresponding to the block
            block_data = _localutil.query_dataframe(dataframe, base_query_str + str(idx))

            # Actually make within block calculations
            sat_value, eps_to_sat, _ = _localutil.get_block_saturation_performance(block_data)

            # Record them in the correct row
            phase_info.loc[phase_info_index, "saturation_value"] = sat_value
            phase_info.loc[phase_info_index, "eps_to_saturation"] = eps_to_sat

        metric_to_return = {'global_within_block_saturation': np.mean(phase_info.loc[:, "saturation_value"]),
                            'global_num_eps_to_saturation': np.mean(phase_info.loc[:, "eps_to_saturation"])}
        return metric_to_return, phase_info


class AgentMetric3(AgentMetric):
    name = "Performance relative to S.T.E"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_subtype': 'ANT_B'}
    description = "Calculates the performance of each task, in each evaluation, relative to a single task expert"
    
    def calculate(self, dataframe):
        ste_dict = util.load_default_ste_data()
        unique_tasks = dataframe.loc[:, 'class_name'].unique()
        result = dict()

        # Make sure I have STE baselines available for all tasks, else complain
        if unique_tasks.any() not in ste_dict:
            raise Exception

        for task in unique_tasks:
            # TODO: Do STE comparison work here. I can assume that if I tell the metrics to first calculate the within block
            # performance, I can then have access to that information in the phase_info dataframe passed in
            rows = dataframe['phase'] == "eval" & dataframe['task'] == task
            result[task] = np.mean(rows["perf"])
        return result


class AgentMetricsReport(core.MetricsReport):
    """
    Aggregates a list of metrics for an Agent learner
    """
    def __init__(self, **kwargs):
        # Defines log_dir, syllabus_subtype, and initializes the _metrics list
        super().__init__(**kwargs)

        # Gets all data from the relevant log files
        self._log_data = util.read_log_data(util.get_l2root_base_dirs('logs', self.log_dir))

        # Adds default metrics to list based on passed syllabus subtype
        self.default_metrics()

        # Initializes a results dictionary that can be returned at the end of the calculation step
        self._results = {}

    def default_metrics(self):
        # TODO: Add preliminary sanity checks to make sure syllabus has expected structure
        if self.syllabus_subtype == "CL":
            self.add(AgentMetric2)

        elif self.syllabus_subtype == "ANT_A":
            self.add(AgentMetric2)

        elif self.syllabus_subtype == "ANT_B":
            self.add(AgentMetric2)

        elif self.syllabus_subtype == "ANT_C":
            self.add(AgentMetric2)

        else:
            raise NotImplementedError

    def calculate(self):
        _, phase_info = _localutil.parse_blocks(self._log_data)

        for metric in self._metrics:
            result, phase_info = metric.calculate(metric, self._log_data, phase_info=phase_info)
            self._results[metric.name] = result

    def plot(self):
        # TODO: Actually, you know, implement plotting
        metric_results = [self._results[r_key] for r_key in self._results]
        print(metric_results)

    def add(self, metrics_lst):
        self._metrics.append(metrics_lst)




import argparse
import l2metrics
from l2metrics import _localutil


class MyCustomAgentMetric(l2metrics.AgentMetric):
    name = "An Example Custom Metric"
    capability = "continual_learning"
    requires = {'syllabus_type': 'agent'}
    description = "A Custom Metric"
    
    def calculate(self, dataframe, phase_info, metrics_dict):
        return {'global_perf': dataframe.loc[:, "reward"].mean()}, {'global_perf': dataframe.loc[:, "reward"].mean()}

    def plot(self, result):
        pass

    def validate(self, phase_info):
        # TODO: Add structure validation of phase_info
        pass


class MyCustomClassMetric(l2metrics.ClassificationMetric):
    name = "An Example Custom Metric"
    capability = "continual_learning"
    requires = {'syllabus_type': 'class'}
    description = "A Custom Metric"

    def calculate(self, dataframe, phase_info, metrics_dict):
        source_column = "GET_LABELS"

        # This could be moved to the validate method in the future
        relevant_columns = _localutil.extract_relevant_columns(dataframe, keyword='score')
        if len(relevant_columns) < 1:
            raise Exception('Not enough performance columns!')

        metrics_dict['avg_across_blocks'] = {}

        for col in relevant_columns:
            data_rows = dataframe.loc[dataframe["source"] == source_column]
            global_perf_cross_blocks = data_rows[col].mean()
            metrics_dict['avg_across_blocks'][col] = global_perf_cross_blocks

        metric_to_return = {c: metrics_dict['avg_across_blocks'][c] for c in relevant_columns}

        return {'global_perf_per_column': metric_to_return}, metrics_dict

    def plot(self, result):
        pass

    def validate(self, phase_info):
        pass


def run():
    parser = argparse.ArgumentParser(description='Run L2Metrics from the command line')

    # We assume that the logs are found in a subdirectory under $L2DATA/logs - this subdirectory must be passed as a
    # parameter in order to locate the logs which will be parsed by this code
    parser.add_argument('-log_dir', default=None, help='Subdirectory under $L2DATA/logs for the log files')

    # Choose syllabus type "agent" for Agent-based environments, and "class" for Classification-based environments
    parser.add_argument('-syllabus_type', choices=["agent", "class"],  default="agent", help='Type of learner '
                                                                                             'used in the syllabus')

    # Syllabus_subtype refers to the structure of the syllabus and will determine the default list of metrics calculated
    # where CL = Continual Learning; ANT_A = Adapting to New Tasks, type A; ANT_B = Adapting to New Tasks, type B; etc.
    # Please refer to the documentation for more details on the distinction between these types.
    parser.add_argument('-syllabus_subtype', choices=["CL", "ANT_A", "ANT_B", "ANT_C"],  default=None,
                        help='Subtype of syllabus')

    args = parser.parse_args()

    if args.log_dir is None:
        raise Exception('Log directory must be specified!')

    if args.syllabus_type == "class":
        metrics_report = l2metrics.ClassificationMetricsReport(log_dir=args.log_dir, syllabus_subtype=args.syllabus_subtype)
        metrics_report.add(MyCustomClassMetric())
    else:
        metrics_report = l2metrics.AgentMetricsReport(log_dir=args.log_dir, syllabus_subtype=args.syllabus_subtype)
        metrics_report.add(MyCustomAgentMetric())

    metrics_report.calculate()

    # Uncomment this for a plot of performance:
    metrics_report.plot()

    metrics_report.report()


if __name__ == "__main__":
    run()

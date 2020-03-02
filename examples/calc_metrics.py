import argparse
import l2metrics
from l2metrics import _localutil
import os


class MyCustomAgentMetric(l2metrics.AgentMetric):
    name = "An Example Custom Metric for illustration"
    capability = "continual_learning"
    requires = {'syllabus_type': 'agent'}
    description = "Records the maximum value per block in the dataframe"
    
    def calculate(self, dataframe, phase_info, metrics_df):
        max_values = {}

        for idx in range(phase_info.loc[:, 'block'].max() + 1):
            max_block_value = dataframe.loc[dataframe["block"] == idx, 'reward'].max()
            max_values[idx] = max_block_value

        # This is the line that fills the metric into the dataframe. Comment it out to suppress this behavior
        metrics_df = _localutil.fill_metrics_df(max_values, 'max_value', metrics_df)
        return metrics_df

    def plot(self, result):
        pass

    def validate(self, phase_info):
        pass


class MyCustomClassMetric(l2metrics.ClassificationMetric):
    name = "An Example Custom Metric for illustration"
    capability = "continual_learning"
    requires = {'syllabus_type': 'class'}
    description = "Records the maximum value per block in the dataframe"

    def calculate(self, dataframe, phase_info, metrics_df):
        max_values = {}

        # This could be moved to the validate method in the future
        source_column = "GET_LABELS"
        relevant_columns = _localutil.extract_relevant_columns(dataframe, keyword='score')
        if len(relevant_columns) < 1:
            raise Exception('Not enough performance columns!')

        for col in relevant_columns:
            data_rows = dataframe.loc[dataframe["source"] == source_column]
            for idx in range(phase_info.loc[:, 'block'].max() + 1):
                max_values[idx] = data_rows.loc[dataframe['block'] == idx, col].max()

            # This is the line that fills the metric into the dataframe. Comment it out to suppress this behavior
            metrics_df = _localutil.fill_metrics_df(max_values, 'max_value', metrics_df, dict_key=col)

        return metrics_df

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
        # Here is where you add your custom metric to the list of metrics already being calculated
        metrics_report.add(MyCustomClassMetric())
    else:
        metrics_report = l2metrics.AgentMetricsReport(log_dir=args.log_dir, syllabus_subtype=args.syllabus_subtype)
        # Here is where you add your custom metric to the list of metrics already being calculated
        metrics_report.add(MyCustomAgentMetric())

    # Calculate metrics in order of their addition to the metrics list.
    metrics_report.calculate()

    # Comment this line out to suppress the generation of a performance plot. Will save by default (no visualization).
    metrics_report.plot()

    metrics_report.report()


if __name__ == "__main__":
    run()

import argparse
import l2metrics


class MyCustomMetric(l2metrics.AgentMetric):
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
    else:
        metrics_report = l2metrics.AgentMetricsReport(log_dir=args.log_dir, syllabus_subtype=args.syllabus_subtype)

    # metrics_report.add(MyCustomMetric())

    metrics_report.calculate()

    # Uncomment this for a very basic reward over episode plot:
    metrics_report.plot()

    metrics_report.report()


if __name__ == "__main__":
    run()

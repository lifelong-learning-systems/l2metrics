import argparse
import os

import l2metrics
from l2metrics import _localutil


def run():
    parser = argparse.ArgumentParser(description='Run L2Metrics from the command line')

    # We assume that the logs are found in a subdirectory under $L2DATA/logs - this subdirectory must be passed as a
    # parameter in order to locate the logs which will be parsed by this code
    parser.add_argument('-l', '--log_dir', help='Subdirectory under $L2DATA/logs for the log files')

    # Choose application measure to use as performance column
    parser.add_argument('-p', '--perf_measure', default="reward",
                        help='Name of column to use for metrics calculations')

    args = parser.parse_args()

    if args.log_dir is None:
        raise Exception('Log directory must be specified!')

    metrics_report = l2metrics.AgentMetricsReport(log_dir=args.log_dir, perf_measure=args.perf_measure)

    # Calculate metrics in order of their addition to the metrics list.
    metrics_report.calculate()

    # Plot metrics
    metrics_report.plot(save=True)

    # Print table of metrics and save values to file
    metrics_report.report(save=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f'Error: {e}')

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

"""
This script illustrates how to produce a lifelong learning metrics report with a
custom metric.
"""

import argparse

import pandas as pd
from l2metrics import _localutil
from l2metrics.core import Metric
from l2metrics.report import MetricsReport


class MyCustomAgentMetric(Metric):
    name = "An Example Custom Metric for illustration"
    capability = "continual_learning"
    requires = {'syllabus_type': 'agent'}
    description = "Records the maximum value per regime in the dataframe"
    
    def __init__(self, perf_measure: str):
        super().__init__()
        self.perf_measure = perf_measure

    def validate(self, block_info):
        pass

    def calculate(self, dataframe: pd.DataFrame, block_info: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
        max_values = {}

        for idx in range(block_info.loc[:, 'regime_num'].max() + 1):
            max_regime_value = dataframe.loc[dataframe['regime_num'] == idx, self.perf_measure].max()
            max_values[idx] = max_regime_value

        # This is the line that fills the metric into the dataframe. Comment it out to suppress this behavior
        metrics_df = _localutil.fill_metrics_df(
            max_values, 'max_value', metrics_df)
        return metrics_df


def run() -> None:
    # Instantiate parser
    parser = argparse.ArgumentParser(description='Run L2Metrics from the command line')

    # Log directories can be absolute paths, relative paths, or paths found in $L2DATA/logs
    parser.add_argument('-l', '--log-dir', required=True,
                        help='Log directory of scenario')

    # Flag for storing log data as STE data
    parser.add_argument('-s', '--store-ste-data', action='store_true',
                        help='Flag for storing log data as STE')

    # Choose application measure to use as performance column
    parser.add_argument('-p', '--perf-measure', default='reward',
                        help='Name of column to use for metrics calculations')

    # Method for calculating performance maintenance
    parser.add_argument('-m', '--maintenance-method', default='mrlep', choices=['mrtlp', 'mrlep', 'both'],
                        help='Method for computing performance maintenance')

    # Method for calculating forward and backward transfer
    parser.add_argument('-t', '--transfer-method', default='contrast', choices=['contrast', 'ratio', 'both'],
                        help='Method for computing forward and backward transfer')

    # Method for normalization
    parser.add_argument('-n', '--normalization-method', default='task', choices=['task', 'run'],
                        help='Method for normalizing data')

    # TODO: Input file for task ranges

    # Mean and standard deviation for adding noise to log data
    parser.add_argument('--noise', default=[0, 0], metavar=('MEAN', 'STD'), nargs=2, type=float,
                        help='Mean and standard deviation for Gaussian noise in log data')

    # Output filename
    parser.add_argument('-o', '--output', default=None,
                        help='Specify output filename for plot and results')

    # Flag for disabling smoothing
    parser.add_argument('--no-smoothing', action='store_true',
                        help='Do not smooth data for metrics and plotting')

    # Flag for enabling normalization
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize data for metrics')

    # Flag for removing outliers
    parser.add_argument('--remove-outliers', action='store_true',
                        help='Remove outliers in data for metrics')

    # Flag for disabling plotting
    parser.add_argument('--no-plot', action='store_true',
                        help='Do not plot performance')

    # Flag for disabling save
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save metrics outputs')

    # Parse arguments
    args = parser.parse_args()
    do_smoothing = not args.no_smoothing
    do_plot = not args.no_plot
    do_save = not args.no_save

    # Initialize metrics report
    report = MetricsReport(log_dir=args.log_dir, perf_measure=args.perf_measure,
                           maintenance_method=args.maintenance_method,
                           transfer_method=args.transfer_method,
                           normalization_method=args.normalization_method, do_smoothing=do_smoothing,
                           do_normalize=args.normalize, normalization_method=args.normalization_method,
                           remove_outliers=args.remove_outliers)

    # Add example of custom metric
    report.add(MyCustomAgentMetric(args.perf_measure))

    # Add noise to log data if mean or standard deviation is specified
    if args.noise[0] or args.noise[1]:
        report.add_noise(mean=args.noise[0], std=args.noise[1])

    # Calculate metrics in order of their addition to the metrics list.
    report.calculate()

    # Print table of metrics and save values to file
    report.report(save=do_save, output=args.output)

    # Plot metrics
    if do_plot:
        report.plot(save=do_save)
        report.plot_ste_data(save=do_save)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f'Error: {e}')

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
import json
import traceback

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
    parser.add_argument('-l', '--log-dir', default=None, type=str,
                        help='Log directory of scenario')

    # Flag for recursively calculating metrics on valid subdirectories within log directory 
    parser.add_argument('-R', '--recursive', action='store_true',
                        help='Recursively compute metrics on logs found in specified directory')

    # Mode for storing log data as STE data
    parser.add_argument('-s', '--ste-store-mode', default=None, choices=['w', 'a'],
                        help='Mode for storing log data as STE, overwrite (w) or append (a)')

    # Method for handling multiple STE runs
    parser.add_argument('-v', '--ste-averaging-method', default='time', choices=['time', 'metrics'],
                        help='Method for handling STE runs, time-series averaging (time) or '
                        'LL metric averaging (metrics)')

    # Choose application measure to use as performance column
    parser.add_argument('-p', '--perf-measure', default='reward', type=str,
                        help='Name of column to use for metrics calculations')

    # Method for aggregating within-lifetime metrics
    parser.add_argument('-a', '--aggregation-method', default='mean', type=str, choices=['mean', 'median'],
                        help='Method for aggregating within-lifetime metrics')

    # Method for calculating performance maintenance
    parser.add_argument('-m', '--maintenance-method', default='mrlep', type=str, choices=['mrlep', 'mrtlp', 'both'],
                        help='Method for computing performance maintenance')

    # Method for calculating forward and backward transfer
    parser.add_argument('-t', '--transfer-method', default='ratio', type=str, choices=['ratio', 'contrast', 'both'],
                        help='Method for computing forward and backward transfer')

    # Method for normalization
    parser.add_argument('-n', '--normalization-method', default='task', type=str, choices=['task', 'run', 'none'],
                        help='Method for normalizing data')

    # Method for smoothing
    parser.add_argument('-g', '--smoothing-method', default='flat', type=str, choices=['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'none'],
                        help='Method for smoothing data, window type')

    # Window length for smoothing
    parser.add_argument('-w', '--window-length', default=None, type=int,
                        help='Window length for smoothing data')

    # Flag for removing outliers
    parser.add_argument('-x', '--clamp-outliers', action='store_true',
                        help='Remove outliers in data for metrics by clamping to quantiles')

    # Data range file for normalization
    parser.add_argument('-d', '--data-range-file', default=None, type=str,
                        help='JSON file containing task performance ranges for normalization')

    # Mean and standard deviation for adding noise to log data
    parser.add_argument('-N', '--noise', default=[0, 0], metavar=('MEAN', 'STD'), nargs=2, type=float,
                        help='Mean and standard deviation for Gaussian noise in log data')

    # Output filename
    parser.add_argument('-o', '--output', default=None, type=str,
                        help='Specify output filename for plot and results')

    # Flag for showing raw performance data under smoothed data
    parser.add_argument('-r', '--show-raw-data', action='store_true',
                        help='Show raw data points under smoothed data for plotting')

    # Flag for showing evaluation block lines
    parser.add_argument('-e', '--show-eval-lines', dest='show_eval_lines', default=True, action='store_true',
                        help='Show lines between evaluation blocks')
    parser.add_argument('--no-show-eval-lines', dest='show_eval_lines', action='store_false',
                        help='Do not show lines between evaluation blocks')

    # Flag for enabling/disabling plotting
    parser.add_argument('-P', '--do-plot', dest='do_plot', default=True, action='store_true',
                        help='Plot performance')
    parser.add_argument('--no-plot', dest='do_plot', action='store_false',
                        help='Do not plot performance')

    # Flag for enabling/disabling save
    parser.add_argument('-S', '--do-save', dest='do_save', default=True, action='store_true',
                        help='Save metrics outputs')
    parser.add_argument('--no-save', dest='do_save', action='store_false',
                        help='Do not save metrics outputs')

    # Settings file arguments
    parser.add_argument('-c', '--load-settings', default='', type=str,
                        help='Load L2Metrics settings from JSON file')
    parser.add_argument('-C', '--do-save-settings', dest='do_save_settings', default=True, action='store_true',
                        help='Save L2Metrics settings to JSON file')
    parser.add_argument('--no-save-settings', dest='do_save_settings', action='store_false',
                        help='Do not save L2Metrics settings to JSON file')

    # Parse arguments
    args = parser.parse_args()
    kwargs = vars(args)

    if args.load_settings:
        with open(args.load_settings, 'r') as config_file:
            kwargs.update(json.load(config_file))

    # Load data range data for normalization and standardize names to lowercase
    if args.data_range_file:
        with open(args.data_range_file) as config_json:
            data_range = json.load(config_json)
            data_range = {key.lower(): val for key, val in data_range.items()}
    else:
        data_range = None

    kwargs['data_range'] = data_range

    # Initialize metrics report
    report = MetricsReport(**kwargs)

    # Add example of custom metric
    report.add(MyCustomAgentMetric(args.perf_measure))

    # Add noise to log data if mean or standard deviation is specified
    if args.noise[0] or args.noise[1]:
        report.add_noise(mean=args.noise[0], std=args.noise[1])

    # Calculate metrics in order of their addition to the metrics list.
    report.calculate()

    # Print table of metrics
    report.report()

    # Save metrics to file
    if args.do_save:
        report.save_metrics(filename=args.output)
        report.save_data(filename=args.output)

    # Plot metrics
    if args.do_plot:
        report.plot(save=args.do_save, show_raw_data=args.show_raw_data,
                    show_eval_lines=args.show_eval_lines)
        report.plot_ste_data(save=args.do_save)

    # Save settings used to run calculate metrics
        if args.do_save_settings:
            report.save_settings(filename=args.output)

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()

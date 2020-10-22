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

import argparse
import os

import l2metrics
from l2metrics import _localutil


class MyCustomAgentMetric(l2metrics.AgentMetric):
    name = "An Example Custom Metric for illustration"
    capability = "continual_learning"
    requires = {'syllabus_type': 'agent'}
    description = "Records the maximum value per regime in the dataframe"
    
    def __init__(self, perf_measure):
        super().__init__()
        self.perf_measure = perf_measure

    def calculate(self, dataframe, block_info, metrics_df):
        max_values = {}

        for idx in range(block_info.loc[:, 'regime_num'].max() + 1):
            max_regime_value = dataframe.loc[dataframe['regime_num'] == idx, self.perf_measure].max()
            max_values[idx] = max_regime_value

        # This is the line that fills the metric into the dataframe. Comment it out to suppress this behavior
        metrics_df = _localutil.fill_metrics_df(max_values, 'max_value', metrics_df)
        return metrics_df

    def plot(self, result):
        pass

    def validate(self, block_info):
        pass


def run():
    # Instantiate parser
    parser = argparse.ArgumentParser(description='Run L2Metrics from the command line')

    # Log directories can be absolute paths, relative paths, or paths found in $L2DATA/logs
    parser.add_argument('-l', '--log-dir', required=True, help='Log directory of scenario')

    # Flag for storing log data as STE data
    parser.add_argument('-s', '--store-ste-data', action='store_true',
                        help='Flag for storing log data as STE')

    # Choose application measure to use as performance column
    parser.add_argument('-p', '--perf-measure', default='reward',
                        help='Name of column to use for metrics calculations')
    
    # Method for calculating forward and backward transfer
    parser.add_argument('-m', '--transfer-method', default='contrast', choices=['contrast', 'ratio'],
                        help='Mathod for computing forward and backward transfer')

    # Output filename
    parser.add_argument('-o', '--output', default=None,
                        help='Specify output filename for plot and results')

    # Flag for disabling plotting
    parser.add_argument('--no-plot', action='store_true', help='Do not plot performance')

    # Flag for disabling save
    parser.add_argument('--no-save', action='store_true', help='Do not save metrics outputs')

    # Parse arguments
    args = parser.parse_args()
    do_plot = not args.no_plot
    do_save = not args.no_save

    # Initialize metrics report
    metrics_report = l2metrics.AgentMetricsReport(log_dir=args.log_dir, perf_measure=args.perf_measure,
                                              transfer_method=args.transfer_method)

    # Add example of custom metric
    metrics_report.add(MyCustomAgentMetric(args.perf_measure))

    # Calculate metrics in order of their addition to the metrics list.
    metrics_report.calculate()

    # Plot metrics
    if do_plot:
        metrics_report.plot(save=do_save, output=args.output)

    # Print table of metrics and save values to file
    metrics_report.report(save=do_save, output=args.output)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f'Error: {e}')

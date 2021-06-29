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


def init_parser():
    # Instantiate parser
    parser = argparse.ArgumentParser(description='Run L2Metrics from the command line')

    # Log directories can be absolute paths, relative paths, or paths found in $L2DATA/logs
    parser.add_argument('-l', '--log-dir', default=None, type=str,
                        help='Log directory of scenario. Defaults to None.')

    # Flag for recursively calculating metrics on valid subdirectories within log directory
    parser.add_argument('-R', '--recursive', action='store_true',
                        help='Recursively compute metrics on logs found in specified directory. \
                            Defaults to false.')

    # Mode for storing log data as STE data
    parser.add_argument('-s', '--ste-store-mode', default=None, choices=['w', 'a'],
                        help='Mode for storing log data as STE, overwrite (w) or append (a). \
                            Defaults to None.')

    # Method for handling multiple STE runs
    parser.add_argument('-v', '--ste-averaging-method', default='time', choices=['time', 'metrics'],
                        help='Method for handling STE runs, time-series averaging (time) or LL \
                            metric averaging (metrics). Defaults to time.')

    # Choose application measure to use as performance column
    parser.add_argument('-p', '--perf-measure', default='reward', type=str,
                        help='Name of column to use for metrics calculations. Defaults to reward.')

    # Method for aggregating within-lifetime metrics
    parser.add_argument('-a', '--aggregation-method', default='mean', type=str, choices=['mean', 'median'],
                        help='Method for aggregating within-lifetime metrics. Defaults to mean.')

    # Method for calculating performance maintenance
    parser.add_argument('-m', '--maintenance-method', default='mrlep', type=str, choices=['mrlep', 'mrtlp', 'both'],
                        help='Method for computing performance maintenance. Defaults to mrlep.')

    # Method for calculating forward and backward transfer
    parser.add_argument('-t', '--transfer-method', default='ratio', type=str, choices=['ratio', 'contrast', 'both'],
                        help='Method for computing forward and backward transfer. Defaults to ratio.')

    # Method for normalization
    parser.add_argument('-n', '--normalization-method', default='task', type=str, choices=['task', 'run', 'none'],
                        help='Method for normalizing data. Defaults to task.')

    # Method for smoothing
    parser.add_argument('-g', '--smoothing-method', default='flat', type=str, choices=['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'none'],
                        help='Method for smoothing data, window type. Defaults to flat.')

    # Window length for smoothing
    parser.add_argument('-w', '--window-length', default=None, type=int,
                        help='Window length for smoothing data. Defaults to None.')

    # Flag for removing outliers
    parser.add_argument('-x', '--clamp-outliers', action='store_true',
                        help='Remove outliers in data for metrics by clamping to quantiles. Defaults \
                            to false.')

    # Data range file for normalization
    parser.add_argument('-d', '--data-range-file', default=None, type=str,
                        help='JSON file containing task performance ranges for normalization. \
                            Defaults to None.')

    # Mean and standard deviation for adding noise to log data
    parser.add_argument('-N', '--noise', default=[0, 0], metavar=('MEAN', 'STD'), nargs=2, type=float,
                        help='Mean and standard deviation for Gaussian noise in log data. Defaults \
                            to [0, 0].')

    # Output directory
    parser.add_argument('-O', '--output-dir', default='results', type=str,
                        help='Directory for output files. Defaults to results.')

    # Output filename
    parser.add_argument('-o', '--output', default=None, type=str,
                        help='Specify output filename for plot and results. Defaults to None.')

    # Flag for showing raw performance data under smoothed data
    parser.add_argument('-r', '--show-raw-data', action='store_true',
                        help='Show raw data points under smoothed data for plotting. Defaults to \
                            false.')

    # Flag for showing evaluation block lines
    parser.add_argument('-e', '--show-eval-lines', dest='show_eval_lines', default=True, action='store_true',
                        help='Show lines between evaluation blocks. Defaults to true.')
    parser.add_argument('--no-show-eval-lines', dest='show_eval_lines', action='store_false',
                        help='Do not show lines between evaluation blocks')

    # Flag for enabling/disabling plotting
    parser.add_argument('-P', '--do-plot', dest='do_plot', default=True, action='store_true',
                        help='Plot performance. Defaults to true.')
    parser.add_argument('--no-plot', dest='do_plot', action='store_false',
                        help='Do not plot performance')

    # Flag for enabling/disabling save
    parser.add_argument('-S', '--do-save', dest='do_save', default=True, action='store_true',
                        help='Save metrics outputs. Defaults to true.')
    parser.add_argument('--no-save', dest='do_save', action='store_false',
                        help='Do not save metrics outputs')

    # Settings file arguments
    parser.add_argument('-c', '--load-settings', default=None, type=str,
                        help='Load L2Metrics settings from JSON file. Defaults to None.')
    parser.add_argument('-C', '--do-save-settings', dest='do_save_settings', default=True, action='store_true',
                        help='Save L2Metrics settings to JSON file. Defaults to true.')
    parser.add_argument('--no-save-settings', dest='do_save_settings', action='store_false',
                        help='Do not save L2Metrics settings to JSON file')

    return parser

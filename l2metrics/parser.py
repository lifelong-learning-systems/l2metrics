"""
Copyright © 2021 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

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

    # Method for handling task variants
    parser.add_argument('-r', '--variant-mode', default='aware', type=str, choices=['aware', 'agnostic'],
                        help='Mode for computing metrics with respect to task variants. \
                            Defaults to aware.')

    # Mode for storing log data as STE data
    parser.add_argument('-s', '--ste-store-mode', default=None, choices=['w', 'a'],
                        help='Mode for storing log data as STE, overwrite (w) or append (a). \
                            Defaults to None.')

    # Method for handling multiple STE runs
    parser.add_argument('-v', '--ste-averaging-method', default='metrics', choices=['metrics', 'time'],
                        help='Method for handling STE runs, LL metric averaging (metrics) or ' \
                            'time-series averaging (time). Defaults to metrics.')

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

    # Flag for smoothing evaluation block data
    parser.add_argument('-G', '--smooth-eval-data', dest='do_smooth_eval_data', default=False, action='store_true',
                        help='Smooth evaluation block data. Defaults to false.')

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

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
import traceback

from l2metrics import _localutil, util


def run():
    parser = argparse.ArgumentParser(description='Store single task expert data from the command line')

    # We assume that the logs are found in a subdirectory under $L2DATA/logs
    # This subdirectory must be passed as a parameter in order to locate the logs
    # which will be parsed by this code
    parser.add_argument('-l', '--log_dir', help='Subdirectory under $L2DATA/logs for the log files')

    # Choose application measure to use as performance column
    parser.add_argument('-p', '--perf_measure', default="reward",
                        help='Name of column to use for metrics calculations')

    args = parser.parse_args()

    if args.log_dir is None:
        raise Exception('Log directory must be specified!')

    # Do an initial check to make sure that reward is valid
    if args.perf_measure not in util.read_column_info(args.log_dir):
        raise Exception(f'Invalid performance measure: {args.perf_measure}')

    # Load data from ste logs
    log_data = util.read_log_data(args.log_dir, [args.perf_measure])
    log_data = log_data.sort_values(by=['regime_num', 'exp_num']).set_index("regime_num", drop=False)
    log_data = log_data[log_data['block_type'] == 'train']
    _, block_info = _localutil.parse_blocks(log_data)

    # Get task name
    task_name = log_data.task_name.unique()

    # Check for number of tasks in scenario
    if task_name.size != 1:
        raise Exception('Scenario does not only contain one task!')

    # Get base directory to store ste data
    filename = util.get_l2root_base_dirs('taskinfo', task_name[0] + '.pkl')

    # Store ste data in task info directory
    log_data.to_pickle(filename)

    print('Stored STE data for', task_name[0])

    # Plot data
    util.plot_performance(log_data, block_info, do_smoothing=True,
                          do_save_fig=False, max_smoothing_window=100, input_title=args.log_dir)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()

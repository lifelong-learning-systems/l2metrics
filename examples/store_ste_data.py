import argparse
from l2metrics import _localutil, util
import traceback


def run():
    parser = argparse.ArgumentParser(description='Store single task expert data from the command line')

    # We assume that the logs are found in a subdirectory under $L2DATA/logs
    # This subdirectory must be passed as a parameter in order to locate the logs
    # which will be parsed by this code
    parser.add_argument('-l', '--log_dir', help='Subdirectory under $L2DATA/logs for the log files')

    args = parser.parse_args()

    if args.log_dir is None:
        raise Exception('Log directory must be specified!')

    # Load data from ste logs
    log_data = util.read_log_data(args.log_dir)
    log_data = log_data.sort_values(
        by=['regime_num', 'exp_num']).set_index("regime_num", drop=False)

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
    util.plot_performance(log_data, do_smoothing=True, do_task_colors=True, do_save_fig=False,
                          max_smoothing_window=100, input_title=args.log_dir)


if __name__ == "__main__":
    try:
        run()
    except:
        traceback.print_exc()

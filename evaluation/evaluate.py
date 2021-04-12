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
This is a Python script for computing and aggregating lifelong learning metrics
across multiple runs of different learning scenarios.

Additionally, this script contains helper functions for the Jupyter notebook,
evaluation.ipynb.
"""

import argparse
import fnmatch
import json
import os
import traceback
import warnings
from pathlib import Path
from typing import List, Tuple
from zipfile import ZipFile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from IPython import get_ipython
from IPython.display import display
from l2metrics import util
from l2metrics.report import MetricsReport

sns.set_style("dark")
sns.set_context("paper")

if get_ipython() is None:
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm


def load_computational_costs(eval_dir: Path) -> pd.DataFrame:
    """Load the computational cost data from the given log directory.

    Args:
        eval_dir (Path): Path to directory containing computational cost data.

    Returns:
        pd.DataFrame: DataFrame containing computational costs for system and agent.
    """

    # Initialize computational cost dataframe
    comp_cost_df = pd.DataFrame()

    # Concatenate computational cost data
    docs_dir = eval_dir / 'docs'
    comp_cost_files = list(docs_dir.glob('computation*.csv'))

    if comp_cost_files:
        comp_cost_df = pd.concat((pd.read_csv(f) for f in comp_cost_files), ignore_index=True)
    else:
        warnings.warn(f"No computational cost files found in directory: {eval_dir}\n")

    return comp_cost_df


def load_performance_thresholds(eval_dir: Path) -> pd.DataFrame:
    """Load the performance threshold data from the given log directory.

    Args:
        eval_dir (Path): Path to directory containing performance thresholds.

    Returns:
        pd.DataFrame: DataFrame containing performance thresholds for system and agent.
    """

    # Initialize computational cost dataframe
    perf_thresh_df = pd.DataFrame()

    # Concatenate computational cost data
    docs_dir = eval_dir / 'docs'
    perf_thresh_file = docs_dir / 'performance_thresholds.csv'

    if perf_thresh_file.exists():
        perf_thresh_df = pd.read_csv(perf_thresh_file)
    else:
        warnings.warn(f"No performance threshold file found in directory: {eval_dir}\n")

    return perf_thresh_df

def load_task_similarities(eval_dir: Path) -> pd.DataFrame:
    """Load the task similarity matrix from the given log directory.

    Args:
        eval_dir (Path): Path to directory containing task similarity matrix.

    Returns:
        pd.DataFrame: DataFrame containing task similarities.
    """

    # Initialize task similarity dataframe
    task_similarity_df = pd.DataFrame()

    # Concatenate computational cost data
    docs_dir = eval_dir / 'docs'
    task_similarity_file = docs_dir / 'task_relationships.csv'

    if task_similarity_file.exists():
        task_similarity_df = pd.read_csv(task_similarity_file)
    else:
        warnings.warn(f"No task similarity file found in directory: {eval_dir}\n")

    return task_similarity_df

def unzip_logs(eval_dir: Path) -> None:
    """Walk through log directory and unzip log archives.

    Args:
        eval_dir (Path): Path to directory containing log archives.
    """
    for root, _, files in os.walk(eval_dir):
        for filename in fnmatch.filter(files, '*.zip'):
            print(f'Unzipping file: {filename}')
            ZipFile(os.path.join(root, filename)).extractall(root)

def store_ste_data(log_dir: Path) -> None:
    """Save all single-task expert data in provided log directory.

    Args:
        log_dir (Path): Path to agent configuration directory containing STE logs.

    Raises:
        FileNotFoundError: If log directory structure does not follow the expected
            structure described in the evaluation protocol.
    """

    # Check for STE logs
    ste_log_dir = log_dir / 'ste_logs' / 'ste_logs'

    if ste_log_dir.exists():
        # Store all the STE data found in the directory
        print('Storing STE data...')
        for ste_dir in ste_log_dir.iterdir():
            if ste_dir.is_dir():
                # Store STE data in append mode
                util.store_ste_data(log_dir=ste_dir, mode='a')
        print('Done storing STE data!\n')
    else:
        # STE log path not found - possibly because compressed archive has not been
        # extracted in the same location yet
        raise FileNotFoundError(f"STE logs not found in expected location!")


def compute_scenario_metrics(**kwargs) -> Tuple[pd.DataFrame, dict]:
    """Compute lifelong learning metrics for single LL logs found at input path.

    Args:
        log_dir (Path): Path to scenario directory.
        perf_measure (str): Name of column to use for metrics calculations.
        maintenance_method (str): Method for computing maintenance values.
            Valid values are 'mrtlp', 'mrlep', and 'both'.
        transfer_method (str, optional): Method for computing forward and backward transfer.
            Valid values are 'contrast', 'ratio', and 'both'. Defaults to 'both'.
        normalization_method (str, optional): Method for normalizing data.
            Valid values are 'none', 'task', and 'run'. Defaults to 'task'.

        output_dir (str, optional): Output directory of results. Defaults to ''.
        show_raw_data (bool, optional): Flag for enabling raw data in background of smoothed curve.
            Defaults to False.
        show_eval_lines (bool, optional): Flag for enabling lines between evaluation blocks to show
            changing slope of evaluation performance. Defaults to True.
        remove_outliers (bool, optional): Flag for enabling outlier removal. Defaults to False.
        do_plot (bool, optional): Flag for enabling plotting. Defaults to False.
        do_save_plots (bool, optional): Flag for enabling saving of plots. Defaults to False.
        do_save_config (bool, optional): Flag for saving L2Metrics settings to JSON file. Defaults to
            False.

    Returns:
        Tuple[pd.DataFrame, dict]: DataFrame containing lifelong metrics from scenarios.
    """

    if 'log_dir' in kwargs:
        log_dir = Path(kwargs['log_dir'])
    else:
        raise RuntimeError("log_dir is required")

    if 'do_plot' in kwargs:
        do_plot = kwargs['do_plot']
    else:
        do_plot = False

    if 'output_dir' in kwargs:
        output_dir = kwargs['output_dir']
    else:
        output_dir = ''

    if 'do_save_plots' in kwargs:
        do_save_plots = kwargs['do_save_plots']
    else:
        do_save_plots = False
    
    if 'show_raw_data' in kwargs:
        show_raw_data = kwargs['show_raw_data']
    else:
        show_raw_data = False

    if 'show_eval_lines' in kwargs:
        show_eval_lines = kwargs['show_eval_lines']
    else:
        show_eval_lines = True

    if 'do_save_config' in kwargs:
        do_save_config = kwargs['do_save_config']
    else:
        do_save_config = False

    # Initialize metrics report
    report = MetricsReport(**kwargs)

    # Calculate metrics
    report.calculate()
    ll_metrics_df = report.ll_metrics_df
    ll_metrics_dict = report.ll_metrics_dict

    # Append SG name to dataframe
    # TODO: Figure out solution that isn't as hard-coded
    ll_metrics_df['sg_name'] = log_dir.parts[-6].split('_')[1]
    ll_metrics_dict['sg_name'] = log_dir.parts[-6].split('_')[1]

    # Append agent configuration to dataframe
    ll_metrics_df['agent_config'] = log_dir.parts[-4]
    ll_metrics_dict['agent_config'] = log_dir.parts[-4]

    if do_plot:
        report.save_data(filename=str(Path(output_dir) / log_dir.name))
        report.plot(save=do_save_plots, show_raw_data=show_raw_data, show_eval_lines=show_eval_lines,
                    output_dir=output_dir)
        report.plot_ste_data(save=do_save_plots, output_dir=output_dir)
        plt.close('all')
    
    if do_save_config:
        report.save_config(filename=str(Path(output_dir) / log_dir.name))

    return ll_metrics_df, ll_metrics_dict


def compute_eval_metrics(**kwargs) -> Tuple[pd.DataFrame, List]:
    """Compute lifelong learning metrics for all LL logs in provided evaluation log directory.

    This function iterates through all the lifelong learning logs it finds in the provided
    directory, computes the LL metrics for those logs, then sorts the metrics by scenario
    type, complexity, and difficulty. Scenarios with missing scenario information
    might be ignored in the evaluation.

    Args:
        eval_dir (Path): Path to evaluation directory containing LL logs.
        ste_dir (str): Agent configuration directory of STE data. A value of '' will save all STE
            logs in every agent configuration directory.
        perf_measure (str): Name of column to use for metrics calculations.
        maintenance_method (str): Method for computing maintenance values.
            Valid values are 'mrtlp', 'mrlep', and 'both.'
        transfer_method (str): Method for computing forward and backward transfer.
            Valid values are 'contrast', 'ratio', and 'both.'
        normalization_method (str, optional): Method for normalizing data.
            Valid values are 'none', 'task', and 'run'. Defaults to 'task'.
        smoothing_method (str, optional): Method for smoothing data.
            Valid values are 'none', 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'.
            Defaults to 'flat'.
        output_dir (str, optional): Output directory of results. Defaults to ''.
        show_raw_data (bool, optional): Flag for enabling raw data in background of smoothed curve.
            Defaults to False.
        show_eval_lines (bool, optional): Flag for enabling lines between evaluation blocks to show
            changing slope of evaluation performance. Defaults to True.
        remove_outliers (bool, optional): Flag for enabling outlier removal. Defaults to False.
        do_plot (bool, optional): Flag for enabling plotting. Defaults to False.
        do_save_plots (bool, optional): Flag for enabling saving of plots. Defaults to False.
        do_store_ste (bool, optional): Flag for enabling save of STE data. Defaults to True.

    Raises:
        FileNotFoundError: If log directory structure does not follow the expected
            structure described in the evaluation protocol.

    Returns:
        pd.DataFrame: DataFrame containing lifelong metrics from all parsed scenarios, sorted by
            scenario type, complexity, and difficulty.
    """

    if 'eval_dir' in kwargs:
        eval_dir = kwargs['eval_dir']
    else:
        raise RuntimeError("eval_dir is required")

    if 'ste_dir' in kwargs:
        ste_dir = kwargs['ste_dir']
    else:
        ste_dir = ''

    if 'do_store_ste' in kwargs:
        do_store_ste = kwargs['do_store_ste']
    else:
        do_store_ste = False
    
    # Initialize LL metric dataframe
    ll_metrics_df = pd.DataFrame()
    ll_metrics_dicts = []

    # Iterate through agent configuration directories
    for agent_config in tqdm(list(eval_dir.glob('agent_config*')), desc='Agents'):
        # Save STE data if enabled
        if do_store_ste:
            if ste_dir in ['', agent_config.name]:
                store_ste_data(eval_dir / agent_config.name)

        # Check for LL logs
        ll_log_dir = agent_config / 'll_logs'

        if ll_log_dir.exists():
            print(f'Computing metrics from LL logs for {agent_config.name}...')

            # Compute and store the LL metrics for all scenarios found in the directory
            for path in tqdm(list(ll_log_dir.iterdir()), desc=agent_config.name):
                if path.is_dir():
                    # Check if current path is log directory for single run
                    if all(x in [f.name for f in path.glob('*.json')] for x in ['logger_info.json', 'scenario_info.json']):
                        metrics_df, metrics_dict = compute_scenario_metrics(log_dir=path, **kwargs)
                        ll_metrics_df = ll_metrics_df.append(metrics_df, ignore_index=True)
                        ll_metrics_dicts.append(metrics_dict)
                    else:
                        # Iterate through subdirectories containing LL logs
                        for sub_path in tqdm(list(path.iterdir()), desc=path.name):
                            if sub_path.is_dir():
                                metrics_df, metrics_dict = compute_scenario_metrics(log_dir=sub_path, **kwargs)
                                ll_metrics_df = ll_metrics_df.append(metrics_df, ignore_index=True)
                                ll_metrics_dicts.append(metrics_dict)
        else:
            raise FileNotFoundError(f"LL logs not found in expected location!")

        # Sort data by scenario type, complexity, difficulty
        if not ll_metrics_df.empty:
            ll_metrics_df = ll_metrics_df.sort_values(by=['scenario_type', 'complexity', 'difficulty'])

    return ll_metrics_df, ll_metrics_dicts


def evaluate() -> None:
    """Runs an evaluation on the provided log directory with the given parameters.

    This function loops through the subdirectories in the given directory, stores all STE data,
    computes LL metrics on all LL data, sorts the metrics by scenario complexity/difficulty,
    displays the aggregated data as tables, plots the results, then saves the metrics to the given
    output location.

    """
    # Instantiate parser
    parser = argparse.ArgumentParser(
        description='Run L2M evaluation from the command line')

    # Evaluation directory be absolute or relative paths
    parser.add_argument('-l', '--eval-dir', required=True, type=str,
                        help='Evaluation directory containing logs')

    # Evaluation directory be absolute or relative paths
    parser.add_argument('-s', '--ste-dir', default='', type=str,
                        help='Agent configuration directory of STE data')

    # Method for handling multiple STE runs
    parser.add_argument('--ste-averaging-method', default='time', choices=['time', 'metrics'],
                        help='Method for handling STE runs, time-series averaging (time) or'
                        'LL metric averaging (metric)')

    # Choose application measure to use as performance column
    parser.add_argument('-p', '--perf-measure', default='performance', type=str,
                        help='Name of column to use for metrics calculations')

    # Method for aggregating within-lifetime metrics
    parser.add_argument('-a', '--aggregation-method', default='median', choices=['mean', 'median'],
                        help='Method for aggregating within-lifetime metrics')

    # Method for calculating performance maintenance
    parser.add_argument('-m', '--maintenance-method', default='mrlep', choices=['mrtlp', 'mrlep', 'both'],
                        help='Method for computing performance maintenance')

    # Method for calculating forward and backward transfer
    parser.add_argument('-t', '--transfer-method', default='ratio', choices=['contrast', 'ratio', 'both'],
                        help='Method for computing forward and backward transfer')

    # Method for normalization
    parser.add_argument('-n', '--normalization-method', default='task', choices=['task', 'run'],
                        help='Method for normalizing data')

    # Method for smoothing
    parser.add_argument('-w', '--smoothing-method', default='flat', choices=['none', 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'],
                        help='Method for smoothing data')

    # Flag for removing outliers
    parser.add_argument('--remove-outliers', action='store_true',
                        help='Remove outliers in data for metrics')

    # Data range file for normalization
    parser.add_argument('-d', '--data-range-file', default=None, type=str,
                        help='JSON file containing task performance ranges for normalization')

    # Output directory
    parser.add_argument('--output-dir', default='results', type=str,
                        help='Directory for output files')

    # Output file location
    parser.add_argument('-o', '--output', default='ll_metrics', type=str,
                        help='Output filename for results')

    # Flag for enabling unzipping of logs
    parser.add_argument('-u', '--do-unzip', action='store_true',
                        help='Unzip all data found in evaluation directory')

    # Flag for showing raw performance data under smoothed data
    parser.add_argument('-r', '--show-raw-data', action='store_true',
                        help='Show raw data points under smoothed data for plotting')

    # Flag for showing evaluation block lines
    parser.add_argument('--show-eval-lines', dest='show_eval_lines', default=True, action='store_true',
                        help='Show lines between evaluation blocks')
    parser.add_argument('--no-show-eval-lines', dest='show_eval_lines', action='store_false',
                        help='Do not show lines between evaluation blocks')

    # Flag for disabling STE save
    parser.add_argument('--do-store-ste', dest='do_store_ste', default=True, action='store_true',
                        help='Do not store STE data')
    parser.add_argument('--no-store-ste', dest='do_store_ste', action='store_false',
                        help='Do not store STE data')

    # Flag for enabling/disabling plotting
    parser.add_argument('--do-plot', dest='do_plot', default=True, action='store_true',
                        help='Plot performance')
    parser.add_argument('--no-plot', dest='do_plot', action='store_false',
                        help='Do not plot performance')

    # Flag for enabling plot save
    parser.add_argument('--save-plots', dest='do_save_plots', default=True, action='store_true',
                        help='Save scenario and STE plots')
    parser.add_argument('--no-save-plots', dest='do_save_plots', action='store_false',
                        help='Save scenario and STE plots')

    # Flag for enabling/disabling save
    parser.add_argument('--do-save', dest='do_save', default=True, action='store_true',
                        help='Save metrics outputs')
    parser.add_argument('--no-save', dest='do_save', action='store_false',
                        help='Do not save metrics outputs')

    # Configuration file settings
    parser.add_argument('--do-save-config', dest='do_save_config', default=True, action='store_true',
                        help='Save L2Metrics settings to JSON file')
    parser.add_argument('--no-save-config', dest='do_save_config', action='store_false',
                        help='Do not save L2Metrics settings to JSON file')

    # Parse arguments
    args = parser.parse_args()
    kwargs = vars(args)
    kwargs['eval_dir'] = Path(args.eval_dir)
    kwargs['output_dir'] = Path(args.output_dir)

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Unzip logs
    if args.do_unzip:
        unzip_logs(args.eval_dir)

    # Compute LL metric data
    matplotlib.use('Agg')
    ll_metrics_df, ll_metrics_dicts = compute_eval_metrics(**kwargs)

    # Display aggregated data
    display(ll_metrics_df.groupby(by=['scenario_type', 'complexity', 'difficulty']).agg(['mean', 'std']))
    display(ll_metrics_df.groupby(by=['scenario_type', 'complexity', 'difficulty']).agg(['median', scipy.stats.iqr]))

    # Save data
    if args.do_save:
        with open(args.output_dir.parent / (args.output + '.tsv'), 'w', newline='\n') as metrics_file:
            ll_metrics_df.set_index(['sg_name', 'agent_config', 'run_id']).sort_values(
                ['agent_config', 'run_id']).to_csv(metrics_file, sep='\t')
        with open(args.output_dir.parent / (args.output + '.json'), 'w', newline='\n') as metrics_file:
            json.dump(ll_metrics_dicts, metrics_file)


if __name__ == '__main__':
    try:
        evaluate()
    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()

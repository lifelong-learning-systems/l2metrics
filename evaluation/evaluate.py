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
from zipfile import ZipFile

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import seaborn as sns
from IPython.display import display
from l2metrics import util
from l2metrics.agent import AgentMetricsReport

sns.set_style("dark")
sns.set_context("paper")


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
    for root, dirs, files in os.walk(eval_dir):
        for filename in fnmatch.filter(files, '*.zip'):
            print(f'Unzipping file: {filename}')
            ZipFile(os.path.join(root, filename)).extractall(root)

def save_ste_data(log_dir: Path) -> None:
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
                util.save_ste_data(str(ste_dir))
        print('Done storing STE data!\n')
    else:
        # STE log path not found - possibly because compressed archive has not been
        # extracted in the same location yet
        raise FileNotFoundError(f"STE logs not found in expected location!")


def compute_scenario_metrics(log_dir: Path, perf_measure: str, transfer_method: str,
                            output_dir: str = '', do_smoothing: bool = True,
                            do_normalize: bool = False, remove_outliers: bool = False,
                            do_plot: bool = False, save_plots: bool = False) -> pd.DataFrame:
    """Compute lifelong learning metrics for single LL logs found at input path.

    Args:
        log_dir (Path): Path to scenario directory.
        perf_measure (str): Name of column to use for metrics calculations.
        transfer_method (str): Method for computing forward and backward transfer.
            Valid values are 'contrast', 'ratio', and 'both.'
        output_dir (str, optional): Output directory of results. Defaults to ''.
        do_smoothing (bool, optional): Flag for enabling smoothing on performance data for metrics.
            Defaults to True.
        do_normalize (bool, optional): Flag for enabling normalization on performance data.
            Defaults to False.
        remove_outliers (bool, optional): Flag for enabling outlier removal. Defaults to False.
        do_plot (bool, optional): Flag for enabling plotting. Defaults to False.
        save_plots (bool, optional): Flag for enabling saving of plots. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing lifelong metrics from scenarios.
    """

    # Initialize metrics report
    report = AgentMetricsReport(
        log_dir=str(log_dir), perf_measure=perf_measure, transfer_method=transfer_method,
        do_smoothing=do_smoothing, do_normalize=do_normalize, remove_outliers=remove_outliers)

    # Calculate metrics
    report.calculate()
    ll_metrics_df = report.lifetime_metrics_df.copy()
    
    # Append SG name to dataframe
    ll_metrics_df['sg_name'] = log_dir.parts[-6].split('_')[1]

     # Append agent configuration to dataframe
    ll_metrics_df['agent_config'] = log_dir.parts[-4]

    # Append scenario name to dataframe
    ll_metrics_df['run_id'] = log_dir.name

    # Append application-specific metric to dataframe
    ll_metrics_df['metrics_column'] = perf_measure

    # Append min and max of performance data
    ll_metrics_df['min'] = report._log_data[perf_measure].min()
    ll_metrics_df['max'] = report._log_data[perf_measure].max()

    # Append scenario complexity and difficulty
    with open(log_dir / 'scenario_info.json', 'r') as json_file:
        scenario_info = json.load(json_file)
        if 'complexity' in scenario_info:
            ll_metrics_df['complexity'] = scenario_info['complexity']
        if 'difficulty' in scenario_info:
            ll_metrics_df['difficulty'] = scenario_info['difficulty']

    if do_plot:
        report.plot(save=save_plots, output_dir=output_dir)
        report.plot_ste_data(save=save_plots, output_dir=output_dir)

    return ll_metrics_df


def compute_eval_metrics(eval_dir: Path,  ste_dir: str, perf_measure: str, transfer_method: str,
                         output_dir: str = '', do_smoothing: bool = True,
                         do_normalize: bool = False, remove_outliers: bool = False,
                         do_plot: bool = False, save_plots: bool = False, do_save_ste: bool = True) -> pd.DataFrame:
    """Compute lifelong learning metrics for all LL logs in provided evaluation log directory.

    This function iterates through all the lifelong learning logs it finds in the provided
    directory, computes the LL metrics for those logs, then sorts the metrics by scenario
    complexity and difficulty. Scenarios with missing complexity or difficulty information
    might be ignored in the evaluation.

    Args:
        eval_dir (Path): Path to evaluation directory containing LL logs.
        ste_dir (str): Agent configuration directory of STE data. A value of '' will save all STE
            logs in every agent configuration directory.
        perf_measure (str): Name of column to use for metrics calculations.
        transfer_method (str): Method for computing forward and backward transfer.
            Valid values are 'contrast', 'ratio', and 'both.'
        output_dir (str, optional): Output directory of results. Defaults to ''.
        do_smoothing (bool, optional): Flag for enabling smoothing on performance data for metrics.
            Defaults to True.
        do_normalize (bool, optional): Flag for enabling normalization on performance data.
            Defaults to False.
        remove_outliers (bool, optional): Flag for enabling outlier removal. Defaults to False.
        do_plot (bool, optional): Flag for enabling plotting. Defaults to False.
        save_plots (bool, optional): Flag for enabling saving of plots. Defaults to False.
        do_save_ste (bool, optional): Flag for enabling save of STE data. Defaults to True.

    Raises:
        FileNotFoundError: If log directory structure does not follow the expected
            structure described in the evaluation protocol.

    Returns:
        pd.DataFrame: DataFrame containing lifelong metrics from all parsed scenarios, sorted by
            scenario complexity and difficulty.
    """
    
    # Initialize LL metric dataframe
    ll_metrics_df = pd.DataFrame()

    # Iterate through agent configuration directories
    for agent_config in tqdm(list(eval_dir.glob('agent_config*')), desc='Agents'):
        # Save STE data if enabled
        if do_save_ste:
            if ste_dir in ['', agent_config.name]:
                save_ste_data(eval_dir / agent_config.name)

        # Check for LL logs
        ll_log_dir = agent_config / 'll_logs'

        if ll_log_dir.exists():
            print(f'Computing metrics from LL logs for {agent_config.name}...')

            # Compute and store the LL metrics for all scenarios found in the directory
            for path in tqdm(list(ll_log_dir.iterdir()), desc=agent_config.name):
                if path.is_dir():
                    # Check if current path is log directory for single run
                    if all(x in [f.name for f in path.glob('*.json')] for x in ['logger_info.json', 'scenario_info.json']):
                        ll_metrics_df = ll_metrics_df.append(compute_scenario_metrics(
                            log_dir=path, perf_measure=perf_measure, transfer_method=transfer_method,
                            output_dir=output_dir, do_smoothing=do_smoothing, do_normalize=do_normalize,
                            remove_outliers=remove_outliers, do_plot=do_plot, save_plots=save_plots),
                            ignore_index=True)
                    else:
                        # Iterate through subdirectories containing LL logs
                        for sub_path in tqdm(list(path.iterdir()), desc=path.name):
                            if sub_path.is_dir():
                                ll_metrics_df = ll_metrics_df.append(compute_scenario_metrics(
                                    log_dir=sub_path, perf_measure=perf_measure, transfer_method=transfer_method,
                                    output_dir=output_dir, do_smoothing=do_smoothing, do_normalize=do_normalize,
                                    remove_outliers=remove_outliers, do_plot=do_plot, save_plots=save_plots),
                                    ignore_index=True)
        else:
            raise FileNotFoundError(f"LL logs not found in expected location!")

        # Sort data by complexity and difficulty
        ll_metrics_df = ll_metrics_df.sort_values(by=['complexity', 'difficulty'])

    return ll_metrics_df


def plot_summary(ll_metrics_df: pd.DataFrame) -> None:
    """Plot the aggregated lifelong metrics DataFrame as a violin plot.

    The plot should show trends for each of the lifelong metrics based on scenario complexity and
    difficulty.

    Args:
        ll_metrics_df (pd.DataFrame): DataFrame containing lifelong metrics from all parsed
            scenarios, sorted by scenario complexity and difficulty.
    """

    fig = plt.figure(figsize=(12, 8))

    ll_metrics = ['perf_recovery', 'perf_maintenance', 'forward_transfer_contrast',
                  'backward_transfer_contrast', 'forward_transfer_ratio',
                  'backward_transfer_ratio', 'ste_rel_perf', 'sample_efficiency']

    for index, metric in enumerate(ll_metrics, start=1):
        if metric in ll_metrics_df.columns:
            # Create subplot for current metric
            ax = fig.add_subplot(3, 3, index)

            # Create grouped violin plot
            sns.violinplot(x='complexity', y=metric, hue='difficulty',
                        data=ll_metrics_df, palette='muted')

            # Resize legend font
            plt.setp(ax.get_legend().get_title(), fontsize='8')
            plt.setp(ax.get_legend().get_texts(), fontsize='6')

    fig.subplots_adjust(wspace=0.35, hspace=0.35)
    plt.show()


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

    # Choose application measure to use as performance column
    parser.add_argument('-p', '--perf-measure', default='performance', type=str,
                        help='Name of column to use for metrics calculations')

    # Method for calculating forward and backward transfer
    parser.add_argument('-m', '--transfer-method', default='ratio', choices=['contrast', 'ratio', 'both'],
                        help='Method for computing forward and backward transfer')

    # Output directory
    parser.add_argument('--output-dir', default='sg_results', type=str,
                        help='Directory for output files')

    # Output file location
    parser.add_argument('-o', '--output', default='ll_metrics.tsv', type=str,
                        help='Output filename for results')

    # Flag for enabling unzipping of logs
    parser.add_argument('-u', '--unzip', action='store_true',
                        help='Unzip all data found in evaluation directory')

    # Flag for disabling smoothing
    parser.add_argument('--no-smoothing', action='store_true',
                        help='Do not smooth performance data for metrics')

    # Flag for enabling normalization
    parser.add_argument('-n', '--normalize', action='store_true',
                        help='Normalize performance data for metrics')

    # Flag for removing outliers
    parser.add_argument('--remove-outliers', action='store_true',
                        help='Remove outliers in data for metrics')

    # Flag for disabling STE save
    parser.add_argument('--no-save-ste', action='store_true',
                        help='Do not store STE data')

    # Flag for disabling plotting
    parser.add_argument('--no-plot', action='store_true',
                        help='Do not plot metrics report')

    # Flag for enabling plot save
    parser.add_argument('--save-plots', action='store_true',
                        help='Save scenario and STE plots')

    # Flag for disabling save
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save metrics outputs')

    # Parse arguments
    args = parser.parse_args()
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output = Path(args.output)
    do_smoothing = not args.no_smoothing
    do_plot = not args.no_plot
    do_save = not args.no_save
    do_save_ste = not args.no_save_ste

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load computational cost data
    # comp_cost_df = load_computational_costs(eval_dir)

    # Load performance threshold data
    # perf_thresh_df = load_performance_thresholds(eval_dir)

    # Load task similarity data
    # task_similarity_df = load_task_similarities(eval_dir)

    # Unzip logs
    if args.unzip:
        unzip_logs(eval_dir)

    # Compute LL metric data
    matplotlib.use('Agg')
    ll_metrics_df = compute_eval_metrics(eval_dir=eval_dir, ste_dir=args.ste_dir, output_dir=output_dir,
                                         perf_measure=args.perf_measure, transfer_method=args.transfer_method,
                                         do_smoothing=do_smoothing, do_normalize=args.normalize,
                                         remove_outliers=args.remove_outliers, do_plot=do_plot,
                                         save_plots=args.save_plots, do_save_ste=do_save_ste)

    # Display aggregated data
    display(ll_metrics_df.groupby(by=['complexity', 'difficulty']).agg(['mean', 'std']))
    display(ll_metrics_df.groupby(by=['complexity', 'difficulty']).agg(['median', scipy.stats.iqr]))

    # Plot aggregated data
    if do_plot:
        matplotlib.use('TkAgg')
        plot_summary(ll_metrics_df)

    # Save data
    if do_save:
        if output.suffix != '.tsv':
            filename = Path(output.name + '.tsv')
        else:
            filename = output

        with open(output_dir / filename, 'w', newline='\n') as metrics_file:
            ll_metrics_df.set_index(['sg_name', 'agent_config', 'run_id']).sort_values(
                ['agent_config', 'run_id']).to_csv(metrics_file, sep='\t')


if __name__ == '__main__':
    from tqdm import tqdm
    try:
        evaluate()
    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()
else:
    from tqdm.notebook import tqdm

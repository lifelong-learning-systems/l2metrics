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
import json
import os

import l2metrics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import *

sns.set_style("dark")
sns.set_context("paper")


def save_ste_data(log_dir):
    # Check for STE logs
    ste_log_dir = log_dir + "/ste_logs/ste_logs/"

    if os.path.isdir(ste_log_dir):
        # Store all the STE data found in the directory
        print('Storing STE data...')
        for ste_task in os.listdir(ste_log_dir):
            l2metrics.util.save_ste_data(ste_log_dir + ste_task)
        print('Done storing STE data!\n')
    else:
        # STE log path not found - possibly because comrpressed archive has not been
        # extracted in the same location yet
        raise Exception(f"STE logs not found in expected location!")


def compute_metrics(log_dir, perf_measure, transfer_method, do_smoothing):
    # Check for LL logs
    ll_log_dir = log_dir + "/ll_logs/"

    if os.path.isdir(ll_log_dir):
        print('Computing metrics from LL logs...')

        # Initialize LL metric dataframe
        ll_metrics_df = pd.DataFrame()

        # Compute and store the LL metrics for all scenarios found in the directory
        for item in tqdm(os.listdir(ll_log_dir), desc='Overall'):
            if os.path.isdir(ll_log_dir + item):
                for scenario in tqdm(os.listdir(ll_log_dir + item), desc='Scenario'):
                    scenario_dir = ll_log_dir + item + '/' + scenario + '/'

                    # Initialize metrics report
                    report = l2metrics.AgentMetricsReport(
                        log_dir=scenario_dir, perf_measure=perf_measure,
                        transfer_method=transfer_method, do_smoothing=do_smoothing)

                    # Calculate metrics in order of their addition to the metrics list
                    report.calculate()

                    # Append lifetime metrics to dataframe
                    ll_metrics_df = ll_metrics_df.append(
                        report.lifetime_metrics_df, ignore_index=True)

                    # Add scenario name to row
                    # ll_metrics_df.at[ll_metrics_df.index[-1], 'scenario'] = scenario.split('-')[0]

                    # Append scenario complexity and difficulty
                    with open(scenario_dir + 'scenario_info.json', 'r') as json_file:
                        scenario_info = json.load(json_file)
                        if 'complexity' in scenario_info:
                            ll_metrics_df.at[ll_metrics_df.index[-1], 'complexity'] = scenario_info['complexity']
                        if 'difficulty' in scenario_info:
                            ll_metrics_df.at[ll_metrics_df.index[-1], 'difficulty'] = scenario_info['difficulty']

    else:
        raise Exception(f"LL logs not found in expected location!")

    # Sort data by complexity and difficulty
    ll_metrics_df = ll_metrics_df.sort_values(by=['complexity', 'difficulty'])

    return ll_metrics_df


def plot(ll_metrics_df):
    fig = plt.figure(figsize=(12, 8))

    for index, metric in enumerate(ll_metrics_df.drop(columns=['complexity', 'difficulty']).columns, start=1):
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


def evaluate():
    # Instantiate parser
    parser = argparse.ArgumentParser(
        description='Run L2M evaluation from the command line')

    # Log directories can be absolute paths, relative paths, or paths found in $L2DATA/logs
    parser.add_argument('-l', '--log-dir', required=True,
                        help='Log directory for evaluation')

    # Choose application measure to use as performance column
    parser.add_argument('-p', '--perf-measure', default='performance',
                        help='Name of column to use for metrics calculations')

    # Method for calculating forward and backward transfer
    parser.add_argument('-m', '--transfer-method', default='contrast', choices=['contrast', 'ratio', 'both'],
                        help='Method for computing forward and backward transfer')

    # Flag for disabling smoothing
    parser.add_argument('--no-smoothing', action='store_true',
                        help='Do not smooth performance data for metrics')

    # Flag for disabling plotting
    parser.add_argument('--no-plot', action='store_true',
                        help='Do not plot metrics report')

    # Flag for disabling save
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save metrics outputs')

    # Parse arguments
    args = parser.parse_args()
    do_smoothing = not args.no_smoothing
    do_plot = not args.no_plot
    do_save = not args.no_save

    # Store STE log data
    save_ste_data(args.log_dir)

    # Compute LL metric data
    ll_metrics_df = compute_metrics(
        args.log_dir, args.perf_measure, args.transfer_method, do_smoothing)

    # Plot aggregated data
    if do_plot:
        plot(ll_metrics_df)


if __name__ == '__main__':
    try:
        evaluate()
    except Exception as e:
        print(f'Error: {e}')

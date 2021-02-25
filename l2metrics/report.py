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

import os
from collections import defaultdict
from typing import List, Tuple, Union

import l2logger.util as l2l
import numpy as np
import pandas as pd
from tabulate import tabulate

from . import core, util
from .backward_transfer import BackwardTransfer
from .block_saturation import BlockSaturation
from .core import Metric
from .forward_transfer import ForwardTransfer
from .performance_maintenance import PerformanceMaintenance
from .performance_recovery import PerformanceRecovery
from .recovery_time import RecoveryTime
from .sample_efficiency import SampleEfficiency
from .ste_relative_performance import STERelativePerf
from .terminal_performance import TerminalPerformance


class MetricsReport():
    """
    Aggregates a list of metrics for an Agent learner
    """

    def __init__(self, **kwargs) -> None:
        # Defines log_dir and initializes the metrics list
        self._metrics = []

        if 'log_dir' in kwargs:
            self.log_dir = kwargs['log_dir']
        else:
            raise RuntimeError("log_dir is required")

        if 'perf_measure' in kwargs:
            self.perf_measure = kwargs['perf_measure']
        else:
            self.perf_measure = 'reward'

        if 'transfer_method' in kwargs:
            self.transfer_method = kwargs['transfer_method']
        else:
            self.transfer_method = 'contrast'

        if 'do_smoothing' in kwargs:
            self.do_smoothing = kwargs['do_smoothing']
        else:
            self.do_smoothing = True

        if 'do_normalize' in kwargs:
            self.do_normalize = kwargs['do_normalize']
        else:
            self.do_normalize = False

        if 'remove_outliers' in kwargs:
            self.remove_outliers = kwargs['remove_outliers']
        else:
            self.remove_outliers = False

        # Initialize list of LL metrics
        self.task_metrics = ['perf_recovery', 'perf_maintenance']
        if self.transfer_method in ['contrast', 'both']:
            self.task_metrics.extend(['forward_transfer_contrast', 'backward_transfer_contrast'])
        if self.transfer_method in ['ratio', 'both']:
            self.task_metrics.extend(['forward_transfer_ratio', 'backward_transfer_ratio'])
        self.task_metrics.extend(['ste_rel_perf', 'sample_efficiency'])

        # Get metric fields
        metric_fields = l2l.read_logger_info(self.log_dir)

        # Do a check to make sure the performance measure has been logged
        if self.perf_measure not in metric_fields:
            raise Exception(f'Performance measure not found in metrics columns: {self.perf_measure}\n'
                            f'Valid measures are: {metric_fields}')

        # Gets all data from the relevant log files
        self._log_data = l2l.read_log_data(self.log_dir)

        # Validate scenario info
        l2l.validate_scenario_info(self.log_dir)

        # Validate data format
        l2l.validate_log(self._log_data, metric_fields)

        # Filter data by completed experiences
        self._log_data = self._log_data[self._log_data['exp_status'] == 'complete']

        # Fill in regime number and sort
        self._log_data = l2l.fill_regime_num(self._log_data)
        self._log_data = self._log_data.sort_values(
            by=['regime_num', 'exp_num']).set_index("regime_num", drop=False)

        # Remove outliers
        if self.remove_outliers:
            self.filter_outliers(quantiles=(0.1, 0.9))

        # Normalize log data
        self.data_min = np.nanmin(self._log_data[self.perf_measure])
        self.data_max = np.nanmax(self._log_data[self.perf_measure])
        self.data_scale = 100

        if self.do_normalize:
            self.normalize_data(scale=self.data_scale)

        if len(self._log_data) == 0:
            raise Exception('No valid log data to compute metrics')

        # Get block summary
        _, self.block_info = l2l.parse_blocks(self._log_data)

        # Store unique task names
        self._unique_tasks = list(self.block_info.sort_values(
            ['block_type', 'block_num'], ascending=[False, True])['task_name'].unique())

        # Do a check to make sure the performance measure is logged
        if self.perf_measure not in self._log_data.columns:
            raise Exception(f'Performance measure ({self.perf_measure}) not found in the log data')

        # Adds default metrics
        self._add_default_metrics()

        # Initialize a results dictionary that can be returned at the end of the calculation step and an internal
        # dictionary that can be passed around for internal calculations
        block_info_keys_to_include = ['block_num', 'block_type', 'task_name', 'regime_num']
        if len(self.block_info.loc[:, 'task_params'].unique()) > 1:
            block_info_keys_to_include.append('task_params')

        self._metrics_df = self.block_info[block_info_keys_to_include].copy()

    def add(self, metrics_list: Union[Metric, List[Metric]]) -> None:
        self._metrics.append(metrics_list)

    def _add_default_metrics(self) -> None:
        # Default metrics no matter the syllabus type
        self.add(BlockSaturation(self.perf_measure))
        self.add(TerminalPerformance(self.perf_measure, self.do_smoothing))
        self.add(RecoveryTime(self.perf_measure, self.do_smoothing))
        self.add(PerformanceRecovery(self.perf_measure))
        self.add(PerformanceMaintenance(self.perf_measure))
        self.add(ForwardTransfer(self.perf_measure, self.transfer_method))
        self.add(BackwardTransfer(self.perf_measure, self.transfer_method))
        self.add(STERelativePerf(self.perf_measure, self.do_smoothing,
                                 self.do_normalize, (self.data_min, self.data_max, self.data_scale)))
        self.add(SampleEfficiency(self.perf_measure, self.do_normalize,
                                  (self.data_min, self.data_max, self.data_scale)))

    def filter_outliers(self, quantiles: Tuple[float, float] = (0.1, 0.9)) -> None:
        x = self._log_data[self.perf_measure]
        self._log_data = self._log_data[x.between(
            x.quantile(quantiles[0]), x.quantile(quantiles[1]))]

    def normalize_data(self, scale: int = 100) -> None:
        # Save raw data in another variable
        self._raw_data = self._log_data[self.perf_measure].copy()

        # Get data range over scenario and STE data
        unique_tasks = list(self._log_data['task_name'].unique())
        for task in unique_tasks:
            ste_data = util.load_ste_data(task)
            if ste_data is not None:
                if self.perf_measure in ste_data.columns:
                    self.data_min = min(self.data_min, np.nanmin(ste_data[self.perf_measure]))
                    self.data_max = max(self.data_max, np.nanmax(ste_data[self.perf_measure]))

        norm_data = (self._log_data[self.perf_measure].values -
                     self.data_min) / (self.data_max - self.data_min) * scale
        self._log_data[self.perf_measure] = norm_data

    def add_noise(self, mean: float, std: float) -> None:
        # Add Gaussian noise to log data
        noise = np.random.normal(mean, std, len(
            self._log_data[self.perf_measure]))
        self._log_data[self.perf_measure] = self._log_data[self.perf_measure] + noise

    def calculate(self) -> None:
        for metric in self._metrics:
            self._metrics_df = metric.calculate(self._log_data, self.block_info, self._metrics_df)

        self.calculate_regime_metrics()
        self.calculate_task_metrics()
        self.calculate_lifetime_metrics()

    def calculate_regime_metrics(self) -> None:
        # Create dataframe for regime-level metrics
        regime_metrics = ['saturation', 'eps_to_sat', 'term_perf', 'eps_to_term_perf']
        self.regime_metrics_df = self.block_info[['block_num', 'block_type', 'task_name', 'task_params']]

        # Fill regime metrics dataframe
        self.regime_metrics_df = pd.concat(
            [self.regime_metrics_df, self._metrics_df[regime_metrics]], axis=1)
        if self.regime_metrics_df['task_params'].size:
            self.regime_metrics_df['task_params'] = self.regime_metrics_df['task_params'].dropna().apply(
                lambda x: x[:25] + '...' if len(x) > 25 else x)
        else:
            self.regime_metrics_df = self.regime_metrics_df.dropna(axis=1)

    def calculate_task_metrics(self) -> None:
        self.task_metrics_df = pd.DataFrame(index=self._unique_tasks, columns=self.task_metrics)
        self.task_metrics_df.index.name = 'task_name'

        # Initialize transfer arrays to NaNs
        num_tasks = len(self._unique_tasks)

        if self.transfer_method in ['contrast', 'both']:
            self.task_metrics_df['forward_transfer_contrast'] = [[np.nan] * num_tasks] * num_tasks
            self.task_metrics_df['backward_transfer_contrast'] = [[np.nan] * num_tasks] * num_tasks
            self.forward_transfers_contrast = defaultdict(dict)
            self.backward_transfers_contrast = defaultdict(dict)
        if self.transfer_method in ['ratio', 'both']:
            self.task_metrics_df['forward_transfer_ratio'] = [[np.nan] * num_tasks] * num_tasks
            self.task_metrics_df['backward_transfer_ratio'] = [[np.nan] * num_tasks] * num_tasks
            self.forward_transfers_ratio = defaultdict(dict)
            self.backward_transfers_ratio = defaultdict(dict)

        # Create data structures for transfer values
        for _, row in self._metrics_df.iterrows():
            if 'forward_transfer_contrast' in self._metrics_df:
                if type(row['forward_transfer_contrast']) is dict:
                    [(other_task, transfer_value)] = row['forward_transfer_contrast'].items()
                    key = (self._unique_tasks.index(other_task),
                           self._unique_tasks.index(row['task_name']))
                    self.forward_transfers_contrast[key[0]][key[1]] = round(
                        transfer_value, 2)
            if 'forward_transfer_ratio' in self._metrics_df:
                if type(row['forward_transfer_ratio']) is dict:
                    [(other_task, transfer_value)] = row['forward_transfer_ratio'].items()
                    key = (self._unique_tasks.index(other_task),
                           self._unique_tasks.index(row['task_name']))
                    self.forward_transfers_ratio[key[0]][key[1]] = round(
                        transfer_value, 2)

            if 'backward_transfer_contrast' in self._metrics_df:
                if type(row['backward_transfer_contrast']) is dict:
                    [(other_task, transfer_value)] = row['backward_transfer_contrast'].items()
                    key = (self._unique_tasks.index(other_task),
                           self._unique_tasks.index(row['task_name']))
                    if key[0] not in self.backward_transfers_contrast.keys():
                        self.backward_transfers_contrast[key[0]][key[1]] = [transfer_value]
                    elif key[1] not in self.backward_transfers_contrast[key[0]].keys():
                        self.backward_transfers_contrast[key[0]][key[1]] = [transfer_value]
                    else:
                        self.backward_transfers_contrast[key[0]][key[1]].append(transfer_value)
            if 'backward_transfer_ratio' in self._metrics_df:
                if type(row['backward_transfer_ratio']) is dict:
                    [(other_task, transfer_value)] = row['backward_transfer_ratio'].items()
                    key = (self._unique_tasks.index(other_task),
                           self._unique_tasks.index(row['task_name']))
                    if key[0] not in self.backward_transfers_ratio.keys():
                        self.backward_transfers_ratio[key[0]][key[1]] = [transfer_value]
                    elif key[1] not in self.backward_transfers_ratio[key[0]].keys():
                        self.backward_transfers_ratio[key[0]][key[1]] = [transfer_value]
                    else:
                        self.backward_transfers_ratio[key[0]][key[1]].append(transfer_value)

        # Fill task metrics dataframe
        for task in self._unique_tasks:
            # Get task metrics
            tm = self._metrics_df[self._metrics_df['task_name'] == task]

            # Iterate over task metrics
            for metric in self.task_metrics:
                if metric in tm.keys():
                    # Create transfer matrix for forward and backward transfer
                    if metric == 'forward_transfer_contrast':
                        transfer_row = self.task_metrics_df.at[task, metric].copy()
                        for k, v in self.forward_transfers_contrast[self._unique_tasks.index(task)].items():
                            transfer_row[k] = v
                        self.task_metrics_df.at[task, metric] = transfer_row
                    elif metric == 'forward_transfer_ratio':
                        transfer_row = self.task_metrics_df.at[task, metric].copy()
                        for k, v in self.forward_transfers_ratio[self._unique_tasks.index(task)].items():
                            transfer_row[k] = v
                        self.task_metrics_df.at[task, metric] = transfer_row
                    elif metric == 'backward_transfer_contrast':
                        transfer_row = self.task_metrics_df.at[task, metric].copy()
                        for k, v in self.backward_transfers_contrast[self._unique_tasks.index(task)].items():
                            transfer_row[k] = round(np.mean(v), 2)
                        self.task_metrics_df.at[task, metric] = transfer_row
                    elif metric == 'backward_transfer_ratio':
                        transfer_row = self.task_metrics_df.at[task, metric].copy()
                        for k, v in self.backward_transfers_ratio[self._unique_tasks.index(task)].items():
                            transfer_row[k] = round(np.mean(v), 2)
                        self.task_metrics_df.at[task, metric] = transfer_row
                    else:
                        # Drop NaN values
                        metric_values = tm[metric].dropna().values

                        if len(metric_values) == 0:
                            self.task_metrics_df.at[task, metric] = np.NaN
                        elif len(metric_values) == 1:
                            self.task_metrics_df.at[task, metric] = metric_values[0]
                        else:
                            self.task_metrics_df.at[task, metric] = metric_values

    def calculate_lifetime_metrics(self) -> None:
        # Calculate lifetime metrics from task metrics
        self.lifetime_metrics_df = pd.DataFrame(columns=self.task_metrics)

        for metric in self.task_metrics:
            if metric in self.task_metrics_df:
                if metric in ['forward_transfer_contrast', 'forward_transfer_ratio']:
                    metric_vals = self.task_metrics_df[metric].values

                    # Flatten lists
                    metric_vals = np.asarray([item for sublist in metric_vals for item in sublist])

                    # Drop NaNs
                    metric_vals = metric_vals[~np.isnan(metric_vals)]
                elif metric == 'backward_transfer_contrast':
                    # Get the first calculated backward transfer values for each task pair
                    metric_vals = [v2[0] for k, v in self.backward_transfers_contrast.items() for k2, v2 in v.items()]
                elif metric == 'backward_transfer_ratio':
                    # Get the first calculated backward transfer values for each task pair
                    metric_vals = [v2[0] for k, v in self.backward_transfers_ratio.items() for k2, v2 in v.items()]
                else:
                    metric_vals = self.task_metrics_df[metric].dropna().values

                if len(metric_vals):
                    self.lifetime_metrics_df[metric] = [np.median(metric_vals)]

    def report(self, save: bool = False, output: str = None) -> None:
        # TODO: Handle reporting custom metrics

        # Print lifetime metrics
        print('\nLifetime Metrics:')
        print(tabulate(self.lifetime_metrics_df.fillna('N/A'), headers='keys', tablefmt='psql',
                       floatfmt=".2f", showindex=False))

        # Print task-level metrics
        print('\nTask Metrics:')
        print(tabulate(self.task_metrics_df, headers='keys', tablefmt='psql', floatfmt=".2f"))

        # Print regime-level metrics
        # print('\nRegime Metrics:')
        # print(tabulate(self.regime_metrics_df.fillna('N/A'), headers='keys', tablefmt='psql', floatfmt=".2f"))

        if save:
            # Generate filename
            if output is None:
                _, filename = os.path.split(self.log_dir)
            else:
                filename = output.replace(" ", "_")

            # Save metrics to file
            with open(filename + '_metrics.tsv', 'w', newline='\n') as metrics_file:
                self.lifetime_metrics_df.to_csv(metrics_file, sep='\t', index=False)
                metrics_file.write('\n')
                self.task_metrics_df.to_csv(metrics_file, sep='\t')
                metrics_file.write('\n')
                self.regime_metrics_df.to_csv(metrics_file, sep='\t')

    def log_summary(self) -> pd.DataFrame:
        # Get summary of log data
        task_experiences = {'task_name': [], 'LX': [], 'EX': []}

        for task in self._unique_tasks:
            task_experiences['task_name'].append(task)
            task_experiences['LX'].append(self._log_data[(self._log_data['task_name'] == task) & (
                self._log_data['block_type'] == 'train')].shape[0])
            task_experiences['EX'].append(self._log_data[(self._log_data['task_name'] == task) & (
                self._log_data['block_type'] == 'test')].shape[0])

        return pd.DataFrame(task_experiences)

    def plot(self, save: bool = False, output_dir: str = '', input_title: str = None) -> None:
        if input_title is None:
            input_title = os.path.split(self.log_dir)[-1]

        util.plot_performance(self._log_data, self.block_info, unique_tasks=self._unique_tasks,
                              do_smoothing=self.do_smoothing, y_axis_col=self.perf_measure,
                              input_title=input_title, output_dir=output_dir, do_save_fig=save)

    def plot_ste_data(self, window_len: int = None, input_title: str = None,
                      save: bool = False, output_dir: str = '') -> None:
        if input_title is None:
            input_title = 'Performance Relative to STE\n' + os.path.split(self.log_dir)[-1]
        plot_filename = 'ste_' + os.path.split(self.log_dir)[-1]

        util.plot_ste_data(self._log_data, self.block_info, self._unique_tasks,
                           perf_measure=self.perf_measure, do_smoothing=self.do_smoothing,
                           window_len=window_len, do_normalize=self.do_normalize, min_max_scale=(
                               self.data_min, self.data_max, self.data_scale),
                           input_title=input_title, output_dir=output_dir, do_save=save,
                           plot_filename=plot_filename)

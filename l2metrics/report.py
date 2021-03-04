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

from .block_saturation import BlockSaturation
from .core import Metric
from .normalizer import Normalizer
from .performance_maintenance import PerformanceMaintenance
from .performance_recovery import PerformanceRecovery
from .recovery_time import RecoveryTime
from .sample_efficiency import SampleEfficiency
from .ste_relative_performance import STERelativePerf
from .terminal_performance import TerminalPerformance
from .transfer import Transfer
from .util import plot_performance, plot_ste_data


class MetricsReport():
    """
    Aggregates a list of metrics for an agent learner.
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

        if 'maintenance_method' in kwargs:
            self.maintenance_method = kwargs['maintenance_method']
        else:
            self.maintenance_method = 'mrlep'

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

        if 'normalization_method' in kwargs:
            self.normalization_method = kwargs['normalization_method']
        else:
            self.normalization_method = 'task'

        if 'data_range' in kwargs:
            self.data_range = kwargs['data_range']
        else:
            self.data_range = None

        if 'remove_outliers' in kwargs:
            self.remove_outliers = kwargs['remove_outliers']
        else:
            self.remove_outliers = False

        # Initialize list of LL metrics
        self.task_metrics = ['perf_recovery']
        if self.maintenance_method in ['mrtlp', 'both']:
            self.task_metrics.extend(['perf_maintenance_mrtlp'])
        if self.maintenance_method in ['mrlep', 'both']:
            self.task_metrics.extend(['perf_maintenance_mrlep'])
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

        # Do a check to make sure the performance measure is logged
        if self.perf_measure not in self._log_data.columns:
            raise Exception(f'Performance measure ({self.perf_measure}) not found in the log data')

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

        # Check for non-zero log data length
        if len(self._log_data) == 0:
            raise Exception('No valid log data to compute metrics')

        if self.do_normalize:
            # Instantiate normalizer
            self.normalizer = Normalizer(perf_measure=self.perf_measure,
                                         data=self._log_data[['task_name', self.perf_measure]].set_index('task_name'),
                                         data_range=self.data_range, method=self.normalization_method)

            # Save raw data in another variable
            self._raw_data = self._log_data[self.perf_measure].copy()

            # Normalize data
            self._log_data = self.normalizer.normalize(self._log_data)
        else:
            self.normalizer = None

        # Get block summary
        _, self.block_info = l2l.parse_blocks(self._log_data)

        # Store unique task names by order of training
        self._unique_tasks = list(self.block_info.sort_values(
            ['block_type', 'block_num'], ascending=[False, True])['task_name'].unique())

        # Adds default metrics
        self._add_default_metrics()

        # Initialize a results dictionary that can be returned at the end of the calculation step and an internal
        # dictionary that can be passed around for internal calculations
        block_info_keys_to_include = ['block_num', 'block_type', 'task_name', 'regime_num']
        if len(self.block_info.loc[:, 'task_params'].unique()) > 1:
            block_info_keys_to_include.append('task_params')

        self._metrics_df = self.block_info[block_info_keys_to_include].copy()

    def add(self, metrics_list: Union[Metric, List[Metric]]) -> None:
        if isinstance(metrics_list, list):
            self._metrics.extend(metrics_list)
        else:
            self._metrics.append(metrics_list)

    def _add_default_metrics(self) -> None:
        # Default metrics no matter the syllabus type
        self.add(BlockSaturation(self.perf_measure))
        self.add(TerminalPerformance(self.perf_measure, self.do_smoothing))
        self.add(RecoveryTime(self.perf_measure, self.do_smoothing))
        self.add(PerformanceRecovery(self.perf_measure))
        self.add(PerformanceMaintenance(self.perf_measure, self.maintenance_method))
        self.add(Transfer(self.perf_measure, self.transfer_method))
        self.add(STERelativePerf(self.perf_measure, self.do_smoothing, self.do_normalize,
                                 self.normalizer))
        self.add(SampleEfficiency(self.perf_measure, self.do_normalize,
                                  self.normalizer))

    def filter_outliers(self, quantiles: Tuple[float, float] = (0.1, 0.9)) -> None:
        x = self._log_data[self.perf_measure]
        self._log_data = self._log_data[x.between(
            x.quantile(quantiles[0]), x.quantile(quantiles[1]))]

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
        # Initialize task metrics dataframe
        self.task_metrics_df = pd.DataFrame(index=self._unique_tasks, columns=self.task_metrics)
        self.task_metrics_df.index.name = 'task_name'

        # Initialize certain task metrics data objects
        num_tasks = len(self._unique_tasks)
        self.task_metrics_df['recovery_times'] = [[]] * num_tasks
        if self.maintenance_method in ['mrtlp', 'both']:
            self.task_metrics_df['maintenance_val_mrtlp'] = [[]] * num_tasks
        if self.maintenance_method in ['mrlep', 'both']:
            self.task_metrics_df['maintenance_val_mrlep'] = [[]] * num_tasks
        if self.transfer_method in ['contrast', 'both']:
            self.task_metrics_df['forward_transfer_contrast'] = [{}] * num_tasks
            self.task_metrics_df['backward_transfer_contrast'] = [{}] * num_tasks
            self.forward_transfer_contrast = defaultdict(dict)
            self.backward_transfer_contrast = defaultdict(dict)
        if self.transfer_method in ['ratio', 'both']:
            self.task_metrics_df['forward_transfer_ratio'] = [{}] * num_tasks
            self.task_metrics_df['backward_transfer_ratio'] = [{}] * num_tasks
            self.forward_transfer_ratio = defaultdict(dict)
            self.backward_transfer_ratio = defaultdict(dict)

        # Create data structures for transfer values
        for _, row in self._metrics_df.iterrows():
            for transfer_metric in ['forward_transfer_contrast', 'forward_transfer_ratio',
                                    'backward_transfer_contrast', 'backward_transfer_ratio']:
                if transfer_metric in self._metrics_df and type(row[transfer_metric]) is dict:
                    [(other_task, transfer_value)] = row[transfer_metric].items()
                    key = (other_task, row['task_name'])
                    if key[0] not in getattr(self, transfer_metric).keys():
                        getattr(self, transfer_metric)[key[0]][key[1]] = [transfer_value]
                    elif key[1] not in getattr(self, transfer_metric)[key[0]].keys():
                        getattr(self, transfer_metric)[key[0]][key[1]] = [transfer_value]
                    else:
                        getattr(self, transfer_metric)[key[0]][key[1]].append(transfer_value)

        # Fill task metrics dataframe
        for task in self._unique_tasks:
            # Get task metrics
            tm = self._metrics_df[self._metrics_df['task_name'] == task]

            # Iterate over task metrics
            for metric in self.task_metrics:
                if metric in tm.keys():
                    if metric == 'perf_recovery':
                        pr = tm[metric].dropna().values
                        self.task_metrics_df.at[task, metric] = np.NaN if len(pr) == 0 else pr[0]
                        self.task_metrics_df.at[task, 'recovery_times'] = list(
                            tm['recovery_time'].dropna().values)
                    elif metric in ['perf_maintenance_mrtlp', 'perf_maintenance_mrlep']:
                        pm = tm[metric].dropna().values
                        self.task_metrics_df.at[task, metric] = pm[0] if len(pm) else np.NaN
                        maintenance_val_name = 'maintenance_val_' + metric.split('_')[-1]
                        maintenance_values = list(tm[maintenance_val_name].values)
                        self.task_metrics_df.at[task, maintenance_val_name] = [
                            maintenance_values[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(maintenance_values))]
                    elif metric in ['forward_transfer_contrast', 'forward_transfer_ratio',
                                    'backward_transfer_contrast', 'backward_transfer_ratio']:
                        self.task_metrics_df.at[task, metric] = getattr(self, metric)[task]
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
                if metric in ['forward_transfer_contrast', 'forward_transfer_ratio',
                              'backward_transfer_contrast', 'backward_transfer_ratio']:
                    # Get the first calculated transfer values for each task pair
                    metric_vals = [v2[0] for _, v in getattr(self, metric).items() for _, v2 in v.items()]
                else:
                    metric_vals = self.task_metrics_df[metric].dropna().values

                if len(metric_vals):
                    # Aggregate metric values with median operator
                    self.lifetime_metrics_df[metric] = [np.median(metric_vals)]

    def report(self, save: bool = False, output: str = None) -> None:
        # TODO: Handle reporting custom metrics

        # Print lifetime metrics
        print('\nLifetime Metrics:')
        print(tabulate(self.lifetime_metrics_df.fillna('N/A'), headers='keys', tablefmt='psql',
                       floatfmt=".2f", showindex=False))

        if save:
            # Generate filename
            if output is None:
                _, filename = os.path.split(self.log_dir.strip('/\\'))
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

        return pd.DataFrame(task_experiences).set_index('task_name')

    def plot(self, save: bool = False, output_dir: str = '', input_title: str = None) -> None:
        if input_title is None:
            input_title = os.path.split(self.log_dir.strip('/\\'))[-1]

        plot_performance(self._log_data, self.block_info, unique_tasks=self._unique_tasks,
                              do_smoothing=self.do_smoothing, y_axis_col=self.perf_measure,
                              input_title=input_title, output_dir=output_dir, do_save_fig=save)

    def plot_ste_data(self, window_len: int = None, input_title: str = None,
                      save: bool = False, output_dir: str = '') -> None:
        if input_title is None:
            input_title = 'Performance Relative to STE\n' + \
                os.path.split(self.log_dir.strip('/\\'))[-1]
        plot_filename = 'ste_' + os.path.split(self.log_dir.strip('/\\'))[-1]

        plot_ste_data(self._log_data, self.block_info, self._unique_tasks,
                           perf_measure=self.perf_measure, do_smoothing=self.do_smoothing,
                           window_len=window_len, do_normalize=self.do_normalize,
                           normalizer=self.normalizer, input_title=input_title,
                           output_dir=output_dir, do_save=save, plot_filename=plot_filename)

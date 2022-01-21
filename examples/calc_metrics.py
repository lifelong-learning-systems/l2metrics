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

"""
This script illustrates how to produce a lifelong learning metrics report with a
custom metric.
"""

import inspect
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from l2metrics import _localutil, util
from l2metrics.core import Metric
from l2metrics.parser import init_parser
from l2metrics.report import MetricsReport
from tqdm import tqdm

logger = logging.getLogger("Calculate Metrics")


class MyCustomAgentMetric(Metric):
    name = "An Example Custom Metric for illustration"
    capability = "continual_learning"
    requires = {'syllabus_type': 'agent'}
    description = "Records the maximum value per regime in the dataframe"
    
    def __init__(self, perf_measure: str):
        super().__init__()
        self.perf_measure = perf_measure

    def validate(self, block_info):
        pass

    def calculate(self, dataframe: pd.DataFrame, block_info: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
        max_values = {}

        for idx in range(block_info.loc[:, 'regime_num'].max() + 1):
            max_regime_value = dataframe.loc[dataframe['regime_num'] == idx, self.perf_measure].max()
            max_values[idx] = max_regime_value

        # This is the line that fills the metric into the dataframe. Comment it out to suppress this behavior
        metrics_df = _localutil.fill_metrics_df(
            max_values, 'max_value', metrics_df)
        return metrics_df


def run() -> None:
    # Initialize parser
    parser = init_parser()

    # Parse arguments
    args = parser.parse_args()
    kwargs = vars(args)

    if args.load_settings:
        with open(args.load_settings, 'r') as config_file:
            kwargs.update(json.load(config_file))

    kwargs['output_dir'] = Path(args.output_dir)

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data range data for normalization and standardize names to lowercase
    if args.data_range_file:
        with open(args.data_range_file) as config_json:
            data_range = json.load(config_json)
            data_range = {key.lower(): val for key, val in data_range.items()}
    else:
        data_range = None

    kwargs['data_range'] = data_range

    # Check for recursive flag
    if args.recursive:
        ll_metrics_df = pd.DataFrame()
        ll_metrics_dicts = []
        log_data_df = pd.DataFrame()

        # Iterate over all runs found in the directory
        dirs = [p for p in Path(args.log_dir).rglob("*") if p.is_dir()]
        for dir in tqdm(dirs, desc=Path(args.log_dir).name):
            # Check if current path is log directory for single run
            if all(x in [f.name for f in dir.glob('*.json')] for x in ['logger_info.json', 'scenario_info.json']):
                if args.ste_store_mode:
                    # Store STE data
                    try:
                        util.store_ste_data(log_dir=dir, mode=args.ste_store_mode)
                    except ValueError as e:
                        logger.error(e)
                else:
                    # Compute and store the LL metrics
                    kwargs['log_dir'] = dir
                    report = MetricsReport(**kwargs)
                    # Add example of custom metric
                    report.add(MyCustomAgentMetric(args.perf_measure))
                    report.calculate()
                    metrics_df = report.ll_metrics_df
                    metrics_dict = report.ll_metrics_dict
                    ll_metrics_df = ll_metrics_df.append(metrics_df, ignore_index=True)
                    ll_metrics_dicts.append(metrics_dict)
                    df = report._log_data
                    df['run_id'] = dir.name
                    log_data_df = log_data_df.append(df, ignore_index=True)

                    # Plot metrics
                    if args.do_plot:
                        report.plot(save=args.do_save, show_eval_lines=args.show_eval_lines,
                                    output_dir=str(args.output_dir))
                        report.plot_ste_data(save=args.do_save, output_dir=str(args.output_dir))
                        plt.close('all')

        # Assign base filename
        filename = args.output_dir / (args.output if args.output else 'll_metrics')

        # Save settings used to run calculate metrics
        if args.do_save_settings:
            with open(str(filename) + '_settings.json', 'w') as settings_file:
                kwargs['log_dir'] = str(kwargs.get('log_dir', ''))
                kwargs['output_dir'] = str(kwargs.get('output_dir', ''))
                json.dump(kwargs, settings_file)

        # Save data
        if args.do_save and args.ste_store_mode is None:
            if not ll_metrics_df.empty:
                with open(str(filename) + '.tsv', 'w', newline='\n') as metrics_file:
                    ll_metrics_df.set_index(['run_id']).to_csv(metrics_file, sep='\t')
            if ll_metrics_dicts:
                with open(str(filename) + '.json', 'w', newline='\n') as metrics_file:
                    json.dump(ll_metrics_dicts, metrics_file)
            if not log_data_df.empty:
                log_data_df.reset_index(drop=True).to_feather(str(filename) + '_data.feather')
    else:
        if args.ste_store_mode:
            # Store STE data
            util.store_ste_data(log_dir=Path(args.log_dir), mode=args.ste_store_mode)
        else:
            # Initialize metrics report
            report = MetricsReport(**kwargs)

            # Add example of custom metric
            report.add(MyCustomAgentMetric(args.perf_measure))

            # Add noise to log data if mean or standard deviation is specified
            if args.noise[0] or args.noise[1]:
                report.add_noise(mean=args.noise[0], std=args.noise[1])

            # Calculate metrics in order of their addition to the metrics list.
            report.calculate()

            # Print table of metrics
            report.report()

            # Save metrics to file
            if args.do_save:
                report.save_metrics(output_dir=args.output_dir, filename=args.output)
                report.save_data(output_dir=args.output_dir, filename=args.output)

            # Plot metrics
            if args.do_plot:
                report.plot(save=args.do_save, show_eval_lines=args.show_eval_lines,
                            output_dir=str(args.output_dir))
                report.plot_ste_data(save=args.do_save, output_dir=str(args.output_dir))
                plt.show()

            # Save settings used to run calculate metrics
            if args.do_save_settings:
                report.save_settings(output_dir=args.output_dir, filename=args.output)

if __name__ == "__main__":
    # Configure logger
    logging.basicConfig(level=logging.INFO)

    try:
        run()
    except (KeyError, ValueError) as e:
        logger.exception(e)

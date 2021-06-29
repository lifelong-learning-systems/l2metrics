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

import json
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from l2metrics import util
from l2metrics.parser import init_parser
from l2metrics.report import MetricsReport
from tqdm import tqdm


def run() -> None:
    # Initialize parser
    parser = init_parser()

    # Parse arguments
    args = parser.parse_args()
    kwargs = vars(args)

    if args.load_settings:
        with open(args.load_settings, 'r') as settings_file:
            kwargs.update(json.load(settings_file))

    kwargs['output_dir'] = Path(args.output_dir)

    # Create output directory if it doesn't exist
    if (args.do_save or args.do_save_settings) and args.ste_store_mode is None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data range data for normalization and standardize names to lowercase
    if args.data_range_file:
        with open(args.data_range_file) as data_range_file:
            data_range = json.load(data_range_file)
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
                    except Exception as e:
                        print(e)
                else:
                    # Compute and store the LL metrics
                    kwargs['log_dir'] = dir
                    report = MetricsReport(**kwargs)
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
                        report.plot(save=args.do_save, show_raw_data=args.show_raw_data,
                                    show_eval_lines=args.show_eval_lines, output_dir=str(args.output_dir))
                        report.plot_ste_data(save=args.do_save, output_dir=str(args.output_dir))
                        plt.close('all')

        # Assign base filename
        filename = args.output_dir / (args.output if args.output else 'll_metrics')

        # Save settings used to run calculate metrics
        if args.do_save_settings and args.ste_store_mode is None:
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
                report.plot(save=args.do_save, show_raw_data=args.show_raw_data,
                            show_eval_lines=args.show_eval_lines, output_dir=str(args.output_dir))
                report.plot_ste_data(save=args.do_save, output_dir=str(args.output_dir))
                plt.show()

            # Save settings used to run calculate metrics
            if args.do_save_settings:
                report.save_settings(output_dir=args.output_dir, filename=args.output)


if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()

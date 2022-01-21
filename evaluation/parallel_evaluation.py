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

import json
import logging
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import psutil
from tqdm import tqdm

from evaluation.evaluate import compute_eval_metrics

logger = logging.getLogger(__name__)

matplotlib.use('Agg')

perf_measure = {
    'argonne': 'score',
    'hrl': 'reward',
    'sri': 'reward',
    'teledyne': 'object_id_prec',
    'upenn': 'performance'
}


def process_evaluation(args):
    eval_dir, sg_name, processing_mode = args

    # Build file and directory strings
    kwargs = {}
    kwargs['eval_dir'] = Path('../../sg_' + sg_name + '_eval') / eval_dir
    kwargs['output_dir'] = Path('results') / processing_mode / sg_name / eval_dir
    kwargs['output'] = sg_name + '_' + processing_mode
    kwargs['agent_config_dir'] = 'agent_config'
    # kwargs['ste_dir'] = ''
    # kwargs['ste_averaging_method'] = 'metrics'
    kwargs['perf_measure'] = perf_measure[sg_name]
    # kwargs['aggregation_method'] = 'mean'
    kwargs['maintenance_method'] = 'both'
    kwargs['transfer_method'] = 'both'
    # kwargs['window_length'] = None
    # kwargs['show_eval_lines'] = True
    # kwargs['do_smooth_eval_data'] = False
    kwargs['do_store_ste'] = False
    kwargs['do_plot'] = True
    kwargs['do_save_plots'] = True
    kwargs['do_save'] = True
    kwargs['do_save_settings'] = True

    data_range_file = None # 'data_range.json'

    # Load data range data for normalization and standardize names to lowercase
    if data_range_file:
        with open(data_range_file) as f:
            data_range = json.load(f)
            data_range = {key.lower(): val for key, val in data_range.items()}
    else:
        data_range = None
    kwargs['data_range'] = data_range

    # Create output directory if it doesn't exist
    kwargs['output_dir'].mkdir(parents=True, exist_ok=True)

    # Generate other input arguments based on data processing mode
    kwargs['normalization_method'] = 'task' if processing_mode in [
        'normalized', 'normalized_no_outliers'] else 'none'
    kwargs['smoothing_method'] = 'flat' if processing_mode in [
        'smoothed', 'normalized', 'normalized_no_outliers'] else 'none'
    kwargs['clamp_outliers'] = processing_mode in ['normalized_no_outliers']

    ll_metrics_df, ll_metrics_dicts, regime_metrics_df, log_data_df = compute_eval_metrics(**kwargs)

    # Save the lifelong learning metrics DataFrame
    if kwargs['do_save']:
        if not ll_metrics_df.empty:
            with open(kwargs['output_dir'] / (kwargs['output'] + '.tsv'), 'w', newline='\n') as metrics_file:
                ll_metrics_df.set_index(['sg_name', 'agent_config', 'run_id']).sort_values(
                    ['agent_config', 'run_id']).to_csv(metrics_file, sep='\t')
        if ll_metrics_dicts:
            with open(kwargs['output_dir'] / (kwargs['output'] + '.json'), 'w', newline='\n') as metrics_file:
                json.dump(ll_metrics_dicts, metrics_file)
        if not regime_metrics_df.empty:
            with open(kwargs['output_dir'] / (kwargs['output'] + '_regime.tsv'), 'w', newline='\n') as metrics_file:
                regime_metrics_df.set_index(['sg_name', 'agent_config', 'run_id']).sort_values(
                    ['agent_config', 'run_id']).to_csv(metrics_file, sep='\t')
        if not log_data_df.empty:
            log_data_df.reset_index(drop=True).to_feather(kwargs['output_dir'] / (kwargs['output'] + '_data.feather'))
    
    # Save settings for evaluation
    if kwargs['do_save_settings']:
        with open(kwargs['output_dir'] / (kwargs['output'] + '_settings.json'), 'w') as outfile:
            kwargs['eval_dir'] = str(kwargs.get('eval_dir', ''))
            kwargs['output_dir'] = str(kwargs.get('output_dir', ''))
            json.dump(kwargs, outfile)


def run():
    # Configure metrics report
    eval_dirs = ['m15_eval']
    sg_names = ['argonne', 'hrl', 'sri', 'teledyne', 'upenn']
    processing_modes = ['raw', 'smoothed', 'normalized', 'normalized_no_outliers']

    # Parallel processing
    sg_configs = list(product(eval_dirs, sg_names, processing_modes))

    with Pool(psutil.cpu_count(logical=True)) as p:
        list(tqdm(p.imap(process_evaluation, sg_configs), total=len(sg_configs)))


if __name__ == '__main__':
    # Configure logger
    logging.basicConfig(level=logging.INFO)

    try:
        run()
    except (KeyError, ValueError) as e:
        logger.exception(e)

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
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import psutil
from tqdm import tqdm

from evaluation.evaluate import compute_eval_metrics

matplotlib.use('Agg')

perf_measure = {
    'argonne': 'score',
    'hrl': 'norm_reward',
    'sri': 'reward',
    'teledyne': 'id_accuracy_cumulative',
    'upenn': 'performance'
}


def process_evaluation(args):
    sg_name, config, ste_dir, do_save_ste, maintenance_method, transfer_method, \
        normalization_method, do_plot, save_plots, do_save = args

    # Build file and directory strings
    eval_dir = Path('../../sg_' + sg_name + '_eval/m9_eval/')
    output_dir = Path('results/' + config + '/' + sg_name + '_' + config)
    output = sg_name + '_metrics_' + config

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate other input arguments based on configuration
    do_smoothing = config in ['smoothed', 'normalized', 'normalized_no_outliers']
    do_normalize = config in ['normalized', 'normalized_no_outliers']
    remove_outliers = config in ['normalized_no_outliers']

    ll_metrics_df, ll_metrics_dicts = compute_eval_metrics(eval_dir=eval_dir, ste_dir=ste_dir,
                                                           output_dir=output_dir,
                                                           perf_measure=perf_measure[sg_name],
                                                           maintenance_method=maintenance_method,
                                                           transfer_method=transfer_method,
                                                           normalization_method=normalization_method,
                                                           do_smoothing=do_smoothing,
                                                           do_normalize=do_normalize,
                                                           remove_outliers=remove_outliers,
                                                           do_plot=do_plot,
                                                           save_plots=save_plots,
                                                           do_save_ste=do_save_ste)

    # Save the lifelong learning metrics DataFrame
    if do_save:
        with open(output_dir.parent / (output + '.tsv'), 'w', newline='\n') as metrics_file:
            ll_metrics_df.set_index(['sg_name', 'agent_config', 'run_id']).sort_values(
                ['agent_config', 'run_id']).to_csv(metrics_file, sep='\t')
        with open(output_dir.parent / (output + '.json'), 'w', newline='\n') as metrics_file:
            json.dump(ll_metrics_dicts, metrics_file)


def run():
    # Configure metrics report
    sg_names = ['argonne', 'hrl', 'sri', 'teledyne', 'upenn']
    configurations = ['raw', 'smoothed', 'normalized', 'normalized_no_outliers']

    ste_dir = ''
    maintenance_method = 'both'
    transfer_method = 'both'
    normalization_method = 'task'
    do_save_ste = False
    do_plot = True
    save_plots = True
    do_save = True

    # Parallel processing
    sg_configs = list(product(sg_names, configurations))
    other_args = (ste_dir, do_save_ste, maintenance_method, transfer_method,
                  normalization_method, do_plot, save_plots, do_save)
    par_args = [x + other_args for x in sg_configs]

    with Pool(psutil.cpu_count(logical=False)) as p:
        list(tqdm(p.imap(process_evaluation, par_args), total=len(par_args)))


if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()

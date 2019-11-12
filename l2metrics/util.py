import pandas as pd
import os
import json
from learnkit.data_util.utils import get_l2data_root


def get_l2root_base_dirs(directory_to_append, sub_to_get=None):
    # This function, get_l2data_root, references the environment variable, $L2DATA, else uses a default value
    file_info_to_return = os.path.join(get_l2data_root(), directory_to_append)

    if sub_to_get:
        base_dir = file_info_to_return
        file_info_to_return = os.path.join(base_dir, sub_to_get)

    return file_info_to_return


def load_default_ste_data():
    # This function will return a dictionary of the Single-Task-Expert data from all of the available single task
    # baselines that have been stored in this JSON file, located at $L2DATA/taskinfo/info.json
    json_file = get_l2root_base_dirs('taskinfo', 'info.json')

    # Load the defaults from the json file, return them as a dictionary
    with open(json_file) as f:
        ste_dict = json.load(f)

    return ste_dict


def read_log_data(dir, analysis_variables=None):
    logs = None
    blocks = None
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file == 'data-log.tsv':
                task = os.path.split(root)[-1]
                if analysis_variables is not None:
                    df = pd.read_csv(os.path.join(root, file), sep='\t')[
                        ['timestamp', 'class_name', 'phase', 'worker', 'block', 'task', 'seed'] + analysis_variables]
                else:
                    df = pd.read_csv(os.path.join(root, file), sep='\t')
                if logs is None:
                    logs = df
                else:
                    logs = pd.concat([logs, df])
            if file == 'block-report.tsv':
                df = pd.read_csv(os.path.join(root, file), sep='\t')
                if blocks is None:
                    blocks = df
                else:
                    blocks = pd.concat([blocks, df])

    return logs.merge(blocks, on=['phase', 'class_name', 'worker', 'block'])

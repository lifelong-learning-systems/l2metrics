import pandas as pd
import os

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

    return logs.merge(blocks, on = ['phase', 'class_name', 'worker', 'block'])

from collections import defaultdict
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

from typing import Tuple

import pandas as pd

from ._localutil import fill_metrics_df, get_terminal_perf
from .core import Metric


class RecoveryTime(Metric):
    name = "Recovery Time"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates whether the system recovers after a change of task or parameters and \
        calculate how long it takes if recovery is achieved"

    def __init__(self, perf_measure: str) -> None:
        super().__init__()
        self.perf_measure = perf_measure

    def validate(self, block_info: pd.DataFrame) -> Tuple[dict, dict]:
        # Get unique tasks
        unique_tasks = block_info.loc[:, 'task_name'].unique()

        # Determine where we need to assess recovery time for each task
        ref_indices = defaultdict(list)
        assess_indices = defaultdict(list)

        for task in unique_tasks:
            # Get train blocks in order of appearance
            tr_block_info = block_info.sort_index().loc[(block_info['block_type'] == 'train') &
                                                        (block_info['block_subtype'] == 'wake') &
                                                        (block_info['task_name'] == task),
                                                        ['task_name']]
            tr_indices = tr_block_info.index

            # Regimes are defined as new combinations of tasks and params, but can repeat,
            # so check for changes across regimes
            first = True
            for idx, block_idx in enumerate(tr_indices):
                if first:
                    first = False
                    continue
                assess_indices[task].append(block_idx)
                ref_indices[task].append(tr_indices[idx - 1])

        if not (ref_indices and assess_indices):
            raise Exception('Not enough blocks to assess recovery time')

        return ref_indices, assess_indices

    def calculate(self, dataframe: pd.DataFrame, block_info: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Get the places where we should calculate recovery time
            ref_indices, assess_indices = self.validate(block_info)

            # Initialize metric dictionary
            recovery_time = {}

            # Iterate over indices for computing recovery time
            for (_, ref_vals), (_, assess_vals) in zip(ref_indices.items(), assess_indices.items()):
                for ref_ind, assess_ind in zip(ref_vals, assess_vals):
                    prev_val = metrics_df['term_perf'][ref_ind]
                    block_data = dataframe.loc[assess_ind]

                    # Check for proper number of rows in block data
                    if block_data.ndim == 1:
                        block_data = pd.DataFrame(block_data).T

                    _, _, eps_to_rec = get_terminal_perf(block_data,
                                                         col_to_use=self.perf_measure,
                                                         prev_val=prev_val)
                    recovery_time[assess_ind] = eps_to_rec

            return fill_metrics_df(recovery_time, 'recovery_time', metrics_df)
        except Exception as e:
            print(f"Cannot compute {self.name} - {e}")
            return metrics_df

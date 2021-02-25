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

import numpy as np
import pandas as pd
from scipy import stats

from ._localutil import fill_metrics_df
from .core import Metric


class PerformanceRecovery(Metric):
    name = "Performance Recovery"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the performance recovery value corresponding to a change of task or parameters"

    def __init__(self, perf_measure: str) -> None:
        super().__init__()
        self.perf_measure = perf_measure

    def validate(self, metrics_df: pd.DataFrame) -> None:
        # Get number of recovery times
        if 'recovery_time' in metrics_df.columns:
            r = metrics_df['recovery_time']
            r = r[r.notna()]
            r_count = r.count()

            if r_count <= 1:
                raise Exception('Not enough recovery times to assess performance recovery')
        else:
            raise Exception('No recovery times')

    def calculate(self, dataframe: pd.DataFrame, block_info: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Get the places where we should calculate recovery time
            self.validate(metrics_df)

            # Initialize metric dictionary
            pr_values = {}

            # Calculate performance recovery for each task
            for task in block_info.loc[:, 'task_name'].unique():
                r = metrics_df[metrics_df['task_name'] == task]['recovery_time']
                r = r[r.notna()]

                # Get Theil-Sen slope
                y = np.array(r)

                if len(y) > 1:
                    slope, _, _, _ = stats.theilslopes(y)

                    # Set performance recovery value as slope
                    idx = block_info[block_info['task_name'] == task]['regime_num'].max()
                    pr_values[idx] = -slope
                else:
                    print(f"Cannot compute {self.name} for task {task} - Not enough recovery times")

            return fill_metrics_df(pr_values, 'perf_recovery', metrics_df)
        except Exception as e:
            print(f"Cannot compute {self.name} - {e}")
            return metrics_df

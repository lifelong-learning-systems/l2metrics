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

import warnings
from typing import Tuple

import numpy as np
import pandas as pd

from ._localutil import fill_metrics_df, get_block_saturation_perf
from .core import Metric
from .util import get_ste_data_names, load_ste_data


class SampleEfficiency(Metric):
    name = "Sample Efficiency"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'agent'}
    description = "Calculates the sample efficiency relative to the single-task expert"

    def __init__(self, perf_measure: str, do_normalize: bool = False,
                 min_max_scale: Tuple[int, int, int] = (0, 100, 100)) -> None:
        super().__init__()
        self.perf_measure = perf_measure
        self.do_normalize = do_normalize
        self.min_max_scale = min_max_scale

    def validate(self, block_info: pd.DataFrame) -> pd.DataFrame:
        # Check if there is STE data for each task in the scenario
        unique_tasks = block_info.loc[:, 'task_name'].unique()
        ste_names = get_ste_data_names()

        # Raise exception if none of the tasks have STE data
        if ~np.any(np.isin(unique_tasks, ste_names)):
            raise Exception('No STE data available for any task')

        # Make sure STE baselines are available for all tasks, else send warning
        if ~np.all(np.isin(unique_tasks, ste_names)):
            warnings.warn('STE data not available for all tasks')

    def calculate(self, dataframe: pd.DataFrame, block_info: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Validate the STE
            self.validate(block_info)

            # Initialize metric dictionaries
            se_saturation = {}
            se_eps_to_sat = {}
            sample_efficiency = {}

            # Iterate through unique tasks and STE
            unique_tasks = block_info.loc[:, 'task_name'].unique()

            for task in unique_tasks:
                # Get block info for task during training
                task_blocks = block_info[(block_info['task_name'] == task) & (
                    block_info['block_type'] == 'train')]

                # Get data concatenated data for task
                task_data = dataframe[dataframe['regime_num'].isin(task_blocks['regime_num'])]

                if len(task_data):
                    # Load STE data
                    ste_data = load_ste_data(task)

                    if ste_data is not None:
                        # Check if performance measure exists in STE data
                        if self.perf_measure in ste_data.columns:
                            if self.do_normalize:
                                norm_data = (ste_data[self.perf_measure].values - self.min_max_scale[0]) / (
                                    self.min_max_scale[1] - self.min_max_scale[0]) * self.min_max_scale[2]
                                ste_data[self.perf_measure] = norm_data

                            # Get task saturation value and episodes to saturation
                            task_saturation, task_eps_to_sat, _ = get_block_saturation_perf(
                                task_data, col_to_use=self.perf_measure)

                            # Get STE saturation value and episodes to saturation
                            ste_saturation, ste_eps_to_sat, _ = get_block_saturation_perf(
                                ste_data, col_to_use=self.perf_measure)

                            # Compute sample efficiency
                            se_saturation[task_data['regime_num'].iloc[-1]] = task_saturation / ste_saturation
                            se_eps_to_sat[task_data['regime_num'].iloc[-1]] = ste_eps_to_sat / task_eps_to_sat
                            sample_efficiency[task_data['regime_num'].iloc[-1]] = \
                                (task_saturation / ste_saturation) * (ste_eps_to_sat / task_eps_to_sat)
                        else:
                            print(f"Cannot compute {self.name} for task {task} - Performance measure not in STE data")
                    else:
                        print(f"Cannot compute {self.name} for task {task} - No STE data available")

            metrics_df = fill_metrics_df(se_saturation, 'se_saturation', metrics_df)
            metrics_df = fill_metrics_df(se_eps_to_sat, 'se_eps_to_sat', metrics_df)
            return fill_metrics_df(sample_efficiency, 'sample_efficiency', metrics_df)
        except Exception as e:
            print(f"Cannot compute {self.name} - {e}")
            return metrics_df

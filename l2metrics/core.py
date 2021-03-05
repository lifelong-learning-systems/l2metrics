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

import abc

import pandas as pd


class Metric(abc.ABC):
    """
    A single metric
    """
    @property
    def capability(self):
        """
        A string (one of the core capabilities that the metric calculates):
            continual_learning
            adapt_to_new_tasks
            goal_driven_perception
            selective_plasticity
            safety_and_monitoring            
        """
        pass

    @property
    def name(self):
        """
        A short label that uniquely identifies this metric. 
        """
        return ""

    @property
    def description(self):
        """
        A more detailed description for this metric
        """
        return ""

    @property
    def requires(self):
        """
        A dictionary of requirements for this metric. Keys 
          syllabus_type: one of 'type1", "type2", etc.
        """
        return {}

    @abc.abstractmethod
    def calculate(self, dataframe: pd.DataFrame, block_info: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate metric

        Args:
            dataframe (pd.DataFrame): Dataframe containing log data
            block_info (pd.DataFrame): High-level block summary of log data
            metrics_df (pd.DataFrame): Incremental Dataframe with columns corresponding to
                calculated metrics along with some of the block_info information.

        Returns:
            pd.DataFrame: Updated metrics dataframe.
        """
        return None

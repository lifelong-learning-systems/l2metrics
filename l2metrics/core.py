"""
Copyright © 2021-2022 The Johns Hopkins University Applied Physics Laboratory LLC

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
    def calculate(
        self,
        dataframe: pd.DataFrame,
        block_info: pd.DataFrame,
        metrics_df: pd.DataFrame,
    ) -> pd.DataFrame:
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

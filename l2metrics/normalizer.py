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

import logging
from collections import defaultdict
from typing import Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Normalizer:
    """Utility class for normalizing data."""

    valid_methods = ["task", "run"]

    def __init__(
        self,
        perf_measure: str,
        data: pd.DataFrame,
        ste_data: dict = {},
        data_range: defaultdict = None,
        method: str = "task",
        scale: int = 100,
        offset: int = 1,
    ) -> None:
        """Constructor for Normalizer.

        Args:
            perf_measure (str): Name of column to use for metrics calculations.
            data (pd.DataFrame, optional): Reference data for calculating data range. Assumed
                DataFrame with task name as index and one column of performance data.
            ste_data (dict, optional): The STE data for computing quantiles. Defaults to {}.
            data_range (defaultdict, optional): Dictionary object for data range. Defaults to None.
            method (str, optional): Normalization method. Valid values are 'task' and 'run'.
                Defaults to 'task'.
            scale (int, optional): Normalization scale, interpreted as from 0 to this value.
                Defaults to 100.
            offset (int, optional): Offset to normalized data. Defaults to 1.

        Raises:
            ValueError: If data range validation fails.
        """

        self.perf_measure = perf_measure

        # Get unique task names in data
        self.unique_tasks = set(data.index.unique())

        if data_range is not None:
            # Validate and set data range for normalizer
            if self._validate_data_range(
                data_range=data_range, task_names=self.unique_tasks
            ):
                self.data_range = data_range
                self.run_min = min([val["min"] for val in self.data_range.values()])
                self.run_max = max([val["max"] for val in self.data_range.values()])
        elif data is not None:
            self.calculate_data_range(data, ste_data)
        else:
            raise ValueError(
                f"Must provide data or data range to initialize Normalizer"
            )

        if self._validate_method(method):
            self.method = method

        if self._validate_scale(scale):
            self.scale = scale

        if self._validate_offset(offset):
            self.offset = offset

    def calculate_data_range(self, data: pd.DataFrame, ste_data: dict = {}) -> None:
        """Calculates data range per task for given data.

        A task data range is the minimum and maximum value of the task performance.

        Args:
            data (pd.DataFrame): Reference data for calculating data range. Assumed
                DataFrame with task name as index and one column of performance data.
            ste_data (dict, optional): The STE data for computing quantiles. Defaults to {}.

        Raises:
            ValueError: If data contains more than just performance values and task name.
        """

        data_column = data.columns.to_numpy()

        if len(data_column) > 1:
            raise ValueError(
                f"Data must only have one column with performance measures"
            )

        # Initialize data range as empty object
        self.data_range = defaultdict(dict)

        # Get data range over scenario and STE data if not provided as input
        for task in self.unique_tasks:
            # Get feature range for each task
            task_min = np.nanmin(data.loc[task])
            task_max = np.nanmax(data.loc[task])

            if ste_data.get(task):
                x_ste = np.concatenate(
                    [
                        ste_data_df[ste_data_df["block_type"] == "train"][
                            self.perf_measure
                        ].to_numpy()
                        for ste_data_df in ste_data.get(task)
                    ]
                )
                self.data_range[task]["min"] = min(task_min, np.nanmin(x_ste))
                self.data_range[task]["max"] = max(task_max, np.nanmax(x_ste))
            else:
                self.data_range[task]["min"] = task_min
                self.data_range[task]["max"] = task_max

        self.run_min = min([val["min"] for val in self.data_range.values()])
        self.run_max = max([val["max"] for val in self.data_range.values()])

    def _validate_data_range(
        self, data_range: defaultdict, task_names: Set[str]
    ) -> bool:
        """Validates data range object.

        Args:
            data_range (defaultdict): Dictionary object for data range.
            task_names (Set[str]): Set of task names in the data.

        Raises:
            TypeError: If data range is not a dictionary object.
            KeyError: If data range is not defined for all tasks.
            KeyError: If the keys min and max are missing.

        Returns:
            bool: True if validation succeeds.
        """

        if not isinstance(data_range, (dict, defaultdict)):
            raise TypeError(f"Invalid data range type - Must be a dictionary")
        elif not set(data_range.keys()).issuperset(task_names):
            raise KeyError(f"Data range not defined for all tasks: {task_names}")
        elif False in [key.keys() >= {"min", "max"} for key in data_range.values()]:
            raise KeyError(f"Missing required fields: min and max")
        else:
            return True

    def _validate_method(self, method: str) -> bool:
        """Validates normalization method.

        Args:
            method (str): Normalization method.

        Raises:
            ValueError: If method is not in list of valid methods.

        Returns:
            bool: True if validation succeeds.
        """

        if method not in self.valid_methods:
            raise ValueError(
                f"Invalid normalization method: {method}\n"
                f"Valid methods are: {self.valid_methods}"
            )
        else:
            return True

    def _validate_scale(self, scale: int) -> bool:
        """Validates normalization scale.

        Args:
            scale (int): Normalization scale.

        Raises:
            TypeError: If scale is not an integer.
            ValueError: If scale is less than or equal to 0.

        Returns:
            bool: True if validation succeeds.
        """

        if not isinstance(scale, int):
            raise TypeError(f"Invalid scale type: {type(scale)}")
        elif scale <= 0:
            raise ValueError(f"Scale value must be greater than 0: {scale}")
        else:
            return True

    def _validate_offset(self, offset: int) -> bool:
        """Validates normalization offset.

        Args:
            offset (int): Normalization offset.

        Raises:
            TypeError: If offset is not an integer.
            ValueError: If scale is less than or equal to 0.

        Returns:
            bool: True if validation succeeds.
        """

        if not isinstance(offset, int):
            raise TypeError(f"Invalid offset type: {type(offset)}")
        elif offset <= 0:
            raise ValueError(f"Offset value must be greater than 0: {offset}")
        else:
            return True

    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalizes the given data with the current instance method and data range/scale.

        Args:
            data (pd.DataFrame): Dataframe to be normalized.

        Raises:
            KeyError: If there's a missing data range for any task.

        Returns:
            pd.DataFrame: Normalized dataframe.
        """

        if self.method == "task":
            for task in data["task_name"].unique():
                if task in self.data_range.keys():
                    task_min = self.data_range[task]["min"]
                    task_max = self.data_range[task]["max"]
                    task_data = data.loc[
                        data["task_name"] == task, self.perf_measure
                    ].to_numpy()

                    if task_min == task_max:
                        data.loc[data["task_name"] == task, self.perf_measure] = (
                            task_data * 0
                        ) + self.offset
                        logger.warning(
                            f"Performance for task ({task}) is constant - normalizing to 0"
                        )
                    else:
                        data.loc[data["task_name"] == task, self.perf_measure] = (
                            (task_data - task_min) / (task_max - task_min) * self.scale
                        ) + self.offset
                else:
                    raise KeyError(f"Missing data range for task '{task}'")
            return data
        elif self.method == "run":
            data.loc[:, self.perf_measure] = (
                (data[self.perf_measure].to_numpy() - self.run_min)
                / (self.run_max - self.run_min)
                * self.scale
            ) + self.offset
            return data

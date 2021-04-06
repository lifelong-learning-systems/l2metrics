from collections import defaultdict
from typing import Set

import numpy as np
import pandas as pd

from .util import load_ste_data


class Normalizer():
    """Utility class for normalizing data.
    """

    valid_methods = ['task', 'run']

    def __init__(self, perf_measure: str, data: pd.DataFrame, data_range: defaultdict = None,
                 method: str = 'task', scale: int = 100) -> None:
        """Constructor for Normalizer.

        Args:
            perf_measure (str): Name of column to use for metrics calculations.
            data (pd.DataFrame, optional): Reference data for calculating data range. Assumed
                DataFrame with task name as index and one column of performance data.
            data_range (defaultdict, optional): Dictionary object for data range. Defaults to None.
            method (str, optional): Normalization method. Valid values are 'task' and 'run'.
                Defaults to 'task'.
            scale (int, optional): Normalization scale, interpreted as from 0 to this value.
                Defaults to 100.

        Raises:
            Exception: If input validation fails.
        """

        self.perf_measure = perf_measure

        # Get unique task names in data
        self.unique_tasks = set(data.index.unique())

        if data_range is not None:
            # Validate and set data range for normalizer
            if self._validate_data_range(data_range=data_range, task_names=self.unique_tasks):
                self.data_range = data_range
                self.run_min = min([val['min'] for val in self.data_range.values()])
                self.run_max = max([val['max'] for val in self.data_range.values()])
        elif data is not None:
            self.calculate_data_range(data)
        else:
            raise Exception(f'Must provide data or data range to initialize Normalizer')

        if self._validate_method(method):
            self.method = method
        
        if self._validate_scale(scale):
            self.scale = scale

    def calculate_data_range(self, data: pd.DataFrame) -> None:
        """Calculates data range per task for given data.

        A task data range is the minimum and maximum value of the task performance.

        Args:
            data (pd.DataFrame): Reference data for calculating data range. Assumed
                DataFrame with task name as index and one column of performance data.

        Raises:
            Exception: If data contains more than just performance values and task name.
        """

        data_column = data.columns.values

        if len(data_column) > 1:
            raise Exception(f'Data must only have one column with performance measures')

        # Initialize data range as empty object
        self.data_range = defaultdict(dict)

        # Get data range over scenario and STE data if not provided as input
        for task in self.unique_tasks:
            # Get feature range for each task
            task_min = np.nanmin(data.loc[task])
            task_max = np.nanmax(data.loc[task])

            # Load STE data
            ste_data = load_ste_data(task)
            ste_data = ste_data[ste_data['block_type'] == 'train']

            if ste_data is not None:
                if self.perf_measure in ste_data.columns:
                    self.data_range[task]['min'] = min(task_min, np.nanmin(ste_data[self.perf_measure]))
                    self.data_range[task]['max'] = max(task_max, np.nanmax(ste_data[self.perf_measure]))
            else:
                self.data_range[task]['min'] = task_min
                self.data_range[task]['max'] = task_max

        self.run_min = min([val['min'] for val in self.data_range.values()])
        self.run_max = max([val['max'] for val in self.data_range.values()])

    def _validate_data_range(self, data_range: defaultdict, task_names: Set[str]) -> bool:
        """Validates data range object.

        Args:
            data_range (defaultdict): Dictionary object for data range.
            task_names (Set[str]): Set of task names in the data.

        Raises:
            Exception: If data range is not a dictionary object.
            Exception: If data range is not defined for all tasks.
            Exception: If the keys min and max are missing.

        Returns:
            bool: True if validation succeeds.
        """

        if not isinstance(data_range, (dict, defaultdict)):
            raise Exception(f'Invalid data range type - Must be a dictionary')
        elif not set(data_range.keys()).issuperset(task_names):
            raise Exception(f'Data range not defined for all tasks')
        elif False in [key.keys() >= {'min', 'max'} for key in data_range.values()]:
            raise Exception(f'Missing required fields: min and max')
        else:
            return True

    def _validate_method(self, method: str) -> bool:
        """Validates normalization method.

        Args:
            method (str): Normalization method.

        Raises:
            Exception: If method is not in list of valid methods.

        Returns:
            bool: True if validation succeeds.
        """

        if method not in self.valid_methods:
            raise Exception(f'Invalid normalization method: {method}\n'
                            f'Valid methods are: {self.valid_methods}')
        else:
            return True

    def _validate_scale(self, scale: int) -> bool:
        """Validates normalization scale.

        Args:
            scale (int): Normalization scale.

        Raises:
            Exception: If scale is not an integer.
            Exception: If scale is less than or equal to 0.

        Returns:
            bool: True if validation succeeds.
        """

        if not isinstance(scale, int):
            raise Exception(f'Invalid scale type: {type(scale)}')
        elif scale <= 0:
            raise Exception(f'Scale value must be greater than 0: {scale}')
        else:
            return True

    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalizes the given data with the current instance method and data range/scale. 

        Args:
            data (pd.DataFrame): Dataframe to be normalized

        Returns:
            pd.DataFrame: Normalized dataframe.
        """

        if self.method == 'task':
            for task in data['task_name'].unique():
                if task in self.data_range.keys():
                    task_min = self.data_range[task]['min']
                    task_max = self.data_range[task]['max']

                    data.loc[data.task_name == task, self.perf_measure] = \
                        (data[data['task_name'] == task][self.perf_measure].values - task_min) / \
                        (task_max - task_min) * self.scale
                else:
                    raise Exception(f"Missing data range for task '{task}'")
            return data
        elif self.method == 'run':
            data.loc[:, self.perf_measure] = (data[self.perf_measure].values - self.run_min) / \
                (self.run_max - self.run_min) * self.scale
            return data

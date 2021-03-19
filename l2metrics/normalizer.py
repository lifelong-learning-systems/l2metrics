from collections import defaultdict

import numpy as np
import pandas as pd

from .util import load_ste_data


class Normalizer():
    """Utility class for normalizing data.
    """

    valid_methods = ['task', 'run']

    def __init__(self, perf_measure: str, data: pd.DataFrame = None, data_range: defaultdict = None,
                 method: str = 'task', scale: int = 100) -> None:
        """Constructor for Normalizer.

        Args:
            perf_measure (str): Name of column to use for metrics calculations.
            data (pd.DataFrame, optional): Reference data for calculating data range. Assumed
                DataFrame with task name as index and one column of performance data. Defaults to None.
            data_range (defaultdict, optional): Dictionary object for data range. Defaults to None.
            method (str, optional): Normalization method. Valid values are 'task' and 'run'.
                Defaults to 'task'.
            scale (int, optional): Normalization scale, interpreted as from 0 to this value.
                Defaults to 100.

        Raises:
            Exception: If input validation fails.
        """

        self.perf_measure = perf_measure

        if data_range is not None:
            # Validate and set data range for normalizer
            if self._validate_data_range(data_range=data_range):
                self.data_range = data_range
                self.run_min = min([val for _, val in self.data_range['min'].items()])
                self.run_max = max([val for _, val in self.data_range['max'].items()])
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

        # Get unique task names in data
        self.unique_tasks = set(data.index.unique())

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

            if ste_data is not None:
                if self.perf_measure in ste_data.columns:
                    self.data_range['min'][task] = min(task_min, np.nanmin(ste_data[self.perf_measure]))
                    self.data_range['max'][task] = max(task_max, np.nanmax(ste_data[self.perf_measure]))
            else:
                self.data_range['min'][task] = task_min
                self.data_range['max'][task] = task_max

        self.run_min = min(self.data_range['min'].values())
        self.run_max = max(self.data_range['max'].values())

    def _validate_data_range(self, data_range: defaultdict) -> bool:
        """Validates data range object.

        Args:
            data_range (defaultdict): Dictionary object for data range.

        Raises:
            Exception: If data range is not a dictionary object.
            Exception: If the keys min and max are missing.
            Exception: If either min or max objects are empty.

        Returns:
            bool: [description]
        """

        if not isinstance(data_range, (dict, defaultdict)):
            raise Exception(f'Invalid data range type - Must be a dictionary')
        elif not data_range.keys() >= {'min', 'max'}:
            raise Exception(f'Missing required fields: min and max')
        elif not (data_range['min'] and data_range['max']):
            raise Exception(f'Missing values in data range')
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
                try:
                    data.loc[data.task_name == task, self.perf_measure] = \
                        (data[data['task_name'] == task][self.perf_measure].values - self.data_range['min'][task]) / \
                        (self.data_range['max'][task] - self.data_range['min'][task]) * self.scale
                except Exception as e:
                    print(e)
            return data
        elif self.method == 'run':
            data.loc[:, self.perf_measure] = (data[self.perf_measure].values - self.run_min) / \
                (self.run_max - self.run_min) * self.scale
            return data

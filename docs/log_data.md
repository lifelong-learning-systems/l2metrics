# Log Data

If saving is enabled, the framework will generate a [Feather file](https://arrow.apache.org/docs/python/feather.html) containing the raw and preprocessed log data from the scenario. This file can be easily read as a pandas.DataFrame in Python using the `read_feather()` function or as a Table using the `read_table()` function:

```python
import pandas as pd
import pyarrow.feather as feather

# Result is pandas.DataFrame
read_df = pd.read_feather('/path/to/file')
read_df = feather.read_feather('/path/to/file')

# Result is pyarrow.Table
read_arrow = feather.read_table('/path/to/file')
```

The Feather file contains the following columns:

- `regime_num`: Regime number, defined as unique block number, block type, task name, and task parameter combination
- `block_num`: Block number from scenario definition
- `block_type`: Block type from scenario definition
- `block_subtype`: Block subtype from scenario definition
- `exp_num`: Experience number
- `worker_id`: Worker ID
- `task_name`: Task name
- `task_params`: Task parameters
- `exp_status`: Experience status (complete)
- `timestamp`: Timestamp
- `performance`: Application-specific measure of performance, processed and used for computing metrics
- `performance_raw`: Raw application-specific measure of performance
- `performance_smoothed`: Smoothed application-specific measure of performance
- `performance_normalized`: Smoothed and normalized application-specific measure of performance

As alluded to above, the Metrics Framework stores all intermediate values of the performance measure during preprocessing, following the order of operations (smooth -> clamp outliers -> normalize). The original column (e.g., `performance`) is overwritten after each step in the data preprocessing and is used by the framework to compute metrics.

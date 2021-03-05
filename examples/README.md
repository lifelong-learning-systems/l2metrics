# Custom Metrics with Lifelong Learning Metrics (L2Metrics)

This directory contains a Python script with an example for creating custom metrics, plotting, and generating a metrics report:

- `calc_metrics.py`:

```
usage: calc_metrics.py [-h] -l LOG_DIR [-s] [-p PERF_MEASURE]
                   [-m {mrtlp,mrlep,both}] [-t {contrast,ratio,both}]
                   [-n {task,run}] [-f DATA_RANGE_FILE] [--noise MEAN STD]
                   [-o OUTPUT] [--no-smoothing] [-r] [--normalize]
                   [--remove-outliers] [--no-plot] [--no-save]

Run L2Metrics from the command line

required arguments:
  -l LOG_DIR, --log-dir LOG_DIR
                        Log directory of scenario

optional arguments:
  -h, --help            show this help message and exit
  -s, --store-ste-data  Flag for storing log data as STE
  -p PERF_MEASURE, --perf-measure PERF_MEASURE
                        Name of column to use for metrics calculations
  -m {mrtlp,mrlep,both}, --maintenance-method {mrtlp,mrlep,both}
                        Method for computing performance maintenance
  -t {contrast,ratio,both}, --transfer-method {contrast,ratio,both}
                        Method for computing forward and backward transfer
  -n {task,run}, --normalization-method {task,run}
                        Method for normalizing data
  -f DATA_RANGE_FILE, --data-range-file DATA_RANGE_FILE
                        JSON file containing task performance ranges for
                        normalization
  --noise MEAN STD      Mean and standard deviation for Gaussian noise in log
                        data
  -o OUTPUT, --output OUTPUT
                        Specify output filename for plot and results
  --no-smoothing        Do not smooth data for metrics and plotting
  -r, --show-raw-data   Show raw data points under smoothed data for plotting
  --normalize           Normalize data for metrics
  --remove-outliers     Remove outliers in data for metrics
  --no-plot             Do not plot performance
  --no-save             Do not save metrics outputs
```

## Writing a Custom Metric

The file `calc_metrics.py` demonstrates how to add custom metrics to a metrics report. Data from the logs is provided to the calculate method, where the actual calculation of your metric should live. An example of an agent metric, max value, is provided for your edification. To add this metric to the list of metrics calculated for a metrics report, simply invoke the following method:

```Python
metrics_report.add(MyCustomAgentMetric(perf_measure))
```

and it will be added to the end of the list in addition to the defaults.

The custom metric class should have a method with the following name and inputs:

```Python
calculate(self, dataframe, block_info, metrics_df):
```

Inputs:

- :param dataframe: Pandas dataframe of log data collated from log files
- :param block_info: Pandas dataframe with high-level information about blocks extracted from log data; contains no reward/score columns
- :param metrics_df: Pandas dataframe with columns corresponding to calculated metrics along with some of the block_info information. Prints out at the end for reporting

Output:

- :return: metrics_df: Pandas dataframe, updated with columns corresponding to previously calculated metrics

## Notes

1. The calculate methods for each metric in self.\_metrics are called **in the order they were added**. Thus, you may choose to leverage previously calculated metrics for your subsequent calculations.
2. "NaN" is the default value in the metrics_df for blocks which do not receive a value.
3. To avoid adding your computed metric to the metrics_df, simply do not include the call to \_localutil.fill_metrics_df, which takes the values you pass and returns a dataframe with those values.
4. The scenario info files in the example log directories are generated during logging, but are not used for calculating metrics. This is typically where one would save any desired metadata from a scenario such as author, source of logs, scenario version, etc.

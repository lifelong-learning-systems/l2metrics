# Custom Metrics with Lifelong Learning Metrics (L2Metrics)

This directory contains a Python script with an example for creating custom metrics, plotting, and generating a metrics report:

- `calc_metrics.py`:

```
usage: calc_metrics.py [-h] [-l LOG_DIR] [-R] [-s {w,a}] [-v {time,metrics}]
                       [-p PERF_MEASURE] [-a {mean,median}]
                       [-m {mrlep,mrtlp,both}] [-t {ratio,contrast,both}]
                       [-n {task,run,none}]
                       [-g {flat,hanning,hamming,bartlett,blackman,none}]
                       [-w WINDOW_LENGTH] [-x] [-d DATA_RANGE_FILE]
                       [-N MEAN STD] [-O OUTPUT_DIR] [-o OUTPUT] [-r] [-e]
                       [--no-show-eval-lines] [-P] [--no-plot] [-S]
                       [--no-save] [-c LOAD_SETTINGS] [-C]
                       [--no-save-settings]

Run L2Metrics from the command line

optional arguments:
  -h, --help            show this help message and exit
  -l LOG_DIR, --log-dir LOG_DIR
                        Log directory of scenario. Defaults to None.
  -R, --recursive       Recursively compute metrics on logs found in specified
                        directory. Defaults to false.
  -s {w,a}, --ste-store-mode {w,a}
                        Mode for storing log data as STE, overwrite (w) or
                        append (a). Defaults to None.
  -v {time,metrics}, --ste-averaging-method {time,metrics}
                        Method for handling STE runs, time-series averaging
                        (time) or LL metric averaging (metrics). Defaults to
                        time.
  -p PERF_MEASURE, --perf-measure PERF_MEASURE
                        Name of column to use for metrics calculations.
                        Defaults to reward.
  -a {mean,median}, --aggregation-method {mean,median}
                        Method for aggregating within-lifetime metrics.
                        Defaults to mean.
  -m {mrlep,mrtlp,both}, --maintenance-method {mrlep,mrtlp,both}
                        Method for computing performance maintenance. Defaults
                        to mrlep.
  -t {ratio,contrast,both}, --transfer-method {ratio,contrast,both}
                        Method for computing forward and backward transfer.
                        Defaults to ratio.
  -n {task,run,none}, --normalization-method {task,run,none}
                        Method for normalizing data. Defaults to task.
  -g {flat,hanning,hamming,bartlett,blackman,none}, --smoothing-method {flat,hanning,hamming,bartlett,blackman,none}
                        Method for smoothing data, window type. Defaults to
                        flat.
  -w WINDOW_LENGTH, --window-length WINDOW_LENGTH
                        Window length for smoothing data. Defaults to None.
  -x, --clamp-outliers  Remove outliers in data for metrics by clamping to
                        quantiles. Defaults to false.
  -d DATA_RANGE_FILE, --data-range-file DATA_RANGE_FILE
                        JSON file containing task performance ranges for
                        normalization. Defaults to None.
  -N MEAN STD, --noise MEAN STD
                        Mean and standard deviation for Gaussian noise in log
                        data. Defaults to [0, 0].
  -O OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory for output files. Defaults to results.
  -o OUTPUT, --output OUTPUT
                        Specify output filename for plot and results. Defaults
                        to None.
  -r, --show-raw-data   Show raw data points under smoothed data for plotting.
                        Defaults to false.
  -e, --show-eval-lines
                        Show lines between evaluation blocks. Defaults to
                        true.
  --no-show-eval-lines  Do not show lines between evaluation blocks
  -P, --do-plot         Plot performance. Defaults to true.
  --no-plot             Do not plot performance
  -S, --do-save         Save metrics outputs. Defaults to true.
  --no-save             Do not save metrics outputs
  -c LOAD_SETTINGS, --load-settings LOAD_SETTINGS
                        Load L2Metrics settings from JSON file. Defaults to
                        None.
  -C, --do-save-settings
                        Save L2Metrics settings to JSON file. Defaults to
                        true.
  --no-save-settings    Do not save L2Metrics settings to JSON file
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

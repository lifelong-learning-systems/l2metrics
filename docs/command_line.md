# Command-Line Execution

This document describes how to run L2Metrics from the command line.

```text
usage: python -m l2metrics [-h] [-l LOG_DIR] [-R] [-r {aware,agnostic}] [-s {w,a}]
                    [-v {metrics,time}] [-p PERF_MEASURE] [-a {mean,median}]
                    [-m {mrlep,mrtlp,both}] [-t {ratio,contrast,both}] [-n {task,run,none}]
                    [-g {flat,hanning,hamming,bartlett,blackman,none}] [-G]
                    [-w WINDOW_LENGTH] [-x] [-d DATA_RANGE_FILE] [-N MEAN STD]
                    [-O OUTPUT_DIR] [-o OUTPUT] [-e] [--no-show-eval-lines] [-P]
                    [--no-plot] [-S] [--no-save]
                    [-c LOAD_SETTINGS] [-C] [--no-save-settings]

Run L2Metrics from the command line

optional arguments:
  -h, --help            show this help message and exit
  -l LOG_DIR, --log-dir LOG_DIR
                        Log directory of scenario. Defaults to None.
  -R, --recursive       Recursively compute metrics on logs found in specified directory. Defaults to false.
  -r {aware,agnostic}, --variant-mode {aware,agnostic}
                        Mode for computing metrics with respect to task variants. Defaults to aware.
  -s {w,a}, --ste-store-mode {w,a}
                        Mode for storing log data as STE, overwrite (w) or append (a). Defaults to None.
  -v {metrics,time}, --ste-averaging-method {metrics,time}
                        Method for handling STE runs, LL metric averaging (metrics) or time-series averaging (time). Defaults to metrics.
  -p PERF_MEASURE, --perf-measure PERF_MEASURE
                        Name of column to use for metrics calculations. Defaults to reward.
  -a {mean,median}, --aggregation-method {mean,median}
                        Method for aggregating within-lifetime metrics. Defaults to mean.
  -m {mrlep,mrtlp,both}, --maintenance-method {mrlep,mrtlp,both}
                        Method for computing performance maintenance. Defaults to mrlep.
  -t {ratio,contrast,both}, --transfer-method {ratio,contrast,both}
                        Method for computing forward and backward transfer. Defaults to ratio.
  -n {task,run,none}, --normalization-method {task,run,none}
                        Method for normalizing data. Defaults to task.
  -g {flat,hanning,hamming,bartlett,blackman,none}, --smoothing-method {flat,hanning,hamming,bartlett,blackman,none}
                        Method for smoothing data, window type. Defaults to flat.
  -G, --smooth-eval-data
                        Smooth evaluation block data. Defaults to false.
  -w WINDOW_LENGTH, --window-length WINDOW_LENGTH
                        Window length for smoothing data. Defaults to None.
  -x, --clamp-outliers  Remove outliers in data for metrics by clamping to quantiles. Defaults to false.
  -d DATA_RANGE_FILE, --data-range-file DATA_RANGE_FILE
                        JSON file containing task performance ranges for normalization. Defaults to None.
  -N MEAN STD, --noise MEAN STD
                        Mean and standard deviation for Gaussian noise in log data. Defaults to [0, 0].
  -O OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory for output files. Defaults to results.
  -o OUTPUT, --output OUTPUT
                        Specify output filename for plot and results. Defaults to None.
  -e, --show-eval-lines
                        Show lines between evaluation blocks. Defaults to true.
  --no-show-eval-lines  Do not show lines between evaluation blocks
  -P, --do-plot         Plot performance. Defaults to true.
  --no-plot             Do not plot performance
  -S, --do-save         Save metrics outputs. Defaults to true.
  --no-save             Do not save metrics outputs
  -c LOAD_SETTINGS, --load-settings LOAD_SETTINGS
                        Load L2Metrics settings from JSON file. Defaults to None.
  -C, --do-save-settings
                        Save L2Metrics settings to JSON file. Defaults to true.
  --no-save-settings    Do not save L2Metrics settings to JSON file
```

By default, the L2Metrics package will calculate metrics with the following options:

- Recursive mode is `disabled`, which means L2Metrics will interpret the log directory as containing L2Logger data for directly computing metrics.
- Variant mode is `aware`, which treats task variants as separate tasks.
- STE averaging method is `metrics`, which performs relative performance and sample efficiency calculations across multiple runs STE logs then averages the intermediate values to produce the lifetime metric.
- Performance measure is `reward`.
- Aggregation method is `mean`, which reports the lifetime metrics as the mean of task-level metrics as opposed to median.
- Performance maintenance method is `mrlep`, which uses the most recent learning evaluation performance as opposed to the most recent terminal learning performance (`mrtlp`).
- Forward and backward transfer use `ratio`.
- Normalization method is `task`, which computes the per-task data ranges by looking at LL and STE log data, then normalizing to [0, 100].
- Smoothing method is `flat`, which smooths data with a rectangular sliding window. Other available options include [hanning](https://numpy.org/doc/stable/reference/generated/numpy.hanning.html#numpy.hanning), [hamming](https://numpy.org/doc/stable/reference/generated/numpy.hamming.html#numpy.hamming), [bartlett](https://numpy.org/doc/stable/reference/generated/numpy.bartlett.html#numpy.bartlett), and [blackman](https://numpy.org/doc/stable/reference/generated/numpy.blackman.html).
- Smoothing of evaluation block data is `disabled`.
- Smoothing window length is `None`, which defaults to min(int(`block_length` \* 0.2), 100).
- Outlier clamping is `disabled`. When enabled and no data range is provided, the outliers (detected using 0.1, 0.9 quantiles) will be clamped to the quantile bounds. If data ranges are provided, then the task data will be clamped to those specified ranges.
- Data range file is `None`, which means L2Metrics will use the per-task data ranges for normalization and clamping.
- Gaussian noise is `disabled`.
- Output directory is `results/`, which will be in the current working directory.
- Output is `None`, which means L2Metrics will use the L2Logger directory name as the base output file names.
- Plotting is `enabled`.
- Draw dotted lines between evaluation blocks is `enabled`.
- Saving of results and log data is `enabled`.
- Load settings from a JSON file is `disabled` unless a valid file path is provided.
- Saving of settings is `enabled`.

**Note**: Valid values for the performance measure input argument are determined by the `metrics_columns` dictionary in `logger_info.json`.
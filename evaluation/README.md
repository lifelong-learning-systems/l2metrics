# Multi-Lifetime Metrics Evaluation

The L2Metrics evaluation package contains a Jupyter notebook (`evaluation.ipynb`) and Python script (`evaluate.py`) for evaluating multi-lifetime metrics. The Jupyter notebook and Python script perform the same functions, but the script allows users to parse, aggregate, and display LL metrics without having to start a Jupyter server. Additionally, the Jupyter notebook relies on the Python script as it contains helper functions for storing STE data and computing metrics on lifelong learning logs.

## Usage

To evaluate multi-lifetime metrics for a lifetime learning agent, you must first generate multiple log files for varying levels of complexity and difficulty in accordance with the L2Logger format version 1.0. An [example_eval](https://github.com/darpa-l2m/example_eval) repository was created to demonstrate the proper format of logs required for evaluation.

Once logs have been generated or unzipped, the LL agent can be evaluated with either the Jupyter notebook (`evaluation.ipypnb`) or the Python script (`evaluate.py`). Both methods require some configuration or input arguments for properly generating the metrics summary report.

**Note**: The evaluation scripts assume the directory structure described in the example evaluation repository README for finding STE and LL logs. The scripts will raise an error if the directory structure does not match what is expected.

### Command-Line Execution

```
usage: evaluate.py [-h] -l LOG_DIR [-p PERF_MEASURE]
                   [-m {contrast,ratio,both}] [-o OUTPUT] [--no-smoothing]
                   [--no-plot] [--no-save]

Run L2M evaluation from the command line

required arguments:

  -l LOG_DIR --log-dir LOG_DIR
                        Log directory of scenario

optional arguments:
  -h, --help            show this help message and exit
  -p PERF_MEASURE, --perf-measure PERF_MEASURE
                        Name of column to use for metrics calculations
  -m {contrast,ratio,both}, --transfer-method {contrast,ratio,both}
                        Method for computing forward and backward transfer
  -o OUTPUT, --output OUTPUT
                        Output filename for results
  --no-smoothing        Do not smooth performance data for metrics and
                        plotting
  --no-plot             Do not plot performance
  --no-save             Do not save metrics outputs
```

**Note**: Valid values for the performance measure input argument are determined by the `metrics_columns` dictionary in `logger_info.json`.

# Multi-Lifetime Metrics Evaluation

The L2Metrics evaluation package contains a Jupyter notebook and Python script for evaluating multi-lifetime metrics. The Jupyter notebook and Python script perform the same functions, but the script allows users to parse, aggregate, and display LL metrics without having to start a Jupyter server.

## Usage

To evaluate multi-lifetime metrics for a lifetime learning agent, you must first generate multiple log files for varying levels of complexity and difficulty in accordance with the L2Logger format version 1.0. An [example_eval](https://github.com/darpa-l2m/example_eval) repository was created to demonstrate the proper format of logs required for evaluation.

Once logs have been generated or unzipped, the LL agent can be evaluated with either the Jupyter notebook (`evaluation.ipypnb`) or the Python script (`evaluate.py`). Both methods require some configuration or input arguments for properly generating the metrics summary report.

### Command-Line Execution

```
usage: evaluate.py [-h] -l LOG_DIR [-p PERF_MEASURE]
                   [-m {contrast,ratio,both}] [--no-smoothing] [--no-plot]
                   [--no-save]

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
  --no-smoothing        Do not smooth performance data for metrics and
                        plotting
  --no-plot             Do not plot performance
  --no-save             Do not save metrics outputs
```

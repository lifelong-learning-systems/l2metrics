# Multi-Lifetime Metrics Evaluation

The L2Metrics evaluation package contains a Jupyter notebook (`evaluation.ipynb`) and two Python scripts (`evaluate.py` and `parallel_evaluation.py`) for evaluating multi-lifetime metrics. The Jupyter notebook and Python scripts perform the same functions, but the script allows users to parse, aggregate, and display LL metrics without having to start a Jupyter server. Additionally, the Jupyter notebook relies on the Python script as it contains helper functions for storing STE data and computing metrics on lifelong learning logs.

## Usage

To evaluate multi-lifetime metrics for a lifetime learning agent, you must first generate multiple log files for varying levels of complexity and difficulty in accordance with the L2Logger format version 1.0. An [example_eval](https://github.com/darpa-l2m/example_eval) repository was created to demonstrate the proper format of logs required for evaluation.

Once logs have been generated or unzipped, the LL agent can be evaluated with either the Jupyter notebook (`evaluation.ipynb`) or the Python scripts (`evaluate.py` / `parallel_evaluation.py`). Both methods require some configuration or input arguments for properly generating the metrics summary report.

**Note**: The evaluation scripts assume the directory structure described in the example evaluation repository README for finding STE and LL logs. The scripts will raise an error if the directory structure does not match what is expected. You may also need to modify the `eval_dir` variable in `process_evaluation` of `parallel_evaluation.py` in order to properly locate the log files.

### Command-Line Execution

```
usage: python -m evaluation.evaluate [-h] -l EVAL_DIR [-s STE_DIR] [-p PERF_MEASURE]
                   [-m {mrtlp,mrlep,both}] [-t {contrast,ratio,both}]
                   [-n {task,run}] [--output-dir OUTPUT_DIR] [-o OUTPUT] [-u]
                   [--no-smoothing] [--normalize] [--remove-outliers]
                   [--no-save-ste] [--no-plot] [--save-plots] [--no-save]

Run L2M evaluation from the command line

optional arguments:
  -h, --help            show this help message and exit
  -l EVAL_DIR, --eval-dir EVAL_DIR
                        Evaluation directory containing logs
  -s STE_DIR, --ste-dir STE_DIR
                        Agent configuration directory of STE data
  -p PERF_MEASURE, --perf-measure PERF_MEASURE
                        Name of column to use for metrics calculations
  -m {mrtlp,mrlep,both}, --maintenance-method {mrtlp,mrlep,both}
                        Method for computing performance maintenance
  -t {contrast,ratio,both}, --transfer-method {contrast,ratio,both}
                        Method for computing forward and backward transfer
  -n {task,run}, --normalization-method {task,run}
                        Method for normalizing data
  --output-dir OUTPUT_DIR
                        Directory for output files
  -o OUTPUT, --output OUTPUT
                        Output filename for results
  -u, --unzip           Unzip all data found in evaluation directory
  --no-smoothing        Do not smooth performance data for metrics
  --normalize           Normalize performance data for metrics
  --remove-outliers     Remove outliers in data for metrics
  --no-save-ste         Do not store STE data
  --no-plot             Do not plot metrics report
  --save-plots          Save scenario and STE plots
  --no-save             Do not save metrics outputs
```

**Note**: Valid values for the performance measure input argument are determined by the `metrics_columns` dictionary in `logger_info.json`.

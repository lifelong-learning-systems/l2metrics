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
                   [--no-smoothing] [-r] [--normalize] [--remove-outliers]
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
  -r, --show-raw-data   Show raw data points under smoothed data for plotting
  --normalize           Normalize performance data for metrics
  --remove-outliers     Remove outliers in data for metrics
  --no-save-ste         Do not store STE data
  --no-plot             Do not plot metrics report
  --save-plots          Save scenario and STE plots
  --no-save             Do not save metrics outputs
```

**Note**: Valid values for the performance measure input argument are determined by the `metrics_columns` dictionary in `logger_info.json`.

## Example Evaluation

This directory also contains the outputs of an example evaluation produced by running the following command:

```bash
python -m evaluation.evaluate --eval-dir=./example_eval/m9_eval/ --ste-dir=agent_config-0 --perf-measure=performance --maintenance-method=both --transfer-method=both --normalization-method=task --output-dir=example_results/normalized/example_normalized --output=example_metrics_normalized --show-raw-data --normalize --save-plots
```

### Metrics TSV File

The TSV file lists all the computed LL metrics from the scenarios found in the specified evaluation directory. The headers in the file are as follows:

- `sg_name`: Name of system group, extracted from the evaluation directory
- `agent_config`: Corresponding agent configuration of scenario
- `run_id`: Run ID or scenario name, extracted from scenario directory name
- `perf_recovery`: Lifetime performance recovery
- `perf_maintenance_mrtlp`: Lifetime performance maintenance, most recent terminal learning performance
- `perf_maintenance_mrlep`: Lifetime performance maintenance, most recent learning evaluation performance
- `forward_transfer_contrast`: Lifetime forward transfer, contrast
- `backward_transfer_contrast`: Lifetime backward transfer, contrast
- `forward_transfer_ratio`: Lifetime forward transfer, ratio
- `backward_transfer_ratio`: Lifetime backward transfer, ratio
- `ste_rel_perf`: Lifetime relative performance compared to STE
- `sample_efficiency`: Lifetime sample efficiency
- `complexity`: Scenario complexity
- `difficulty`: Scenario difficulty
- `metrics_column`: Application metric used to compute metrics
- `min`: Minimum value of data in scenario
- `max`: Maximum value of data in scenario
- `num_lx`: Total number of LXs in scenario
- `num_ex`: Total number of EXs in scenario

### Metrics JSON File

The JSON file lists all the task-level metrics in addition to all the computed LL metrics from the scenario found in the specified evaluation directory. This file is JSON formatted due to the complex nested structures and varying object types corresponding to each metric. The task-level metrics reported for each scenario are as follows:

- `perf_recovery`: Task performance recovery
- `perf_maintenance_mrtlp`: Task performance maintenance, most recent terminal learning performance
- `perf_maintenance_mrlep`: Task performance maintenance, most recent learning evaluation performance
- `forward_transfer_contrast`: Task forward transfer, contrast
- `backward_transfer_contrast`: Task backward transfer, contrast
- `forward_transfer_ratio`: Task forward transfer, ratio
- `backward_transfer_ratio`: Task backward transfer, ratio
- `ste_rel_perf`: Task relative performance compared to STE
- `sample_efficiency`: Task sample efficiency
- `recovery_times`: List of recovery times used for computing performance recovery
- `maintenance_val_mrtlp`: List of maintenance values used for computing performance maintenance, MRTLP
- `maintenance_val_mrlep`: List of maintenance values used for computing performance maintenance, MRLEP
- `min`: Minimum value of task data in scenario
- `max`: Minimum value of task data in scenario
- `num_lx`: Total number of task LXs in scenario
- `num_ex`: Total number of task EXs in scenario

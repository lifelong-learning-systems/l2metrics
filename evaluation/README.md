# Multi-Lifetime Metrics Evaluation

The L2Metrics evaluation package contains a Jupyter notebook (`evaluation.ipynb`) and two Python scripts (`evaluate.py` and `parallel_evaluation.py`) for evaluating multi-lifetime metrics. The Jupyter notebook and Python scripts perform the same functions, but the script allows users to parse, aggregate, and display LL metrics without having to start a Jupyter server. Additionally, the Jupyter notebook relies on the Python script as it contains helper functions for storing STE data and computing metrics on lifelong learning logs.

The package also contains a `validate` module that checks for the existence of required files in the specified evaluation directory.

## Usage

To evaluate multi-lifetime metrics for a lifetime learning agent, you must first generate multiple log files for varying levels of complexity and difficulty in accordance with the L2Logger format version 1.0. An [example_eval](https://github.com/darpa-l2m/example_eval) repository was created to demonstrate the proper format of logs required for evaluation.

Once logs have been generated or unzipped, the LL agent can be evaluated with either the Jupyter notebook (`evaluation.ipynb`) or the Python scripts (`evaluate.py` / `parallel_evaluation.py`). Both methods require modifying some settings or input arguments for properly generating the metrics summary report.

**Note**: The evaluation scripts assume the directory structure described in the example evaluation repository README for finding STE and LL logs. The scripts will raise an error if the directory structure does not match what is expected. You may also need to modify the `eval_dir` variable in `process_evaluation` of `parallel_evaluation.py` in order to properly locate the log files.

### Command-Line Execution

```
usage: python -m evaluation.evaluate [-h] [-l EVAL_DIR] [-f AGENT_CONFIG_DIR] [-s STE_DIR]   
                   [-v {time,metrics}] [-p PERF_MEASURE] [-a {mean,median}]
                   [-m {mrlep,mrtlp,both}] [-t {ratio,contrast,both}]      
                   [-n {task,run,none}]
                   [-g {flat,hanning,hamming,bartlett,blackman,none}]      
                   [-w WINDOW_LENGTH] [-x] [-d DATA_RANGE_FILE]
                   [-O OUTPUT_DIR] [-o OUTPUT] [-u] [-e]
                   [--no-show-eval-lines] [-T] [--no-store-ste] [-P]       
                   [--no-plot] [-L] [--no-save-plots] [-S] [--no-save]     
                   [-c LOAD_SETTINGS] [-C] [--no-save-settings]

Run L2M evaluation from the command line

optional arguments:
  -h, --help            show this help message and exit
  -l EVAL_DIR, --eval-dir EVAL_DIR
                        Evaluation directory containing logs. Defaults to "".
  -f AGENT_CONFIG_DIR, --agent-config-dir AGENT_CONFIG_DIR
                        Agent configuration directory of data. Defaults to "".
  -s STE_DIR, --ste-dir STE_DIR
                        Agent configuration directory of STE data. Defaults to
                        "".
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
  -O OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory for output files. Defaults to results.
  -o OUTPUT, --output OUTPUT
                        Output filename for results. Defaults to ll_metrics.
  -u, --do-unzip        Unzip all data found in evaluation directory
  -e, --show-eval-lines
                        Show lines between evaluation blocks. Defaults to
                        true.
  --no-show-eval-lines  Do not show lines between evaluation blocks
  -T, --do-store-ste    Store STE data. Defaults to true.
  --no-store-ste        Do not store STE data
  -P, --do-plot         Plot performance. Defaults to true.
  --no-plot             Do not plot performance
  -L, --do-save-plots   Save scenario and STE plots. Defaults to true.
  --no-save-plots       Do not save scenario and STE plots
  -S, --do-save         Save metrics outputs. Defaults to true.
  --no-save             Do not save metrics outputs
  -c LOAD_SETTINGS, --load-settings LOAD_SETTINGS
                        Load evaluation settings from JSON file. Defaults to
                        None.
  -C, --do-save-settings
                        Save L2Metrics settings to JSON file. Defaults to
                        true.
  --no-save-settings    Do not save L2Metrics settings to JSON file
```

**Note**: Valid values for the performance measure input argument are determined by the `metrics_columns` dictionary in `logger_info.json`.

### Validation

An evaluation directory can be validated from the command line:

```
usage: python -m evaluation.validate [-h] eval_dir

Validate evaluation submission from the command line

positional arguments:
  eval_dir    Evaluation directory for certain month (e.g., ./m12_eval/)

optional arguments:
  -h, --help  show this help message and exit
```

## Example Evaluation

This directory also contains the outputs of an example evaluation produced by running the following command:

```bash
python -m evaluation.evaluate -l ../../example_eval/m12_eval/ -O ./example_results/ -o example_metrics
```

Similarly to the `l2metrics` package, you may also specify a JSON file containing the desired evaluation settings instead of using the command-line arguments. The settings loaded from the JSON file will take precedence over any arguments specified on the command line.

```bash
python -m evaluation.evaluate -c evaluation_settings.json
```

### Metrics TSV File

The TSV file lists all the computed LL metrics from the scenarios found in the specified evaluation directory. The headers in the file are as follows:

- `sg_name`: Name of system group, extracted from the evaluation directory
- `agent_config`: Corresponding agent configuration of scenario
- `run_id`: Run ID or scenario name, extracted from scenario directory name
- `perf_recovery`: Lifetime performance recovery
- `perf_maintenance_mrlep`: Lifetime performance maintenance, most recent learning evaluation performance
- `perf_maintenance_mrtlp`: Lifetime performance maintenance, most recent terminal learning performance
- `forward_transfer_ratio`: Lifetime forward transfer, ratio
- `backward_transfer_ratio`: Lifetime backward transfer, ratio
- `forward_transfer_contrast`: Lifetime forward transfer, contrast
- `backward_transfer_contrast`: Lifetime backward transfer, contrast
- `ste_rel_perf`: Lifetime relative performance compared to STE
- `sample_efficiency`: Lifetime sample efficiency
- `complexity`: Scenario complexity
- `difficulty`: Scenario difficulty
- `scenario_type`: Scenario type
- `metrics_column`: Application metric used to compute metrics
- `min`: Minimum value of data in scenario
- `max`: Maximum value of data in scenario
- `num_lx`: Total number of LXs in scenario
- `num_ex`: Total number of EXs in scenario

### Metrics JSON File

The JSON file lists all the task-level metrics in addition to all the computed LL metrics from the scenario found in the specified evaluation directory. This file is JSON formatted due to the complex nested structures and varying object types corresponding to each metric. The task-level metrics reported for each scenario are as follows:

- `perf_recovery`: Task performance recovery
- `perf_maintenance_mrlep`: Task performance maintenance, most recent learning evaluation performance
- `perf_maintenance_mrtlp`: Task performance maintenance, most recent terminal learning performance
- `forward_transfer_ratio`: List of task forward transfer, ratio
- `backward_transfer_ratio`: List of task backward transfer, ratio
- `forward_transfer_contrast`: List of task forward transfer, contrast
- `backward_transfer_contrast`: List of task backward transfer, contrast
- `ste_rel_perf`: Task relative performance compared to STE
- `sample_efficiency`: Task sample efficiency
- `recovery_times`: List of recovery times used for computing performance recovery
- `maintenance_val_mrtlp`: List of maintenance values used for computing performance maintenance, MRTLP
  - Each sub-list represents a different reference TLP
- `maintenance_val_mrlep`: List of maintenance values used for computing performance maintenance, MRLEP
  - Each sub-list represents a different reference LEP
- `min`: Minimum value of task data in scenario
- `max`: Minimum value of task data in scenario
- `num_lx`: Total number of task LXs in scenario
- `num_ex`: Total number of task EXs in scenario

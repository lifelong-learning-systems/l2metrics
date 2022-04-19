# File Descriptions

## Lifetime Metrics TSV File

The TSV file lists all the computed LL metrics from the scenarios found in the specified evaluation directory. The headers in the file are as follows:

- `run_id`: Run ID or scenario name, extracted from scenario directory name
- `perf_recovery`: Lifetime performance recovery
- `avg_train_perf`: Lifetime mean training performance
- `avg_eval_perf`: Lifetime mean evaluation performance
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
- `runtime`: Total runtime, in seconds, calculated as difference between max and min timestamps in log data

## Lifetime Metrics JSON File

The JSON file lists all the task-level metrics in addition to all the computed LL metrics from the scenario found in the specified evaluation directory. This file is JSON formatted due to the complex nested structures and varying object types corresponding to each metric. The task-level metrics reported for each scenario are as follows:

- `run_id`: Run ID or scenario name, extracted from scenario directory name
- `perf_recovery`: Task performance recovery
- `avg_train_perf`: Lifetime mean training performance
- `avg_eval_perf`: Lifetime mean evaluation performance
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
- `runtime`: Total runtime, in seconds, calculated as difference between max and min timestamps in log data
- `normalization_data_range`: Task data ranges used for normalization

## Regime Metrics TSV File

The regime metrics TSV file includes the block summary of lifetimes in addition to the regime-level metrics as separate columns. The headers in the file as as follows:

- `run_id`
- `block_num`
- `block_type`
- `block_subtype`
- `task_name`
- `regime_num`
- `task_params`
- `saturation`: Regime saturation value
- `exp_to_sat`: Number of experience to saturation
- `term_perf`: Terminal performance of regime, defined as the mean of the last 10% of training values or the mean of all values in evaluation blocks
- `exp_to_term_perf`: Number of experiences to terminal performance, defined as the 90% index of training blocks or 50% for evaluation blocks
- `avg_perf`: Average performance of the values in the regime

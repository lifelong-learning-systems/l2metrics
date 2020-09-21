Getting Started with Lifelong Learning Metrics
==

Background
--

To calculate metrics on the performance of your system, you must first generate log files in accordance with the TEF format. Please see /docs/metrics_documentation.pdf for more details on how to generate compatible logs.

Once these logs are generated, you'll need to store Single Task Expert (STE) data and pass the log directory as well as the performance measurement to run the metrics. An example log directory is provided to get you started.

Getting Started
--

To store STE data, run the following command from the project source:

> `python examples/store_ste_data.py --log_dir=ste_scenario-1582671493-285338`

Replace the log directory with logs for other STE tasks and repeat until all STE data is stored.

To generate a metrics plot and report, run the following command from the project source:

> `python examples/calc_metrics.py --log_dir=examples/syllabus_ANT_multiple_tasks-1600709972-1635876 --perf_measure=reward`

If you do not wish to provide a fully qualified path to your log directory, you may copy it to your $L2DATA/logs directory. This is the default location for logs generated using the TEF.

The output figure of reward over episodes (saved by default) should look like this:

![diagram](syllabus_ANT_multiple_tasks-1600709972-1635876.png)

Writing a Custom Metric
==

Background
--

The file examples/calc_metrics.py demonstrates how to add custom metrics to a Metrics Report. Data from the logs is provided to the calculate method, where the actual calculation of your metric should live. An example of both a classification and agent metric are provided for your edificiation. To add this metric to the default metrics calculated for a Metrics Report, simply invoke metrics_report.add(MyCustomAgentMetric(perf_measure)) and it will be added to the end of the list in addition to the defaults.

calculate(dataframe, block_info, metrics_dict):

Inputs:

- :param dataframe: Pandas dataframe of log data, collated from log files
- :param block_info: Pandas dataframe with high level information about blocks/phases extracted from log data. Contains no reward/score columns
- :param metrics_df: Pandas dataframe with columns corresponding to calculated metrics along with some of the block_info information. Prints out at the end for reporting.

Output:

- :return: metrics_df: Pandas dataframe, updated with columns corresponding to previously calculated metrics

A few important notes
--

1) The calculate methods for each metric in self._metrics are called **in the order they were added** - thus, you may choose to leverage previously calculated metrics for your subsequent calculations.
2) "nan" is the default value in the metrics_df for blocks which do not recieve a value.
3) To avoid adding your computed metric to the metrics_df, simply do not include the call to _localutil.fill_metrics_df, which takes the values you pass and returns a dataframe with those values.
4) The Agent Metrics are more mature than the classification metrics. Please do not hesitate to reach out if you have questions, particularly about the classification metrics.

import numpy as np
import pandas as pd
import l2metrics

class CustomMetric1(l2metrics.AgentMetric):
    capability = "continual_learning"
    requires = {'syllabus_type': 'type2'}
    description = "Custom metric - Calculates the performance across all tasks and phases"
    
    def calculate(self, dataframe):
        return {'global_perf': mean(dataframe["perf"])}


metrics = l2metrics.AgentMetricsReport(
        log_rootdir="~/custom_log_location/")

metrics.add([CustomMetric1])
results = metrics.calculate()

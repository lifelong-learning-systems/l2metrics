from . import core, util

"""
Standard metrics for Classification tasks
"""

class ClassificationMetric(core.Metric):
    pass


class ClassificationMetric1(ClassificationMetric):
    name = "Global Mean Classification Performance"
    capability = "continual_learning"
    requires = {'syllabus_type': 'type2'}
    description = "Calculates the performance across all tasks and phases"
    
    def calculate(self, dataframe):
        return {'global_perf': mean(dataframe["perf"])}


class ClassificationMetric2(ClassificationMetric):
    name = "Classification Performance relative to S.E.E"
    capability = "adapt_to_new_tasks"
    requires = {'syllabus_type': 'type1'}
    description = "Calculates the performance of each task, in each evaluation, relative to a single task expert"
    
    def calculate(self, dataframe):
        unique_tasks = dataframe['task'].unique()
        result = dict()
        for task in unique_tasks:
            rows = dataframe['phase'] == "eval" & dataframe['task'] == task
            result[task] = mean(rows["perf"])
        return result


class ClassificationMetricsReport(core.MetricsReport):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._metrics = [ClassificationMetric1, ClassificationMetric2]

    def calculate(self):
        pass

    def plot(self):
        pass

    def add(self, metrics_lst):
        self._metrics.append(metrics_lst)


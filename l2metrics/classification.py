"""
(c) 2019 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).
All Rights Reserved. This material may be only be used, modified, or reproduced
by or for the U.S. Government pursuant to the license rights granted under the
clauses at DFARS 252.227-7013/7014 or FAR 52.227-14. For any other permission,
please contact the Office of Technology Transfer at JHU/APL.

NO WARRANTY, NO LIABILITY. THIS MATERIAL IS PROVIDED “AS IS.” JHU/APL MAKES NO
REPRESENTATION OR WARRANTY WITH RESPECT TO THE PERFORMANCE OF THE MATERIALS,
INCLUDING THEIR SAFETY, EFFECTIVENESS, OR COMMERCIAL VIABILITY, AND DISCLAIMS
ALL WARRANTIES IN THE MATERIAL, WHETHER EXPRESS OR IMPLIED, INCLUDING (BUT NOT
LIMITED TO) ANY AND ALL IMPLIED WARRANTIES OF PERFORMANCE, MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF INTELLECTUAL PROPERTY
OR OTHER THIRD PARTY RIGHTS. ANY USER OF THE MATERIAL ASSUMES THE ENTIRE RISK
AND LIABILITY FOR USING THE MATERIAL. IN NO EVENT SHALL JHU/APL BE LIABLE TO ANY
USER OF THE MATERIAL FOR ANY ACTUAL, INDIRECT, CONSEQUENTIAL, SPECIAL OR OTHER
DAMAGES ARISING FROM THE USE OF, OR INABILITY TO USE, THE MATERIAL, INCLUDING,
BUT NOT LIMITED TO, ANY DAMAGES FOR LOST PROFITS.
"""
from . import core, util
import numpy as np

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
        return {'global_perf': np.mean(dataframe["perf"])}


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
            result[task] = np.mean(rows["perf"])
        return result


class ClassificationMetricsReport(core.MetricsReport):

    def __init__(self, **kwargs):
        # Defines log_dir, syllabus_subtype, and initializes the _metrics list
        super().__init__(**kwargs)
        self._log_data = util.read_log_data(self.log_dir)
        self._metrics = [ClassificationMetric1, ClassificationMetric2]

    def default_metrics(self):
        pass

    def calculate(self):
        pass

    def plot(self):
        pass

    def add(self, metrics_lst):
        self._metrics.append(metrics_lst)


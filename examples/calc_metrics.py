import numpy as np
import l2metrics


class MyCustomMetric(l2metrics.AgentMetric):
    capability = "continual_learning"
    requires = {'syllabus_type': 'agent'}
    description = "A Custom Metric"
    
    def calculate(self, dataframe, phase_info, metrics_dict):
        return {'global_perf': dataframe.loc[:,"reward"].mean()}, {'global_perf': dataframe.loc[:,"reward"].mean()}

    def plot(self, result):
        pass

    def validate(self, phase_info):
        # TODO: Add structure validation of phase_info
        pass


metrics_report = l2metrics.AgentMetricsReport(
        log_dir="syllabus_ANT-1574806881648/", syllabus_subtype="CL")

metrics_report.add(MyCustomMetric())
metrics_report.calculate()
metrics_report.report()

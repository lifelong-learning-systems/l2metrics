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
import numpy as np
import l2metrics


class CustomMetric1(l2metrics.AgentMetric):
    capability = "continual_learning"
    requires = {'syllabus_type': 'type2'}
    description = "Custom metric - Calculates the performance across all tasks and phases"
    
    def calculate(self, dataframe):
        return {'global_perf': np.mean(dataframe["perf"])}


metrics = l2metrics.AgentMetricsReport(
        log_dir="~/custom_log_location/", syllabus_subtype="CL")

metrics.add([CustomMetric1])
results = metrics.calculate()

# (c) 2019 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).
# All Rights Reserved. This material may be only be used, modified, or reproduced
# by or for the U.S. Government pursuant to the license rights granted under the
# clauses at DFARS 252.227-7013/7014 or FAR 52.227-14. For any other permission,
# please contact the Office of Technology Transfer at JHU/APL.

# NO WARRANTY, NO LIABILITY. THIS MATERIAL IS PROVIDED “AS IS.” JHU/APL MAKES NO
# REPRESENTATION OR WARRANTY WITH RESPECT TO THE PERFORMANCE OF THE MATERIALS,
# INCLUDING THEIR SAFETY, EFFECTIVENESS, OR COMMERCIAL VIABILITY, AND DISCLAIMS
# ALL WARRANTIES IN THE MATERIAL, WHETHER EXPRESS OR IMPLIED, INCLUDING (BUT NOT
# LIMITED TO) ANY AND ALL IMPLIED WARRANTIES OF PERFORMANCE, MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF INTELLECTUAL PROPERTY
# OR OTHER THIRD PARTY RIGHTS. ANY USER OF THE MATERIAL ASSUMES THE ENTIRE RISK
# AND LIABILITY FOR USING THE MATERIAL. IN NO EVENT SHALL JHU/APL BE LIABLE TO ANY
# USER OF THE MATERIAL FOR ANY ACTUAL, INDIRECT, CONSEQUENTIAL, SPECIAL OR OTHER
# DAMAGES ARISING FROM THE USE OF, OR INABILITY TO USE, THE MATERIAL, INCLUDING,
# BUT NOT LIMITED TO, ANY DAMAGES FOR LOST PROFITS.

import abc


class Metric(abc.ABC):        
    """
    A single metric
    """
    @property
    def capability(self):
        """
        A string (one of the core capabilities that the metric calculates):
            continual_learning
            adapt_to_new_tasks
            goal_driven_perception
            selective_plasticity
            safety_and_monitoring            
        """
        pass

    @property
    def name(self):
        """
        A short label that uniquely identifies this metric. 
        """
        return ""

    @property
    def description(self):
        """
        A more detailed description for this metric
        """
        return ""

    @property
    def requires(self):
        """
        A dictionary of requirements for this metric. Keys 
          syllabus_type: one of 'type1", "type2", etc.
        """
        return {}

    @abc.abstractmethod    
    def calculate(self, data, metadata, metrics_dict):
        """
        Returns a dict of values
        """
        return None

    @abc.abstractmethod
    def plot(self, result):
        """
        Visualizes the metric using matplotlib visualizations
        """
        pass


class MetricsReport(object):
    """
    Aggregates a list of metrics for a learner
    """
    def __init__(self, **kwargs):

        self._metrics = []

        if 'log_dir' in kwargs:
            self.log_dir = kwargs['log_dir']
        else:
            raise RuntimeError("log_dir is required")

        if 'syllabus_subtype' in kwargs:
            self.syllabus_subtype = kwargs['syllabus_subtype']
        else:
            # TODO: Log/warn that we are using the default syllabus_subtype = CL
            self.syllabus_subtype = "CL"

    def calculate(self):
        pass

    def plot(self):
        pass

    def add(self, metrics_lst):
        self._metrics.extend(metrics_lst)

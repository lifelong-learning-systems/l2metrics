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
    def calculate(self, data, metadata):
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

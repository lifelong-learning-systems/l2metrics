import abc

class Metric(abc.ABC):        
    """
    A single metric
    """
    @abc.abstractproperty
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

    @abc.abstractproperty
    def name(self):
        """
        A short label that uniquely identifies this metric. 
        """
        return ""

    @abc.abstractproperty
    def description(self):
        """
        A more detailed description for this metric
        """
        return ""

    @abc.abstractproperty
    def requires(self):
        """
        A dictionary of requirements for this metric. Keys 
          syllabus_type: one of 'type1", "type2", etc.
        """
        return {}


    @abc.abstractmethod    
    def calculate(self):
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
        if 'log_rootdir' in kwargs:
            kwargs['log_rootdir']
        else:
            raise RuntimeError("log_rootdir is required")
        self._metrics = []

    def calculate(self):
        for metric in self._metrics:
            result = metric.calculate()


    def plot(self):
        pass

    def add(self, metrics_lst):
        self._metrics.append(metrics_lst)



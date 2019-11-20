# Overview of Metrics Calculation for Lifelong Learning
## Reinforcement Learning Metric Documentation, v0.1

Assumptions and Requirements:
------

+ Log files __must__ include logged reward, and the column in the data file __must__ be named reward. 
    Without this column, the metrics code will fail.
    This is assumed to be logged per episode.
+ Single Task Expert Saturation values for each task __must__ be included in a JSON file found in $L2DATA/taskinfo/info.json 
    Without this file, the metric "Comparison to STE" cannot be calculated 
    
    The task names must match the names in the log files exactly. The format for this file will be: 
    
    {
    "task_name_1" : 0.8746,
    "task_name_2" : 0.9315,
    ...,
    "task_name_n" : 0.8089
    }
        
+ Syllabi used to generate the log files __must__ include annotations with phase information and shall conform to the following convention:
    
    Phase annotation format:

    {"$phase":  "1.train"}, {"$phase":  "1.test"}, {"$phase":  "2.train"}, {"$phase":  "2.test"}, etc

    Structure:
    
    + CL

    Consists only of a single task with parametric variations exercised throughout the syllabus. Testing phase is optional, 
    but recommended. The purpose of this type of syllabus is to assess whether the agent can adjust to changes in the 
    environment and maintain performance on the previous parameters when new ones are introduced

    + ANT, subtype A

    Consists of multiple tasks with no parametric variations exercised throughout the syllabus. Testing phase is mandatory.
    The purpose of this type of syllabus is to assess whether the agent can learn a new task without forgetting the old task
    and does not seek to assess knowledge transfer.

    + ANT, subtype B

    Consists of multiple tasks with no parametric variations exercised throughout the syllabus. Testing phase is mandatory.
    The purpose of this type of syllabus is to assess whether the agent can transfer knowledge from a learned task to a new task.


Metric definitions:
------

+ Phase
    - Phase number should be incremented, starting with phase 1
    - Phase type can be either "train" or "test"
+ Block
    - A unique combination of task and parameters. Subcomponent of phase. 
    These are automatically assigned by the logging code and do not need to be annotated by the syllabus designer.
+ Task
    - ADD DEFINITION FROM RELEASED DOCUMENTATION HERE
+ Episode
    - ADD DEFINITION FROM RELEASED DOCUMENTATION HERE



Metric calculation descriptions:
------

1) Saturation Value
    + Purpose: The saturation value is computed to quantify the maximum maintained value by the agent.
    + Calculated by: Since multiple rewards may be logged per episode, the mean is taken per episode. 
    Then, the max of the rolling average is taken of the mean reward per episode with a smoothing parameter, s (default = 0.1)
    + Compared to: Future or single task expert saturation values of the same task
    + Computed for: {CL, ANT_A, ANT_B}

2) Time to Saturation
    + Purpose: Time to saturation is used to quantify how quickly, in number of episodes, the agent took to achieve the 
    saturation value computed above.
    + Calculated by: The first time the saturation value (or above) is seen, that episode number is recorded
    + Compared to: Future times to saturation of the same task
    + Computed for: {CL, ANT_A, ANT_B}

3) Normalized Integral of Reward/Time
    - Purpose: Taking the Integral of Accumulated Reward over Time allows for a more robust comparison of the time to learn a particular task, 
    taking into account both the shape and saturation of the learning for future comparison. Has limitations; must be normalized by length; 
    only training phases can be compared to each other
    + Calculated by: Integrating reward over time, then dividing by the number of episodes used to accumulate the reward
    + Compared to: Future training instances of this metric
    + Computed for: {ANT_B}

4) Recovery Time
    + Purpose: 
    + Calculated by:
    + Compared to:
    + Computed for: {CL, ANT_A, ANT_B}

5) Performance Maintenance on Test Sets
    + Purpose:
    + Calculated by:
    + Compared to:
    + Computed for: {CL, ANT_A, ANT_B}

6) Performance relative to STE (training) - Saturation Value, Time, Integral
    + Purpose:
    + Calculated by:
    + Compared to:
    + Computed for: {CL, ANT_A, ANT_B}

7) Forward Transfer (cross tasks)
    + Purpose:
    + Calculated by:
    + Compared to:
    + Computed for: {CL, ANT_A, ANT_B}

8) Time to Learn (cross tasks)
    + Purpose:
    + Calculated by:
    + Compared to:
    + Computed for: {CL, ANT_A, ANT_B}


To Add Your Own Metrics:
------

The steps required to run your new metric in the existing metrics pipeline are as follows:

1. Write your custom metric, MyCustomMetric in agent.py according to the structure set in l2metrics/core.py

    More details regarding the structure of the AgentMetrics class can be found in l2metrics/core.py - your metric code to 
    actually compute a custom metric goes in the required calculate method. Note that the required arguments are the log data,
    the phase_info, and the metrics_dict. The metrics dict is filled by each metric in its turn, whereas the log data and the
    phase info are extracted in the AgentMetricsReport constructor from the logs via two helper functions
    
    l2metrics/util.py - (read_log_data): scrapes the logs and returns a pandas dataframe of the logs and task parameters
    l2metrics/_localutil.py - (parse_blocks): builds a pandas dataframe of the phase information contained in the log data


2. Insert MyCustomMetric to be added into the appropriate default metric lists for any syllabus type you want the metric
to be calculated for - CL, ANT_A, and/or ANT_B

    The AgentMetricsReport adds the default list of metrics based on the passed syllabus type (a mandatory parameter to run 
    the metrics code via command line). AgentMetricsReport will only add your new metric to the default list if you insert 
    it in the _add_default_metrics method for your desired syllabus type

3. Run l2metrics/__ main __.py with the appropriate command line parameters and watch your metric be reported when you 
call the report method





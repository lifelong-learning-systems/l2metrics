# Overview of Metrics Calculation for Lifelong Learning
## Reinforcement Learning Metric Documentation, v0.1

Overview:
------

All of the information needed to get started with this code is contained in metrics_documentation.pdf - please see this document
for a more complete picture of the code


The most relevant piece of the Metric class is the calculate method, which has the following
required arguments: the log data, a phase_info dataframe, and a metrics_dict dataframe.
The metrics_dict starts blank and is filled by each metric in its turn, whereas the log data
and the phase info are extracted in the MetricsReport constructor from the logs via two
helper functions:

l2metrics/util.py - (read_log_data): scrapes the logs and returns a pandas dataframe of the logs and task parameters
l2metrics/_localutil.py - (parse_blocks): builds a pandas dataframe of the phase information contained in the log data

These dataframes are passed along by the MetricsReport to the appropriate metric and thus
the Metrics and MetricsReport classes should be utilized in conjuction with each other.
Though there are a list of default metrics which the MetricsReport uses for the Core
Capabilties being exercised at this time, you may choose to add your own metric to this list
by using the add method on MetricsReport. Please see the calc_metrics.py file for more
details on how to get started with writing your own custom metric. An extremely simple
whole-syllabus-mean is currently implemented as an example to help get you started.


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
    
    + ANT, subtype C

    Consists of multiple tasks with parametric variations exercised throughout the syllabus. Testing phase is mandatory.
    The purpose of this type of syllabus is to assess whether the agent can transfer knowledge from a learned task with 
    parametric variation to a new task with parametric variation.


Metric definitions:
------

Key Concepts

Task: A single abstract capability (or skill) that a performer system must learn
Episode: A concrete instance of a task
Syllabus: A sequence of episodes
Learning Lifetime: One syllabus or multiple syllabi in sequence


Metric Specific Terms for Syllabus Design

Phase: A subcomponent of a syllabus during which either training or evaluation takes place
Block: A unique combination of task and parameters. It is a subcomponent of phase that is automatically assigned by the logging code and does not need to be annotated by the syllabus designer.


Metric calculation descriptions:
------

1. Saturation Value

Purpose: The saturation value is computed to quantify the maximum maintained value by
the agent.
Calculated by: Since multiple rewards may be logged per episode, the mean is taken per
episode. Then, the max of the rolling average is taken of the mean reward per episode with
a smoothing parameter, s (default, 0.1)
Compared to: Future or single task expert saturation values of the same task

2. Time to Saturation

Purpose: Time to saturation is used to quantify how quickly, in number of episodes, the
agent took to achieve the saturation value computed above.
Calculated by: The first time the saturation value (or above) is seen, that episode number is
recorded
Compared to: Future times to saturation of the same task

3. Normalized Integral of Reward/Time

Purpose: Taking the Integral of Accumulated Reward over Time allows for a more robust
comparison of the time to learn a particular task, taking into account both the shape and
saturation of the learning for future comparison. Has limitations; must be normalized by
length; only training phases can be compared to each other
Calculated by: Integrating reward over time, then dividing by the number of episodes used
to accumulate the reward
Compared to: Future training instances of this metric

4. Recovery Time

Purpose: Recovery time is calculated to determine how quickly (if at all) an agent can
"bounce back" after a change is introduced to its environment
Calculated by: After some training phase achieves a saturation value, determine how many
episodes, if any, it takes for the agent to regain the same performance on the same task
Compared to: An agent’s recovery time is comparable across tasks

5. Performance Maintenance on Test Sets

Purpose: Performance maintenance on test sets is calculated to determine whether an agent
catastrophically forgets a previously learned task
Calculated by: Comparing all computed metrics on the train set (saturation values, time to
saturation, etc) on the test set and computing the difference in performance

6. Performance relative to STE (training)

Purpose: STE Relative performance assesses whether a lifelong learner outperforms a
traditional learner.
Calculated by: Normalizing metrics computed on the lifelong learner by the same metrics
computed on the traditional learner



To Get Started:
------

Get started by first generating some log files. You may do this by the following:
1. Download and install the minigridkit repo, located here: PUT A LINK HERE
2. Configure your environment variable $L2DATA to wherever you want your logs to end up.
3. Run minigrid_learnkit/minigrid_train_ppo.py
4. Your logs should appear in $L2DATA\logs\"YOUR_LOG_DIRECTORY"

Then, you should be able to:11

5. Pass "YOUR_LOG_DIRECTORY" as the log_dir parameter in the calc_metrics.py file, 
and you should get an output printed to console that looks something like this:

Console output:

Metric: Average Within Block Saturation Calculation
Value: {'global_within_block_saturation': 0.6201868186374054, 'global_num_eps_to_saturation': 15.833333333333334}

Metric: 
Value: {'global_perf': 0.9247070921895495}

Process finished with exit code 0



To Add Your Own Metrics:
------

Please see the calc_metrics.py file for more details on how to get started with writing your own custom metric. 
An extremely simple whole-syllabus-mean is currently implemented as an example to help get you started.



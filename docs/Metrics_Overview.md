# Overview of Metrics Calculation for Lifelong Learning


Reinforcement Learning Metric Documentation, v0.1

Assumptions and Requirements:

1) Log files MUST include logged reward per episode, otherwise the metrics code will not run.
2) Syllabi used to generate the log files shall include annotations with phase information and shall conform to the following convention:

    Phase annotation format:
    {"$phase":  "1.train"}, {"$phase":  "1.test"}, etc

    Structure:
    CL
    - Consists only of a single task with parametric variations exercised throughout the syllabus. Testing phase is optional, but recommended to enable computing performance maintenance metrics

    ANT, subtype A
    - Consists of multiple tasks with no parametric variations exercised throughout the syllabus. Testing phase is mandatory.

    ANT, subtype B
    - Consists of multiple tasks with no parametric variations exercised throughout the syllabus. Testing phase is mandatory.


Metric definitions:

Phase
Block
Task
Episode



Metric calculation descriptions:

Saturation Value
    - Purpose: The saturation value is computed to quantify the maximum maintained value by the agent.
    - Calculated by: Since multiple rewards may be logged per episode, the mean is taken per episode. Then, the max of the rolling average is taken of the mean reward per episode with a smoothing parameter, s (default = 0.1)
    - Compared to: Future or single task expert saturation values of the same task
    - Computed for: {CL, ANT_A, ANT_B}

Time to Saturation
    - Purpose: Time to saturation is used to quantify how quickly, in number of episodes, the agent took to achieve the saturation value computed above.
    - Calculated by: The first time the saturation value (or above) is seen, that episode number is recorded
    - Compared to: Future times to saturation of the same task
    - Computed for: {CL, ANT_A, ANT_B}

(Normalized) Integral of Reward/Time
    - Purpose:
    - Calculated by:
    - Compared to:
    - Computed for: {CL, ANT_A, ANT_B}

Recovery Time
    - Purpose:
    - Calculated by:
    - Compared to:
    - Computed for: {CL, ANT_A, ANT_B}

Performance on evaluation vs training sets
    - Purpose:
    - Calculated by:
    - Compared to:
    - Computed for: {CL, ANT_A, ANT_B}

Comparison to STE (training) - Saturation Value, Time, Integral
    - Purpose:
    - Calculated by:
    - Compared to:
    - Computed for: {CL, ANT_A, ANT_B}

Forward Transfer (cross tasks)
    - Purpose:
    - Calculated by:
    - Compared to:
    - Computed for: {CL, ANT_A, ANT_B}

Time to Learn (cross tasks)
    - Purpose:
    - Calculated by:
    - Compared to:
    - Computed for: {CL, ANT_A, ANT_B}

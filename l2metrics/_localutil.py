import pandas as pd
import re
import numpy as np


def parse_blocks(data):
    # Want to get the unique phases, split out training/testing info, and return the split info
    phases_list = []
    test_task_nums = []

    phases_from_logs = data.loc[:, 'phase'].unique()

    for p in phases_from_logs:

        # Extract information
        x = re.match(r'^(\d+)[.-_]*(\w+)', p)
        phase_number = x.group(1)
        phase_type = x.group(2)

        # Now must account for the multiple tasks, parameters
        phase_query_str = "phase == '" + p + "'"
        d1 = query_dataframe(data, phase_query_str)
        blocks_within_phases = d1.loc[:, 'block'].unique()
        param_set = d1.loc[:, 'params'].unique()

        # TODO: Check the phase_type for what I expect and make sure to complain if it's not what's expected
        # TODO: Make more rigid expectations AND document them!
        # TODO: Generate my own phase numbers? Or complain if phases skip numbers?

        # Save the block numbers involved in testing for subsequent metrics
        if phase_type == 'test':
            test_task_nums.extend(blocks_within_phases)

        for b in blocks_within_phases:
            block_query_str = "block == '" + str(b) + "'"
            d2 = query_dataframe(d1, block_query_str)
            task_name = d2.loc[:, 'class_name'].unique()[0]

            phase_block = {'phase': p, 'phase_number': phase_number, 'phase_type': phase_type, 'task_name': task_name,
                           'block': b}

            if len(param_set) > 1:
                # There is parameter variation exercised in the syllabus and we need to record it
                task_specific_param_set = d2.loc[:, 'params'].unique()[0]
                phase_block['param_set'] = task_specific_param_set
            elif len(param_set) == 1:
                # Every task in this block has the same parameter set
                phase_block['param_set'] = param_set[0]
            else:
                # TODO: ERROR MESSAGE HERE
                raise KeyError()

            phases_list.append(phase_block)

    # Convenient for future dev to have the block id be the same as the index of the dataframe
    phases_df = pd.DataFrame(phases_list).sort_values(by=['block']).set_index("block", drop=False)

    return test_task_nums, phases_df


def query_dataframe(df, query_str, get_last=False):
    if get_last:
        query_return = df.loc(-1, 'block')
    else:
        query_return = df.query(query_str)

    return query_return


def moving_average(values, window):
    if window:
        weights = np.repeat(1.0, window)/window
        sma = np.convolve(values, weights, 'valid')
    else:
        sma = values
    return sma


def get_block_saturation_performance(data, previous_saturation_value=None):
    # Calculate the "saturation" value
    # Calculate the number of episodes to "saturation"
    smoothing_param = 0.1

    mean_reward_per_episode = data.loc[:, ['task', 'reward']].groupby('task').mean()
    mean_data = np.ravel(mean_reward_per_episode.values)

    # Take the rolling average of the mean of the data
    smoothed_data = moving_average(mean_data, int(round(smoothing_param*len(data))))
    saturation_value = np.max(smoothed_data)

    # Calculate the number of episodes to "saturation", which we define as the maximum of the rolling average
    inds = np.where(smoothed_data == saturation_value)
    episodes_to_saturation = inds[0][0]
    episodes_to_recovery = np.nan

    if previous_saturation_value:
        inds = np.where(smoothed_data >= previous_saturation_value)
        if len(inds[0]):
            episodes_to_recovery = inds[0][0]
        else:
            episodes_to_recovery = -999

    return saturation_value, episodes_to_saturation, episodes_to_recovery

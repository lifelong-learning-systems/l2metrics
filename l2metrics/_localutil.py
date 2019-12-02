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

import pandas as pd
import re
import numpy as np


def parse_blocks(data):
    # Want to get the unique phases, split out training/testing info, and return the split info
    phases_list = []
    test_task_nums = []
    all_block_nums = []

    phases_from_logs = data.loc[:, 'phase'].unique()

    for p in phases_from_logs:

        # Extract information
        x = re.match(r'^(\d+)[.-_]*(\w+)', p)

        if x.group(1) or x.group(2) is None:
            raise Exception('Unsupported phase annotation: {:s}! Supported format is phase_number.'
                            'phase_type, which must be (int).(string)'.format(p))
        
        phase_number = x.group(1)     
        phase_type = x.group(2)

        if str.lower(phase_type) not in ["train", "test"]:
            raise Exception('Unsupported phase type: {:s}! Supported phase types are "train" and "test"'
                            .format(phase_type))

        # Now must account for the multiple tasks, parameters
        d1 = data[data["phase"] == p]
        blocks_within_phases = d1.loc[:, 'block'].unique()
        param_set = d1.loc[:, 'params'].unique()

        # Save the block numbers involved in testing for subsequent metrics
        if phase_type == 'test':
            test_task_nums.extend(blocks_within_phases)

        for b in blocks_within_phases:
            all_block_nums.append(b)
            d2 = d1[d1["block"] == b]
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
                raise Exception("Error parsing the parameter set for this task: {:s}".format(param_set))

            phases_list.append(phase_block)

    # Convenient for future dev to have the block id be the same as the index of the dataframe
    phases_df = pd.DataFrame(phases_list).sort_values(by=['block']).set_index("block", drop=False)

    # Quick check to make sure the block numbers (zero indexed) aren't a mismatch on the length of the block nums array
    if (max(all_block_nums)+1)/len(all_block_nums) != 1:
        raise Warning("Phase number: {:d} and length {:d} mismatch!".format(max(all_block_nums), len(all_block_nums)))

    return test_task_nums, phases_df


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

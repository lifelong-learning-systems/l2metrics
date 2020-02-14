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

        if x is not None:
            if x.re.groups != 2:
                raise Exception('Unsupported phase annotation: "{:s}"! Supported format is phase_number.'
                                'phase_type, which must be (int).(string)'.format(p))
        else:
            raise Exception('Unsupported phase annotation: "{:s}"! Supported format is phase_number.'
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
        Warning("Phase number: {:f} and length {:d} mismatch!".format(max(all_block_nums), len(all_block_nums)))

    return test_task_nums, phases_df


def smooth(x, window_len=11, window='hanning'):
    # """smooth the data using a window with requested size.
    # Code from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    # This method is based on the convolution of a scaled window with the signal.
    # The signal is prepared by introducing reflected copies of the signal
    # (with the window size) in both ends so that transient parts are minimized
    # in the beginning and end part of the output signal.
    # input:
    #    x: the input signal
    #    window_len: the dimension of the smoothing window; should be an odd integer
    #    window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
    #        flat window will produce a moving average smoothing.
    # output:
    #    the smoothed signal
    # example:
    # t=linspace(-2,2,0.1)
    # x=sin(t)+randn(len(t))*0.1
    # y=smooth(x)
    # see also:
    # numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    # scipy.signal.lfilter
    # NOTE: length(output) != length(input), to correct this: return
    # y[(window_len/2-1):-(window_len/2)] instead of just y.

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        # raise(ValueError, "Input vector needs to be bigger than window size.")
        window_len = int(np.floor(x.size / 2))

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    # Changed to return output of same length as input
    start_ind = int(np.floor(window_len/2-1))
    end_ind = -int(np.ceil(window_len/2))
    return y[start_ind:end_ind]


def get_block_saturation_perf(data, column_to_use=None, previous_saturation_value=None):
    # Calculate the "saturation" value
    # Calculate the number of episodes to "saturation"

    mean_reward_per_episode = data.loc[:, ['task', column_to_use]].groupby('task').mean()
    mean_data = np.ravel(mean_reward_per_episode.values)

    # Take the moving average of the mean of the per episode reward
    smoothed_data = smooth(mean_data, window='flat')
    saturation_value = np.max(smoothed_data)

    # Calculate the number of episodes to "saturation", which we define as the max of the moving average
    inds = np.where(smoothed_data == saturation_value)
    episodes_to_saturation = inds[0][0]
    episodes_to_recovery = np.nan

    if previous_saturation_value:
        inds = np.where(smoothed_data >= previous_saturation_value)
        if len(inds[0]):
            episodes_to_recovery = inds[0][0]
        else:
            episodes_to_recovery = np.NAN

    return saturation_value, episodes_to_saturation, episodes_to_recovery


def extract_relevant_columns(dataframe, keyword):
    # Parse the dataframe and get out the appropriate column names for Classification performance assessment
    relevant_cols = []

    for col in dataframe.columns:
        if col.startswith(keyword):
            relevant_cols.append(col)

    return relevant_cols


def simplify_task_names(unique_task_names, phase_info):
    # Capture the correspondence between the Classification Train/Test tasks
    name_map = {'full_name_map': {}}
    task_map = {}
    type_map = {}
    block_list = {}

    # First find the blocks for each task
    for t in unique_task_names:
        this_instance_blocks = phase_info.loc[phase_info['task_name'] == t, 'block']

        # Then get the base class name by finding the phase/task type annotation and getting the string that comes after
        x = re.search(r'(train|test)(\w+)', t)
        if x.re.groups != 2:
            raise Exception('Improperly formatted task names! Classification tasks should include '
                            '"train" or "test," but this one was: {:s}'.format(t))

        task_type = x.group(1)
        task_name = x.group(2)

        # Record which tasks were used for training for future metric computation
        if task_type == 'train':
            name_map[task_name] = t

        name_map['full_name_map'][t] = task_name

        # Add to the task map
        if task_name not in task_map.keys():
            task_map[task_name] = this_instance_blocks.values
        else:
            existing_blocks = task_map.get(task_name)
            tmp = {task_name: np.append(existing_blocks, this_instance_blocks.values)}
            task_map.update(tmp)

        # And update the block and type lists
        block_list.update({v: task_name for v in this_instance_blocks.values})
        type_map.update({v: task_type for v in this_instance_blocks.values})

    return task_map, block_list, name_map, type_map


def plot_transfer_matrix(phase_info, forward, reverse):
    # Want to build a table of data showing the forward/reverse transfer from one task to another. Phases are rows,
    # Tasks are columns.
    train_blocks = phase_info.loc[phase_info['phase_type'] == 'train', 'block'].to_numpy()
    test_blocks = phase_info.loc[phase_info['phase_type'] == 'test', 'block'].to_numpy()

    # We'll produce a table with unique_phase_nums rows and unique_test_blocks columns.
    unique_phase_nums = phase_info.loc[:, 'phase_number'].unique()
    unique_test_blocks = phase_info.loc[phase_info['phase_type'] == 'test', 'task_name'].unique()

    table_data = np.empty((len(unique_phase_nums), len(unique_test_blocks)))

    for n in unique_phase_nums:
        blocks_within_phase = phase_info.loc[phase_info['phase_number'] == n, 'block'].to_numpy()
        this_phase_test_blocks = [b for b in blocks_within_phase if b in test_blocks]

        # Now check the forward and reverse transfer dicts for a key with these blocks

        for task_name, block_num in reverse.keys():
            # Fill in the table,
            table_data[n, block_num] = reverse.get((task_name, block_num))

    pass

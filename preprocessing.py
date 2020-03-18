import json
import glob
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import deque
import copy

DATA_PATH = './json_data/'
JSON_FILES_PATHS = glob.glob("./json_data/*.json")


#  Borrow from Dana's script
def parseJson(filename):
    # Try to load the json file
    try:
        with open(filename) as f:
            jsonData = json.load(f)

    except:
        sys.stderr.write("Unable to open file '%s'" % filename)
        return

    # The json file contains game_states, player_input, messages, and ui_events
    gameStatesJson = jsonData['game_states']

    # Parse the game states
    frames, missiles, shells = parseGameStates(gameStatesJson)

    # Collate all the data into a dictionary
    gameData = {'frames': frames,
                'missiles': missiles,
                'shells': shells}
    return gameData


#  Borrow from Dana's script
def parseGameStates(gameStates, fortress_ids=['0'], player_ids=['0', '1']):
    """
    Convert a json representation of the game states to three numpy arrays:
    frames - tickNum, time, score, pose of fortresses, players, and border
    missiles - tickNum, id, and pose of each missile
    shells - tickNum, id, and pose of each shell
    """

    # Keep as list of frame, missile and shell data extracted from each frame
    frames, missiles, shells = [], [], []

    for state in gameStates:
        _frame, _missiles, _shells = parseGameFrame(state, fortress_ids, player_ids)
        frames.append(_frame)
        missiles.append(_missiles)
        shells.append(_shells)

    # Convert the list to 2D numpy arrays
    frames = np.vstack(frames)
    missiles = np.vstack(missiles)
    shells = np.vstack(shells)

    return frames, missiles, shells


#  Borrow from Dana's script
def parseGameFrame(frame, fortress_ids=['0'], player_ids=['0', '1']):
    """
    Convert a json representation of a single game frame to a npy representation

    """

    ## Game data and objects - can be put in a single line in a np array

    # Frame-level data (e.g., tick number, time, etc)
    tickNum = frame['tick']
    time = frame['time']
    score = frame['score']
    frame_gamedata = np.array([tickNum, time, score])

    # Get the fortress and player states
    fortresses = [parseFortress(frame['fortresses'][_id]) for _id in fortress_ids]
    players = [parsePlayer(frame['players'][_id]) for _id in player_ids]
    border = np.array([frame['border'][pos] for pos in ['xMin', 'xMax', 'yMin', 'yMax']])

    # Concatenate everything to a flat representation
    frame_data = np.concatenate([frame_gamedata] + fortresses + players + [border])

    missiles = []
    shells = []

    ## Missile and shells
    if frame['missiles'] is not None:
        _missiles = frame['missiles']
        missiles = [np.concatenate([np.array([tickNum, int(_id)]), parseProjectile(_missiles[_id])]) for _id in
                    _missiles.keys()]

    if frame['shells'] is not None:
        _shells = frame['shells']
        shells = [np.concatenate([np.array([tickNum, int(_id)]), parseProjectile(_shells[_id])]) for _id in
                  _shells.keys()]

    # Convert shells and missiles to np arrays, possibly with 0 elements
    missiles = np.reshape(np.array(missiles), (-1, 5))
    shells = np.reshape(np.array(shells), (-1, 5))

    return frame_data, missiles, shells


#  Borrow from Dana's script
def parsePlayer(player):
    """
    Convert player data from json to numpy format

    Array index to value:
        0	x
        1	y
        2	angle
        3	vx
        4	vy
        5	alive
    """

    player_npy = np.zeros((6,), dtype=np.float32)

    player_npy[0] = player['position']['x']
    player_npy[1] = player['position']['y']
    player_npy[2] = player['angle']
    player_npy[3] = player['velocity']['x']
    player_npy[4] = player['velocity']['y']
    player_npy[5] = 1.0 if player['alive'] else 0.0

    return player_npy


#  Borrow from Dana's script
def parseFortress(fortress):
    """
    Convert fortress data from json to numpy format

    Array index to value:
        0	x
        1	y
        2	angle
        3	alive
        4	target_id
        5	vulnerable
        6	shield x
        7	shield y
        8	shield angle
        9	shield radius
        10	activation region x
        11	activation region y
        12	activation region angle
        13	activation region radius
    """

    fortress_npy = np.zeros((14,), dtype=np.float32)

    # Fortress Data
    fortress_npy[0] = fortress['x']
    fortress_npy[1] = fortress['y']
    fortress_npy[2] = fortress['angle']
    fortress_npy[3] = 1.0 if fortress['alive'] else 0.0
    fortress_npy[4] = fortress['target']

    # Shield Data
    fortress_npy[5] = 1.0 if fortress['shield']['vulnerable'] else 0.0
    fortress_npy[6] = fortress['shield']['position']['x']
    fortress_npy[7] = fortress['shield']['position']['y']
    fortress_npy[8] = fortress['shield']['angle']
    fortress_npy[9] = fortress['shield']['radius']

    # Activation Region Data
    fortress_npy[10] = fortress['activationRegion']['position']['x']
    fortress_npy[11] = fortress['activationRegion']['position']['y']
    fortress_npy[12] = fortress['activationRegion']['angle']
    fortress_npy[13] = fortress['activationRegion']['radius']

    return fortress_npy


#  Borrow from Dana's script
def parseProjectile(projectile):
    """
    Convert shell json data to numpy format

    Array index to value
        0	x
        1	y
        2	angle
    """

    projectile_npy = np.zeros((3,), dtype=np.float32)

    projectile_npy[0] = projectile['position']['x']
    projectile_npy[1] = projectile['position']['y']
    projectile_npy[2] = projectile['angle']

    return projectile_npy


#  Read json file and return a dictionary
def read_json_file(file: str):
    with open(file) as f:
        json_data = json.load(f)
    return json_data


#  Function 1
# TODO: it will return empty list
def generate_candidate_clips(data: dict) -> list:
    """
    :param data: game_data from function parseJson
    :return: a list of tuples satisfied some constraints
    """
    event_tick = []
    alive = True
    for frame in data['frames']:
        if frame[6] == 0 and alive == True:
            alive = False
            event_tick.append(int(frame[0]))
        elif frame[6] == 1 and alive == False:
            alive = True

    clip_max_len = 300
    clip_min_len = 150
    clip_ticks = []

    # generate clips
    for event in event_tick:
        start = event - min(clip_max_len, event)
        interest_start = start
        for frame in data['frames'][event:start:-1]:
            #  vulnerable or player0 isn't alive or player1 isn't alive
            #  TODO Isn't alive?
            if frame[8] == 0 or frame[22] == 0 or frame[28] == 0:
                interest_start = int(frame[0])
                break
        if event - interest_start > clip_min_len:
            clip_ticks.append((interest_start, event))
    return clip_ticks


#  Convert Cartesian to Polar
def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)
    return rho, phi


#  Convert Polar to Cartesian
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


# Function 2
def in_trust_region(data: dict, tuples: list, theta=np.pi / 3):
    """
    :param data: game_data from function parseJson
    :param tuples: a list of tuples generated by generate_candidate_tuples
    :param theta: hyper-parameter for the angle, the total angle should be 2 * theta
    :return: result: list of True or False
    """

    #  TODO Here I assume player1 is shooter
    shooter_angles = np.array(
        [cart2pol(data['frames'][t[0]:t[1] + 1, 23], data['frames'][t[0]:t[1] + 1, 24])[1] for t in tuples])
    fortress_angles = np.array([np.radians(data['frames'][t[0]:t[1] + 1, 5]) for t in tuples])
    relative_angles = (shooter_angles - fortress_angles + 2 * np.pi) % (2 * np.pi)
    result = [np.bitwise_and(i >= np.pi - theta, i <= np.pi + theta) for i in relative_angles]
    # result = relative_angles >= np.pi - theta and relative_angles <= np.pi + theta
    return result


#  TODO function3
def count_actions(data: dict, tuples: list, lists: list):
    """
    :param data: a dictionary read from corresponding json file
    :param tuples: a list of tuples of candidate clips, e.g [(1600, 1833), ...]
    :param lists: a list of true or false [np.array([True, True, False, False,...]), np.array([...]), ...]
    :return: counts: a list of counts for each candidate clip

    """
    ticks = np.array([i['tick'] for i in np.array(data['ui_events'])])
    commands = np.array([i['command'] for i in np.array(data['ui_events'])])
    counts = []
    for index, t in enumerate(tuples):
        count = 0
        start_index = np.where((ticks >= t[0]) == True)[0][0]
        end_index = np.where((ticks <= t[1]) == True)[0][-1]
        tmp_ticks_array = ticks[start_index: end_index].copy()
        # tmp_commands_array = commands[start_index : end_index].copy()
        tmp_index = ticks[start_index: end_index].copy() - ticks[start_index]
        for index2, j in enumerate(tmp_ticks_array):
            if lists[index][tmp_index[index2]]:
                # TODO: only consider fire action
                # TODO: consider duplicate actions
                if commands[j] == 'fire':
                    count += 1
        counts.append(count)
    return counts

    # raise NotImplementedError
    # return counts


#  Use function 1,2,3 to select best clip
def generate_clips(file: str):
    """
    :param file: the name of json file
    :return result: a clip with the most APM, the form is (file_name, (clip_start, clip_end))

    game_data['frames'] has 33 attributes:

    0   tick
    1   time
    2   scores

    fortress attributes:
    3	x
    4	y
    5	angle
    6	alive
    7	target_id
    8	vulnerable
    9	shield x
    10	shield y
    11	shield angle
    12	shield radius
    13	activation region x
    14	activation region y
    15	activation region angle
    16	activation region radius

    player 0 attributes:
    17	x
    18	y
    19	angle
    20	vx
    21	vy
    22	alive

    player 1 attributes:
    23	x
    24	y
    25	angle
    26	vx
    27	vy
    28	alive

    border attributes:
    29  xMin
    30  xMax
    31  yMin
    32  yMax
    """
    game_data = parseJson(file)
    json_data = read_json_file(file)
    candidate_tuples = generate_candidate_clips(game_data)
    if len(candidate_tuples) == 0:
        return None
    candidate_in_trust_region = in_trust_region(game_data, candidate_tuples, theta=np.pi / 3)
    #  TODO: Here is the function 3
    counts = count_actions(json_data, candidate_tuples, candidate_in_trust_region)
    idx_with_most_actions = np.argmax(np.array(counts))
    clipped_json_file = extract_Json_clips(json_data, [candidate_tuples[idx_with_most_actions]])
    return file, candidate_tuples[idx_with_most_actions], clipped_json_file


#  Iterate all json file
def main():
    best_clips = {}
    for index, file in enumerate(JSON_FILES_PATHS):
        print('%d / %d, Processing %s' % (index, len(JSON_FILES_PATHS), file.split('/')[2]))
        result = generate_clips(file)
        if not result:
            print('passed...')
            continue
        best_clips[result[0]] = result[1]
        with open('./clipped/' + file[12:-5] + '_clipped.json', 'w', encoding='utf-8') as f:
            json.dump(result[2], f, ensure_ascii=False, indent=4)
    return best_clips


def extract_Json_clips(JsonData, clip_ticks):
    new_Json = {'game_states': [], 'messages': [], 'player_input': JsonData['player_input'], 'ui_events': []}

    for start, end in clip_ticks:
        game_states_tmp = JsonData['game_states'][start:end]
        for g in game_states_tmp:
            g['tick'] -= start
        new_Json['game_states'].extend(game_states_tmp)

        for messages in JsonData['messages']:
            if messages['tick'] >= start and messages['tick'] < end:
                messages_tmp = copy.deepcopy(messages)
                messages_tmp['tick'] -= start
                new_Json['messages'].extend(messages_tmp)

        for event in JsonData['ui_events']:
            if event['tick'] >= start and event['tick'] < end:
                event_tmp = copy.deepcopy(event)
                event_tmp['tick'] -= start
                new_Json['ui_events'].extend(event_tmp)

    return new_Json


if __name__ == '__main__':
    best_clip_dict = main()

import json
import pandas as pd

action_mapping = {"0": "0",
                  "1": "1",
                  "2": "3",
                  "3": "4",
                  "4": "11",
                  "5": "12"}

def write_game_record(game_record, filename):
    with open(filename, 'w') as f:
        f.write("frame\treward\tscore\tterminal\taction\n")
        for record in game_record:
            f.write(record)

def read_data(directory, filename):
    path = directory + filename
    conv_lines = []
    with open(path, 'r') as f:
        f.readline()
        lines = f.readlines()
        for line in lines:
            frame, reward, score, terminal, action = line.split()
            action = action_mapping[action]
            reward = str(int(float(reward)))
            score = str(int(float(score)))
            conv_lines.append("\t".join([frame, reward, score, terminal, action]) + "\n")
    return conv_lines

# files = ["145.0.txt", "220.0.txt", "240.0.txt", "360.0.txt", "805.0.txt", "960.0.txt"]
# for file in files:
#     print(file)
#     lines = read_data("viz_data/", file)
#     write_game_record(lines, "viz_data/SI_mix/" + file)

files = ["3-260.0-self-noscreen.txt", "9-520.0-self-noscreen.txt",
         "10-240.0-self-noscreen.txt", "11-655.0-self-noscreen.txt",
         "14-340.0-self-noscreen.txt", "15-395.0-self-noscreen.txt"]
for file in files:
    print(file)
    lines = read_data("DQNfromDemo/Test/SI_screens/", file)
    write_game_record(lines, "viz_data/SI_self/" + file)

from os import path
import os
import sys
import time

local = path.abspath(__file__)
root = path.dirname(path.dirname(path.dirname(local)))
if root not in sys.path:
    sys.path.append(root)

import gym
import random
import torch
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import torch.nn.functional as F
from DQNwithNoisyNet.NoisyLayer import NoisyLinear
from DQNfromDemo import DQfD
import numpy as np

from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames
import json
from PIL import Image
import time

def plotJE(dqn,color):
    tree=dqn.replay.tree
    data = [[d] for d in tree.data[0:500]]
    JE = list(map(dqn.JE, data))
    plt.plot(JE, color=color)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_s = nn.Linear(4, 40)
        self.fc1_a = nn.Linear(1, 40)
        self.fc2 = nn.Linear(40, 1)

    def forward(self, s, a):
        x = self.fc1_s(s) + self.fc1_a(a)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)  # For 128-byte RAM states
        self.fc2 = nn.Linear(64, 6)  # Action space of 6

    def forward(self, s):
        x = self.fc1(s)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class NoisyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_s = NoisyLinear(4, 40)
        self.fc1_a = NoisyLinear(1, 40)
        self.fc2 = NoisyLinear(40, 1)

    def forward(self, s, a):
        x = self.fc1_s(s) + self.fc1_a(a)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def sample(self):
        for layer in self.children():
            if hasattr(layer, "sample"):
                layer.sample()


class NoisyNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = NoisyLinear(4, 40)
        self.fc2 = NoisyLinear(40, 2)

    def forward(self, s):
        x = self.fc1(s)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def sample(self):
        for layer in self.children():
            if hasattr(layer, "sample"):
                layer.sample()

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 32, 32), nn.Conv2d(32, 16, 53))
        self.fc = nn.Linear(16, 6)

    def forward(self, s):
        x = self.conv(s)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        # print("##### x shape #####", x.shape)
        return x.squeeze()


def write_game_record(game_record, filename):
    with open(filename, 'w') as f:
        f.write("frame\treward\tscore\tterminal\taction\n")
        for record in game_record:
            line = ""
            for i, ent in enumerate(record):
                if i == 4:
                    line += str(ent) + "\n"
                    # print(line)
                else:
                    line += str(ent) + "\t"
            f.write(line)


def preprocess_frame(frame):
    # Greyscale frame
    # gray = rgb2gray(frame)

    # Crop the screen (remove the part below the player)
    # [Up: Down, Left: right]
    cropped_frame = frame[8:-12, 4:-12]

    # Normalize Pixel Values
    normalized_frame = cropped_frame / 255.0

    # Resize
    # Thanks to Mikołaj Walkowiak
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])
    permute_preprocessed_frame = torch.from_numpy(preprocessed_frame).unsqueeze(0).permute(0,3,1,2).float()
    return permute_preprocessed_frame  # 110x84x1 frame


log_file_name = "SI_screens/SIPreLoss.txt"

if __name__ == "__main__":
    env = gym.make('SpaceInvaders-v0')
    s = env.reset()
    s = preprocess_frame(s)
    # print("state: ")
    # print(s.shape)
    A = [[0], [1], [2], [3], [4], [5]]
    af = lambda x:A
    dqn = DQfD.DeepQL(ConvNet, lr=0.002, gamma=1.0, actionFinder=None, N=50000,n_step=1)
    process = []
    randomness = []
    epoch = 10
    eps_start = 0.05
    eps_end = 0.95
    N = 1 - eps_start
    lam = -math.log((1 - eps_end) / N) / epoch
    total = 0
    count = 0  # successful count
    start = 0

    # screenshots = np.load("283.npy")
    #
    # with open("SpaceInvadersDemo283.txt", "r") as file:
    #     data = json.load(file)
    #     for k, v in data.items():
    #         for _, a, r, _, done in v:
    #             start += 1
    #             s = screenshots[start + 1][3:, :, :]
    #             s = preprocess_frame(s)
    #             s_ = screenshots[start + 2][3:, :, :]
    #             s_ = preprocess_frame(s_)
    #             dqn.storeDemoTransition(s, a, r, s_, done, int(k))
    #
    # dqn.replay.tree.start = start
    # loss = 0
    # # with open(log_file_name, 'a') as f:
    # #     f.write("[" + str(time.time()) + "]\t" + "Pre-train:\n")
    #
    # for i in range(100):
    #     loss += dqn.Jloss
    #     if i % 10 == 0:
    #         print(loss / 10)
    #         # with open(log_file_name, 'a') as f:
    #         #     f.write(str(i) + ":\t" + str(loss / 10) + "\n")
    #         loss = 0
    #         print("pretraining:", i)
    #         # print("Save the model...")
    #         # torch.save(dqn.vc.predictNet.state_dict(), "SI_screens/SIPrePredict2-" + str(i) + ".txt")
    #         # torch.save(dqn.vc.targetNet.state_dict(), "SI_screens/SIPreTarget2-" + str(i) + ".txt")
    #     try:
    #         dqn.update()
    #     except:
    #         print("uh-oh")
    #         continue
    #
    # # dqn.vc.predictNet.load_state_dict(torch.load("SI_screens/SIPrePredict2-290.txt"))
    # # dqn.vc.targetNet.load_state_dict(torch.load("SI_screens/SIPreTarget2-290.txt"))
    #
    # with open(log_file_name, 'a') as f:
    #     f.write("[" + str(time.time()) + "]\t" + "Self-train:\n")
    #
    # loss = 0
    # for i in range(epoch):
    #     print(i)
    #     with open(log_file_name, 'a') as f:
    #         f.write("[" + str(time.time()) + "]:\tRound " + str(i) + "\n")
    #     # if i % 10 == 0:
    #     #     print("Save the model...")
    #     #     torch.save(dqn.vc.predictNet.state_dict(), "./SpaceInvadersPredict.txt")
    #     #     torch.save(dqn.vc.targetNet.state_dict(), "./SpaceInvadersTarget.txt")
    #     dqn.eps = 1 - N * math.exp(-lam * i)
    #     dqn.eps = 0.9
    #     total = 0
    #     lives = 3
    #     steps = 0
    #     while True:
    #         loss += dqn.Jloss
    #         print(dqn.Jloss)
    #         if steps % 10 == 0:
    #             if steps % 100 == 0:
    #                 torch.save(dqn.vc.predictNet.state_dict(), "SI_screens/SIPredict2-" + str(steps) + ".txt")
    #                 torch.save(dqn.vc.targetNet.state_dict(), "SI_screens/SITarget2-" + str(steps) + ".txt")
    #             print("step: ", steps, "\tloss: ", loss / 10)
    #             with open(log_file_name, 'a') as f:
    #                 f.write(str(steps) + ":\t" + str(loss / 10) + "\n")
    #             loss = 0
    #         a = dqn.act(s)
    #         s_, r, done, info = env.step(a[0])
    #         s_ = preprocess_frame(s_)
    #         # print("s_:", s_.shape)
    #         total += r
    #         if info['ale.lives'] < lives:
    #             r = -10
    #             lives = info['ale.lives']
    #         dqn.storeTransition(s, a, r, s_, done)
    #         try:
    #             dqn.update()
    #         except:
    #             print("uh-oh again")
    #             continue
    #         s = s_
    #         steps += 1
    #         if done:
    #             s = env.reset()
    #             s = preprocess_frame(s)
    #             print('total:', total)
    #             with open(log_file_name, 'a') as f:
    #                 f.write("Game finished. Total Score:\t" + str(total))
    #             process.append(total)
    #             break
    #     torch.save(dqn.vc.predictNet.state_dict(), "SI_screens/SIPredict2.txt")
    #     torch.save(dqn.vc.targetNet.state_dict(), "SI_screens/SITarget2.txt")
    #
    # with open('SI_screens/SIScores.txt', 'w') as f:
    #     for s in process:
    #         f.write("%s\n" % s)

    dqn.vc.predictNet.load_state_dict(torch.load("SI_screens/SIPrePredict2-90.txt"))
    dqn.vc.targetNet.load_state_dict(torch.load("SI_screens/SIPreTarget2-90.txt"))
    # dqn.vc.predictNet.load_state_dict(torch.load("SI_screens/SIPrePredict.txt"))
    # dqn.vc.targetNet.load_state_dict(torch.load("SI_screens/SIPreTarget.txt"))

    # plt.plot(range(epoch), process)
    # plt.xlabel("Epoch")
    # plt.ylabel("Score")
    # plt.savefig('SpaceInvadersScore.jpg')

    # plt.show()
    total = 0
    s = env.reset()
    s = preprocess_frame(s)
    dqn.eps = 1
    avg_score = 0
    rounds = 0
    game_record = []
    frame = 0
    write_count = 0
    while True:
        if rounds > 20:
            break
        a = dqn.act(s)[0]
        # a = env.action_space.sample()
        # if random.random() > 0.5:
        #     a = 4  # left
        # a = random.choices([1, 3, 4], [0.5, 0.4, 0.1])
        s, r, done, _ = env.step(a)
        s = preprocess_frame(s)
        game_record.append([frame, r, total, done, a])
        total += r
        frame += 1
        env.render()
        if done:
            s = env.reset()
            s = preprocess_frame(s)
            avg_score = (avg_score * rounds + total) / (rounds + 1)
            print("Score: " + str(total) + "    Avg: " + str(avg_score))
            if total > 200:
                write_count += 1
                write_game_record(game_record, "SI_screens/" + str(rounds) + "-" + str(total) + "-full-noscreen.txt")
            total = 0
            frame = 0
            game_record = []
            rounds += 1

    env.close()


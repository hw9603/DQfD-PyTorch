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
import json


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
        print("xxxxxxxx")
        print(x.shape)
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


def write_game_record(game_record, filename):
    with open(filename, 'w') as f:
        f.write("frame\treward\tscore\tterminal\taction\tram_state\n")
        for record in game_record:
            line = ""
            for i, ent in enumerate(record):
                if i == 5:
                    line += str(list(ent)) + "\n"
                    print(line)
                else:
                    line += str(ent) + "\t"
            f.write(line)

if __name__ == "__main__":
    env = gym.make('SpaceInvaders-ram-v0')
    s = env.reset()
    A = [[0], [1], [2], [3], [4], [5]]
    af = lambda x:A
    dqn = DQfD.DeepQL(Net2, lr=0.002, gamma=1.0, actionFinder=None, N=50000,n_step=1)
    process = []
    randomness = []
    epoch = 100
    eps_start = 0.05
    eps_end = 0.95
    N = 1 - eps_start
    lam = -math.log((1 - eps_end) / N) / epoch
    total = 0
    count = 0  # successful count
    start = 0
    # with open("SpaceInvadersDemoNonZero.txt", "r") as file:
    #     data = json.load(file)
    #     for k, v in data.items():
    #         for s, a, r, s_, done in v:
    #             start += 1
    #             dqn.storeDemoTransition(s, a, r, s_, done, int(k))
    #
    # dqn.replay.tree.start = start
    # loss = 0
    # for i in range(3000):
    #     loss += dqn.Jloss
    #     if i % 100 == 0:
    #         print(loss / 100)
    #         # if loss / 100 < 50 and i != 0:
    #         #     break
    #         loss = 0
    #         print("pretraining:", i)
    #     dqn.update()
    #
    # print("Save the model...")
    # torch.save(dqn.vc.predictNet.state_dict(), "./SIPrePredict.txt")
    # torch.save(dqn.vc.targetNet.state_dict(), "./SIPreTarget.txt")

    # dqn.vc.predictNet.load_state_dict(torch.load("./SIPrePredict330.txt"))
    # dqn.vc.targetNet.load_state_dict(torch.load("./SIPreTarget330.txt"))

    for i in range(epoch):
        print(i)
        # if i % 10 == 0:
        #     print("Save the model...")
            # torch.save(dqn.vc.predictNet.state_dict(), "./SpaceInvadersPredict.txt")
            # torch.save(dqn.vc.targetNet.state_dict(), "./SpaceInvadersTarget.txt")
        dqn.eps = 1 - N * math.exp(-lam * i)
        dqn.eps = 0.9
        total = 0
        lives = 3
        while True:
            a = dqn.act(s)
            s_, r, done, info = env.step(a[0])
            total += r
            if info['ale.lives'] < lives:
                r = -10
                lives = info['ale.lives']
            dqn.storeTransition(s, a, r, s_, done)
            dqn.update()
            s = s_
            if done:
                s = env.reset()
                print('total:', total)
                process.append(total)
                break

    # with open('SpaceInvadersScores.txt', 'w') as f:
    #     for s in process:
    #         f.write("%s\n" % s)

    # dqn.vc.predictNet.load_state_dict(torch.load("./SIPrePredict330.txt"))
    # dqn.vc.targetNet.load_state_dict(torch.load("./SIPreTarget330.txt"))

    # plt.plot(range(epoch), process)
    # plt.xlabel("Epoch")
    # plt.ylabel("Score")
    # plt.savefig('SpaceInvadersScore.jpg')

    # plt.show()
    total = 0
    s = env.reset()
    dqn.eps = 1
    avg_score = 0
    rounds = 0
    game_record = []
    frame = 0
    write_count = 0
    while True:
        # if rounds > 20:
        #     break
        a = dqn.act(s)[0]
        # a = env.action_space.sample()
        # if random.random() > 0.5:
        #     a = 4  # left
        # a = random.choices([1, 3, 4], [0.5, 0.4, 0.1])
        s, r, done, _ = env.step(a)
        game_record.append([frame, r, total, done, a, s])
        total += r
        frame += 1
        env.render()
        if done:
            s = env.reset()
            avg_score = (avg_score * rounds + total) / (rounds + 1)
            print("Score: " + str(total) + "    Avg: " + str(avg_score))
            if total > 10:
                write_count += 1
                write_game_record(game_record, str(rounds) + "-" + str(total) + ".txt")
                if write_count > 1:
                    break
            total = 0
            frame = 0
            game_record = []
            rounds += 1

    env.close()


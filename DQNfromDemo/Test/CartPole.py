from os import path
import sys

local = path.abspath(__file__)
root = path.dirname(path.dirname(path.dirname(local)))
if root not in sys.path:
    sys.path.append(root)

import gym
import torch
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import torch.nn.functional as F
from DQNwithNoisyNet.NoisyLayer import NoisyLinear
from DQNfromDemo import DQfD
import json
from tensorboardX import SummaryWriter


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
        self.fc1 = nn.Linear(4, 40)
        self.fc2 = nn.Linear(40, 2)

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


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    s = env.reset()
    A = [[0], [1]]
    af = lambda x:A
    dqn = DQfD.DeepQL(Net2, lr=0.002, gamma=1.0, actionFinder=None, N=5000,n_step=1)
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
    # writer = SummaryWriter('logs/CartPole')
    with open("CartPoleDemo.txt", "r") as file:
        data = json.load(file)
        for k, v in data.items():
            for s, a, r, s_, done in v:
                start += 1
                dqn.storeDemoTransition(s, a, r, s_, done, int(k))

    dqn.replay.tree.start = start
    for i in range(500):
        if i % 100 == 0:
            print("pretraining:", i)
        dqn.update()

    print("Save the model...")
    torch.save(dqn.vc.predictNet.state_dict(), "./CartPolePredict.txt")
    torch.save(dqn.vc.targetNet.state_dict(), "./CartPoleTarget.txt")

    for i in range(epoch):
        print(i)
        if i % 10 == 0:
            print("Save the model...")
            torch.save(dqn.vc.predictNet.state_dict(), "./CartPolePredict.txt")
            torch.save(dqn.vc.targetNet.state_dict(), "./CartPoleTarget.txt")
        dqn.eps = 1 - N * math.exp(-lam * i)
        dqn.eps = 0.9
        count = count + 1 if total >= 500 else 0
        # if count >= 2:
        #     dqn.eps = 1
        #     break
        total = 0
        while True:
            a = dqn.act(s)
            s_, r, done, _ = env.step(a[0])
            total += r
            r = -1 if done and total < 500 else 0.002
            dqn.storeTransition(s, a, r, s_, done)
            dqn.update()
            s = s_
            if done:
                s = env.reset()
                print('total:', total)
                process.append(total)
                break
    with open('CartPoleScores.txt', 'w') as f:
        for s in process:
            f.write("%s\n" % s)
    plt.plot(range(epoch), process)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig('CartPoleScore.jpg')

    dqn.vc.predictNet.load_state_dict(torch.load("./CartPolePredict.txt"))
    dqn.vc.targetNet.load_state_dict(torch.load("./CartPoleTarget.txt"))
    total = 0
    s = env.reset()
    dqn.eps = 1
    while True:
        a = dqn.act(s)[0]
        s, r, done, _ = env.step(a)
        total += 1
        env.render()
        if done:
            s = env.reset()
            print(total)
            total = 0

    env.close()

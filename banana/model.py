# coding:utf-8

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, alpha, seed=12345):
        super(QNetwork, self).__init__()
        self.seed = T.manual_seed(seed)
        self.h1 = nn.Linear(state_size, 256)
        self.h2 = nn.Linear(256, 128)
        self.h3 = nn.Linear(128, action_size)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print("GPU:", T.cuda.is_available())
        self.to(self.device)

    def forward(self, state):
        state_tensor = T.Tensor(state).to(self.device)
        l1 = F.relu(self.h1(state_tensor))
        l2 = F.relu(self.h2(l1))
        action_ = self.h3(l2)
        return action_


class Agent(object):
    def __init__(self, action_size, state_size, gamma, epsilon, alpha, maxMemorySize, epsEnd=0.05, repalce=10000):
        self.GAMMA =gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.action_space = action_size
        self.memSize = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.memCntr = 0
        self.replace_target_cnt = repalce
        self.Q_eval = QNetwork(state_size, action_size, alpha)
        self.Q_next = QNetwork(state_size, action_size, alpha)

    def storeTransition(self, state, action, reward, state_, done):
        if self.memCntr < self.memSize:
            self.memory.append([state, action, reward, state_, done])
        else:
            self.memory[self.memCntr % self.memSize] = [state, action, reward, state_, done]
        self.memCntr += 1

    def chooseAction(self, state):
        rand = np.random.random()
        actions = self.Q_eval.forward(state)
        if rand < 1 - self.EPSILON:
            action = T.argmax(actions[1]).item()
        else:
            action = np.random.choice(self.action_space, p=[0.4, 0.2, 0.2, 0.2])
        self.steps += 1
        return action

    def learn(self, batch_size):
        self.Q_eval.optimizer.zero_grad()
        if self.replace_target_cnt is not None and \
           self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

        if self.memCntr + batch_size < self.memSize:
            memStart = int(np.random.choice(range(self.memCntr)))
        else:
            memStart = int(np.random.choice(range(self.memSize-batch_size-1)))

        miniBatch = self.memory[memStart:memStart + batch_size]
        memory = np.array(miniBatch)

        Qpred = self.Q_eval.forward(list(memory[:, 0][:])).to(self.Q_eval.device)
        Qnext = self.Q_next.forward(list(memory[:, 3][:])).to(self.Q_eval.device)

        maxA = T.argmax(Qnext, dim=1).to(self.Q_eval.device)
        rewards = T.Tensor(list(memory[:, 2])).to(self.Q_eval.device)
        Qtarget = Qpred.clone().to(self.Q_eval.device)
        Qtarget[:, maxA] = rewards + self.GAMMA * T.max(Qnext, dim=1)[0]

        if self.steps > 500:
            if self.EPSILON - 1e-4 > self.EPS_END:
                self.EPSILON -= 1e-3
            else:
                self.EPSILON = self.EPS_END

        loss_f = self.Q_eval.loss(Qpred, Qtarget).to(self.Q_eval.device)
        loss_f.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1

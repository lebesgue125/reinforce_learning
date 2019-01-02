# coding:utf-8

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import random

dev = T.device('cuda:0' if T.cuda.is_available() else 'cpu')


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.seed = T.manual_seed(0)
        self.h1 = nn.Linear(state_size, 128)
        self.h2 = nn.Linear(128, 64)
        self.h3 = nn.Linear(64, action_size)

    def forward(self, state):
        l1 = F.relu(self.h1(state))
        l2 = F.relu(self.h2(l1))
        action_ = self.h3(l2)
        return action_


class Agent(object):
    def __init__(self, action_size, state_size, gamma, alpha,
                 maxMemorySize, repalce, path=None, continue_train=True):
        self.GAMMA =gamma
        self.action_space = action_size
        self.memSize = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.memCntr = 0
        self.path = path
        self.replace_target_cnt = repalce
        self.Q_eval = QNetwork(state_size, action_size).to(dev)
        self.Q_next = QNetwork(state_size, action_size).to(dev)
        self.optimizer = optim.Adam(self.Q_eval.parameters(), lr=alpha)
        # self.Q_next.load_state_dict(self.Q_eval.state_dict())
        if path is not None and continue_train:
            if os.path.exists(path + 'q_next'):
                self.Q_next.load_state_dict(T.load(path + 'q_next'))
                self.Q_next.to(dev)
            if os.path.exists(path + 'q_eval'):
                self.Q_eval.load_state_dict(T.load(path + 'q_eval'))
                self.Q_eval.to(dev)

    def storeTransition(self, state, action, reward, state_, done):
        if self.memCntr < self.memSize:
            self.memory.append([state, action, reward, state_, done])
        else:
            self.memory[self.memCntr % self.memSize] = [state, action, reward, state_, done]
        self.memCntr += 1

    def chooseAction(self, state, eps):
        if np.random.random() > eps:
            self.Q_eval.eval()
            with T.no_grad():
                action = T.argmax(self.Q_eval.forward(T.from_numpy(state).float().to(dev))).item()
            self.Q_eval.train()
        else:
            action = np.random.choice(self.action_space, p=[0.4, 0.2, 0.2, 0.2])
        self.steps += 1
        return action

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return
        memory = np.array(random.sample(self.memory, k=batch_size))
        Qpred = self.Q_eval.forward(T.from_numpy(np.vstack(memory[:, 0])).float().to(dev)).to(dev)
        Qnext = self.Q_next.forward(T.from_numpy(np.vstack(memory[:, 3])).float().to(dev)).to(dev)

        # maxA = T.argmax(Qnext, dim=1).to(self.Q_eval.device)
        rewards = T.from_numpy(np.vstack(memory[:, 2])).float().to(dev)
        dones = T.from_numpy(np.vstack(memory[:, 4]).astype(dtype=np.float32)).float().to(dev)
        actions = T.from_numpy(np.vstack(memory[:, 1])).long().to(dev)
        # Qtarget = Qpred.clone().to(self.Q_eval.device)
        # Qtarget[:, maxA] = rewards + self.GAMMA * T.max(Qnext, dim=1)[0]
        Qtarget = rewards + self.GAMMA * Qnext.detach().max(1)[0].unsqueeze(1) * (1 - dones)
        Q_e = Qpred.gather(1, actions.view(batch_size, 1))
        loss_f = F.mse_loss(Q_e, Qtarget)
        self.optimizer.zero_grad()
        loss_f.backward()
        self.optimizer.step()

        for eval_param, target_param in zip(self.Q_eval.parameters(), self.Q_next.parameters()):
            target_param.data.copy_(0.001 * eval_param.data + 0.999 * target_param.data)
        self.learn_step_counter += 1

    def save(self):
        T.save(self.Q_eval.state_dict(), self.path + '/q_eval')
        T.save(self.Q_next.state_dict(), self.path + '/q_next')

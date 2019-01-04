# coding:utf-8
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from util.pytorch_param import dev
from util.HistoryStorage import HistoryStored


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
                 max_memory_size, tau):
        self.gamma = gamma
        self.tau = tau
        self.action_size = action_size
        self.state_size = state_size
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = HistoryStored('GameRecord',
                                    ['states', 'actions', 'rewards', 'state_', 'dones'],
                                    max_memory_size)
        self.memCntr = 0
        self.Q_eval = QNetwork(state_size, action_size).to(dev)
        self.Q_next = QNetwork(state_size, action_size).to(dev)
        self.optimizer = optim.Adam(self.Q_eval.parameters(), lr=alpha)
        # self.Q_next.load_state_dict(self.Q_eval.state_dict())

    def store(self, states, actions, rewards, next_states, dones):
        trajectory = dict()
        trajectory['actions'] = actions
        trajectory['rewards'] = np.array(rewards, dtype=np.float32)
        trajectory['dones'] = [0 if d else 1 for d in dones]
        trajectory['states'] = states
        trajectory['state_'] = next_states
        self.memory.add(trajectory)

    def choose_action(self, state, eps):
        if np.random.random() > eps:
            self.Q_eval.eval()
            with T.no_grad():
                action = T.argmax(self.Q_eval.forward(T.from_numpy(state).float().to(dev))).item()
            self.Q_eval.train()
        else:
            action = np.random.choice(self.action_size)
        self.steps += 1
        return action

    def learn(self, batch_size):
        if self.memory.total_record < batch_size + 2:
            return 0

        train_data = self.memory.take_sample(batch_size)
        states = train_data['states'].reshape([-1, self.state_size])
        actions = train_data['actions'].reshape([-1, 1])
        rewards = train_data['rewards'].reshape([-1, 1])
        state_ = train_data['state_'].reshape([-1, self.state_size])
        dones = train_data['dones'].reshape([-1, 1])

        Qpred = self.Q_eval.forward(T.from_numpy(states).float().to(dev)).to(dev)
        Qnext = self.Q_next.forward(T.from_numpy(state_).float().to(dev)).to(dev)

        # maxA = T.argmax(Qnext, dim=1).to(self.Q_eval.device)
        rewards = T.from_numpy(np.vstack(rewards)).float().to(dev)
        dones = T.from_numpy(np.vstack(dones).astype(dtype=np.float32)).float().to(dev)
        actions = T.from_numpy(np.vstack(actions)).long().to(dev)
        # Qtarget = Qpred.clone().to(self.Q_eval.device)
        # Qtarget[:, maxA] = rewards + self.GAMMA * T.max(Qnext, dim=1)[0]
        Qtarget = rewards + self.gamma * Qnext.detach().max(1)[0].unsqueeze(1) * dones
        Q_e = Qpred.gather(1, actions.view(batch_size, 1))
        loss_f = F.mse_loss(Q_e, Qtarget)
        self.optimizer.zero_grad()
        loss_f.backward()
        self.optimizer.step()

        for eval_param, target_param in zip(self.Q_eval.parameters(), self.Q_next.parameters()):
            target_param.data.copy_(self.tau * eval_param.data + (1 - self.tau) * target_param.data)
        self.learn_step_counter += 1

    def save(self, path):
        T.save(self.Q_eval.state_dict(), path + 'q_eval')
        T.save(self.Q_next.state_dict(), path + 'q_next')

    def load(self, path):
        if os.path.exists(path + 'q_next'):
            self.Q_next.load_state_dict(T.load(path + 'q_next'))
            self.Q_next.to(dev)
        if os.path.exists(path + 'q_eval'):
            self.Q_eval.load_state_dict(T.load(path + 'q_eval'))
            self.Q_eval.to(dev)

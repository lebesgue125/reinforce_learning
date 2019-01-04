import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from util.pytorch_param import dev
from torch.autograd import Variable
from util.HistoryStorage import HistoryStored
from util.noise import *


class ActorNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super(ActorNetwork, self).__init__()
        self.seed = T.manual_seed(0)
        self.action_size = action_size
        self.state_size = state_size

        self.h1 = nn.Linear(state_size, 64)
        self.h2 = nn.Linear(64, 128)
        self.h3 = nn.Linear(128, self.action_size)

    def forward(self, states):
        l1 = F.relu(T.from_numpy(states).float().to(dev))
        l2 = F.relu(l1)
        return F.tanh(l2)


class Actor(object):

    def __init__(self, state_size, action_size, ALPHA, TAU):
        self.action_size = action_size
        self.state_size = state_size
        self.alpha = ALPHA
        self.tau = TAU
        self.actor = ActorNetwork(state_size, action_size).to(dev)
        self.actor_ = ActorNetwork(state_size, action_size).to(dev)
        self.optimizer = T.optim.Adam(self.alpha)

    def train(self, state, action_grads):
        grads = T.autograd.grad(self.actor.forward(state), Variable(self.actor.parameters()), -action_grads)
        self.optimizer.zero_grad()
        for (p, g) in zip(self.actor.parameters(), grads):
            p.grad = g
        self.optimizer.step()

    def target_train(self):
        for eval_param, target_param in zip(self.actor.parameters(), self.actor_.parameters()):
            target_param.data.copy_(self.tau * eval_param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        T.save(self.actor.state_dict(), path + '/actor')
        T.save(self.actor_.state_dict(), path + '/actor_')

    def load(self, path):
        p_a = path + '/actor'
        if os.path.exists():
            self.actor.load_state_dict(T.load(p_a))
            self.actor.to(dev)
        p_a_ = path + '/actor_'
        if os.path.exists(p_a_):
            self.actor_.load_state_dict(T.load(p_a_))
            self.actor_.to(dev)


class CriticNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super(CriticNetwork, self).__init__()
        self.seed = T.manual_seed(0)
        self.action_size = action_size
        self.state_size = state_size

        self.s_heed = nn.Linear(self.state_size, 64)
        self.s_h1 = nn.Linear(64, 128)
        self.a_h1 = nn.Linear(self.state_size, 128)
        self.h2 = nn.Linear(128, 128)
        self.h3 = nn.Linear(128, self.action_size)

    def forward(self, states, actions):
        s = T.from_numpy(states).float().to(dev)
        a = T.from_numpy(actions).float().to(dev)
        s_s = F.relu(self.s_heed(s))
        s_1 = self.s_h1(s_s)
        a_1 = self.a_h1(a)
        s_a = s_1 + a_1
        s_a_2 = F.relu(self.h2(s_a))
        return self.h3(s_a_2)


class Critic(object):

    def __init__(self, state_size, action_size, ALPHA, TAU):
        self.state_size = state_size
        self.action_size = action_size
        self.tau = TAU
        self.alpha = ALPHA
        self.critic = CriticNetwork(state_size, action_size).to(dev)
        self.critic_ = CriticNetwork(state_size, action_size).to(dev)
        self.optimizer = T.optim.Adam(self.alpha)

    def gradients(self, states, actions):
        return T.autograd.grad(self.critic.forward(states, actions), actions)

    def target_train(self):
        for eval_param, target_param in zip(self.critic.parameters(), self.critic_.parameters()):
            target_param.data.copy_(self.tau * eval_param.data + (1 - self.tau) * target_param.data)

    def train_on_batch(self, states, actions, y_t):
        loss = F.mse_loss(self.critic.forward(states, actions), y_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def save(self, path):
        T.save(self.critic.state_dict(), path + '/critic')
        T.save(self.critic_.state_dict(), path + '/critic_')

    def load(self, path):
        p_c = path + '/critic'
        if os.path.exists(p_c):
            self.critic.load_state_dict(T.load(p_c))
            self.critic.to(dev)
        p_c_ = path + '/critic_'
        if os.path.exists(p_c_):
            self.critic_.load_state_dict(T.load(p_c_))
            self.critic_.to(dev)


class DDPG_AGENT(object):

    def __init__(self, action_size, state_size, ALPHA, TAU, max_memory_size):
        self.action_size = action_size
        self.state_size = state_size
        self.alpha = ALPHA
        self.tau = TAU
        self.actor = Actor(state_size, action_size, ALPHA, TAU)
        self.critic = Critic(state_size, action_size, ALPHA, TAU)
        self.memory = HistoryStored('GameRecord',
                                    ['states', 'actions', 'rewards', 'state_', 'dones'],
                                    max_memory_size)
        self.step = 0

    def store(self, states, actions, rewards, next_states, dones):
        trajectory = dict()
        trajectory['actions'] = actions
        trajectory['rewards'] = np.array(rewards)
        trajectory['dones'] = [0 if d else 1 for d in dones]
        trajectory['states'] = states
        trajectory['state_'] = next_states
        self.memory.add(trajectory)

    def choose_action(self, state, epsilon, agent_num, train=True):
        action = self.actor.actor.forward(state.reshape([agent_num, self.state_size]))
        if train:
            action += max(epsilon, 0) * ou_generate_noise(action, 0.0, 0.60, 0.30)
        self.step += 1

        return action

    def learn(self, batch_size, gamma):
        if self.memory.total_record < batch_size + 2:
            return 0
        train_data = self.memory.take_sample(batch_size)
        states = train_data['states'].reshape([-1, self.state_size])
        actions = train_data['actions'].reshape([-1, self.action_size])
        rewards = train_data['rewards'].reshape([-1, 1])
        state_ = train_data['state_'].reshape([-1, self.state_size])
        dones = train_data['dones'].reshape([-1, 1])

        q_value_ = self.critic.critic_.forward(state_, self.actor.actor_.forward(state_))
        y_t = rewards + gamma * dones * q_value_
        loss = self.critic.train_on_batch(states, actions, y_t)
        a_for_grad = self.actor.actor.forward(states)
        grads = self.critic.gradients(states, a_for_grad)
        self.actor.train(states, grads)
        self.actor.target_train()
        self.critic.target_train()
        return loss

    def save(self, path):
        self.actor.save(path)
        self.critic.save(path)

    def load(self, path):
        try:
            self.actor.load(path)
            self.critic.load(path)
        except Exception as e:
            print('loading data encounter an error.', e)


import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from util.pytorch_param import dev, numpy_to_tensor
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
        l1 = F.relu(self.h1(states))
        l2 = F.relu(self.h2(l1))
        return F.tanh(self.h3(l2))


class Actor(object):

    def __init__(self, state_size, action_size, alpha, tau):
        self.action_size = action_size
        self.state_size = state_size
        self.alpha = alpha
        self.tau = tau
        self.actor = ActorNetwork(state_size, action_size).to(dev)
        self.actor_ = ActorNetwork(state_size, action_size).to(dev)
        self.optimizer = T.optim.Adam(self.actor.parameters(), lr=self.alpha)

    def train(self, policy_loss):
        policy_loss = -policy_loss.sum()
        policy_loss.backward()
        self.optimizer.step()

    def target_train(self):
        for eval_param, target_param in zip(self.actor.parameters(), self.actor_.parameters()):
            target_param.data.copy_(self.tau * eval_param.data + (1 - self.tau) * target_param.data)

    def hard_update(self):
        for eval_param, target_param in zip(self.actor.parameters(), self.actor_.parameters()):
            target_param.data.copy_(eval_param.data)

    def save(self, path):
        T.save(self.actor.state_dict(), path + '/actor')
        T.save(self.actor_.state_dict(), path + '/actor_')

    def load(self, path):
        p_a = path + '/actor'
        if os.path.exists(p_a):
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
        self.a_h1 = nn.Linear(self.action_size, 128)
        self.h2 = nn.Linear(256, 128)
        self.h3 = nn.Linear(128, 1)

    def forward(self, states, actions):
        s_s = F.relu(self.s_heed(states))
        s_1 = self.s_h1(s_s)
        a_1 = F.relu(self.a_h1(actions))
        s_a = T.cat((s_1, a_1), dim=1)
        s_a_2 = F.relu(self.h2(s_a))
        return self.h3(s_a_2)


class Critic(object):

    def __init__(self, state_size, action_size, alpha, tau):
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.alpha = alpha
        self.critic = CriticNetwork(state_size, action_size).to(dev)
        self.critic_ = CriticNetwork(state_size, action_size).to(dev)
        self.optimizer = T.optim.Adam(self.critic.parameters(), lr=self.alpha)

    def gradient(self, states, a_for_grad):
        return self.critic.forward(states, a_for_grad)

    def target_train(self):
        for eval_param, target_param in zip(self.critic.parameters(), self.critic_.parameters()):
            target_param.data.copy_(self.tau * eval_param.data + (1 - self.tau) * target_param.data)

    def hard_update(self):
        for eval_param, target_param in zip(self.critic.parameters(), self.critic_.parameters()):
            target_param.data.copy_(eval_param.data)

    def train_on_batch(self, states, actions, y_t):
        self.optimizer.zero_grad()
        loss = F.mse_loss(self.critic.forward(states, actions), y_t)
        loss.backward()
        self.optimizer.step()

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


class DDPGAGENT(object):

    def __init__(self, action_size, state_size, actor_alpha, critic_alpha, tau, max_memory_size):
        self.action_size = action_size
        self.state_size = state_size
        self.actor = Actor(state_size, action_size, actor_alpha, tau)
        self.critic = Critic(state_size, action_size, critic_alpha, tau)
        self.actor.hard_update()
        self.critic.hard_update()
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
        action_t = numpy_to_tensor(state.reshape([agent_num, self.state_size]), 'float')
        self.actor.actor.eval()
        with T.no_grad():
            action = self.actor.actor.forward(action_t).cpu().data.numpy()
        self.actor.actor.train()
        if train:
            action += max(epsilon, 0) * ou_generate_noise(action, 0.0, 0.60, 0.30)
        self.step += 1

        return action

    def learn(self, batch_size, gamma):
        if self.memory.total_record < batch_size + 2:
            return 0
        train_data = self.memory.take_sample(batch_size)
        states = numpy_to_tensor(train_data['states'].reshape([-1, self.state_size]), 'float')
        actions = numpy_to_tensor(train_data['actions'].reshape([-1, self.action_size]), 'float')
        rewards = numpy_to_tensor(train_data['rewards'].reshape([-1, 1]), 'float')
        state_ = numpy_to_tensor(train_data['state_'].reshape([-1, self.state_size]), 'float')
        dones = numpy_to_tensor(train_data['dones'].reshape([-1, 1]), 'float')

        q_value_ = self.critic.critic_.forward(state_, self.actor.actor_.forward(state_))
        y_t = rewards + gamma * dones * q_value_
        self.critic.train_on_batch(states, actions, y_t)
        self.actor.optimizer.zero_grad()
        a_for_grad = self.actor.actor.forward(states)
        grads = self.critic.gradient(states, a_for_grad)
        self.actor.train(grads)
        self.actor.target_train()
        self.critic.target_train()

    def save(self, path):
        self.actor.save(path)
        self.critic.save(path)

    def load(self, path):
        try:
            self.actor.load(path)
            self.critic.load(path)
        except Exception as e:
            print('loading data encounter an error.', e)


import tensorflow as tf
from util.HistoryStorage import HistoryStored
import numpy as np


class QNetwork(object):

    def __init__(self, state_size, action_size, alpha, tua, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.model, self.state = self.create_dqn()
        self.model_, self.state_ = self.create_dqn()
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='rewards')
        self.dones = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='dones')
        self.actions = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='actions')
        self.alpha = alpha
        self.tua = tua
        self.gamma = gamma
        self.target = tf.py_func(self.q_function,
                                 [self.model.output, self.model_.output, self.rewards, self.actions, self.dones],
                                 [tf.float32])
        self.loss = tf.losses.mean_squared_error(self.target, self.model.output)
        self.optimize = tf.train.AdamOptimizer(alpha).minimize(self.loss)

    def train(self, session, state, state_, rewards, actions, dones):
        return session.run({'opt': self.optimize, 'loss': self.loss},
                           {self.state: state, self.state_: state_,
                            self.rewards: rewards, self.dones: dones, self.actions: actions})

    def create_dqn(self):
        state = tf.keras.layers.Input(shape=[self.state_size])
        h1 = tf.keras.layers.Dense(128, activation='relu')(state)
        h2 = tf.keras.layers.Dense(128, activation='relu')(h1)
        R = tf.keras.layers.Dense(self.action_size, activation='linear')(h2)
        model = tf.keras.Model(inputs=state, outputs=R)
        return model, state

    def train_target(self):
        m_v = self.model.get_weights()
        m_v_ = self.model_.get_weights()
        for i in range(len(m_v)):
            m_v_[i] = m_v[i] * self.tua + (1 - self.tua) * m_v_[i]
        self.model_.set_weights(m_v_)

    def __index_along_every_row(self, array, index):
        N, _ = array.shape
        t = np.zeros([N, 1], dtype=np.float32)
        for i in range(N):
            t[i, 0] = array[i, index[i, 0]]
        return t

    def q_function(self, eval, target, rewards, actions, dones):
        N, _ = eval.shape
        for i in range(N):
            eval[i, actions[i, 0]] = rewards[i, 0] + self.gamma * max(target[i, :]) * dones[i, 0]
        return eval


class DQNAgent(object):

    def __init__(self, session, state_size, action_size, max_memory_size, gamma=0.99, alpha=1e-4, tua=1e-3):
        self.session = session
        tf.keras.backend.set_session(session)
        self.state_size = state_size
        self.action_size = action_size
        self.dqn = QNetwork(state_size, action_size, alpha, tua, gamma)
        self.memory = HistoryStored('GameRecord',
                                    ['states', 'actions', 'rewards', 'state_', 'dones'],
                                    max_memory_size)
        self.step = 0
        self.session.run(tf.global_variables_initializer())

    def choose_action(self, state):
        self.step += 1
        return self.dqn.model.predict(state.reshape([-1, self.state_size]))

    def store(self, states, actions, rewards, next_states, dones):
        trajectory = dict()
        trajectory['actions'] = actions
        trajectory['rewards'] = np.array(rewards)
        trajectory['dones'] = [0 if d else 1 for d in dones]
        trajectory['states'] = states
        trajectory['state_'] = next_states
        self.memory.add(trajectory)

    def learn(self, batch_size):
        if self.memory.total_record < batch_size + 2:
            return 0

        train_data = self.memory.take_sample(batch_size)
        states = train_data['states'].reshape([-1, self.state_size])
        actions = train_data['actions'].reshape([-1, 1])
        rewards = train_data['rewards'].reshape([-1, 1])
        state_ = train_data['state_'].reshape([-1, self.state_size])
        dones = train_data['dones'].reshape([-1, 1])

        loss = self.dqn.train(self.session, states, state_, rewards, actions, dones)
        self.dqn.train_target()

        return loss['loss']





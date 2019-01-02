import tensorflow as tf
from util.HistoryStorage import HistoryStored
from util.noise import *

class Actor(object):

    def __init__(self, state_size, action_size, session, ALPHA, TAU):
        self.action_size = action_size
        self.state_size = state_size
        self.alpha = ALPHA
        self.tau = TAU
        self.session = session
        tf.keras.backend.set_session(self.session)
        self.model, self.weight, self.state = self.create_network()
        self.model_, self.weight_, self.state_ = self.create_network()

        self.action_grad = tf.placeholder(dtype=tf.float32, shape=[None, self.action_size])
        self.param_grad = tf.gradients(self.model.output, self.weight, -self.action_grad)
        grads = zip(self.param_grad, self.weight)
        self.optimize = tf.train.AdamOptimizer(self.alpha).apply_gradients(grads)
        self.session.run(tf.global_variables_initializer())

    def train(self, state, action_grads):
        self.session.run(self.optimize, {self.state: state, self.action_grad:action_grads})

    def target_train(self):
        actor_w = self.model.get_weights()
        actor_w_ = self.model_.get_weights()
        for i in range(len(actor_w)):
            actor_w_[i] = self.tau * actor_w[i] + (1 - self.tau) * actor_w_[i]
        self.model_.set_weights(actor_w_)


    def create_network(self):
        state = tf.keras.Input(shape=[self.state_size], dtype='float32')
        h0 = tf.keras.layers.Dense(64, activation='relu')(state)
        h1 = tf.keras.layers.Dense(128, activation='relu')(h0)
        R = tf.keras.layers.Dense(self.action_size, activation='tanh')(h1)
        # a_2 = tf.keras.layers.Dense(1, activation='tanh')(h1)
        # a_3 = tf.keras.layers.Dense(1, activation='tanh')(h1)
        # a_4 = tf.keras.layers.Dense(1, activation='tanh')(h1)
        # R = tf.keras.layers.concatenate([a_1, a_2, a_3, a_4], axis=1)
        model = tf.keras.Model(inputs=state, outputs=R)
        return model, model.trainable_weights, state


class Critic(object):

    def __init__(self, state_size, action_size, session, ALPHA, TAU):
        self.state_size = state_size
        self.action_size = action_size
        self.tau = TAU
        self.alpha = ALPHA
        self.session = session
        tf.keras.backend.set_session(session)
        self.model, self.state, self.action = self.create_network()
        self.model_, self.state_, self.action_ = self.create_network()
        self.action_grads = tf.gradients(self.model.output, self.action)
        self.session.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.session.run(self.action_grads, {self.state: states, self.action: actions})[0]

    def target_train(self):
        t_w = self.model.get_weights()
        t_w_ = self.model_.get_weights()
        for i in range(len(t_w)):
            t_w_[i] = self.tau * t_w[i] + (1 - self.tau) * t_w_[i]
        self.model_.set_weights(t_w_)

    def create_network(self):
        state = tf.keras.Input(shape=[self.state_size])
        action = tf.keras.Input(shape=[self.action_size], name='action_input')
        s_head = tf.keras.layers.Dense(64, activation='relu')(state)
        a_h1 = tf.keras.layers.Dense(128, activation='linear')(action)
        s_h1 = tf.keras.layers.Dense(128, activation='linear')(s_head)
        m_h2 = tf.keras.layers.add([a_h1, s_h1])
        h3 = tf.keras.layers.Dense(128, activation='relu')(m_h2)
        v = tf.keras.layers.Dense(self.action_size, activation='linear')(h3)
        model = tf.keras.Model(inputs=[state, action], outputs=v)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(self.alpha))
        return model, state, action


class DDPG_AGENT(object):

    def __init__(self, session, action_size, state_size, ALPHA, TAU, max_memory_size):
        self.session = session
        tf.keras.backend.set_session(session)
        self.action_size = action_size
        self.state_size = state_size
        self.alpha = ALPHA
        self.tau = TAU
        self.actor = Actor(state_size, action_size, session, ALPHA, TAU)
        self.critic = Critic(state_size, action_size, session, ALPHA, TAU)
        self.memory = HistoryStored('GameRecord',
                                    ['states', 'actions', 'rewards', 'state_', 'dones'],
                                    max_memory_size)
        self.step = 0

    def store(self, dict_data):
        self.memory.add(dict_data)

    def choose_action(self, state, epsilon, train=True):
        action = self.actor.model.predict(state.reshape([1, self.state_size]))
        if train:
            action += max(epsilon, 0) * ou_generate_noise(action, 0.0, 0.60, 0.30)
        self.step += 1
        return action

    def learn(self, batch_size, gamma):
        if self.memory.total_record < batch_size + 2:
            return 0
        train_data = self.memory.take_sample(batch_size)
        states = train_data['states']
        actions = train_data['actions']
        rewards = train_data['rewards']
        state_ = train_data['state_']
        dones = train_data['dones']
        q_value_ = self.critic.model_.predict([state_, self.actor.model_.predict(state_)])

        y_t = rewards + gamma * dones * q_value_
        loss = self.critic.model.train_on_batch([states, actions], y_t)
        a_for_grad = self.actor.model.predict(states)
        grads = self.critic.gradients(states, a_for_grad)
        self.actor.train(states, grads)
        self.actor.target_train()
        self.critic.target_train()
        return loss





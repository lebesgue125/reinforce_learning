import tensorflow as tf
from util import networktool as nt


class FullConnectNetwork(object):

    def __init__(self, name, hidden_size, state_size, agent_num=1, gate=tf.nn.relu):
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.agent_num = agent_num
        self.name = name
        self.gate = gate
        self.state = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size], name=name)
        self.dims = hidden_size[-1]

    def forward(self):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            h1 = self.gate(nt.linear_layer('L%d' % 0, self.state, self.state_size, 64))
            h2 = self.gate(nt.linear_layer('L%d' % 1, h1, 64, 64))
            return h2


class GaussianActorCritic(object):

    def __init__(self, actor_n, critic_n, action_size, alpha):
        self.actor_n = actor_n
        self.critic_n = critic_n
        self.alpha = alpha
        self.action_size = action_size
        self.actor_mean = nt.linear_layer('out_action_mean', self.actor_n.forward(), self.actor_n.dims, action_size)
        self.actor_std = nt.linear_layer('out_action_std', self.actor_n.forward(), self.actor_n.dims, action_size)
        self.critic_fc = nt.linear_layer('out_critic', self.critic_n.forward(), self.critic_n.dims, 1)
        # with tf.variable_scope('AC_std', reuse=tf.AUTO_REUSE):
        #     self.std = tf.get_variable('action_std',
        #                                initializer=tf.zeros(shape=[1, action_size]),
        #                                dtype=tf.float32,
        #                                trainable=False)
        # self.log_prob = tf.placeholder(dtype=tf.float32,
        #                                shape=[None, self.actor_n.agent_num],
        #                                name='log_prob')
        #
        # self.critic_v = tf.placeholder(dtype=tf.float32,
        #                                shape=[None, self.actor_n.agent_num, 1],
        #                                name='critic_v')
        self.log_prob = None
        self.ret = tf.placeholder(dtype=tf.float32,
                                  shape=[None, 1],
                                  name='ret')

        self.advantage = tf.placeholder(dtype=tf.float32,
                                        shape=[None, 1],
                                        name='adv')
        #
        # self.entropy = tf.placeholder(dtype=tf.float32,
        #                               shape=[None, self.actor_n.agent_num],
        #                               name='entropy')
        self.action = None
        self.entropy = None
        self.loss = None
        self.train_op = None
        self.policy_loss = None
        self.entropy_loss = None
        self.value_loss = None

    def forward(self, action=None):
        mean = tf.tanh(self.actor_mean)
        dist = tf.distributions.Normal(mean, tf.sqrt(tf.nn.softplus(self.actor_std)))
        if action is None:
            self.action = dist.sample()
        else:
            self.action = action
        self.log_prob = tf.reduce_sum(dist.log_prob(self.action), axis=-1)
        self.entropy = tf.reduce_sum(dist.entropy(), axis=-1)

        return {'actions': self.action,
                'log_prob': self.log_prob,
                'entropy': self.entropy,
                'mean': mean,
                'critic_v': self.critic_fc}

    def loss_fun(self, entropy_loss_weight, value_loss_weight):
        self.policy_loss = -tf.reduce_mean(tf.multiply(self.log_prob, self.advantage))
        self.value_loss = 0.5 * tf.reduce_mean(tf.squared_difference(self.ret, self.critic_fc))
        self.entropy_loss = tf.reduce_mean(self.entropy)
        self.loss = self.policy_loss - self.entropy_loss * entropy_loss_weight + self.value_loss * value_loss_weight

    def train(self):
        # self.train_op = (tf.train.RMSPropOptimizer(self.alpha).minimize(self.policy_loss),
        #                  tf.train.RMSPropOptimizer(self.alpha).minimize(self.value_loss))
        self.train_op = tf.train.RMSPropOptimizer(self.alpha)
        self._clip_grad()
        return {'policy_loss': self.policy_loss,
                'value_loss': self.value_loss,
                'entropy_loss': self.entropy,
                'train': self.train_op}

    def _clip_grad(self):
        if self.train_op is not None and self.loss is not None:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            grad_and_vars = self.train_op.compute_gradients(self.loss, tf.trainable_variables())
            self.train_op = self.train_op.apply_gradients([[
                tf.clip_by_value(g, -0.1, 0.1), v]
                for g, v in grad_and_vars
            ], global_step=self.global_step)








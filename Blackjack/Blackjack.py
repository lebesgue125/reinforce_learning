import sys
from collections import defaultdict

import gym
import numpy as np
from util import calculate_G
from Blackjack.plot_utils import plot_blackjack_values, plot_policy

env = gym.make('Blackjack-v0')

def generate_episode_from_limit_stochastic(bj_env):
    episode = []
    state = bj_env.reset()
    while True:
        probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p=probs)
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def generate_episode(bj_env, Q, epsilon):
    episode = []
    state = bj_env.reset()
    nA = bj_env .action_space.n
    while True:
        p_s = np.ones(nA) * epsilon / nA
        best_one = np.argmax(Q[state])
        p_s[best_one] = 1 - epsilon + (epsilon / nA)

        action = np.random.choice(np.array(nA), p=p_s) if state in Q else bj_env.action_space.sample()
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0):
    # initialize empty dictionaries of arrays
    # returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.\n".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        episode = generate_episode(env)
        reward = calculate_G(list(map(lambda per: per[-1], episode)), gamma)
        for k in range(len(episode)):
            state, action, _ = episode[k]
            N[state][action] += 1
            Q[state][action] += (reward[k] - Q[state][action]) / N[state][action]
    return Q





def mc_control(env, num_episodes, alpha, eps_s=1.0, eps_decay=.99999, eps_min=0.05, gamma=1.0):
    nA = env.action_space.n
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    epsilon = eps_s
    # loop over episodes
    win_counter = 0
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.\n".format(i_episode, num_episodes), end="")
            print("win probability %d, %f" % (win_counter,(win_counter / 1000)))
            win_counter = 0
            sys.stdout.flush()
        epsilon = max(epsilon * eps_decay, eps_min)
        episode = generate_episode(env, Q, epsilon)
        if episode[-1][-1] == 1:
            win_counter += 1
        reward = calculate_G(list(map(lambda per: per[-1], episode)), gamma)
        for k in range(len(episode)):
            state, action, _ = episode[k]
            Q[state][action] += (reward[k] - Q[state][action]) * alpha
    policy = dict((k, np.argmax(v)) for k, v in Q.items())
    return policy, Q


def example1():
    Q = mc_prediction_q(env, 500000, generate_episode_from_limit_stochastic, 0.8)
    #obtain the corresponding state-value function
    V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) \
                      for k, v in Q.items())
    # plot the state-value function
    plot_blackjack_values(V_to_plot)


def example2():

    policy, Q = mc_control(env, 5000000, 0.02)
    # obtain the corresponding state-value function
    V = dict((k, np.max(v)) for k, v in Q.items())

    # plot the state-value function
    plot_blackjack_values(V)
    # plot the policy
    plot_policy(policy)

if __name__ == '__main__':
    example2()
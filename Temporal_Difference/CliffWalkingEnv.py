# coding:utf-8
import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from util import calculate_G
from Temporal_Difference import check_test
from Temporal_Difference.plot_utils import plot_values

env = gym.make('CliffWalking-v0')

# print(env.action_space)
# print(env.observation_space)

# V_opt = np.zeros((4, 12))
# V_opt[0][0:13] = -np.arange(3, 15)[::-1]
# V_opt[1][0:13] = -np.arange(3, 15)[::-1] + 1
# V_opt[2][0:13] = -np.arange(3, 15)[::-1] + 2
# V_opt[3][0] = -13
#
# plot_values(V_opt)

# V_opt = np.zeros((4, 12))
# V_opt[0][0:13] = np.arange(0, 12)
# V_opt[1][0:13] = np.arange(12, 24)
# V_opt[2][0:13] = np.arange(24, 36)
# V_opt[3][0:13] = np.arange(36, 48)
# plot_values(V_opt)


def choose_action(env_cw, Q, state, episode_i, eps=None):
    nA = env_cw.action_space.n
    if eps is None:
        eps = 1 / episode_i
    p_s = np.ones(nA) * eps / nA
    best_one = np.argmax(Q[state])
    p_s[best_one] = 1 - eps + (eps / nA)
    action = np.random.choice(np.array(nA), p=p_s)
    return action, p_s


def update_sarsa(Q_curr, Q_next, alpha, reward, gramma):
    return Q_curr + alpha * (reward + gramma * Q_next - Q_curr)


def generate_episode(env_cw, Q, alpha, update_fun, gramma, episode_i):
    state = env_cw.reset()
    action, _ = choose_action(env_cw, Q, state, episode_i)
    while True:
        next_state, reward, done, _ = env_cw.step(action)
        if not done:
            next_action, _ = choose_action(env_cw, Q, next_state, episode_i)
            Q[state][action] = update_fun(Q[state][action], Q[next_state][next_action], alpha, reward, gramma)
            state = next_state
            action = next_action
        if done:
            Q[state][action] = update_fun(Q[state][action], 0, alpha, reward, gramma)
            break
    return Q


def sarsa(env, num_episodes, alpha, gamma=1.0):
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    # initialize performance monitor
    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        Q = generate_episode(env, Q, alpha, update_sarsa, gamma, i_episode)
    return Q


def part1():
    # obtain the estimated optimal policy and corresponding action-value function
    Q_sarsa = sarsa(env, 5000, .01)

    # print the estimated optimal policy
    policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
    check_test.run_check('td_control_check', policy_sarsa)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_sarsa)

    # plot the estimated optimal state-value function
    V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
    plot_values(V_sarsa)


def update_q_learning(Q_curr, max_next_state, alpha, reward, gamma):
    return Q_curr + alpha * (reward + gamma * max_next_state - Q_curr)


def generate_episode_q_learning(env_cw, Q, alpha, gamma, episode_i):
    state = env_cw.reset()
    while True:
        action, _ = choose_action(env_cw, Q, state, episode_i)
        next_state, reward, done, _ = env_cw.step(action)
        if not done:
            Q[state][action] = update_q_learning(Q[state][action], Q[next_state][np.argmax(Q[next_state])], alpha, reward, gamma)
            state = next_state
        if done:
            Q[state][action] = update_q_learning(Q[state][action], 0, alpha, reward, gamma)
            break
    return Q


def q_learning(env, num_episodes, alpha, gamma=1.0):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))
    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        generate_episode_q_learning(env, Q, alpha, gamma, i_episode)
    return Q


def part2():
    # obtain the estimated optimal policy and corresponding action-value function
    Q_sarsamax = q_learning(env, 5000, .01)

    # print the estimated optimal policy
    policy_sarsamax = np.array([np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4,12))
    check_test.run_check('td_control_check', policy_sarsamax)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_sarsamax)

    # plot the estimated optimal state-value function
    plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])


def generate_episode_expected_sarsa(env_cw, Q, alpha, gamma):
    state = env_cw.reset()
    action, p_s = choose_action(env_cw, Q, state, 0, 0.005)
    while True:
        next_state, reward, done, _ = env_cw.step(action)
        if not done:
            next_action, p_s = choose_action(env_cw, Q, next_state, 0, 0.005)
            Q[state][action] = update_q_learning(Q[state][action], np.sum(p_s * Q[next_state]), alpha, reward, gamma)
            state = next_state
            action = next_action
            print(state)
        if done:
            Q[state][action] = update_q_learning(Q[state][action], 0, alpha, reward, gamma)
            break
    return Q


def expected_sarsa(env, num_episodes, alpha, gamma=1.0):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))
    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        generate_episode_expected_sarsa(env, Q, alpha, gamma)
    return Q


def part3():
    # obtain the estimated optimal policy and corresponding action-value function
    Q_expsarsa = expected_sarsa(env, 10000, 1)

    # print the estimated optimal policy
    policy_expsarsa = np.array([np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4,12)
    check_test.run_check('td_control_check', policy_expsarsa)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_expsarsa)

    # plot the estimated optimal state-value function
    plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])


if __name__ == '__main__':
    part3()
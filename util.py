# coding:utf-8


def calculate_G(rewards, gamma=1.0):
    b_reward = 0
    r_l = []
    rewards.reverse()
    for r in rewards:
        G = r + gamma * b_reward
        r_l.append(G)
        b_reward = G
    r_l.reverse()
    return r_l


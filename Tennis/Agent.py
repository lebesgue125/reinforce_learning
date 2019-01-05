from unityagents import UnityEnvironment
from network.DDPG import DDPG_AGENT
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

env = UnityEnvironment(file_name="../environment/Tennis_Windows_x86_64/Tennis.exe")
save_path = '../result/tennis/'

brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

count = 0
epsilon = 0
decay = 0.99
TAU = 1e-3
ALPHA = 4e-5
gamma = 0.99
batch_size = 64

mean_score = 0
scores_list = []
scores_total = []
max_memory_size = 50000
train = True    # change True to False to skip training process and load the saved weights.


def view(agent, num=1):
    count = 0
    for i in range(num):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        start_time = time.time()
        count += 1
        while True:
            actions = agent.choose_action(states, epsilon, num_agents)
            actions = np.clip(actions, -1, 1)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += rewards
            states = next_states
            if np.any(dones):
                break
        scores_list.append(scores)
        print("\rNo.{} score this episode: {:.4f},\tmean_scores: {:.4},\ttime: {:.4f}"
              .format(count, np.mean(scores), np.mean(scores_list), time.time()-start_time), end='')
        if count % 100 == 0:
            print("\rNo.{} score this episode: {:.4f}, ".format(count, np.mean(scores_list)))
            scores_list.clear()


with tf.Session() as session:
    agent = DDPG_AGENT(session, action_size, state_size, ALPHA, TAU, max_memory_size)
    saver = tf.train.Saver()
    saver.restore(session, save_path)
    while mean_score < 2 and train:
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        start_time = time.time()
        count += 1
        loss = 0
        while True:
            actions = agent.choose_action(states, epsilon, num_agents)
            actions = np.clip(actions, -1, 1)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += rewards
            agent.store(states, actions, rewards, next_states, dones)
            states = next_states
            if np.any(dones):
                break
            if agent.step % 4 == 0:
                loss += agent.learn(batch_size, gamma)
        if count > 300:
            epsilon = max(epsilon * decay, 0.001)
        scores_list.append(scores)
        print("\rNo.{} score this episode: {:.4f},\tloss: {:.4f},\tmean_scores: {:.4},\tepsilon: {:.4},\ttime: {:.4f}"
              .format(count, np.mean(scores), loss, np.mean(scores_list), epsilon, time.time()-start_time), end='')
        if count % 100 == 0:
            mean_score = np.mean(scores_list)
            if mean_score > 1.5:
                saver.save(session, save_path)
            scores_total.extend(scores_list)
            scores_list.clear()
            print("\rNo.{} score this episode: {:.4f}".format(count, mean_score))

    if train:
        saver.save(session, save_path)
        fig = plt.figure()
        plt.plot(range(len(scores_total)), scores_total)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
        view(agent, 10)
    else:
        saver.restore(session, save_path)
        view(agent, 10)

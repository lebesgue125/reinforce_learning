from unityagents import UnityEnvironment
import numpy as np
from network.DDPG_NEW import DDPG_AGENT
import tensorflow as tf
import matplotlib.pyplot as plt
import time


env = UnityEnvironment(file_name="../environment/Reacher_one_agent/Reacher.exe")
save_path = "../result/reacher-single/"
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


env_info = env.reset(train_mode=False)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
state = env_info.vector_observations
state_size = state.shape[1]

max_memory_size = 50000
ALPHA = 5e-4
TAU = 1e-3
gamma = 0.99
epsilon = 1
decay = 0.999
mean_scores = 0
batch_size = 64
scores_list = []
scores_total = []
count = 0
train = True

with tf.Session() as session:
    agent = DDPG_AGENT(session, action_size, state_size, ALPHA, TAU, max_memory_size)
    saver = tf.train.Saver()
    while mean_scores < 30 and train:
        dones = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        count += 1
        rewards = np.zeros(num_agents, dtype=np.float32)
        start_time = time.time()
        loss = 0
        while not np.any(dones):
            actions = agent.choose_action(states, epsilon, num_agents)
            actions = np.clip(actions, -1, 1)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            reward = env_info.rewards
            rewards += reward
            dones = env_info.local_done
            agent.store(states, actions, reward, next_states, dones)
            states = next_states
            if agent.step % 4 == 0:
                loss += agent.learn(batch_size, gamma)
        epsilon = max(epsilon * decay, 0.001)
        scores_list.append(rewards)
        print("\rNo.{} score this episode: {:.4f},\tloss: {:.4f},\tmean_scores: {:.4},\tepsilon: {:.4},\ttime: {:.4f}"
              .format(count, rewards[0], loss / 250.0, np.mean(scores_list), epsilon, time.time()-start_time), end='')
        if count % 100 == 0:
            mean_scores = np.mean(scores_list)
            print("\rNo.{} score this episode: {:.4f}, ".format(count, mean_scores))
            scores_total.extend(scores_list)
            scores_list.clear()

    if train:
        saver.save(session, save_path)
        fig = plt.figure()
        plt.plot(range(len(scores_total)), scores_total)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
    else:
        saver.restore(session, save_path)

    for i in range(10):
        dones = np.zeros(num_agents)
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        count += 1
        rewards = np.zeros(num_agents, dtype=np.float32)
        reward = np.zeros(num_agents, dtype=np.float32)
        start_time = time.time()
        while not np.any(dones):
            actions = agent.choose_action(states, 0, num_agents, False)
            actions = np.clip(actions, -1, 1)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            reward = env_info.rewards
            rewards += reward
            dones = env_info.local_done
            states = next_states
        scores_list.append(rewards)
        print("\rNo.{} score this episode: {:.4f},\tmean_scores: {:.4},\ttime: {:.4f}"
              .format(count, rewards[0], np.mean(scores_list), time.time()-start_time), end='')
        if count % 100 == 0:
            print("\rNo.{} score this episode: {:.4f}, ".format(count, np.mean(scores_list)))
            scores_total.extend(scores_list)
            scores_list.clear()

env.close()

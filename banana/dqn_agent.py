# coding:utf-8
from unityagents import UnityEnvironment
import numpy as np
from network.DQN import DQNAgent
import matplotlib.pyplot as plt
import tensorflow as tf
import time

env = UnityEnvironment(file_name="../environment/Banana_Windows_x86_64/Banana.exe")
path = "../result/banana/"
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=False)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

total_scores = []
scores = []
batch_size = 64
mean = 0
count = 0
eps = 1.0
eps_end = 0.01
decay = 0.999
max_t = 1000
gamma = 0.99
alpha = 1e-4
tua = 1e-3
max_memory_size = 50000
train = False

with tf.Session() as session:
    brain_agent = DQNAgent(session, state_size, action_size, max_memory_size, gamma, alpha, tua)
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    while mean < 13 and train:
        env_info = env.reset(train_mode=True)[brain_name]
        score = 0
        time_b = time.time()
        loss = 0
        for i in range(max_t):
            if np.random.random() > eps:
                action = np.argmax(brain_agent.choose_action(state), axis=1)
            else:
                action = np.random.choice(action_size)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            brain_agent.store(state, action, reward, next_state, [done])
            state = next_state
            if brain_agent.step % 4 == 0:
                loss += brain_agent.learn(batch_size)
            if done:
                break
        scores.append(score)
        total_scores.append(score)
        eps = max(eps * decay, eps_end)
        print("\rEpisode: {},\tCurr Score: {},\tAverage Score: {:.2f},\tLoss:{:.4},\tEPS:{:.4},\tTime: {:.4}".format(count, score, np.mean(scores), loss/250.0, eps, time.time()-time_b), end="")
        if count % 100 == 0 and count > 0:
            mean = np.mean(scores)
            print("\rEpisode: {}, \tAverage Score: {:.2f}".format(count, mean))
            scores.clear()
        count += 1

    if train:
        saver.save(session, path)
        fig = plt.figure()
        plt.plot(range(len(total_scores)), total_scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
    else:
        saver.restore(session, path)

    saver.restore(session, path)
    for _ in range(10):
        done = False
        env_info = env.reset(train_mode=False)[brain_name]
        score = 0
        state = env_info.vector_observations[0]
        while not done:

            action = brain_agent.action = np.argmax(brain_agent.choose_action(state), axis=1)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            state = next_state
        print("Score is ", score)

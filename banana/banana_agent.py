# coding:utf-8
from unityagents import UnityEnvironment
import numpy as np
from network.DQN_PyTorch import Agent
import matplotlib.pyplot as plt
import time

env = UnityEnvironment(file_name="../environment/Banana_Windows_x86_64/Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=False)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
# print('States look like:', state)
state_size = len(state)
# print('States have length:', state_size)

batch_size = 64
eps = 1.0
eps_end = 0.01
decay = 0.999
max_memory_size = 100000
gamma = 0.99
alpha = 5e-4
tau = 1e-3
update_every = 4

train = False
save_path = '../result/banana_pytorch/'
mean = 0
count = 0
total_scores = []
scores = []
max_t = 1000

brain_agent = Agent(action_size, state_size, gamma, alpha, max_memory_size, tau)
while mean < 13 and train:
    env_info = env.reset(train_mode=True)[brain_name]
    score = 0
    state = env_info.vector_observations[0]
    time_b = time.time()
    for i in range(max_t):
        action = brain_agent.choose_action(state, eps)  # select an action
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        brain_agent.store(state, action, reward, next_state, [done])
        state = next_state  # roll over the state to next time step
        if brain_agent.steps % update_every == 0:
            brain_agent.learn(batch_size)
        if done:
            break
    scores.append(score)
    total_scores.append(score)
    eps = max(eps * decay, eps_end)
    print("\rEpisode: {}, \tCurr Score: {}, \tAverage Score: {:.4f}, \tEPS:{:.4f}, \tTime: {}".format(count, score, np.mean(scores), eps, time.time()-time_b), end="")
    if count % 100 == 0 and count > 0:
        mean = np.mean(scores)
        brain_agent.save(save_path)
        print("\rEpisode: {}, \tAverage Score: {:.4f}".format(count, mean))
        scores.clear()
    count += 1


def show_game(episode_num):
    for _ in range(episode_num):
        done = False
        env_info = env.reset(train_mode=False)[brain_name]
        score = 0
        state = env_info.vector_observations[0]
        while not done:
            action = brain_agent.choose_action(state, 0)
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            score += reward
            state = next_state
        print("Score is ", score)

if train:
    brain_agent.save(save_path)
    fig = plt.figure()
    plt.plot(range(len(total_scores)), total_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(save_path + 'banana.png')
    plt.show()
    show_game(10)
else:
    brain_agent.load(save_path)
    show_game(10)




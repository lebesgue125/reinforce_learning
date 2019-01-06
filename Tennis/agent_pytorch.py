from unityagents import UnityEnvironment
from network.DDPG_PyTorch import DDPGAGENT
import numpy as np
import time
import matplotlib.pyplot as plt

env = UnityEnvironment(file_name="../environment/Tennis_Windows_x86_64/Tennis.exe")


brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

count = 0
epsilon = 1.0
decay = 0.9999
actor_alpha = 1e-4
critic_alpha = 1e-4
tau = 1e-3
gamma = 0.99
batch_size = 64

mean_score = 0
scores_list = []
scores_total = []
max_memory_size = 50000
train = False    # change True to False to skip training process and load the saved weights.
save_path = '../result/tennis_pytorch/'


def view(agent, num=1):
    count = 0
    for i in range(num):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        start_time = time.time()
        count += 1
        while True:
            actions = agent.choose_action(states, 0, num_agents)
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
            print("\rEpisode {} Average Score: {:.4f}".format(count, np.mean(scores_list)))
            scores_list.clear()


agent = DDPGAGENT(action_size, state_size, actor_alpha, critic_alpha, tau, max_memory_size)
while mean_score < 2 and train:
    env_info = env.reset(train_mode=True)[brain_name]
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
        agent.store(states, actions, rewards, next_states, dones)
        states = next_states
        if np.any(dones):
            break
        if agent.step % 4 == 0:
            agent.learn(batch_size, gamma)
    if count > 300:
        epsilon = max(epsilon * decay, 0.001)
    scores_list.append(scores)
    print("\rNo.{} score this episode: {:.4f},\tmean_scores: {:.4},\tepsilon: {:.4},\ttime: {:.4f}"
          .format(count, np.mean(scores), np.mean(scores_list), epsilon, time.time()-start_time), end='')
    if count % 100 == 0:
        mean_score = np.mean(scores_list)
        if mean_score > 1.0:
            agent.save(save_path)
        scores_total.extend(scores_list)
        scores_list.clear()
        print("\rEpisode {} Average Score: {:.4f}".format(count, mean_score))

if train:
    agent.save(save_path)
    fig = plt.figure()
    plt.plot(range(len(scores_total)), scores_total)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(save_path + 'tennis.png')
    plt.show()
    view(agent, 10)
else:
    agent.load(save_path)
    view(agent, 10)

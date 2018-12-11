# coding:utf-8

from unityagents import UnityEnvironment
import numpy as np
from banana.model import QNetwork, Agent

env = UnityEnvironment(file_name="E:\MyWork\gitpython\deep-reinforcement-learning\p1_navigation\Banana_Windows_x86_64\Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=False)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


brain_agent = Agent(action_size, state_size, maxMemorySize=10000, gamma=0.95, epsilon=1.0, alpha=0.03, repalce=None)

while brain_agent.memCntr < brain_agent.memSize:
    env_info = env.reset(train_mode=True)[brain_name]
    done = False
    count = 0
    score = 0
    while not done:
        action = np.random.choice(action_size, p=[0.4, 0.2, 0.2, 0.2])  # select an action
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        count += 1
        if done and score < 13:
            reward -= 100
        if reward == 0:
            reward -= 0.5
        brain_agent.storeTransition(state, action, reward, next_state, done)
        state = next_state  # roll over the state to next time step

print('done the initializing memory')

scores = []
epsHistory = []
numGames = 2000
batch_size = 200

for i in range(numGames):
    print('starting game ' + str(i+1), 'epsilon: %.4f' % brain_agent.EPSILON)
    epsHistory.append(brain_agent.EPSILON)
    done = False
    env_info = env.reset(train_mode=True)[brain_name]
    score = 0
    state = env_info.vector_observations[0]

    while not done:

        action = brain_agent.chooseAction(state)
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward
        if done and score < 13:
            reward -= 100
        if reward == 0:
            reward -= 0.5
        brain_agent.storeTransition(state, action, reward, next_state, done)
        state = next_state
    brain_agent.learn(batch_size)
    scores.append(score)
    print("Score: {}".format(score))
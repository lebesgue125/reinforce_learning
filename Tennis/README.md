# Collaboration and Competition
## 1. Introduction
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.
## 2. Solution
We use DDPG to train the two agent simultaneously.
the code of DDPG and its agent is in 'network/DDPG.py' 
## 3. How to execute.
You can execute the Agent.py.
Change the parameter "train" in Agent.py to False to skip the training process and load the available weight in 
result/tennis directory.
## 3. Experiment Report
1. This is the episode and average score during the running time.
```text
No.100 score this episode: -0.0020, 
...... 
No.7000 score this episode: 0.0359, 
No.7100 score this episode: 0.0245, 
No.7200 score this episode: 0.0165, 
No.7300 score this episode: 0.0130, 
No.7400 score this episode: 0.0325, 
No.7500 score this episode: 0.0240, 
No.7600 score this episode: 0.0240, 
No.7700 score this episode: 0.0280, 
No.7800 score this episode: 0.0210, 
No.7900 score this episode: 0.0270, 
No.8000 score this episode: 0.0325, 
No.8100 score this episode: 0.0410, 
No.8200 score this episode: 0.0440, 
No.8300 score this episode: 0.0385, 
No.8400 score this episode: 0.0360, 
No.8500 score this episode: 0.0345, 
No.8600 score this episode: 0.0405, 
No.8700 score this episode: 0.0460, 
No.8800 score this episode: 0.0355, 
No.8900 score this episode: 0.0420, 
No.9000 score this episode: 0.0475, 
No.9100 score this episode: 0.0435, 
No.9200 score this episode: 0.0380, 
No.9300 score this episode: 0.0485, 
No.9400 score this episode: 0.0530, 
No.9500 score this episode: 0.0565, 
No.9600 score this episode: 0.0610, 
No.9700 score this episode: 0.0505, 
No.9800 score this episode: 0.0465, 
No.9900 score this episode: 0.0560, 
No.10000 score this episode: 0.0490, 
No.10100 score this episode: 0.0575, 
No.10200 score this episode: 0.0550, 
No.10300 score this episode: 0.0480, 
No.10400 score this episode: 0.0565, 
No.10500 score this episode: 0.0620, 
No.10600 score this episode: 0.0590, 
No.10700 score this episode: 0.0645, 
No.10800 score this episode: 0.0870, 
No.10900 score this episode: 0.0965, 
No.11000 score this episode: 0.0980, 
No.11100 score this episode: 0.1010, 
No.11200 score this episode: 0.1290, 
No.11300 score this episode: 0.1270, 
No.11400 score this episode: 0.0975, 
No.11500 score this episode: 0.1490, 
No.11600 score this episode: 0.1250, 
No.11700 score this episode: 0.0985, 
No.11800 score this episode: 0.1450, 
No.11900 score this episode: 0.2020, 
No.12000 score this episode: 0.1435, 
No.12100 score this episode: 0.1630, 
No.12200 score this episode: 0.1795, 
No.12300 score this episode: 0.2590, 
No.12400 score this episode: 0.2485, 
No.12500 score this episode: 0.2015, 
No.12600 score this episode: 0.2541, 
No.12700 score this episode: 0.3976, 
No.12800 score this episode: 1.1521, 
```
>2. This is the picture of these scores that generate during the training process.
>![multiple-reacher](https://i.ibb.co/gDdk3Kj/Tennis.png)


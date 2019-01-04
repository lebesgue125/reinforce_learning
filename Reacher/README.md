# Continuous Control
## Introduction:
### One Double-jointed Arm
>#### 1. Problem:
>You need to train a double-jointed arm to move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
>#### 2. Solution:
>We adopt the DDPG algorithm to solve this problem. 
>#### 3. How to Execute.
>You can execute the one_reacher.py.
>Change the parameter 'train' to False to skip the training process and load the available weight in  
>result/reacher-single directory.
>#### 3. Experiment Report
>1. This is the episode and average score during the running time.
>```text
>No.100 score this episode: 0.9096, 
>No.200 score this episode: 1.0587, 
>No.300 score this episode: 2.5118, 
>No.400 score this episode: 5.5385, 
>No.500 score this episode: 19.0347, 
>No.600 score this episode: 29.2395, 
>No.700 score this episode: 28.5232, 
>No.800 score this episode: 30.1664,
>```
>2. This is the picture of these scores that generate during the training process.
>![reacher-single](https://i.ibb.co/DMBzBMz/reacher-single.png)

### Multiple Double-jointed Arm
>### 1. Problem
>You need to train twenty double-jointed arm to move to target locations simultaneously. The reward is the same with One double-jointed arm . 
>### 2. Solution:
>We use DDPG algorithm to train these arms.
>#### 3. How to execute.
>You can execute the multiple_reacher.py.
>Change the parameter 'train' to False to skip the training process and load the available weight in 
>result/reacher_multiple directory.
>#### 4. Experiment Report
>1. This is the episode and average score during the running time.
>```text
>No.100 score this episode: 1.0235, 
>No.200 score this episode: 2.6423, 
>No.300 score this episode: 4.6524, 
>No.400 score this episode: 22.5428, 
>No.500 score this episode: 35.1221,
>```
>2. This is the picture of these scores that generate during the training process.
>![multiple-reacher](https://i.ibb.co/ZWgrcD8/multiple-reacher.png)

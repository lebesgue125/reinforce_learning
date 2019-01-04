# Navigation
## Introduction:
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

> - **0** move forward.
> - **1** move backward.
> - **2** turn left.
> - **3** turn right.

In this project, you need to control a agent to collect bananas. When your agent collect a yellow banana, you 
get +1 reward. Or get -1 reward when you collect a blue one. So we need to train a agent to get more than 13 rewards in 
one episode. 

we use dqn to solve this problem and mse for loss function.
We use PyTorch for neural network frame.
The code of the network in network/DQN_PyTorch.py

## Getting Started
Running environment.
#### 1. Python
> The python version is 3.6
#### 2. UnityEnvironment
>```text
>pip3 install UnityEnvironment
>```
#### 3. GPU-Environment:
> 
>CUDA 9.0
>
#### 4. Pytorch-GPU
>torch-1.0.0-cp36-cp36m-win_amd64 
>```text
>pip3 install https://download.pytorch.org/whl/cu90/torch-1.0.0-cp36-cp36m-win_amd64.whl
>pip3 install torchvision
>```

## How to execute.
- You can execute the banana_agent.py file to train the model.
- You can change the parameter 'train' in banana_agent.py file to False to skip the training process and load the available weight in 
result/banana_pytorch/ directory.

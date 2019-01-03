#Navigation
##Introduction:
In this project, you need to controll a agent to collect bananas. When your agent collect a yellow banana, you 
get +1 reward. Or get -1 reward when you collect a blue one. So we need to train a agent to get more than 13 rewards in 
one episode. 

In my project, you can run in dqn_agent.py or Navigation.ipynb. 
we use dqn to solve this problem and mse for loss function.
We use tensorflow-gpu for neuarl network frame.

## How to execute.
You can execute the dqn_agent.py file or Navigation.ipynb.
You can change the parameter 'train' to False to ship the training process and load the available weight in 
result/banana directory.
##Experiment Report    
1. This is the episode and average score during the running time.
```text
    Episode: 100,   Average Score: -0.11
    Episode: 200,   Average Score: 1.27
    Episode: 300, 	Average Score: 2.63
    Episode: 400, 	Average Score: 3.83
    Episode: 500, 	Average Score: 4.58
    Episode: 600, 	Average Score: 5.54
    Episode: 700, 	Average Score: 6.29
    Episode: 800, 	Average Score: 7.78
    Episode: 900, 	Average Score: 8.59
    Episode: 1000, 	Average Score: 8.85
    Episode: 1100, 	Average Score: 9.32
    Episode: 1200, 	Average Score: 10.42
    Episode: 1300, 	Average Score: 9.91
    Episode: 1400, 	Average Score: 10.41
    Episode: 1500, 	Average Score: 10.46
    Episode: 1600, 	Average Score: 10.92
    Episode: 1700, 	Average Score: 11.60
    Episode: 1800, 	Average Score: 11.83
    Episode: 1900, 	Average Score: 12.00
    Episode: 2000, 	Average Score: 11.85
    Episode: 2100, 	Average Score: 12.03
    Episode: 2200, 	Average Score: 12.60
    Episode: 2300, 	Average Score: 12.42
    Episode: 2400, 	Average Score: 12.23
    Episode: 2500, 	Average Score: 12.96
    Episode: 2600, 	Average Score: 13.39
```  
2. This is the picture of these scores that generate during the training process.
![banana](https://i.ibb.co/qx48qrm/banana.png)

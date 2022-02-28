# Bananavigation
A deep reinforcement learning implementation to collect bananas in an ML-Agents Unity environment, built for the Udacity nanodegree program.

# Project 1: Navigation
![Collecting bananas GIF](CollectingBananas.gif)

### Introduction

This project trains an agent to navigate and collect bananas in a large, square world set up within the ML-Agents Unity environment.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal is to have the agent collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and the environment is considered solved when the agent gets an average score of +13 over 100 consecutive episodes per project requirements, although +16 was used here to stress the agent slightly more as this is still inclusive of the required +13.

### Getting Started

1. Download the OS-appropriate environment from one of the links below:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Clone Udacity's Deep Reinforcment Learning Nanodegree repository and place the file in this pository, in the `p1_navigation/` folder, and unzip (or decompress) the file. The repository is available at [https://github.com/udacity/deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning).

### Instructions

Open the `Navigation.ipynb` Jupyter notebook and follow the instructions in the cells to train and test the agent.

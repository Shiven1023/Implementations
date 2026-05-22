#DQN on Vanilla Policy Gradient(REINFORCE)
import numpy as np 
import matplotlib.pyplot as plt
import gymnasium as gym
import random 
import math
import torch
import torch.nn as nn
from collections import deque
import torch.optim as optim
"""
PSEUDOCODE: 1) Env setup 
2)Network built for producing policy pi(a|s) --> takes in state features outputs probability distribution over actions
3) exploration --> perform full trajectories based on sampling from the policy of the newtork
4)for episode in range(n_episodes)-->while(not done):
    env.step() --> store reward attained and log of prob of the action sampled

5) after finishing trajectory we compute loss as - torch.sum(log_prob()*returns) (Here we dont want the returns to be paert of the computational graph and have
 gradients produced with respect to it so use .detach())
 Also we must normalise the returns array before using for loss calculation
 6) optimizer.zero_grad()
    loss.backward()
    optimizer.step()

"""

env = gym.make("CartPole-v1")
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]
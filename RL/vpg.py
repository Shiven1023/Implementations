#Vanilla Policy Gradient(REINFORCE) on Cartpole  
import numpy as np 
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
"""
This differs from Q-learning methods in the fact that we learn the policy function directly rather than optimisign for a value function
hence we dont need e-greedy since sampling from the policy already involves stochasticity and we perform training after each episode rather than during it (like in q-learning)
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


Hyperparamters: 1) No. of episodes
2) learning rate 3) discount factor 4)choosing the optimiser(Adam most likely) 5) epsilon(to prevent the normalised return from have division by zero)

"""

env = gym.make("CartPole-v1")
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_states,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,n_actions), #we arent applying softmax here instead we use dist.Categorical
        )
    def forward(self,x):
        out = self.network(x)
        return out


gamma= 0.99
epsilon = 1e-8
episodes = 4000
done = False
policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(),lr = 1e-3)
state,_=env.reset(seed = 42)
episode_rewards = []
for i in range(episodes):
    state,_=env.reset()
    done = False
    log_probs = []
    rewards = []
    
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0) #converting the numpy array that env gives us and converting into tensor of shape(1,4)
        logits = policy(state_tensor)
        distribution = dist.Categorical(logits=logits)
        action = distribution.sample() 
        log_prob = distribution.log_prob(action)
        log_probs.append(log_prob)
        next_obs,reward,terminated,truncated,info = env.step(action.item())
        rewards.append(reward)
        state = next_obs
        if terminated or truncated:
            done = True
    discounted_returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        discounted_returns.insert(0, G) 
    discounted_returns = np.array(discounted_returns)
    normalized_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + epsilon)
    returns = torch.FloatTensor(normalized_returns)
    episode_rewards.append(sum(rewards))
    if i % 50 == 0:
        print(f"Episode {i} | Avg Reward (last 50): {np.mean(episode_rewards[-50:]):.1f}")
    log_probs_tensor = torch.stack(log_probs)
    loss = -torch.sum(log_probs_tensor * returns)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


         


                
    
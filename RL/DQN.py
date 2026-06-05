#DQN on CartPole
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
PSEUDOCODE:
Make two networks; target(provides y value(R+gamma*Q(s',a')) and online network (gives current q value) to update (through backprop of online netwrok)
Make Replay buffer which stores lists of (S,A,R,gamma,S') to train online network on after collection
--> Define hyperparameters
--> Build replay buffer via multiple episodes(similar to the way in Q-learning leaving out the update equation)

--> Sample from episodes and train online network on those samples
--> Q(s,a) is computed using online network Q(s',a') through target network
"""
# 1. Network class 
# 2. ReplayBuffer class (stores transitions, has sample() method)
# 3. Hyperparameters
# 4. Training loop:
#    - step, store transition
#    - if buffer large enough: sample batch, compute loss, backprop
#    - every N steps: sync target network
#    - epsilon decay

env = gym.make("CartPole-v1")
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_states,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,n_actions),
        )
    def forward(self,x):
        out = self.network(x)
        return out
    
class ReplayBuffer():
    def __init__(self,mxlen):
        self.mxlen = mxlen
        self.buffer = deque(maxlen = self.mxlen)

    def append(self,replay:list) -> None:
        self.buffer.append(replay)

    def sample(self,batch_size):
        return random.sample(self.buffer,batch_size)
"""
batch_size --> number of samples taken from buffer during training
mxlen --> max buffer size
min_size --> minimum size of buffer required to start sampling
"""
#Hyperparameters
gamma = 0.95
episodes = 2000
epsilon = 1
eps_decay = 0.995
batch_size = 64 
min_size = 1500 #min buffer size to start sampling
#Models
OnlineNetwork = DQN()
TargetNetwork = DQN()
TargetNetwork.load_state_dict(OnlineNetwork.state_dict())
optimizer = optim.Adam(OnlineNetwork.parameters(),lr = 0.01)
RB = ReplayBuffer(mxlen = 10000)

done = False
for i in range(episodes):
    state,_ = env.reset()
    done = False
    if i!=0:
        epsilon *= eps_decay
    while(not done):

        if(np.random.rand()<epsilon):
            action = env.action_space.sample()
        else:
            action = np.argmax(OnlineNetwork.forward(state))
        next_obs,reward,terminated,truncated,info = env.step(action)
        if(terminated or truncated):
            done = True

        RB.append([state,action,reward,next_obs,done])
        # condition for sampling
        if(len(RB)>= min_size):
            train  = RB.sample(batch_size) #64 rows of 5 columns
            train = np.array(train)
            for  j in range(train.shape[1]):
                col = train[:,j]
                if j == 0:
                    state_tensor = torch.tensor(col,dtype = torch.float)
                elif j ==1:
                    action_tensor = torch.tensor(col,dtype = torch.long)
                elif j ==2:
                    reward_tensor = torch.tensor(col,dtype = torch.float)
                elif j ==3:
                    next_state_tensor = torch.tensor(col,dtype = torch.float)
                else:
                    done_tensor = torch.tensor(1-col,dtype = torch.long)



                




        state = next_obs
        


import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
"""
PSEUDOCODE:
initialise q table--> ZEROS(dimesion is (states,actions))
define hyperparaameters--> epsilon decay, number of episode, discount factor, learning rate
for i in episode: 
    env.reset
    while(not done):
        sample action(e-greedy)
        take action
        update q value (TD equation)

"""
env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n
q_table = np.zeros((n_states,n_actions))
gamma= 0.95 #discount
lr = 0.1
epsilon = 1
episodes = 50000
eps_decay= 0.9999
done = False
wins =0

for i in range(episodes):

    state,_=env.reset(seed = 42)
    done  = False
    while(not done):
        if(np.random.rand()<epsilon):
            action = env.action_space.sample()
        else:
            action = np.argmax((q_table[state]))
            
        next_obs,reward,terminated,truncated,info = env.step(action)
        if terminated:
            q_table[state][action]+= lr*(reward-q_table[state][action])
        else:
            q_table[state][action]+= lr*(reward+gamma*(np.max(q_table[next_obs]))-q_table[state][action])

        if terminated or truncated:
            if reward:
                #print(f"The {i}th episode was a success")
                wins +=1

            #else:
                #print(f"The {i}th episode was a failure")
            done = True
            epsilon = max(0.01, epsilon * eps_decay)
        state = next_obs
eval_wins = 0
eval_episodes = 1000
for i in range(eval_episodes):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state])  # pure greedy, no exploration
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_obs
    if reward == 1:
        eval_wins += 1
print(f"Eval win:{eval_wins/eval_episodes*100}")

q_grid = np.max(q_table, axis=1).reshape(4,4)
plt.imshow(q_grid, cmap='coolwarm')
plt.colorbar()
plt.title("Q-table heatmap")
plt.show()



        
    


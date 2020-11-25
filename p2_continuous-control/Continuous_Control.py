#!/usr/bin/env python
# coding: utf-8

# # Continuous Control
# 
# ---
# 
# You are welcome to use this coding environment to train your agent for the project.  Follow the instructions below to get started!
# 
# ### 1. Start the Environment
# 
# Run the next code cell to install a few packages.  This line will take a few minutes to run!

# In[1]:


get_ipython().system('pip -q install ./python')


# The environments corresponding to both versions of the environment are already saved in the Workspace and can be accessed at the file paths provided below.  
# 
# Please select one of the two options below for loading the environment.

# In[2]:


from unityagents import UnityEnvironment
import numpy as np
import workspace_utils
from workspace_utils import active_session

# select this option to load version 1 (with a single agent) of the environment
# env = UnityEnvironment(file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')

# select this option to load version 2 (with 20 agents) of the environment
env = UnityEnvironment(file_name='/data/Reacher_Linux_NoVis/Reacher.x86_64')


# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[3]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# ### 2. Examine the State and Action Spaces
# 
# Run the code cell below to print some information about the environment.

# In[4]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


# ### 3. Take Random Actions in the Environment
# 
# In the next code cell, you will learn how to use the Python API to control the agent and receive feedback from the environment.
# 
# Note that **in this coding environment, you will not be able to watch the agents while they are training**, and you should set `train_mode=True` to restart the environment.

# In[ ]:


env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
    actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


# When finished, you can close the environment.

# In[ ]:


env.close()


# ### 4. It's Your Turn!
# 
# Now it's your turn to train your own agent to solve the environment!  A few **important notes**:
# - When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
# ```python
# env_info = env.reset(train_mode=True)[brain_name]
# ```
# - To structure your work, you're welcome to work directly in this Jupyter notebook, or you might like to start over with a new file!  You can see the list of files in the workspace by clicking on **_Jupyter_** in the top left corner of the notebook.
# - In this coding environment, you will not be able to watch the agents while they are training.  However, **_after training the agents_**, you can download the saved model weights to watch the agents on your own machine! 

# In[5]:


env_info = env.reset(train_mode=True)[brain_name]


# In[6]:


import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from agent import Agent


# In[7]:


load_agent = False

def load_saved_agent(local_actor_pth, local_critic_pth):
    local_actor = torch.load(local_actor_pth)
    local_critic = torch.load(local_critic_pth)
    
    loaded_agent = Agent(action_size, state_size, num_agents, random_seed=0)
    
    loaded_agent.local_actor.load_state_dict(local_actor)
    loaded_agent.local_critic.load_state_dict(local_critic)
    
    return loaded_agent

if load_agent:
    agent = load_saved_agent('checkpoint_local_actor.pth', 'checkpoint_local_critic.pth')
    print('Agent loaded')
else:
    agent = Agent(state_size, action_size, num_agents, random_seed=0)
    print('Agent created')


# In[25]:


agent.local_actor.load_state_dict(torch.load('checkpoint_local_actor.pth'))
agent.local_critic.load_state_dict(torch.load('checkpoint_local_critic.pth'))


# In[8]:


def ddpg_train(agent, num_agents, n_episodes=10000, max_t=1000, print_every=100):
    scores = []
    scores_window = deque(maxlen=100)
    scores_mean = []
    n_episodes = 1000

    for episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]            # reset the environment
        states = env_info.vector_observations
        agent.reset()                                                # reset the agent noise
        score = np.zeros(num_agents)
        
        for t in range(max_t):
            actions = agent.act(states)
        
            env_info = env.step( actions )[brain_name]               # send the action to the environment                            
            next_states = env_info.vector_observations               # get the next state        
            rewards = env_info.rewards                               # get the reward        
            dones = env_info.local_done                              # see if episode has finished        

            agent.step(states, actions, rewards, next_states, dones)

            score += rewards                                         # update the score
        
            states = next_states                                     # roll over the state to next time step        
                                                        
            if np.any( dones ):                                          # exit loop if episode finished        
                 break                                        

        scores.append(np.mean(score))
        scores_window.append(np.mean(score))
        scores_mean.append(np.mean(scores_window))

        print('\rEpisode: \t{} \tScore: \t{:.2f} \tAverage Score: \t{:.2f}'.format(episode, np.mean(score), np.mean(scores_window)), end="")
        
        if episode % print_every == 0:            
            torch.save(agent.local_actor.state_dict(), 'checkpoint_local_actor.pth')           # save local actor
            torch.save(agent.local_critic.state_dict(), 'checkpoint_local_critic.pth')         # save local critic
        
        if np.mean(scores_window) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
            torch.save(agent.local_actor.state_dict(), 'checkpoint_local_actor.pth')           # save local actor
            torch.save(agent.local_critic.state_dict(), 'checkpoint_local_critic.pth')         # save local critic
            break    

    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.plot(np.arange(1, len(scores_mean)+1), scores_mean)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(('Score', 'Mean'), fontsize='xx-large')
    plt.show()

with active_session():
    ddpg_train(agent, num_agents)


# In[ ]:





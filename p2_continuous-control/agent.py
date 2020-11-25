"""Contains all code related to the Agent."""
from typing import List

import copy
import numpy as np
import random
from collections import namedtuple, deque

from numpy.core.defchararray import add

from models import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 2        # update target networks every two gradient ascent steps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
  """Interacts and learns from the environment."""
  def __init__(self,
               state_size: int,
               action_size: int,
               n_agents: int=1,
               random_seed: int=123) -> None:
    """Initialize agent object."""
    self.n_agents = n_agents
    self.action_size = action_size
    self.state_size = state_size
    self.seed = random.seed(random_seed)

    self.local_actor = Actor(
        action_size,
        state_size,
        random_seed,
    ).to(device)
    self.target_actor = Actor(
        action_size,
        state_size,
        random_seed,
    ).to(device)
    self.actor_optimizer = optim.Adam(
        self.local_actor.parameters(), lr=LR_ACTOR)

    self.local_critic = Critic(
        action_size,
        state_size,
        random_seed,
    ).to(device)
    self.target_critic = Critic(
        action_size,
        state_size,
        random_seed,
    ).to(device)
    self.critic_optimizer = optim.Adam(
        self.local_critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

    self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, n_agents, random_seed)

    # noise process
    self.noise = OUNoise((n_agents, action_size), random_seed)

  def step(self, state, action, reward, next_state, done: bool) -> None:
    """Taking a step within the environment."""
    for i in range(self.n_agents):
      self.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])

    # Learn if enough samples in memory buffer
    if len(self.memory) > BATCH_SIZE:
      experiences = self.memory.sample()
      self.learn(experiences)


  def act(self, state, add_noise: bool=True) -> np.array:
    """Returns actions based on the current given state."""
    state = torch.from_numpy(state).float().to(device)

    self.local_actor.eval()  # set network to eval mode.
    with torch.no_grad():
      action_values = self.local_actor(state).cpu().data.numpy()

    self.local_actor.train()
    if add_noise:
      action_values += self.noise.sample()

    return np.clip(action_values, -1, 1)

  def learn(self, experience, gamma: float=GAMMA) -> None:
    """Update value params given experience tuples using gradient ascent."""
    states, actions, rewards, next_states, dones = experience

    # compute critic loss
    actions_next = self.target_actor(next_states)
    q_targets_next = self.target_critic(next_states, actions_next)
    q_targets = rewards + gamma * q_targets_next * (1- dones)
    q_expected = self.local_critic(states, actions)
    critic_loss = F.mse_loss(q_expected, q_targets)

    # minimize loss
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    # gradient ascent for actor
    actions_pred = self.local_actor(states)
    actor_loss = -self.local_critic(states, actions_pred).mean()

    # minimize loss
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    # soft updates for target networks
    self.soft_update(self.local_critic, self.target_critic, TAU)
    self.soft_update(self.local_actor, self.target_actor, TAU)

  def soft_update(self, local_model, target_model, tau: float) -> None:
    """Soft updates to model parameters."""
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

  def reset(self) -> None:
    """Reset the noise for the agent."""
    self.noise.reset()


class OUNoise:
  """Ornstein-Uhlenbeck noise process to be added to the actions."""
  def __init__(self, size: int, seed: int, mu:float=0., theta:float=0.15, sigma:float=0.2) -> None:
    """Initialize noise parameters."""
    self.size = size
    self.mu = mu * np.ones(size)
    self.theta = theta
    self.sigma = sigma
    self.seed = random.seed(seed)
    self.reset()

  def reset(self) -> None:
    """Reset noise state to mean."""
    self.state = copy.copy(self.mu)

  def sample(self) -> List:
    """Update internal state and return is as a noise sample."""
    x = self.state
    dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
    self.state = x + dx
    return self.state


class ReplayBuffer:
  """Fixed sized buffer (array) which stores experience tuples of the agent."""
  def __init__(
      self, buffer_size: int, batch_size: int, n_agents: int, seed: int) -> None:
    """Initializes replay buffer object."""
    self.n_agents = n_agents
    self.memory = deque(maxlen=buffer_size)
    self.batch_size = batch_size
    self.experience = namedtuple(
        "Experience",
        field_names=["state", "action", "reward", "next_state", "done"]
    )
    self.seed = random.seed(seed)

  def add(self, state, action: int, reward: int, next_state, done: bool) -> None:
    """Add new experience to memory."""
    e = self.experience(state, action, reward, next_state, done)
    self.memory.append(e)

  def sample(self) -> namedtuple:
    """Randomly sample batch of the experiences from memory (buffer)."""
    experiences = random.sample(self.memory, k=self.batch_size)

    states = (torch.from_numpy(
      np.vstack([e.state for e in experiences if e is not None]))
      .float().to(device))
    actions = (torch.from_numpy(
        np.vstack([e.action for e in experiences if e is not None]))
        .float().to(device))
    rewards = (torch.from_numpy(
        np.vstack([e.reward for e in experiences if e is not None]))
        .float().to(device))
    next_states = (torch.from_numpy(
        np.vstack([e.next_state for e in experiences if e is not None]))
        .float().to(device))
    dones = (torch.from_numpy(
        np.vstack([e.done for e in experiences if e is not None])
        .astype(np.uint8)).float().to(device))

    return (states, actions, rewards, next_states, dones)

  def __len__(self) -> int:
    """Returns current size of the replay buffer."""
    return len(self.memory)

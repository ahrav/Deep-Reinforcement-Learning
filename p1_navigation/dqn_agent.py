from collections import namedtuple, deque
import numpy as np
import random

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e5)  # Replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-2  # soft update of target params
LR = 5.5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Agent:
  """Object that interacts and learns from it's environment."""
  def __init__(self, state_size, action_size, seed) -> None:
    """Initialize the Agent object.

    Params:
      state_size (int): dimensions of each state
      action_size (int): dimension of each action
      seed (int): random seed
    """
    self.state_size = state_size
    self.action_size = action_size
    self.seed = random.seed(seed)

    # Q-Network
    self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
    self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
    self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

    # Replay memory (buffer)
    self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    # Initialize time step (for updating every UPDATE_EVERY steps)
    self.t_step = 0

  def step(self, state, action, reward, next_state, done) -> None:
    """Takes a step within the environment."""
    # Save experience in replay buffer (memory)
    self.memory.add(state, action, reward, next_state, done)

    # Learn every UPDATE_EVERY time step
    self.t_step = (self.t_step + 1) % UPDATE_EVERY
    if self.t_step == 0:
      # If enough subsets avail in replay buffer, randomly select and learn
      if len(self.memory) > BATCH_SIZE:
        self.learn(self.memory.sample(), GAMMA)

  def act(self, state, eps=0.) -> int:
    """Returns actions for a given state dependent on current policy.

    Args:
      state (List): current state
      eps (float): epsilon, for epsilon-greedy action selection
    """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    self.qnetwork_local.eval()
    with torch.no_grad():
      action_values = self.qnetwork_local(state)
    self.qnetwork_local.train()

    # Epsilon-greedy action selection
    if random.random() > eps:
      return np.argmax(action_values.cpu().data.numpy())
    return random.choice(np.arange(self.action_size))

  def learn(self, experience, gamma) -> None:
    """Update value parameters using given batch of experience tuples.

    Args:
      experiences (Tuple(torch.Tensor)): tuple of (s, a, r s', done) tuples
      gamma (float): discount factor
    """
    states, actions, rewards, next_states, dones = experience

    # Get max predicted Q values for next states from target model
    q_target_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
    # Compute Q targets for current states
    q_targets = rewards + (gamma * q_target_next * (1 - dones))

    # Get expected Q value from local model
    q_expected = self.qnetwork_local(states).gather(1, actions)

    # Compute loss
    loss = F.mse_loss(q_expected, q_targets)
    # Minimize loss
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

  def soft_update(self, local_model, target_model, tau) -> None:
    """Soft update to model parameters.

    Args:
      local_model (Pytorch model): weights will be copied from
      target_model (Pytorch model): weights will be copie to
      tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
  """Fixed size buffer to store experience tuples."""
  def __init__(self, action_size, buffer_size, batch_size, seed) -> None:
    """Initialize ReplayBuffer object.

    Params:
      action_size (int): dimensions of each action
      buffer_size (int): maximum size of buffer
      batch_size (int): size of each training batch
      seed (int): random seed
    """
    self.action_size = action_size
    self.memory = deque(maxlen=buffer_size)
    self.batch_size = batch_size
    self.experience = namedtuple(
        'Experience',
        field_names=['state', 'action', 'reward', 'next_state', 'done'])
    self.seed = random.seed(seed)

  def add(self, state, action, reward, next_state, done) -> None:
    """Add a new experience to the memory buffer object.

    Args:
      state (List): state of current object
      action (int): choice for the object to make
      reward (int): reward given for completing action
      next_state (List): next state of object
      done (bool): flag to determine whether object has completed experience
    """
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
        .long().to(device))
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

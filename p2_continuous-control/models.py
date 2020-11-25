"""All the models used within the project."""
from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer: List) -> tuple:
  fan_in = layer.weight.data.size()[0]
  lim = 1. / np.sqrt(fan_in)
  return (-lim, lim)


class Actor(nn.Module):
  """Actor (Policy) model - Neural net that determines action for an agent."""
  def __init__(self,
               action_size: int,
               state_size: int,
               seed: int,
               fc1_units: int=128,
               fc2_units: int=128) -> None:
    """Initialize parameters and build the model."""
    super(Actor, self).__init__()
    self.seed = torch.manual_seed(seed)

    self.fc1 = nn.Linear(state_size, fc1_units)
    self.fc2 = nn.Linear(fc1_units, fc2_units)
    self.fc3 = nn.Linear(fc2_units, action_size)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    """Set weights for the network."""
    self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
    self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
    self.fc3.weight.data.uniform_(-3e-3, 3e-3)

  def forward(self, state) -> torch.Tensor:
    """Forward propogation through network, mapping states to actions."""
    # forward through each layer in `hidden_layers`, with ReLU activation
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))

    # forward final layer with tanh activation (-1, 1)
    return F.tanh(self.fc3(x))


class Critic(nn.Module):
  """Critic (Value) model - Neural net estimates total expected return."""
  def __init__(self,
               action_size: int,
               state_size: int,
               seed: int,
               fc1_units: int=128,
               fc2_units: int=128) -> None:
    super(Critic, self).__init__()
    self.seed = torch.manual_seed(seed)

    self.fc1 = nn.Linear(state_size, fc1_units)
    self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
    self.fc3 = nn.Linear(fc2_units, 1)
    self.reset_parameters()

  def reset_parameters(self):
    self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
    self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
    self.fc3.weight.data.uniform_(-3e-3, 3e-3)

  def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Build a critic (value) network - maps state, action pairs -> Q-values."""
    xs = F.relu(self.fc1(state))
    x = torch.cat((xs, action), dim=1)
    x = F.relu(self.fc2(x))
    return self.fc3(x)

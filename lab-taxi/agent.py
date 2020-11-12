import numpy as np
import random
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 0.005
        self.gamma = 0.8213937505783256
        self.alpha = 0.0898281065178629
        # self.eps_decay = 0.8978159313202051
        # self.eps_min = 0.05

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random() > self.epsilon:
          return np.argmax(self.Q[state])
        else:
          return random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # self.epsilon = max(self.epsilon*self.eps_decay, self.eps_min)
        current = self.Q[state][action]
        policy_s = np.zeros(self.nA) * self.epsilon / self.nA
        policy_s[np.argmax(self.Q[next_state])] = 1 - self.epsilon + (self.epsilon / self.nA)
        Qsa_next = np.dot(self.Q[next_state], policy_s)
        target = reward + (self.gamma * Qsa_next)
        self.Q[state][action]  = current + (self.alpha * (target - current))
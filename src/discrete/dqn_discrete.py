import torch
import torch.nn as nn
import numpy as np
import random
from cnn_discrete import CarCNN
from replay_memory import ReplayMemory

class DQNDiscrete():
    def __init__(self, action_space, eps_start, eps_end, eps_decay, gamma, lr):
        self.action_space = action_space
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.lr = lr
        self.target_network = CarCNN()
        self.policy_network = CarCNN()
        #Initialise les deux réseaux avec les mêmes poids
        self.target_network(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr)

        self.memory = ReplayMemory(10000)
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()

        eps_threshold = max(self.eps_end, self.eps_start/max(self.steps_done, 1))

        if sample < eps_threshold:
            action = random.randint(1, 5)
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)

                q_values = self.policy_network(state_tensor)

                action = np.argmax(q_values)
        self.steps_done += 1
        return action 
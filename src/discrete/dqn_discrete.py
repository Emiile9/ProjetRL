import torch
import torch.nn as nn
import numpy as np
import random
from cnn_discrete import CarCNN
from replay_memory import ReplayMemory

class DQNDiscrete():
    def __init__(self, action_space, eps_start = 0.9, eps_end = 0.01, eps_decay = 1000, gamma = 0.9, lr = 0.01, batch_size = 256):
        self.action_space = action_space
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.lr = lr
        self.target_network = CarCNN()
        self.policy_network = CarCNN()
        self.batch_size = batch_size

        #Initialise les deux réseaux avec les mêmes poids
        self.target_network(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr)

        self.memory = ReplayMemory(10000)
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()

        eps_threshold = max(self.eps_end, self.eps_start)

        if sample < eps_threshold:
            action = random.randint(1, 5)
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)

                q_values = self.policy_network(state_tensor)

                action = np.argmax(q_values)
        self.steps_done += 1
        return action 
    
    def upadte(self):
        #Entraine que si on a assez de données
        if self.memory.get_len() < self.batch_size:
            return
        
        batch = random.sample(self.memory.memory, self.batch_size)
    
        # Zip le batch pour récupérer les composantes
        states, actions, rewards, next_states, dones = zip(*batch)

        #Conversion en tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
import torch
import torch.nn as nn
import numpy as np
import random
from cnn_discrete import CarCNN
from replay_memory import ReplayMemory

class DQNDiscrete():
    def __init__(self, action_space, eps_start=0.9, eps_end=0.05, eps_decay=0.9999, gamma=0.95, lr=0.001, batch_size=64, tau = 0.005):
        self.action_space = action_space
        self.epsilon = eps_start 
        self.eps_end = eps_end
        self.eps_decay = eps_decay 
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.tau = tau

        self.policy_network = CarCNN()
        self.target_network = CarCNN()
        
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayMemory(10000)
        self.steps_done = 0

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 
        self.policy_network.to(self.device)

    def select_action(self, state):
        sample = random.random()
        
        if sample < self.epsilon:
            action = random.randint(0, 4) 
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                q_values = self.policy_network(state_tensor)
                action = q_values.argmax(dim=1).item()
        
        #Update epsilon
        if self.epsilon > self.eps_end:
            self.epsilon *= self.eps_decay
            
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

        current_q = self.policy_network(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            # If done=1, only the reward counts (no future rewards)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        loss = self.loss_fn(current_q, target_q)

        #Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #update le reseau cible petit à petit 
        for target_param, policy_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, filename):
        torch.save(self.policy_network.state_dict(), filename)

    def load(self, filename):
        self.policy_network.load_state_dict(torch.load(filename))
        self.target_network.load_state_dict(self.policy_network.state_dict())
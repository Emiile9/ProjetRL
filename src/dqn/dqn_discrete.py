import torch
import torch.nn as nn
import numpy as np
import random
from src.dqn.cnn_discrete import CarCNN
from src.dqn.replay_memory import ReplayMemory

class DQNDiscrete():
    def __init__(self, action_space, eps_start=0.9, eps_end=0.01, eps_decay=0.9999, gamma=0.95, lr=0.0005, batch_size=64, tau = 0.05):
        self.action_space = action_space
        self.epsilon = eps_start 
        self.eps_end = eps_end
        self.eps_start = eps_start
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

        self.transition_memory = ReplayMemory(10000)
        self.steps_done = 0

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 
        self.policy_network.to(self.device)
        self.target_network.to(self.device) 

    def select_action(self, state):
        sample = random.random()
        
        if sample < self.epsilon:
            action = random.randint(0, 4) 
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                q_values = self.policy_network(state_tensor)
                action = q_values.argmax(dim=1).item()
        
        #Update epsilon
        if self.epsilon > self.eps_end:
            self.epsilon = max(self.eps_end, self.epsilon - (self.eps_start - self.eps_end) / 100000)
            
        self.steps_done += 1
        return action
    
    def update(self):
        #Entraine que si on a assez de données
        if self.transition_memory.get_len() < self.batch_size:
            return
        
        batch = random.sample(self.transition_memory.memory, self.batch_size)
    
        # Zip le batch pour récupérer les composantes
        states, actions, rewards, next_states, dones = zip(*batch)
        #Conversion en tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        current_q = self.policy_network(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            # If done=1, only the reward counts (no future rewards)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        loss = self.loss_fn(current_q, target_q)

        #Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()
    
    def copy_weights_to_target(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def save(self, filename):
        torch.save(self.policy_network.state_dict(), filename)

    def load(self, filename):
        self.policy_network.load_state_dict(torch.load(filename))
        self.target_network.load_state_dict(self.policy_network.state_dict())
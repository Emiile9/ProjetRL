import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
from collections import deque
import random

class ActorCritic(nn.Module):
    def __init__(self, observation_shape, action_dim, continuous=True):
        super(ActorCritic, self).__init__()
        self.continuous = continuous
        
        # Encodeur CNN partagé
        self.encoder = nn.Sequential(
            nn.Conv2d(observation_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculer la taille de sortie du CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_shape)
            cnn_output_size = self.encoder(dummy_input).shape[1]
        
        # Couche commune
        self.fc_common = nn.Sequential(
            nn.Linear(cnn_output_size, 512),
            nn.ReLU()
        )
        
        # Acteur (politique)
        if continuous:
            self.actor_mean = nn.Linear(512, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        else:
            self.actor = nn.Linear(512, action_dim)
        
        # Critique (valeur)
        self.critic = nn.Linear(512, 1)
        
        # Initialisation des poids
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = x.float() / 255.0  # Normalisation
        features = self.encoder(x)
        features = self.fc_common(features)
        
        value = self.critic(features)
        
        if self.continuous:
            action_mean = torch.tanh(self.actor_mean(features))
            action_std = torch.exp(self.actor_log_std.expand_as(action_mean))
            return action_mean, action_std, value
        else:
            action_logits = self.actor(features)
            return action_logits, value
    
    def get_action(self, x, deterministic=False):
        if self.continuous:
            action_mean, action_std, value = self.forward(x)
            
            if deterministic:
                return action_mean, None, value
            
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            
            return action, log_prob, value
        else:
            action_logits, value = self.forward(x)
            
            if deterministic:
                action = torch.argmax(action_logits, dim=-1, keepdim=True)
                return action, None, value
            
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).unsqueeze(-1)
            
            return action.unsqueeze(-1), log_prob, value
    
    def evaluate_actions(self, x, actions):
        if self.continuous:
            action_mean, action_std, value = self.forward(x)
            dist = Normal(action_mean, action_std)
            log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
        else:
            action_logits, value = self.forward(x)
            dist = Categorical(logits=action_logits)
            log_prob = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
            entropy = dist.entropy().unsqueeze(-1)
        
        return log_prob, value, entropy

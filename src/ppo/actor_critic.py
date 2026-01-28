import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
        )
        
        # Actor head: Mean values for [Steer, Gas, Brake]
        self.actor_mu = nn.Sequential(
            nn.Linear(512, 3), 
            nn.Tanh() 
        )
        self.log_std = nn.Parameter(torch.zeros(1, 3)) 
        
        # Critic head: State Value
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        features = self.fc(self.conv(x))
        return self.actor_mu(features), self.critic(features)

    def get_action(self, x):
        mu, value = self.forward(x)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        
        return action, log_prob, entropy, value
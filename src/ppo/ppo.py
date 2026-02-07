import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from .actor_critic import ActorCritic
from .rollout_buffer import RolloutBuffer
class PPO:
    def __init__(
        self,
        env,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        epochs=10,
        batch_size=64,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Déterminer si l'environnement est continu
        self.continuous = isinstance(env.action_space, gym.spaces.Box)
        
        if self.continuous:
            self.action_dim = env.action_space.shape[0]
        else:
            self.action_dim = env.action_space.n
        
        observation_shape = env.observation_space.shape
        
        # Créer le réseau acteur-critique
        self.actor_critic = ActorCritic(
            observation_shape, self.action_dim, continuous=self.continuous
        ).to(device)
        
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        self.buffer = RolloutBuffer()
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Calcule les avantages avec GAE (Generalized Advantage Estimation)"""
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self):
        """Met à jour la politique avec PPO"""
        states, actions, rewards, old_log_probs, values, dones = self.buffer.get()
        
        # Calculer la valeur du dernier état
        with torch.no_grad():
            last_state = torch.FloatTensor(states[-1]).unsqueeze(0).to(self.device)
            if self.continuous:
                _, _, next_value = self.actor_critic(last_state)
            else:
                _, next_value = self.actor_critic(last_state)
            next_value = next_value.cpu().numpy().flatten()[0]
        
        # Calculer les avantages et les retours
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Normaliser les avantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convertir en tenseurs
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(-1).to(self.device)
        returns = torch.FloatTensor(returns).unsqueeze(-1).to(self.device)
        
        # Optimisation PPO
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Évaluer les actions
                log_probs, state_values, entropy = self.actor_critic.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Ratio de politique
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Perte de politique (clipped)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Perte de valeur
                value_loss = nn.MSELoss()(state_values, batch_returns)
                
                # Perte d'entropie
                entropy_loss = -entropy.mean()
                
                # Perte totale
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Mise à jour
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        self.buffer.clear()
    
    def train(self, total_timesteps, rollout_length=2048, log_interval=10):
        """Entraîne l'agent PPO"""
        state, _ = self.env.reset()
        episode_reward = 0
        episode_count = 0
        timestep = 0
        episode_rewards = []
        
        while timestep < total_timesteps:
            # Collecte de données
            for _ in range(rollout_length):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, log_prob, value = self.actor_critic.get_action(state_tensor)
                
                action_np = action.cpu().numpy()[0]
                log_prob_np = log_prob.cpu().numpy()[0] if log_prob is not None else 0
                value_np = value.cpu().numpy()[0][0]
                
                next_state, reward, done, truncated, _ = self.env.step(action_np)
                
                self.buffer.add(state, action_np, reward, log_prob_np, value_np, done or truncated)
                
                state = next_state
                episode_reward += reward
                timestep += 1
                
                if done or truncated:
                    episode_rewards.append(episode_reward)
                    episode_count += 1
                    
                    if episode_count % log_interval == 0:
                        avg_reward = np.mean(episode_rewards[-log_interval:])
                        print(f"Timestep: {timestep}/{total_timesteps} | "
                              f"Episode: {episode_count} | "
                              f"Avg Reward (last {log_interval}): {avg_reward:.2f}")
                    
                    state, _ = self.env.reset()
                    episode_reward = 0
                
                if timestep >= total_timesteps:
                    break
            
            # Mise à jour de la politique
            self.update()
        
        return episode_rewards
    
    def save(self, filepath):
        """Sauvegarde le modèle"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Modèle sauvegardé dans {filepath}")
    
    def load(self, filepath):
        """Charge le modèle"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Modèle chargé depuis {filepath}")
import gymnasium as gym
from src.ppo.actor_critic import ActorCritic
from src.ppo.ppo import PPO
from src.utils.make_custom_env import make_env
import torch
from torch.distributions import Normal


import gymnasium as gym
import numpy as np

def main():
    # Paramètres
    CONTINUOUS = True  # True pour actions continues, False pour discrètes
    TOTAL_TIMESTEPS = 1_000_000
    ROLLOUT_LENGTH = 2048
    
    # Créer l'environnement
    env = make_env(continuous=CONTINUOUS, mode="train")
    
    # Créer l'agent PPO
    agent = PPO(
        env=env,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        epochs=10,
        batch_size=64,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5
    )
    
    print("Début de l'entraînement...")
    print(f"Type d'action: {'Continu' if CONTINUOUS else 'Discret'}")
    print(f"Device: {agent.device}")
    
    # Entraîner l'agent
    rewards = agent.train(
        total_timesteps=TOTAL_TIMESTEPS,
        rollout_length=ROLLOUT_LENGTH,
        log_interval=10
    )
    
    # Sauvegarder le modèle
    agent.save("ppo_carracing.pth")
    
    print("Entraînement terminé!")
    
    # Test du modèle
    print("\nTest du modèle entraîné...")
    test_env = make_env(continuous=CONTINUOUS, mode="watch")
    state, _ = test_env.reset()
    
    total_reward = 0
    done = False
    
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            action, _, _ = agent.actor_critic.get_action(state_tensor, deterministic=True)
        action_np = action.cpu().numpy()[0]
        
        state, reward, done, truncated, _ = test_env.step(action_np)
        total_reward += reward
        done = done or truncated
    
    print(f"Récompense totale du test: {total_reward:.2f}")
    test_env.close()


if __name__ == "__main__":
    main()

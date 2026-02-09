import gymnasium as gym
from src.ppo.actor_critic import ActorCritic
from src.ppo.ppo import PPO
from src.utils.make_custom_env import make_env
import torch
from torch.distributions import Normal
import pickle


import gymnasium as gym
import numpy as np

def main():
    # ✅ HYPERPARAMÈTRES OPTIMISÉS pour CarRacing
    CONTINUOUS = True
    TOTAL_TIMESTEPS = 2_000_000  # Plus d'entraînement
    ROLLOUT_LENGTH = 2048
    
    # Créer l'environnement
    env = make_env(continuous=CONTINUOUS, mode="train")
    
    # ✅ HYPERPARAMÈTRES AMÉLIORÉS
    agent = PPO(
        env=env,
        lr=1e-4,              # ✅ Learning rate plus faible
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        epochs=4,             # ✅ Moins d'epochs (éviter overfitting)
        batch_size=128,       # ✅ Batch plus grand
        value_coef=0.5,
        entropy_coef=0.01,    # ✅ Plus d'exploration
        max_grad_norm=0.5
    )
    
    print("Début de l'entraînement OPTIMISÉ...")
    print(f"Type d'action: {'Continu' if CONTINUOUS else 'Discret'}")
    print(f"Device: {agent.device}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    
    # Entraîner l'agent
    rewards = agent.train(
        total_timesteps=TOTAL_TIMESTEPS,
        rollout_length=ROLLOUT_LENGTH,
        log_interval=10,
        save_interval=100
    )
    
    # Sauvegarder le modèle final
    agent.save("ppo_carracing_final.pth")
    
    
    print("Entraînement terminé!")
    print(f"Meilleure récompense moyenne: {max([np.mean(rewards[i:i+10]) for i in range(len(rewards)-10)]):.2f}")
    with open("stats_ppo.pkl", "wb") as f:
        pickle.dump(rewards, f)

if __name__ == "__main__":
    main()

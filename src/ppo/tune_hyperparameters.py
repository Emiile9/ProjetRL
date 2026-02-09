from src.ppo.main import main
import pickle

for lr in [1e-3, 1e-4, 5e-5]:
    stats = main(lr=lr, epochs=4, batch_size=128, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, gae_lambda=0.95, gamma=0.99)
    stats_dict = {"episode": list(range(len(stats))), "reward": stats}
    with open(f"/Users/emile/Desktop/IASD/TP_M2/ProjetRL/stats/ppo/stats_ppo_lr_{lr}.pkl", "wb") as f:
        pickle.dump(stats_dict, f)
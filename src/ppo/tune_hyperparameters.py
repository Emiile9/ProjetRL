from src.ppo.main import main
import pickle

for ec in [0.01, 0.05, 0.001]:
    print(f"Training with entropy_coef={ec}")
    stats = main(lr=1e-4, epochs=4, batch_size=128, value_coef=0.5, entropy_coef=ec, max_grad_norm=0.5, gae_lambda=0.95, gamma=0.99)
    print(stats)
    with open(f"/Users/emile/Desktop/IASD/TP_M2/ProjetRL/stats/ppo/stats_ppo_entropy_coef_{ec}.pkl", "wb") as f:
        pickle.dump(stats, f)


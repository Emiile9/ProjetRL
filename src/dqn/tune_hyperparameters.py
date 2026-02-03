import pickle
from src.dqn.main import train_agent

eps_decays = [1000, 10000, 100000]
learning_rates = [0.001, 0.0005, 0.0001]

for eps_divider in eps_decays:
    for lr in learning_rates:
        print(f"Training with eps_divider={eps_divider}, lr={lr}")
        stats = train_agent(
            lr=lr,
            eps_start=0.9,
            eps_end=0.01,
            eps_divider=eps_divider,
            num_episodes=500,
        )
        # Save stats to a file for later analysis
        filename = f"stats_epsdiv{eps_divider}_lr{lr}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(stats, f)
        print(f"Saved stats to {filename}")
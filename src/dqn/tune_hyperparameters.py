import pickle
from src.dqn.main import train_agent

C = [1, 5, 10, 50]

for c in C:
    print(f"Training with c={c}")
    stats = train_agent(
        lr=0.0005,
        eps_start=0.9,
        eps_end=0.01,
        eps_divider=10000,
        num_episodes=1000,
    )
    # Save stats to a file for later analysis
    filename = f"stats_c{c}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(stats, f)
    print(f"Saved stats to {filename}")
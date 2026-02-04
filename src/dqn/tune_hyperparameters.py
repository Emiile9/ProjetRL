import pickle
from src.dqn.main import train_agent

eps_decays = [1000, 10000, 100000]

for eps_divider in eps_decays:
    print(f"Training with eps_divider={eps_divider}")
    stats = train_agent(
        lr=0.0005,
        eps_start=0.9,
        eps_end=0.01,
        eps_divider=eps_divider,
        num_episodes=1500,
    )
    # Save stats to a file for later analysis
    filename = f"stats_epsdiv{eps_divider}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(stats, f)
    print(f"Saved stats to {filename}")
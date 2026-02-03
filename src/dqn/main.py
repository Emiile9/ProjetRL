import gymnasium as gym
import pickle
import argparse

from ..utils.make_custom_env import make_env
from src.dqn.dqn_discrete import DQNDiscrete

parser = argparse.ArgumentParser(
    prog="RL Project Emile Descroix",
    description="Training script for a DQN agent on a custom environment",
)

parser.add_argument(
    "--lr", type=float, default=0.0005, help="learning rate for the optimizer"
)
parser.add_argument(
    "--eps_start",
    type=float,
    default=0.9,
    help="starting epsilon for the epsilon-greedy policy",
)
parser.add_argument(
    "--eps_end",
    type=float,
    default=0.01,
    help="ending epsilon for the epsilon-greedy policy",
)
parser.add_argument(
    "--eps_divider", type=int, default=100000, help="divisor for epsilon decay"
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=500,
    help="number of episodes to train the agent for",
)

args = parser.parse_args()


def train_agent(
    lr=0.0005, eps_start=0.9, eps_end=0.01, eps_divider=100000, num_episodes=20
):
    env = make_env(continuous=False)
    agent = DQNDiscrete(
        action_space=env.action_space.n,
        lr=lr,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_divider=eps_divider,
    )
    stats = {"episode": [], "reward": [], "epsilon": []}

    for e in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        for _ in range(10000):
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.transition_memory.push_transition(
                (state, action, reward, next_state, done)
            )
            
            agent.update()

            state = next_state
            total_reward += reward

            if done:
                break
        stats["episode"].append(e)
        stats["reward"].append(total_reward)
        stats["epsilon"].append(agent.epsilon)

        print(f"Episode {e}: Reward = {total_reward}, Epsilon = {agent.epsilon:.2f}")

        if e % 5 == 0:
            agent.copy_weights_to_target()
        if e % 50 == 0:
            agent.save(f"car_dqn_{args.lr}_{e}.pth")
    env.close()
    return stats


stats_0_05 = train_agent(
    lr=args.lr,
    eps_start=args.eps_start,
    eps_end=args.eps_end,
    eps_divider=args.eps_divider,
    num_episodes=args.num_episodes,
)

with open("training_stats_continuous.pkl", "wb") as f:
    pickle.dump(stats_0_05, f)

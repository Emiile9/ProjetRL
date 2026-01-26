import gymnasium as gym
import pickle

from make_custom_env import make_env
from dqn_discrete import DQNDiscrete

def train_agent():
    env = make_env(continuous=False)
    agent = DQNDiscrete(action_space=env.action_space.n)
    num_episodes = 2000
    stats = {"episode" : [], "reward" : [], "epsilon" : []}

    for e in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        for t in range(10000):
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 4. Store experience
            agent.transition_memory.push_transition((state, action, reward, next_state, done))
            
            # 5. Train
            agent.update() # This now includes the target network sync logic
            
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
            agent.save(f"car_dqn_{str(agent.tau).replace('.', '_')}_{e}.pth")
    env.close()
    return stats

stats_0_05 = train_agent()

with open("training_stats.pkl", "wb") as f:
    pickle.dump(stats_0_05, f)
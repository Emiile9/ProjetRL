import gymnasium as gym

from make_custom_env import make_env
from dqn_discrete import DQNDiscrete

env = make_env(continuous=False)
agent = DQNDiscrete(action_space=env.action_space.n)
num_episodes = 500

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
            
    print(f"Episode {e}: Reward = {total_reward}, Epsilon = {agent.epsilon:.2f}")
    if e % 20 == 0:
        agent.save("car_dqn_v1.pth")
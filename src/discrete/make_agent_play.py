from make_custom_env import make_env
from dqn_discrete import DQNDiscrete
import torch

def watch_agent(path, episodes=5):
    env = make_env(render_mode="human")
    
    agent = DQNDiscrete(action_space=5)
    agent.load(path)
    agent.policy_network.eval() # Set to evaluation mode
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                action = agent.policy_network(state_tensor).argmax().item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()
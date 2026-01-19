from make_custom_env import make_env
from dqn_discrete import DQNDiscrete
import torch
import numpy as np

def watch_agent(path, episodes=100):
    env = make_env(render_mode="human", continuous=False)
    
    agent = DQNDiscrete(action_space=5)
    agent.load(path)
    agent.policy_network.eval() # Set to evaluation mode
    
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            device = agent.device 

            # 2. Inside your loop:
            with torch.no_grad():
                # Convert to tensor AND move to device
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
                
                # Now this will work!
                q_values = agent.policy_network(state_tensor)
                action = q_values.argmax().item()
            
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()

watch_agent("/Users/emile/Desktop/IASD/TP_M2/ProjetRL/car_dqn_v1.pth")
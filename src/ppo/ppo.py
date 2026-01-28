import torch

class PPO():
    def __init__(self, actor_critic, trajectories):
        self.actor_critic = actor_critic
        self.trajectories = trajectories

    def collect_trajectories(self):
        observations = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []
        t = 0

        obs, _ = self.env.reset() # Reset returns (obs, info)
            
        while t < self.trajectories:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device) / 255.0
            
            with torch.no_grad():
                action, log_prob, _, value = self.actor_critic.get_action(obs_t)

            next_obs, reward, term, trunc, _ = self.env.step(action.cpu().numpy()[0])
            done = term or trunc
            
            observations.append(obs_t.squeeze())
            actions.append(action.squeeze())
            rewards.append(torch.tensor(reward, device=self.device))
            dones.append(torch.tensor(done, dtype=torch.float32, device=self.device))
            log_probs.append(log_prob.squeeze())
            values.append(value.squeeze())

            obs = next_obs
            if done:
                obs, _ = self.env.reset()
            
            t += 1

        last_obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device) / 255.0
        _, last_value = self.actor_critic(last_obs_t)
        
        return (torch.stack(observations), torch.stack(actions), torch.stack(rewards), 
                torch.stack(dones), torch.stack(log_probs), torch.stack(values), 
                last_value.detach().squeeze())
import torch
import torch.nn as nn
import numpy as np


class PPO:
    def __init__(
        self,
        actor_critic,
        trajectories,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        ppo_epochs=10,
        mini_batch_size=64,
        lr=3e-4,
    ):
        self.actor_critic = actor_critic
        self.trajectories = trajectories
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.lr = lr
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=3e-4, eps=1e-5, weight_decay=1e-3
        )

    def collect_trajectories(self, env):
        observations = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []
        t = 0

        obs, _ = env.reset()  # Reset returns (obs, info)

        while t < self.trajectories:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device) / 255.0

            with torch.no_grad():
                action, log_prob, _, value = self.actor_critic.get_action(obs_t)

            next_obs, reward, term, trunc, _ = env.step(action.cpu().numpy()[0])
            done = term or trunc

            observations.append(obs_t.squeeze())
            actions.append(action.squeeze())
            rewards.append(torch.tensor(reward, device=self.device))
            dones.append(torch.tensor(done, dtype=torch.float32, device=self.device))
            log_probs.append(log_prob.squeeze())
            values.append(value.squeeze())

            obs = next_obs
            if done:
                obs, _ = env.reset()

            t += 1

        last_obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device) / 255.0
        _, last_value = self.actor_critic(last_obs_t)

        return (
            torch.stack(observations),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(dones),
            torch.stack(log_probs),
            torch.stack(values),
            last_value.detach().squeeze(),
        )

    def compute_advantages(self, rewards, values, dones, last_value):
        # last_value is the predicted value of the state AFTER the rollout ends
        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae_lam = 0

        # We walk backwards from the end of the rollout to the start
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]

            # 1. Calculate TD Error (delta)
            # delta = reward + (gamma * next_value) - current_value
            delta = (
                rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            )

            # 2. Calculate GAE
            # gae = delta + (gamma * lambda * gae_from_next_step)
            advantages[t] = last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )

        # 3. Calculate Returns (The target for the Critic network)
        returns = advantages + values

        # 4. Standardize Advantages (Crucial for stability!)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self, s, a, lp, ret, adv):
        # s: obs, a: actions, lp: log_probs, ret: returns, adv: advantages
        for _ in range(self.ppo_epochs):
            # Create random mini-batches
            indices = np.arange(self.trajectories)
            np.random.shuffle(indices)

            for start in range(0, self.trajectories, self.mini_batch_size):
                end = start + self.mini_batch_size
                idx = indices[start:end]

                # Fetch mini-batch
                batch_s, batch_a, batch_lp, batch_ret, batch_adv = (
                    s[idx],
                    a[idx],
                    lp[idx],
                    ret[idx],
                    adv[idx],
                )

                # Get new policy values
                mu, val = self.actor_critic(batch_s)
                dist = torch.distributions.Normal(mu, self.actor_critic.log_std.exp())
                new_lp = dist.log_prob(batch_a).sum(axis=-1)
                entropy = dist.entropy().sum(axis=-1)

                # 1. Policy Loss (Clipped)
                ratio = torch.exp(new_lp - batch_lp)
                surr1 = ratio * batch_adv
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * batch_adv
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # 2. Value Loss (MSE)
                value_loss = nn.MSELoss()(val.squeeze(), batch_ret)

                # 3. Total Loss
                # We subtract entropy to encourage exploration
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save(self, filename):
        torch.save(self.actor_critic.state_dict(), filename)

    def load(self, filename):
        self.actor_critic.load_state_dict(torch.load(filename))

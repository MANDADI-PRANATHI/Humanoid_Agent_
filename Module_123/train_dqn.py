# train_dqn.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# --------------------------
# Q-Network
# --------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_joints: int, num_bins: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_joints = num_joints
        self.num_bins = num_bins
        hidden = 512
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_joints * num_bins),
        )

    def forward(self, obs):
        x = self.net(obs)
        return x.view(-1, self.num_joints, self.num_bins)

# --------------------------
# Replay Buffer
# --------------------------
class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action_bins, reward, next_obs, done):
        self.buffer.append((obs, action_bins, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.stack(obs),
            np.stack(actions),
            np.stack(rewards),
            np.stack(next_obs),
            np.stack(dones),
        )

    def __len__(self):
        return len(self.buffer)

# --------------------------
# Discretizer
# --------------------------
class Discretizer:
    def __init__(self, num_joints: int, num_bins: int):
        self.num_joints = num_joints
        self.num_bins = num_bins
        self.bin_values = np.linspace(-1.0, 1.0, num_bins)

    def bins_to_torque(self, action_bins):
        return self.bin_values[action_bins]

# --------------------------
# Reward function
# --------------------------
def compute_reward(env, torques):
    base_vel = env.get_base_velocity()
    r_vel = base_vel[0]
    alive = 1.0 if env.is_upright() else 0.0
    r_energy = np.sum(np.square(torques))
    w_vel, w_alive, w_energy = 2.0, 0.5, 0.01
    fall_penalty = -50.0
    reward = w_vel * r_vel + w_alive * alive - w_energy * r_energy
    if not env.is_upright():
        reward += fall_penalty
    return reward

# --------------------------
# DQN Agent
# --------------------------
class DQNAgent:
    def __init__(self, obs_dim, num_joints, num_bins, lr=1e-4, gamma=0.99, batch_size=64, device="cpu"):
        self.obs_dim = obs_dim
        self.num_joints = num_joints
        self.num_bins = num_bins
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        self.q = QNetwork(obs_dim, num_joints, num_bins).to(device)
        self.q_target = QNetwork(obs_dim, num_joints, num_bins).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=lr)

        self.discretizer = Discretizer(num_joints, num_bins)
        self.replay = ReplayBuffer()
        self.buffer = self.replay  # alias for old code

        self.eps = 1.0
        self.eps_min = 0.1
        self.eps_decay = 0.9995

    # --------------------------
    # Convert bins to torques
    # --------------------------
    def bins_to_action(self, action_bins):
        return self.discretizer.bins_to_torque(action_bins)

    # --------------------------
    # Select action
    # --------------------------
    def select_action(self, obs):
        if np.random.rand() < self.eps:
            return np.random.randint(0, self.num_bins, size=self.num_joints)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qvals = self.q(obs_t)[0]
        return torch.argmax(qvals, dim=1).cpu().numpy()

    # --------------------------
    # Store transition
    # --------------------------
    def push(self, *args):
        self.replay.push(*args)

    # --------------------------
    # Training step
    # --------------------------
    def train_step(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if len(self.replay) < batch_size:
            return 0.0

        obs, actions, rewards, next_obs, dones = self.replay.sample(batch_size)

        obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_vals = self.q(obs_t)
        chosen_q = q_vals.gather(2, actions_t.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_vals = self.q(next_obs_t)
            next_actions = torch.argmax(next_q_vals, dim=2)
            next_q_target = self.q_target(next_obs_t)
            target_q = next_q_target.gather(2, next_actions.unsqueeze(-1)).squeeze(-1)
            
            rewards_exp = rewards_t.unsqueeze(1).expand(-1, self.num_joints)
            dones_exp = dones_t.unsqueeze(1).expand(-1, self.num_joints)
            target = rewards_exp + self.gamma * (1 - dones_exp) * target_q
        loss = nn.MSELoss()(chosen_q, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # epsilon decay
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        return loss.item()

    # --------------------------
    # Update target network
    # --------------------------
    def update_target(self):
        self.q_target.load_state_dict(self.q.state_dict())

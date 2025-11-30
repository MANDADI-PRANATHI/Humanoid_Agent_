import argparse
import datetime
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from agent_ppo import PPOAgent
from buffer_ppo import PPOBuffer

from humanoid_env import HumanoidWalkEnv


def make_arg_parser():
    parser = argparse.ArgumentParser(description="PPO training for HumanoidWalkEnv")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--n_epochs", type=int, default=3)  # <--- keep small by default for debug; change as needed
    parser.add_argument("--train_iters", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--target_kl", type=float, default=0.03)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--render_epoch", type=int, default=10)
    parser.add_argument("--urdf_path", type=str, default="./assets/humanoid_10theta.urdf")
    return parser


DEFAULT_JOINT_MAP = {
    0: "right_knee",
    1: "left_knee",
    2: "right_hip_pitch",
    3: "left_hip_pitch",
    4: "right_ankle",
    5: "left_ankle",
    6: "right_shoulder_pitch",
    7: "left_shoulder_pitch",
    8: "right_elbow",
    9: "left_elbow",
    10: "right_wrist",
    11: "left_wrist",
    12: "right_hip_roll",
    13: "left_hip_roll",
    14: "right_shoulder_roll",
    15: "left_shoulder_roll",
}


def make_env(urdf_path, joint_map):
    def _f():
        return HumanoidWalkEnv(urdf_path=urdf_path, joint_map=joint_map, gui=False)
    return _f


def ppo_update(agent, optimizer, scaler,
               batch_obs, batch_actions, batch_returns,
               batch_old_log_probs, batch_adv,
               clip_epsilon, vf_coef, ent_coef, device):

    agent.train()
    optimizer.zero_grad()

    with torch.amp.autocast(device_type="cpu" ,enabled=False):
        _, new_log_probs, entropies, new_values = agent.get_action_and_value(batch_obs.to(device), batch_actions.to(device))
        ratio = torch.exp(new_log_probs - batch_old_log_probs.to(device))
        kl = ((batch_old_log_probs.to(device) - new_log_probs) / max(1, batch_actions.size(-1))).mean()

        surr1 = ratio * batch_adv.to(device)
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_adv.to(device)
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.MSELoss()(new_values.squeeze(1), batch_returns.to(device))
        entropy = entropies.mean()

        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item(), kl.item()


def train_main():
    args = make_arg_parser().parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    current_dir = os.path.dirname(__file__)
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoints_dir = os.path.join(current_dir, "checkpoints", run_name)
    os.makedirs(checkpoints_dir, exist_ok=True)
    logs_dir = os.path.join(current_dir, "logs", run_name)
    writer = SummaryWriter(logs_dir)

    n_envs = max(1, args.n_envs)
    envs = gym.vector.SyncVectorEnv([make_env(args.urdf_path, DEFAULT_JOINT_MAP) for _ in range(n_envs)])
    test_env = make_env(args.urdf_path, DEFAULT_JOINT_MAP)()

    obs_dim = envs.single_observation_space.shape
    act_dim = envs.single_action_space.shape

    agent = PPOAgent(obs_dim[0], act_dim[0]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    scaler = torch.amp.GradScaler()

    buffer = PPOBuffer(obs_dim, act_dim, args.n_steps, n_envs, device, args.gamma, args.gae_lambda)

    reset_out = envs.reset()
    next_obs = torch.tensor(np.array(reset_out[0], dtype=np.float32), device=device)
    next_terminateds = torch.zeros(n_envs, dtype=torch.float32, device=device)
    next_truncateds = torch.zeros(n_envs, dtype=torch.float32, device=device)

    best_mean_reward = -np.inf
    reward_list = []

    for epoch in range(1, args.n_epochs + 1):
        for _ in range(args.n_steps):
            obs = next_obs
            terminateds = next_terminateds
            truncateds = next_truncateds

            with torch.no_grad():
                actions, logprobs, _, values = agent.get_action_and_value(obs)
                values = values.reshape(-1)

            next_obs_np, rewards, next_terminateds, next_truncateds, _ = envs.step(actions.cpu().numpy())

            next_obs = torch.tensor(np.array(next_obs_np, dtype=np.float32), device=device)
            reward_list.extend(rewards)
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            next_terminateds = torch.as_tensor(next_terminateds, dtype=torch.float32, device=device)
            next_truncateds = torch.as_tensor(next_truncateds, dtype=torch.float32, device=device)

            buffer.store(obs, actions, rewards_t, values, terminateds, truncateds, logprobs)

        with torch.no_grad():
            next_values = agent.get_value(next_obs).reshape(1, -1)
            next_terminateds = next_terminateds.reshape(1, -1)
            next_truncateds = next_truncateds.reshape(1, -1)
            traj_adv, traj_ret = buffer.calculate_advantages(next_values, next_terminateds, next_truncateds)

        traj_obs, traj_act, traj_logprob = buffer.get()
        traj_obs = traj_obs.view(-1, *obs_dim)
        traj_act = traj_act.view(-1, *act_dim)
        traj_logprob = traj_logprob.view(-1)
        traj_adv = traj_adv.view(-1)
        traj_ret = traj_ret.view(-1)
        traj_adv = (traj_adv - traj_adv.mean()) / (traj_adv.std() + 1e-8)

        dataset_size = args.n_steps * n_envs
        traj_indices = np.arange(dataset_size)

        losses_total = []
        kl_list = []
        kl_early_stop = False

        for _ in range(args.train_iters):
            np.random.shuffle(traj_indices)
            for start_idx in range(0, dataset_size, args.batch_size):
                end_idx = start_idx + args.batch_size
                batch_indices = traj_indices[start_idx:end_idx]

                batch_obs = traj_obs[batch_indices].to(device)
                batch_actions = traj_act[batch_indices].to(device)
                batch_returns = traj_ret[batch_indices].to(device)
                batch_old_log_probs = traj_logprob[batch_indices].to(device)
                batch_adv = traj_adv[batch_indices].to(device)

                loss, policy_loss, value_loss, entropy, kl = ppo_update(agent, optimizer, scaler, batch_obs,
                                                                        batch_actions, batch_returns,
                                                                        batch_old_log_probs, batch_adv,
                                                                        args.clip_ratio, args.vf_coef,
                                                                        args.ent_coef, device)

                losses_total.append(loss)
                kl_list.append(kl)

                if kl > args.target_kl:
                    kl_early_stop = True
                    break
            if kl_early_stop:
                break

        mean_reward = float(np.mean(reward_list)) if reward_list else 0.0
        reward_list = []
        writer.add_scalar("reward/mean", mean_reward, epoch)
        writer.add_scalar("loss/total", np.mean(losses_total) if losses_total else 0.0, epoch)

        # save best + last
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            torch.save(agent.state_dict(), os.path.join(checkpoints_dir, "best.pt"))
        torch.save(agent.state_dict(), os.path.join(checkpoints_dir, "last.pt"))

        # optional short eval (no video by default)
        if epoch % args.render_epoch == 0:
            obs_eval, _ = test_env.reset()
            done_eval = False
            steps_eval = 0
            while not done_eval and steps_eval < 240:
                with torch.no_grad():
                    action_eval, _, _, _ = agent.get_action_and_value(torch.tensor(np.array([obs_eval], dtype=np.float32), device=device))
                action_np = action_eval.squeeze(0).cpu().numpy()
                obs_eval, _, term, trunc, _ = test_env.step(action_np)
                done_eval = bool(term or trunc)
                steps_eval += 1

    # Save metadata file to help test script find latest run
    meta_path = os.path.join(checkpoints_dir, "META.txt")
    with open(meta_path, "w") as f:
        f.write(f"run_name={run_name}\n")
        f.write(f"checkpoint_dir={checkpoints_dir}\n")

    try:
        envs.close()
    except Exception:
        pass
    try:
        test_env.close()
    except Exception:
        pass
    writer.close()
    print(f"Training finished. Checkpoints in: {checkpoints_dir}")


if __name__ == "__main__":
    train_main()


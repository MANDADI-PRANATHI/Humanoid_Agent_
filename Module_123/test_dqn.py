# test_full_pipeline_with_dqn.py
import os
import time
import argparse
import traceback
import numpy as np
import cv2
import torch

from module import PoseExtractor
from humanoid_env import HumanoidWalkEnv
from train_dqn import DQNAgent

def ensure_dirs(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def build_agent_from_env(env, num_bins=5, lr=1e-4, device=None):
    obs_dim = env.observation_space.shape[0]
    num_joints = env.mapped_theta_len
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return DQNAgent(obs_dim=obs_dim, num_joints=num_joints, num_bins=num_bins, device=device)

def safe_reset_env(env, initial_pose):
    try:
        out = env.reset(initial_pose=initial_pose)
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out
    try:
        obs = env._get_observation()
    except Exception:
        obs = np.zeros(env.observation_space.shape, dtype=np.float32)
    return obs, {}

def train_loop(env, agent, initial_pose, episodes=10, steps_per_episode=150, batch_size=64, render=False, save_path="./dqn_saved.pth"):
    ensure_dirs(save_path)
    replay = agent.buffer
    logs = []

    for ep in range(1, episodes+1):
        obs, info = safe_reset_env(env, initial_pose)
        total_reward, steps = 0.0, 0

        for step in range(steps_per_episode):
            if render and env.gui:
                time.sleep(0.05)

            action_bins = agent.select_action(obs)
            action = agent.bins_to_action(action_bins)

            try:
                next_obs, reward, terminated, truncated, info = env.step(action)
            except TypeError:
                tmp = env.step(action)
                if len(tmp) == 4:
                    next_obs, reward, terminated, info = tmp
                    truncated = False
                else:
                    raise

            replay.push(obs, action_bins.astype(np.int32), float(reward), next_obs, float(terminated or truncated))
            loss = agent.train_step(batch_size=batch_size)

            obs = next_obs
            total_reward += float(reward)
            steps += 1

            if terminated or truncated:
                break

        agent.update_target()
        print(f"[Episode {ep}] reward={total_reward:.3f} steps={steps} replay_size={len(replay)} eps={agent.eps:.3f} loss={loss if loss else 'N/A'}")
        logs.append((ep, total_reward, steps, len(replay)))

        if ep % 50 == 0 or ep == episodes:
            try:
                torch.save(agent.q.state_dict(), save_path)
                print(f" -> Saved network to {save_path}")
            except Exception as e:
                print("Could not save model:", e)

    return logs

def evaluate_greedy(env, agent, initial_pose, rollouts=2, max_steps=500, render=False):
    prev_eps = agent.eps
    agent.eps = 0.0
    results = []

    for r in range(rollouts):
        obs, info = safe_reset_env(env, initial_pose)
        total_reward, fell = 0.0, False

        for step in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q = agent.q(obs_t)[0]
                action_bins = torch.argmax(q, dim=1).cpu().numpy()
            action = agent.bins_to_action(action_bins)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            obs = next_obs
            if terminated or truncated:
                fell = terminated
                break
            if render and env.gui:
                time.sleep(env.time_step)

        results.append((total_reward, fell))
        print(f"Eval {r}: total_reward={total_reward:.3f} fell={fell}")

    agent.eps = prev_eps
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, default="./models/")
    parser.add_argument("--image_path", type=str, default="./images/test8.jpg")
    parser.add_argument("--urdf_path", type=str, default="./assets/humanoid_10theta.urdf")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="./dqn_humanoid.pth")
    parser.add_argument("--no_gui", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.image_path) or not os.path.exists(args.urdf_path):
        print("Image or URDF not found."); return

    try:
        extractor = PoseExtractor(model_folder=args.model_folder)
    except Exception as e:
        print("Failed to init PoseExtractor:", e); traceback.print_exc(); return

    res = extractor.get_initial_pose(args.image_path, return_drawn=True)
    if res is None or res.theta_init_vector is None:
        print("Failed to get initial pose."); return

    theta = np.array(res.theta_init_vector, dtype=np.float32)
    if np.mean(np.abs(theta)) > 6.5:
        theta = np.deg2rad(theta)
        print("Converted theta from degrees to radians.")

    joint_map = {i: name for i, name in enumerate([
        "right_knee","left_knee","right_hip_pitch","left_hip_pitch",
        "right_ankle","left_ankle","right_shoulder_pitch","left_shoulder_pitch",
        "right_elbow","left_elbow","right_wrist","left_wrist",
        "right_hip_roll","left_hip_roll","right_shoulder_roll","left_shoulder_roll"
    ])}

    env = HumanoidWalkEnv(urdf_path=args.urdf_path, joint_map=joint_map, gui=not args.no_gui)
    obs0, _ = safe_reset_env(env, initial_pose=theta)
    agent = build_agent_from_env(env, num_bins=args.bins)

    print(f"Starting training for {args.episodes} episodes...")
    logs = train_loop(env, agent, initial_pose=theta, episodes=args.episodes, steps_per_episode=args.steps, batch_size=64, render=args.render)
    print("Training finished.")

    print("Evaluating greedy policy...")
    eval_res = evaluate_greedy(env, agent, initial_pose=theta, rollouts=2, max_steps= 100, render=args.render)
    env.close()

if __name__ == "__main__":
    main()

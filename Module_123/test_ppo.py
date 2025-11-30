
import argparse
import os
import sys
import subprocess
import time
import glob
import numpy as np
import torch
import cv2

from module import PoseExtractor
from humanoid_env import HumanoidWalkEnv
from agent_ppo import PPOAgent


def make_arg_parser():
    parser = argparse.ArgumentParser(description="Test PPO with image-initialized pose and run training automatically")
    parser.add_argument("--model_folder", type=str, default="./models/", help="OpenPose model folder")
    parser.add_argument("--image_path", type=str, default="./images/test8.jpg", help="Image for initial pose")
    parser.add_argument("--urdf_path", type=str, default="./assets/humanoid_10theta.urdf", help="URDF path")
    parser.add_argument("--policy_path", type=str, default=None, help="If provided, skip training and use this policy file")
    parser.add_argument("--train_args", type=str, default="", help="Extra args to pass to training script (quoted string)")
    return parser


JOINT_MAP = {
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
    15: "left_shoulder_roll"
}


def clamp_to_limits(env, theta):
    t = np.array(theta, dtype=np.float32).copy()
    for t_idx, j_idx in env.joint_map.items():
        if t_idx >= len(t):
            continue
        lo, hi = env.joint_limits.get(j_idx, (None, None))
        if lo is None or hi is None:
            continue
        t[t_idx] = max(lo, min(hi, t[t_idx]))
    return t


def find_latest_best_checkpoint(base_dir="."):
    # Search checkpoints/*/best.pt and return the most-recent run dir
    candidates = glob.glob(os.path.join(base_dir, "checkpoints", "*", "best.pt"))
    if not candidates:
        return None
    # choose by file mtime
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


if __name__ == "__main__":
    args = make_arg_parser().parse_args()

    print("Using:")
    print("  model_folder:", args.model_folder)
    print("  image_path:", args.image_path)
    print("  urdf_path:", args.urdf_path)
    print("  policy_path (optional):", args.policy_path)

    # 1) Extract pose from image
    extractor = PoseExtractor(model_folder=args.model_folder)
    res = extractor.get_initial_pose(args.image_path, return_drawn=True)

    # save visualization
    os.makedirs("outputs", exist_ok=True)
    vis_path = os.path.join("outputs", "pose_debug_test.jpg")
    if hasattr(res, "image") and res.image is not None:
        cv2.imwrite(vis_path, res.image)
        print("Saved pose visualization to:", vis_path)

    theta_vec = None
    if res and res.theta_init_vector is not None:
        theta_vec = np.array(res.theta_init_vector, dtype=np.float32)
        if np.mean(np.abs(theta_vec)) > 6.5:
            theta_vec = np.deg2rad(theta_vec)
            print("Converted theta from degrees to radians.")

    # 2) If policy_path not provided, run trainer (Option A)
    policy_file = None
    if args.policy_path:
        policy_file = args.policy_path
        print("Using provided policy file:", policy_file)
    else:
        print("No policy_path provided — running training script now (this may take a while)...")
        # call python ppo_train_humanoid.py with urdf_path argument and any user extras
        cmd = [sys.executable, "ppo_train_humanoid.py", "--urdf_path", args.urdf_path]
        if args.train_args:
            cmd.extend(args.train_args.strip().split())
        print("Calling:", " ".join(cmd))
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            raise RuntimeError("Training script failed (non-zero exit). Check stdout/stderr for details.")
        # find most recent best.pt
        best = find_latest_best_checkpoint(base_dir=".")
        if best is None:
            raise FileNotFoundError("No checkpoints/*/best.pt found after training.")
        policy_file = best
        print("Found trained policy:", policy_file)

    # 3) Create env and reset with extracted pose (GUI on)
    env = HumanoidWalkEnv(urdf_path=args.urdf_path, joint_map=JOINT_MAP, gui=True)

    if theta_vec is not None:
        theta_vec = clamp_to_limits(env, theta_vec)
        obs, info = env.reset(initial_pose=theta_vec)
    else:
        obs, info = env.reset()

    # 4) Load policy and run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = PPOAgent(obs_dim, act_dim).to(device)
    policy.load_state_dict(torch.load(policy_file, map_location=device))
    policy.eval()

    print("Policy loaded. Running policy in GUI. Close window or CTRL+C to stop.")
    try:
        while True:
            with torch.no_grad():
                action, _, _, _ = policy.get_action_and_value(torch.tensor([obs], dtype=torch.float32, device=device))
            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, term, trunc, info = env.step(action_np)
            if term or trunc:
                obs, info = env.reset(initial_pose=theta_vec if theta_vec is not None else None)
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        print("Exiting.")
    finally:
        try:
            env.close()
        except Exception:
            pass

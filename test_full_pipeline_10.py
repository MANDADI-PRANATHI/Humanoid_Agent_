# test_full_pipeline_10_instrumented.py
import os
import time
import math
import traceback
import numpy as np
import pybullet as p

from pose_init.module1 import PoseExtractor
from sim_env.humanoid_env import HumanoidWalkEnv

MODEL_FOLDER = "/home/prana/openpose/models/"   # change if different
IMAGE_PATH = "test.jpg"                         # put example image here
URDF_PATH = os.path.join("assets", "humanoid_10theta.urdf")

def clamp_to_limits(env, theta_vector):
    """Clamp theta values to the env's mapped joint limits (if available)."""
    t = np.array(theta_vector, dtype=np.float64).copy()
    for t_idx, joint_idx in env.joint_map.items():
        if t_idx >= len(t): 
            continue
        lim = env.joint_limits.get(joint_idx, None)
        if lim is None:
            # no finite limits known
            continue
        lo, hi = lim
        if lo is None or hi is None:
            continue
        # clamp
        t[t_idx] = float(max(lo, min(hi, t[t_idx])))
    return t

def print_mapping_and_limits(env):
    print("=== Joint mapping (theta_index -> joint_index -> joint_name) ===")
    # build reverse name map
    rev = {v: k for k, v in env.joint_name_to_index.items()}
    for theta_idx, joint_idx in sorted(env.joint_map.items()):
        jname = env.joint_name_to_index.keys()
        # find name safely
        jname = None
        for name, idx in env.joint_name_to_index.items():
            if idx == joint_idx:
                jname = name; break
        lim = env.joint_limits.get(joint_idx, None)
        print(f" theta[{theta_idx}] -> joint_index {joint_idx} name='{jname}' limits={lim}")
    print("===============================================================")

def main():
    extractor = PoseExtractor(model_folder=MODEL_FOLDER)
    res = extractor.get_initial_pose(IMAGE_PATH, return_drawn=False)
    theta = np.array(res.theta_init_vector, dtype=np.float64)
    print("raw theta vector (len={}):\n{}".format(len(theta), theta))

    # assume PoseExtractor might give degrees; convert to radians if values seem large
    if np.mean(np.abs(theta)) > 6.5:  # average > ~374 deg -> probably degrees
        print("Converting theta from degrees to radians (mean magnitude too large).")
        theta = np.deg2rad(theta)

    # map theta indices -> URDF joint names (must match URDF)
    joint_map = {
        0: "right_knee",
        1: "left_knee",
        2: "right_hip_pitch",
        3: "left_hip_pitch",
        4: "right_ankle",
        5: "left_ankle",
        6: "right_shoulder_pitch",
        7: "left_shoulder_pitch",
        8: "right_elbow",
        9: "left_elbow"
    }

    # create env with GUI so we can see it
    env = HumanoidWalkEnv(urdf_path=URDF_PATH, gui=True, joint_map=joint_map, smoothing_steps=12)

    try:
        # show mapping & limits
        print_mapping_and_limits(env)

        # clamp theta to limits reported by env (use env.joint_map indices)
        theta_clamped = clamp_to_limits(env, theta)
        print("theta after clamping (len={}):\n{}".format(len(theta_clamped), theta_clamped))

        # suggestion: for debugging start with smaller/safer targets
        print("Using settle_steps=200 for safer initialization...")
        obs = env.reset(initial_pose=theta_clamped, settle_steps=200)
        print("Reset done. Observation length:", len(obs))

        # print base position and contact points right after reset
        base_pos, base_ori = p.getBasePositionAndOrientation(env.robot_id)
        print("Base position after reset:", base_pos)
        contacts = p.getContactPoints(env.robot_id)
        print(f"Number of contact points after reset: {len(contacts)}")
        if len(contacts) > 0:
            print("Sample contacts (first 10):")
            for c in contacts[:10]:
                print(c)

        # run a controlled loop using zero actions first (do not command large random moves)
        n_steps = 480
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
        for i in range(n_steps):
            obs, reward, done, _, _ = env.step(zero_action)
            if i % 60 == 0:
                base_pos, _ = p.getBasePositionAndOrientation(env.robot_id)
                contacts = p.getContactPoints(env.robot_id)
                print(f"step {i} | reward={reward:.4f} | base_z={base_pos[2]:.3f} | contacts={len(contacts)}")
            if done:
                print("Env reported done at step", i)
                break

        print("Main loop finished without exception.")

    except Exception as e:
        print("Exception caught in main():")
        traceback.print_exc()
        # keep GUI open to inspect the moment of exception
        input("Exception occurred — press Enter to quit and close GUI...")

    else:
        # keep GUI open for manual inspection before closing
        input("Finished running. Press Enter to close GUI and exit...")

    finally:
        try:
            env.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()

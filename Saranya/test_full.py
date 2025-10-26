# test_full_pipeline_10_instrumented.py
import os
import time
import math
import traceback
import numpy as np
import pybullet as p

from module import PoseExtractor
from humanoid_env import HumanoidWalkEnv

MODEL_FOLDER = "/home/prana/openpose/models/"   # change if different
IMAGE_PATH = "../test4.png"                         # put example image here
URDF_PATH = os.path.join("../assets", "humanoid_10theta.urdf")

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

def debug_pose_application(env, theta):
    print("\n--- DEBUG: theta -> joint targets check ---")
    for t_idx, joint_idx in sorted(env.joint_map.items()):
        # find joint name in env
        jname = None
        for nm, idx in env.joint_name_to_index.items():
            if idx == joint_idx:
                jname = nm; break
        val = float(theta[t_idx]) if t_idx < len(theta) else None
        lim = env.joint_limits.get(joint_idx, None)
        clamped = None
        if val is not None and lim is not None:
            lo, hi = lim
            if lo is not None and hi is not None:
                clamped = max(lo, min(hi, val))
        print(f" theta[{t_idx:2d}] -> joint_idx {joint_idx:2d} name='{jname}'  value={val}  limits={lim}  clamped={clamped}")
    print("--- end debug ---\n")


def main():
    extractor = PoseExtractor(model_folder=MODEL_FOLDER)
    res = extractor.get_initial_pose(IMAGE_PATH, return_drawn=False)
    theta = np.array(res.theta_init_vector_3d, dtype=np.float64)
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
    print_mapping_and_limits(env)
    print("theta length:", len(theta), "mapped indices:", sorted(joint_map.keys()))
    debug_pose_application(env, theta)
    obs = env.reset(initial_pose=theta)   # <--- important: set the initial pose here
    print("Applied initial pose; observation length:", obs.shape)
    print("\nGUI is active — press Ctrl+C or close the window to exit.")
    try:
        while True:
            #p.stepSimulation()
            time.sleep(1. / 240.)
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
        p.disconnect()
if __name__ == "__main__":
    main()

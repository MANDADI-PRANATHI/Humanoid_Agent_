# test_full_pipeline_10_instrumented.py
import os
import time
import math
import traceback
import numpy as np
import pybullet as p
import argparse
from module import PoseExtractor
from humanoid_env import HumanoidWalkEnv
import cv2


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
    parser = argparse.ArgumentParser(description="Run humanoid pose simulation.")
    parser.add_argument(
        "--model_folder",
        type=str,
        default="./models/",
        help="Path to the model folder (default: ./models/)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./images/test8.jpg",
        help="Path to the input image (default: ./images/test8.jpg)"
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        default="./assets/humanoid_10theta.urdf",
        help="Path to the humanoid URDF file (default: ./assets/humanoid_10theta.urdf)"
    )

    args = parser.parse_args()

    MODEL_FOLDER = args.model_folder
    IMAGE_PATH = args.image_path
    URDF_PATH = args.urdf_path

    print(f"\nUsing paths:")
    print(f"  MODEL_FOLDER = {MODEL_FOLDER}")
    print(f"  IMAGE_PATH   = {IMAGE_PATH}")
    print(f"  URDF_PATH    = {URDF_PATH}")    
    extractor = PoseExtractor(model_folder=MODEL_FOLDER)
    res = extractor.get_initial_pose(IMAGE_PATH, return_drawn=True)
    # --- Save the drawn pose image ---
    if hasattr(res, "image") and res.image is not None:
        from datetime import datetime

        os.makedirs("outputs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("outputs", f"pose_debug_{ts}.jpg")
        cv2.imwrite(out_path, res.image)
        print(f"✅ Saved pose visualization image to: {out_path}")
    else:
        print("⚠️ No drawn pose image found in result — check get_initial_pose output.")

    theta = np.array(res.theta_init_vector_3d, dtype=np.float64)
    print("raw theta vector (len={}):\n{}".format(len(theta), theta))

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
        9: "left_elbow",
        10: 'right_wrist',
        11: 'left_wrist'
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

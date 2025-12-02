# test_full_pipeline_10_instrumented.py

import os
import time
import math
import traceback
import numpy as np
import pybullet as p
import argparse
import cv2

from module import PoseExtractor
from humanoid_env import HumanoidWalkEnv
from Biped_Pybullet.main import walk
from train_dqn import DQNAgent


# ---------------------------
# Utility: clamp to joint limits
# ---------------------------
def clamp_to_limits(env, theta_vector):
    t = np.array(theta_vector, dtype=np.float64).copy()
    for t_idx, joint_idx in env.joint_map.items():
        if t_idx >= len(t):
            continue
        lim = env.joint_limits.get(joint_idx, None)
        if lim is None:
            continue
        lo, hi = lim
        if lo is None or hi is None:
            continue
        t[t_idx] = float(max(lo, min(hi, t[t_idx])))
    return t


# ---------------------------
# Debug helpers
# ---------------------------
def print_mapping_and_limits(env):
    print("=== Joint mapping (theta_index -> joint_index -> joint_name) ===")
    for theta_idx, joint_idx in sorted(env.joint_map.items()):
        jname = None
        for nm, idx in env.joint_name_to_index.items():
            if idx == joint_idx:
                jname = nm
                break
        lim = env.joint_limits.get(joint_idx, None)
        print(f" theta[{theta_idx}] -> joint_index {joint_idx} name='{jname}' limits={lim}")
    print("===============================================================")


def debug_pose_application(env, theta):
    print("\n--- DEBUG: theta -> joint targets check ---")
    for t_idx, joint_idx in sorted(env.joint_map.items()):
        jname = None
        for nm, idx in env.joint_name_to_index.items():
            if idx == joint_idx:
                jname = nm
                break
        val = float(theta[t_idx]) if t_idx < len(theta) else None
        lim = env.joint_limits.get(joint_idx, None)
        clamped = None
        if val is not None and lim is not None:
            lo, hi = lim
            if lo is not None and hi is not None:
                clamped = max(lo, min(hi, val))
        print(f" theta[{t_idx:2d}] -> joint {joint_idx:2d} name={jname} value={val} clamped={clamped}")
    print("--- end debug ---\n")


# ---------------------------
# FALL HANDLER (KEEP GUI OPEN)
# ---------------------------
def handle_fall(env, message):
    """
    Handle a fall without disconnecting the PyBullet GUI.
    Strategy:
      1. Try env.close(keep_connection=True) if available.
      2. If that fails, call p.resetSimulation() directly (keeps GUI).
      3. Then call the walking controller (walk) which will reuse the GUI and load the biped.
    """
    print("\n❗ FALL DETECTED:", message)

    # Attempt env-level graceful close that keeps the connection
    try:
        # prefer env's close if it supports keep_connection
        env.close(keep_connection=True)
        print("Environment closed (kept connection).")
    except TypeError:
        # env.close doesn't accept keep_connection — fallback to p.resetSimulation
        print("env.close() has no keep_connection option; falling back to p.resetSimulation().")
        try:
            # remove existing objects but keep GUI
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            print("p.resetSimulation() executed successfully.")
        except Exception as e:
            print("Error during p.resetSimulation():", e)
            traceback.print_exc()
    except Exception as e:
        print("Exception while trying env.close(keep_connection=True):", e)
        traceback.print_exc()
        # as a last resort, try resetting simulation directly
        try:
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            print("p.resetSimulation() executed successfully (after exception).")
        except Exception as e2:
            print("Also failed to reset simulation:", e2)
            traceback.print_exc()

    # Now launch the walking controller which reuses the same GUI connection
    try:
        print("Launching Biped walking controller in the same GUI...")
        walk()  # this should reuse the existing connection and load the biped
    except Exception as e:
        print("Error while starting walking controller:")
        print(e)
        traceback.print_exc()

    # Exit current script after handing over to the walker
    print("Exiting current script (walking controller now running).")
    exit(0)


# ---------------------------
# MAIN
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Run humanoid pose simulation.")
    parser.add_argument("--model_folder", default="./models/")
    parser.add_argument("--image_path", default="./images/test8.jpg")
    parser.add_argument("--urdf_path", default="./assets/humanoid_10theta.urdf")
    args = parser.parse_args()

    MODEL_FOLDER = args.model_folder
    IMAGE_PATH = args.image_path
    URDF_PATH = args.urdf_path

    print(f"\nUsing paths:\n  MODEL_FOLDER={MODEL_FOLDER}\n  IMAGE_PATH={IMAGE_PATH}\n  URDF_PATH={URDF_PATH}")

    # Extract pose from image
    extractor = PoseExtractor(model_folder=MODEL_FOLDER)
    res = extractor.get_initial_pose(IMAGE_PATH, return_drawn=True)

    # Save pose visualization
    if hasattr(res, "image") and res.image is not None:
        os.makedirs("outputs", exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"outputs/pose_debug_{ts}.jpg"
        cv2.imwrite(out_path, res.image)
        print(f"✓ Saved pose image to: {out_path}")

    theta = np.array(res.theta_init_vector, dtype=np.float64)
    print("\nRaw θ vector:", theta)

    # Convert deg → rad if needed
    if np.mean(np.abs(theta)) > 6.5:
        theta = np.deg2rad(theta)
        print("Converted degrees → radians.")

    # ---------------------------
    # Pose → URDF joint mapping
    # ---------------------------
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
        10: "right_wrist",
        11: "left_wrist",

        12: "right_hip_roll",
        13: "left_hip_roll",
        14: "right_shoulder_roll",
        15: "left_shoulder_roll"
    }

    # Start environment
    env = HumanoidWalkEnv(urdf_path=URDF_PATH, joint_map=joint_map, gui=True)
    obs, info = env.reset()

    print_mapping_and_limits(env)
    debug_pose_application(env, theta)

    # ------------------------------------
    # APPLY INITIAL POSE
    # ------------------------------------
    obs, info = env.reset(initial_pose=theta)
    if info.get("fell", False):
        handle_fall(env, "Fell immediately after applying pose")

    # ------------------------------------
    # ONE STEP FORWARD
    # ------------------------------------
    action = np.zeros(env.mapped_theta_len, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        handle_fall(env, "Fell during first step")

    # ------------------------------------
    # STABILITY TEST (500 steps)
    # ------------------------------------
    print("\n→ Stability test (500 steps)")
    for i in range(500):
        obs, r, done, trunc, info = env.step(action)
        if done:
            handle_fall(env, f"Fell during stability test at step {i}")

    print("✓ Stable for 500 steps.")

    # ------------------------------------
    # SINGLE JOINT TEST
    # ------------------------------------
    print("\n→ Testing each joint...")

    test_indices = list(env.joint_map.keys())[:10]
    for idx in test_indices:
        print(f"  Testing joint index {idx}")
        a = np.zeros(env.mapped_theta_len, dtype=np.float32)
        a[idx] = 0.3

        for _ in range(40):
            obs, r, done, trunc, info = env.step(a)
            if done:
                handle_fall(env, f"Fell while actuating joint {idx}")

        time.sleep(0.1)

    print("\nAll tests finished. Robot stable.")

    # ------------------------------------
    # KEEP GUI OPEN
    # ------------------------------------
    print("\nGUI active. Close window or Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1 / 240.)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        # close without disconnect by default (to keep consistent behavior) if supported
        try:
            env.close(keep_connection=True)
        except TypeError:
            env.close()
        except Exception:
            env.close()


if __name__ == "__main__":
    main()

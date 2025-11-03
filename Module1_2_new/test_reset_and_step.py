# test_reset_and_step.py
import numpy as np
from humanoid_env import HumanoidWalkEnv

URDF_PATH = "../assets/humanoid_10theta.urdf"   # change if needed

def run_test(gui=True):
    env = HumanoidWalkEnv(urdf_path=URDF_PATH, gui=gui)
    # create a simple theta vector (zeros or small offset)
    theta = np.zeros(env.mapped_theta_len, dtype=np.float64) if env.mapped_theta_len > 0 else np.zeros((0,), dtype=np.float64)

    # 1) reset with initial pose
    obs, info = env.reset(initial_pose=theta)
    print("After reset:")
    print("  step_count:", env.step_count)          # expected 0
    print("  ref_angles shape:", None if env.ref_angles is None else env.ref_angles.shape)
    print("  ref_angles (first 8):", env.ref_angles[:8] if env.ref_angles.size >= 8 else env.ref_angles)
    print("  last_action:", env.last_action)

    # 2) take one zero action step
    if env.mapped_theta_len > 0:
        action = np.zeros(env.mapped_theta_len, dtype=np.float32)
    else:
        action = np.zeros((0,), dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)
    print("\nAfter one step():")
    print("  step_count:", env.step_count)          # expected 1
    print("  last_action (first 8):", env.last_action[:8] if env.last_action.size >= 8 else env.last_action)
    print("  reward:", reward)
    print("  terminated:", terminated, " truncated:", truncated)
    print("  info keys:", list(info.keys()))

    # 3) take a few more steps to verify increment and stable obs
    for i in range(3):
        obs, reward, terminated, truncated, info = env.step(action)
    print("\nAfter 4 total steps:")
    print("  step_count:", env.step_count)          # expected 4
    env.close()

if __name__ == "__main__":
    run_test(gui=True)

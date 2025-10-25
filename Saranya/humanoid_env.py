# sim_env/humanoid_env.py

import os
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import time
import math
import warnings

class HumanoidWalkEnv(gym.Env):
    def __init__(self, urdf_path: str, gui: bool = False, timestep: float = 1/240.0,
                 joint_map: dict | None = None, smoothing_steps: int = 12):
        self.urdf_path = urdf_path
        self.gui = gui
        self.timestep = timestep
        self._raw_joint_map = joint_map
        self.smoothing_steps = int(smoothing_steps)

        self.physics_client = None
        self.robot_id = None
        self.joint_name_to_index = {}
        self.controllable_joints = []
        self.joint_limits = {}

        # Default standing pose (easy to tweak)
        self.default_pose = {
            "right_hip_roll": 0.05,
            "left_hip_roll": -0.05,
            "right_knee": 0.15,
            "left_knee": 0.15,
            "right_ankle_pitch": 0.1,
            "left_ankle_pitch": 0.1,
            "right_ankle_roll": -0.05,
            "left_ankle_roll": 0.05
        }

        # Start physics
        self._start_physics()
        # Load robot once to build joint maps
        self._load_robot_and_build_maps(initial_load=True)

        n_ctrl = len(self.controllable_joints)
        self.action_space = spaces.Box(low=-3.14, high=3.14, shape=(n_ctrl,), dtype=np.float32)
        obs_dim = n_ctrl * 2 + 12  # pos(n) + vel(n) + base_pos(3) + base_euler(3) + base_lin(3)+base_ang(3) = 2n+12
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32)

    def _start_physics(self):
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.timestep)
        p.setGravity(0,0,-9.81)

    def _safe_float(self, x):
        try:
            return float(x)
        except Exception:
            return None

    def _is_limit_finite(self, lo, hi):
        if lo is None or hi is None:
            return False
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return False
        if abs(lo) > 1e6 or abs(hi) > 1e6:
            return False
        return True

    def _load_robot_and_build_maps(self, initial_load: bool = False):
        p.resetSimulation()
        p.setGravity(0,0,-9.81)
        p.loadURDF("plane.urdf")
        start_pos = [0,0,0.5]
        start_ori = p.getQuaternionFromEuler([0,0,0])
        self.robot_id = p.loadURDF(self.urdf_path, start_pos, start_ori,
                                   useFixedBase=False, flags=p.URDF_USE_SELF_COLLISION)

        self.joint_name_to_index = {}
        self.controllable_joints = []
        self.joint_limits = {}

        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            jname = info[1].decode('utf-8')
            jtype = info[2]
            lower = self._safe_float(info[8])
            upper = self._safe_float(info[9])
            self.joint_name_to_index[jname] = i
            if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC, p.JOINT_SPHERICAL):
                self.controllable_joints.append(i)
                if self._is_limit_finite(lower, upper):
                    self.joint_limits[i] = (lower, upper)
                else:
                    self.joint_limits[i] = None

        # convert joint_map to indices
        self.joint_map = {}
        if self._raw_joint_map is not None:
            for t_idx, j in self._raw_joint_map.items():
                if isinstance(j, str):
                    if j in self.joint_name_to_index:
                        self.joint_map[int(t_idx)] = self.joint_name_to_index[j]
                    else:
                        raise ValueError(f"joint_map refers to unknown joint name '{j}'")
                else:
                    self.joint_map[int(t_idx)] = int(j)
        else:
            for idx, ji in enumerate(self.controllable_joints[:10]):
                self.joint_map[idx] = ji

        if initial_load:
            pass

    def _clamp_to_limits(self, joint_index, value):
        lim = self.joint_limits.get(joint_index, None)
        if lim is None:
            return value
        lo, hi = lim
        if lo is None or hi is None:
            return value
        return max(lo, min(hi, value))

    def _read_positions(self):
        pos = []
        for j in self.controllable_joints:
            s = p.getJointState(self.robot_id, j)
            pos.append(s[0])
        return np.array(pos, dtype=np.float32)

    def reset(self, initial_pose: np.ndarray | None = None, settle_steps: int = 30):
        p.resetSimulation()
        p.setGravity(0,0,-9.81)
        p.loadURDF("plane.urdf")
        start_pos = [0,0,1.2]
        start_ori = p.getQuaternionFromEuler([0,0,0])
        self.robot_id = p.loadURDF(self.urdf_path, start_pos, start_ori,
                                   useFixedBase=False, flags=p.URDF_USE_SELF_COLLISION)

        # rebuild maps/limits
        self.joint_name_to_index = {}
        self.controllable_joints = []
        self.joint_limits = {}
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            jname = info[1].decode('utf-8')
            jtype = info[2]
            lower = self._safe_float(info[8])
            upper = self._safe_float(info[9])
            self.joint_name_to_index[jname] = i
            if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC, p.JOINT_SPHERICAL):
                self.controllable_joints.append(i)
                if self._is_limit_finite(lower, upper):
                    self.joint_limits[i] = (lower, upper)
                else:
                    self.joint_limits[i] = None

        # validate joint_map
        for t_idx, jidx in list(self.joint_map.items()):
            if jidx not in self.controllable_joints:
                warnings.warn(f"joint_map target index {jidx} not in controllable joints; removing mapping")
                del self.joint_map[t_idx]

        # read current positions
        current = self._read_positions()
        target = current.copy()

        # apply user-provided initial_pose
        if initial_pose is not None:
            for t_idx, joint_idx in self.joint_map.items():
                if t_idx >= len(initial_pose):
                    continue
                val = float(initial_pose[t_idx])
                val = self._clamp_to_limits(joint_idx, val)
                try:
                    pos_idx = self.controllable_joints.index(joint_idx)
                    target[pos_idx] = val
                except ValueError:
                    continue

        # --- Set default standing pose ---
        joints = self.joint_name_to_index
        def set_if_present(name, val):
            idx = joints.get(name)
            if idx is not None and idx in self.controllable_joints:
                target[self.controllable_joints.index(idx)] = val

        for name, val in self.default_pose.items():
            set_if_present(name, val)

        # set joints instantly
        for idx, j in enumerate(self.controllable_joints):
            p.resetJointState(self.robot_id, j, target[idx], targetVelocity=0.0)

        # allow physics to settle
        for _ in range(settle_steps):
            p.stepSimulation()
            if self.gui:
                time.sleep(self.timestep)

        return self._get_observation()

    def _get_observation(self):
        pos, vel = [], []
        for j in self.controllable_joints:
            s = p.getJointState(self.robot_id, j)
            pos.append(s[0])
            vel.append(s[1])
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot_id)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)
        base_euler = p.getEulerFromQuaternion(base_ori)
        obs = np.concatenate([
            np.array(pos,dtype=np.float32),
            np.array(vel,dtype=np.float32),
            np.array(base_pos,dtype=np.float32),
            np.array(base_euler,dtype=np.float32),
            np.array(base_lin_vel,dtype=np.float32),
            np.array(base_ang_vel,dtype=np.float32)
        ], axis=0)
        return obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        n = len(self.controllable_joints)
        if action.shape[0] != n:
            if action.shape[0] < n:
                a = np.zeros(n, dtype=np.float32)
                a[:action.shape[0]] = action
                action = a
            else:
                action = action[:n]
        p.setJointMotorControlArray(self.robot_id,
                                    jointIndices=self.controllable_joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=action.tolist(),
                                    positionGains=[0.7] * n)
        p.stepSimulation()
        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_fallen()
        return obs, reward, done, False, {}

    def _compute_reward(self):
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot_id)
        base_lin_vel, _ = p.getBaseVelocity(self.robot_id)
        vx = base_lin_vel[0]
        roll, pitch, _ = p.getEulerFromQuaternion(base_ori)
        upright_penalty = abs(roll) + abs(pitch)
        return float(vx - 0.5 * upright_penalty)

    def _check_fallen(self):
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        return base_pos[2] < 0.4

    def close(self):
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except Exception:
                pass
            self.physics_client = None


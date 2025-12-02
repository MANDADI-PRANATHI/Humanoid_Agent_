from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium.spaces import Box


class HumanoidWalkEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240}

    def __init__(
        self,
        urdf_path: str,
        gui: bool = True,
        joint_map: Optional[Dict[int, str]] = None,
        smoothing_steps: int = 0,
        time_step: float = 1.0 / 240.0,
        base_link_index: int = -1,
        start_height: float = 0.6,
    ):
        self.urdf_path = urdf_path
        self.gui = gui
        self.smoothing_steps = int(max(0, smoothing_steps))
        self.time_step = float(time_step)
        self.base_link_index = base_link_index

        self.start_height = float(start_height)
        self.start_pos = [0.0, 0.0, self.start_height]
        self.start_ori = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

        self.physics_client = None
        self.robot_id = None

        self.sim_substeps = 4
        self.kp = 0.7
        self.kd = 0.1
        self.max_episode_steps = 2000
        self.step_count = 0

        self.actions_are_normalized = True
        self.max_delta = 0.6

        self.fall_height = 0.5
        self.tilt_threshold = 1.2
        self.w_vel = 1.0
        self.w_upright = 2.0
        self.w_energy = 1e-3

        self.ref_angles = None
        self.last_action = None
        self.foot_link_indices = []

        self.user_joint_map = joint_map if joint_map is not None else {}

        self.joint_name_to_index: Dict[str, int] = {}
        self.joint_limits: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
        self.joint_map: Dict[int, int] = {}

        self.mapped_theta_len = max(self.user_joint_map.keys()) + 1 if self.user_joint_map else 0

        self.action_space = None
        self.observation_space = None

        self._connect()
        self._load_urdf_and_build_mapping()
        self._build_spaces()

    # ------------------
    # PyBullet helpers
    # ------------------
    def _connect(self):
        if self.physics_client is not None:
            return
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(numSolverIterations=10)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)

    def _load_urdf_and_build_mapping(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # ------------------------------
        # ADD PLANE
        # ------------------------------
        p.loadURDF("plane.urdf")

        flags = p.URDF_USE_INERTIA_FROM_FILE
        self.start_pos[2] = max(self.start_pos[2], 1.0)
        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=self.start_pos,
            baseOrientation=self.start_ori,
            useFixedBase=False,
            flags=flags,
        )

        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        print(f"[DEBUG] Robot loaded at height: {base_pos[2]:.3f}")

        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            name = info[1].decode("utf-8")
            joint_type = info[2]
            lo = hi = None
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                lo = info[8]
                hi = info[9]
            self.joint_name_to_index[name] = i
            self.joint_limits[i] = (lo, hi)

        self.joint_map = {}
        for t_idx, jname in self.user_joint_map.items():
            if jname not in self.joint_name_to_index:
                raise KeyError(f"Provided joint name '{jname}' not found in URDF.")
            self.joint_map[int(t_idx)] = self.joint_name_to_index[jname]

        if self.joint_map:
            self.mapped_theta_len = max(self.joint_map.keys()) + 1
        else:
            self.mapped_theta_len = 0

        self.joint_limits = {j: self.joint_limits[j] for j in set(self.joint_map.values())}

        self.foot_link_indices = []
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            link_name = info[12].decode("utf-8") if isinstance(info[12], bytes) else str(info[12])
            if any(k in link_name.lower() for k in ("foot", "toe")):
                self.foot_link_indices.append(i)

        print(f"[DEBUG] Foot link indices detected: {self.foot_link_indices}")

    # ------------------
    # Spaces & obs
    # ------------------
    def _build_spaces(self):
        if self.mapped_theta_len > 0:
            lows = []
            highs = []
            for t_idx in range(self.mapped_theta_len):
                j_idx = self.joint_map.get(t_idx)
                lo, hi = self.joint_limits.get(j_idx, (None, None))
                if lo is None or hi is None or lo >= hi:
                    lows.append(-10.0)
                    highs.append(10.0)
                else:
                    lows.append(lo)
                    highs.append(hi)
            self.action_space = Box(np.array(lows, np.float32), np.array(highs, np.float32))
        else:
            self.action_space = Box(low=np.zeros((0,), np.float32), high=np.zeros((0,), np.float32))

        obs_low = []
        obs_high = []
        for _ in range(self.mapped_theta_len):
            obs_low.extend([-np.inf, -np.inf])
            obs_high.extend([np.inf, np.inf])
        obs_low.extend([-np.inf] * 9)
        obs_high.extend([np.inf] * 9)
        self.observation_space = Box(np.array(obs_low, np.float32), np.array(obs_high, np.float32))

    def _get_observation(self) -> np.ndarray:
        parts = []
        for t_idx in range(self.mapped_theta_len):
            j_idx = self.joint_map.get(t_idx)
            st = p.getJointState(self.robot_id, j_idx)
            pos = st[0] if st else 0.0
            vel = st[1] if st else 0.0
            parts.append(float(pos))
            parts.append(float(vel))

        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_lin, base_ang = p.getBaseVelocity(self.robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)

        parts.extend([*base_pos])
        parts.extend([*base_lin])
        parts.extend([roll, pitch, yaw])
        return np.array(parts, np.float32)

    # ------------------
    # Reset / Step
    # ------------------
    def reset(self, *, initial_pose=None, seed=None, options=None):

        if seed is not None:
            np.random.seed(seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)

        # ------------------------------
        # ADD PLANE HERE ALSO
        # ------------------------------
        p.loadURDF("plane.urdf")

        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=self.start_pos,
            baseOrientation=self.start_ori,
            useFixedBase=False,
        )

        for j_idx in self.joint_name_to_index.values():
            try:
                p.resetJointState(self.robot_id, j_idx, 0.0, 0.0)
            except:
                pass

        if initial_pose is not None:
            theta = np.array(initial_pose, float)
            for t_idx, j_idx in self.joint_map.items():
                if t_idx < len(theta):
                    lo, hi = self.joint_limits.get(j_idx, (None, None))
                    if lo is not None and hi is not None:
                        theta[t_idx] = np.clip(theta[t_idx], lo, hi)

            for t_idx, j_idx in self.joint_map.items():
                if t_idx < len(theta):
                    p.resetJointState(self.robot_id, j_idx, float(theta[t_idx]), 0.0)

        self.ref_angles = np.zeros(self.mapped_theta_len, np.float32)
        for t_idx, j_idx in self.joint_map.items():
            st = p.getJointState(self.robot_id, j_idx)
            self.ref_angles[t_idx] = st[0] if st else 0.0

        self.step_count = 0
        self.last_action = np.zeros(self.mapped_theta_len, np.float32)

        obs = self._get_observation()
        return obs, {"robot_id": self.robot_id}

    def step(self, action: np.ndarray):
        if self.mapped_theta_len == 0:
            for _ in range(self.sim_substeps):
                p.stepSimulation()
            return self._get_observation(), 0.0, False, False, {}

        a = np.array(action, np.float32)
        if a.size != self.mapped_theta_len:
            tmp = np.zeros(self.mapped_theta_len, np.float32)
            tmp[: len(a)] = a[: len(tmp)]
            a = tmp

        if self.action_space is not None:
            a = np.clip(a, self.action_space.low, self.action_space.high)

        if self.actions_are_normalized:
            max_delta_arr = np.ones(self.mapped_theta_len, np.float32) * self.max_delta
            for t_idx, j_idx in self.joint_map.items():
                lo, hi = self.joint_limits.get(j_idx, (None, None))
                if lo is not None and hi is not None:
                    half_range = 0.5 * (hi - lo)
                    max_delta_arr[t_idx] = min(max_delta_arr[t_idx], half_range)
            targets = self.ref_angles + a * max_delta_arr
        else:
            targets = a

        for t_idx, j_idx in self.joint_map.items():
            p.setJointMotorControl2(
                self.robot_id,
                j_idx,
                p.POSITION_CONTROL,
                targetPosition=float(targets[t_idx]),
                positionGain=self.kp,
                velocityGain=self.kd,
                force=200.0,
            )

        for _ in range(self.sim_substeps):
            p.stepSimulation()
            if self.gui:
                time.sleep(self.time_step)

        obs = self._get_observation()

        base_lin = np.array(obs[self.mapped_theta_len * 2 + 3: self.mapped_theta_len * 2 + 6])
        yaw = float(obs[self.mapped_theta_len * 2 + 8])
        forward_vec = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        forward_vel = float(np.dot(base_lin, forward_vec))

        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        rot = p.getMatrixFromQuaternion(base_orn)
        base_up = np.array([rot[6], rot[7], rot[8]])
        upright_score = float(np.dot(base_up, np.array([0, 0, 1])))
        upright_penalty = max(0.0, 1.0 - upright_score)

        energy = 0.0
        for t_idx, j_idx in self.joint_map.items():
            st = p.getJointState(self.robot_id, j_idx)
            if st:
                torque = st[3]
                vel = st[1]
                energy += abs(torque * vel)

        energy_cost = float(energy)
        action_penalty = 0.001 * np.sum(np.abs(a))

        reward = (
            self.w_vel * forward_vel
            - self.w_upright * upright_penalty
            - self.w_energy * energy_cost
            - action_penalty
        )

        base_z = float(obs[self.mapped_theta_len * 2 + 2])
        roll = float(obs[self.mapped_theta_len * 2 + 6])
        pitch = float(obs[self.mapped_theta_len * 2 + 7])

        terminated = base_z < self.fall_height or abs(roll) > self.tilt_threshold or abs(pitch) > self.tilt_threshold
        truncated = self.step_count >= self.max_episode_steps

        contacts = p.getContactPoints(self.robot_id)
        in_contact = any(c[3] in self.foot_link_indices and c[9] > 0 for c in contacts)

        info = {
            "forward_vel": forward_vel,
            "upright": upright_score,
            "energy": energy_cost,
            "action_penalty": action_penalty,
            "step_count": self.step_count,
            "in_contact": in_contact,
        }

        self.last_action = a.copy()
        self.step_count += 1

        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self, keep_connection=False):
        """
        Close environment.
        If keep_connection=True, resetSimulation() is used to clear objects but the
        GUI connection (display) remains open so other modules can reuse it.
        If keep_connection=False, fully disconnect.
        """
        try:
            if keep_connection:
                # reset the world but keep GUI/connection open
                p.resetSimulation()
            else:
                p.disconnect()
        except Exception as e:
            print("Error during env.close():", e)
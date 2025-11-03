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
        start_height: float = 0.6,  # <-- new, default a bit above plane
    ):
        self.urdf_path = urdf_path
        self.gui = gui
        self.smoothing_steps = int(max(0, smoothing_steps))
        self.time_step = float(time_step)
        self.base_link_index = base_link_index

        # new: configurable start pose (you can supply start_height when creating env)
        self.start_height = float(start_height)
        self.start_pos = [0.0, 0.0, self.start_height]
        self.start_ori = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

        self.physics_client = None
        self.robot_id = None
        # control / sim params
        self.sim_substeps = 4             # physics micro-steps per control step
        self.kp = 0.7                     # position PD gain (used by POSITION_CONTROL)
        self.kd = 0.1
        self.max_episode_steps = 2000     # truncate after this many steps
        self.step_count = 0

        # action handling
        # if True, assume incoming actions are normalized in [-1, 1] and map to
        # reference_angle + action * max_delta; otherwise expect absolute joint positions.
        self.actions_are_normalized = True
        # per-joint maximum delta (radians) when actions_are_normalized==True
        self.max_delta = 0.6  # default for all joints; can make per-joint later

        # termination / reward weights
        self.fall_height = 0.5
        self.tilt_threshold = 1.2  # rad
        self.w_vel = 1.0
        self.w_upright = 2.0
        self.w_energy = 1e-3

        # bookkeeping
        self.ref_angles = None      # reference angles (set at reset)
        self.last_action = None
        self.foot_link_indices = []  # will be discovered in _load_urdf_and_build_mapping

        # user-provided mapping theta_idx -> joint_name
        self.user_joint_map = joint_map if joint_map is not None else {}

        # these are populated after loading URDF
        self.joint_name_to_index: Dict[str, int] = {}
        self.joint_limits: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
        # mapping theta index -> joint index (int)
        self.joint_map: Dict[int, int] = {}

        # will be set after loading
        self.mapped_theta_len = max(self.user_joint_map.keys()) + 1 if self.user_joint_map else 0

        # place-holder for action/obs spaces (recreated after load)
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
        # reset to ensure clean state
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # ✅ add a ground plane to prevent falling through
        flags = p.URDF_USE_INERTIA_FROM_FILE
        # use configured start_pos/start_ori so the robot is not spawned intersecting the plane
        self.start_pos[2] = max(self.start_pos[2], 1.0)
        self.robot_id = p.loadURDF(self.urdf_path, basePosition=self.start_pos, baseOrientation=self.start_ori, useFixedBase=False, flags=flags)
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        print(f"[DEBUG] Robot loaded at height: {base_pos[2]:.3f}")

        # build joint name -> index map and limits
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            # info[1] -> name bytes
            name = info[1].decode('utf-8')
            joint_type = info[2]
            lo = None
            hi = None
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                lo = info[8]
                hi = info[9]
            self.joint_name_to_index[name] = i
            self.joint_limits[i] = (lo, hi)

        # build resolved mapping from theta index -> joint index
        self.joint_map = {}
        for t_idx, jname in self.user_joint_map.items():
            if jname not in self.joint_name_to_index:
                raise KeyError(f"Provided joint name '{jname}' not found in URDF joints.")
            self.joint_map[int(t_idx)] = self.joint_name_to_index[jname]

        # set mapped length
        if self.joint_map:
            self.mapped_theta_len = max(self.joint_map.keys()) + 1
        else:
            self.mapped_theta_len = 0

        # store joint limits only for mapped joints for quick lookup
        self.joint_limits = {j_idx: self.joint_limits[j_idx] for j_idx in set(self.joint_map.values())}
        # --- detect foot link indices heuristically (search link names containing 'foot' or 'toe') ---
        self.foot_link_indices = []
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            link_name_bytes = info[12]  # link name bytes
            try:
                link_name = link_name_bytes.decode('utf-8')
            except Exception:
                link_name = str(link_name_bytes)
            if any(k in link_name.lower() for k in ("foot", "toe")):
                # joint i controls a link that likely represents a foot (link index == i)
                self.foot_link_indices.append(i)
        print(f"[DEBUG] Foot link indices detected: {self.foot_link_indices}")

    # ------------------
    # Spaces & obs
    # ------------------
    def _build_spaces(self):
        # (unchanged)
        if self.mapped_theta_len > 0:
            lows = []
            highs = []
            for t_idx in range(self.mapped_theta_len):
                j_idx = self.joint_map.get(t_idx, None)
                if j_idx is None:
                    lows.append(-10.0)
                    highs.append(10.0)
                else:
                    lo, hi = self.joint_limits.get(j_idx, (None, None))
                    if lo is None or hi is None or lo >= hi:
                        lows.append(-10.0)
                        highs.append(10.0)
                    else:
                        lows.append(lo)
                        highs.append(hi)
            self.action_space = Box(low=np.array(lows, dtype=np.float32), high=np.array(highs, dtype=np.float32), dtype=np.float32)
        else:
            self.action_space = Box(low=np.zeros((0,), dtype=np.float32), high=np.zeros((0,), dtype=np.float32), dtype=np.float32)

        obs_low = []
        obs_high = []
        for t_idx in range(self.mapped_theta_len):
            obs_low.extend([-np.inf, -np.inf])  # pos, vel
            obs_high.extend([np.inf, np.inf])
        obs_low.extend([-np.inf] * 9)
        obs_high.extend([np.inf] * 9)
        self.observation_space = Box(low=np.array(obs_low, dtype=np.float32), high=np.array(obs_high, dtype=np.float32), dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        parts = []
        for t_idx in range(self.mapped_theta_len):
            j_idx = self.joint_map.get(t_idx, None)
            if j_idx is None:
                parts.append(0.0)
                parts.append(0.0)
            else:
                st = p.getJointState(self.robot_id, j_idx)
                pos = st[0] if st is not None else 0.0
                vel = st[1] if st is not None else 0.0
                parts.append(float(pos))
                parts.append(float(vel))
        if self.base_link_index == -1:
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            base_lin, base_ang = p.getBaseVelocity(self.robot_id)
            roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)
        else:
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            base_lin, base_ang = p.getBaseVelocity(self.robot_id)
            roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)
        parts.extend([float(base_pos[0]), float(base_pos[1]), float(base_pos[2])])
        parts.extend([float(base_lin[0]), float(base_lin[1]), float(base_lin[2])])
        parts.extend([float(roll), float(pitch), float(yaw)])
        return np.array(parts, dtype=np.float32)

    # ------------------
    # Reset / step
    # ------------------
    def reset(self, *, initial_pose: Optional[np.ndarray] = None, seed: Optional[int] = None, options: dict = None):

        if seed is not None:
            np.random.seed(seed)

        # reset physics & reload URDF to ensure clean state
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        # use configured start_pos/start_ori so resets place robot above plane
        self.robot_id = p.loadURDF(self.urdf_path, basePosition=self.start_pos, baseOrientation=self.start_ori, useFixedBase=False)

        # zero all joints
        for j_idx in self.joint_name_to_index.values():
            try:
                p.resetJointState(self.robot_id, j_idx, targetValue=0.0, targetVelocity=0.0)
            except Exception:
                pass

        # apply initial pose if provided
        if initial_pose is not None:
            theta = np.array(initial_pose, dtype=np.float64).copy()
            # clamp against limits
            for t_idx, j_idx in self.joint_map.items():
                if t_idx >= len(theta):
                    continue
                val = float(theta[t_idx])
                lim = self.joint_limits.get(j_idx, (None, None))
                lo, hi = lim
                if lo is not None and hi is not None and lo < hi:
                    val = max(lo, min(hi, val))
                theta[t_idx] = val

            if self.smoothing_steps <= 0:
                for t_idx, j_idx in self.joint_map.items():
                    if t_idx >= len(theta):
                        continue
                    try:
                        p.resetJointState(self.robot_id, j_idx, theta[t_idx], 0.0)
                    except Exception:
                        pass
            else:
                current = np.zeros((self.mapped_theta_len,), dtype=np.float64)
                for t_idx, j_idx in self.joint_map.items():
                    st = p.getJointState(self.robot_id, j_idx)
                    current[t_idx] = st[0] if st is not None else 0.0

                for step in range(self.smoothing_steps):
                    alpha = float(step + 1) / float(self.smoothing_steps)
                    for t_idx, j_idx in self.joint_map.items():
                        if t_idx >= len(theta):
                            continue
                        tgt = (1.0 - alpha) * current[t_idx] + alpha * theta[t_idx]
                        p.setJointMotorControl2(self.robot_id, j_idx, p.POSITION_CONTROL, targetPosition=float(tgt), force=1000)
                    p.stepSimulation()
                    if self.gui:
                        time.sleep(self.time_step)
        # --- store reference angles after reset (used if actions are normalized) ---
        self.ref_angles = np.zeros((self.mapped_theta_len,), dtype=np.float32)
        for t_idx, j_idx in self.joint_map.items():
            st = p.getJointState(self.robot_id, j_idx)
            self.ref_angles[t_idx] = float(st[0]) if st is not None else 0.0

        # reset step counter and last action
        self.step_count = 0
        self.last_action = np.zeros((self.mapped_theta_len,), dtype=np.float32)

        # now return initial observation/info
        obs = self._get_observation()
        info = {"robot_id": self.robot_id}
        return obs, info

        obs = self._get_observation()
        info = {"robot_id": self.robot_id}
        return obs, info

    def step(self, action: np.ndarray):
        """
        Control step:
          - map/clip action
          - apply to motors (POSITION_CONTROL)
          - step the physics (self.sim_substeps)
          - compute obs, reward, terminated/truncated, info
        """

        # quick path if no mapped joints
        if self.mapped_theta_len == 0:
            for _ in range(self.sim_substeps):
                p.stepSimulation()
            obs = self._get_observation()
            return obs, 0.0, False, False, {}

        # cast action
        a = np.array(action, dtype=np.float32)
        if a.size != self.mapped_theta_len:
            # pad/truncate to expected shape
            tmp = np.zeros((self.mapped_theta_len,), dtype=np.float32)
            tmp[: min(len(a), len(tmp))] = a[: len(tmp)]
            a = tmp

        # Optionally clip to action_space if available
        if self.action_space is not None and hasattr(self.action_space, "low"):
            # if action_space uses huge infinities, clip only if bounds finite
            low = np.where(np.isfinite(self.action_space.low), self.action_space.low, -np.inf)
            high = np.where(np.isfinite(self.action_space.high), self.action_space.high, np.inf)
            a = np.minimum(np.maximum(a, low), high)

        # map normalized actions -> target joint positions (or treat as absolute)
        if self.actions_are_normalized:
            # actions assumed in [-1, 1]
            # target = ref + a * max_delta
            max_delta_arr = np.ones((self.mapped_theta_len,), dtype=np.float32) * float(self.max_delta)
            # If joint limits known, reduce delta to not exceed range
            for t_idx, j_idx in self.joint_map.items():
                lo, hi = self.joint_limits.get(j_idx, (None, None))
                if lo is not None and hi is not None and lo < hi:
                    # allow at most half-range as delta (so ref +/- delta stays in bounds)
                    half_range = 0.5 * (hi - lo)
                    max_delta_arr[t_idx] = min(max_delta_arr[t_idx], half_range)
            targets = (self.ref_angles + a * max_delta_arr).astype(np.float32)
        else:
            targets = a.astype(np.float32)

        # apply motor commands (POSITION_CONTROL)
        for t_idx, j_idx in self.joint_map.items():
            tgt = float(targets[t_idx]) if t_idx < len(targets) else 0.0
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=j_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=tgt,
                positionGain=self.kp,
                velocityGain=self.kd,
                force=200.0
            )

        # step simulation micro-steps
        for _ in range(self.sim_substeps):
            p.stepSimulation()
            if self.gui:
                time.sleep(self.time_step)

        # observation
        obs = self._get_observation()

        # compute forward velocity in robot heading
        base_lin = np.array(obs[self.mapped_theta_len * 2 + 3:self.mapped_theta_len * 2 + 6], dtype=np.float32)
        yaw = float(obs[self.mapped_theta_len * 2 + 8])
        forward_vec = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
        forward_vel = float(np.dot(base_lin, forward_vec))

        # upright penalty: 1 - dot(base_up, world_up); compute base_up from quaternion
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        rot_mat = p.getMatrixFromQuaternion(base_orn)
        base_up = np.array([rot_mat[6], rot_mat[7], rot_mat[8]], dtype=np.float32)  # third column
        upright_score = float(np.dot(base_up, np.array([0.0, 0.0, 1.0], dtype=np.float32)))
        upright_penalty = max(0.0, 1.0 - upright_score)

        # energy penalty (sum |torque * vel|) across mapped joints (use getJointState torque at [3])
        energy = 0.0
        for t_idx, j_idx in self.joint_map.items():
            st = p.getJointState(self.robot_id, j_idx)
            if st is None:
                continue
            pos, vel, _, applied_torque = st[0], st[1], st[2], st[3]
            energy += abs(float(applied_torque) * float(vel))
        energy_cost = float(energy)

        # action magnitude penalty (small)
        action_penalty = 0.001 * float(np.sum(np.abs(a)))

        # reward combination
        reward = float(self.w_vel * forward_vel - self.w_upright * upright_penalty - self.w_energy * energy_cost - action_penalty)

        # termination checks
        base_z = float(obs[self.mapped_theta_len * 2 + 2])
        roll = float(obs[self.mapped_theta_len * 2 + 6])
        pitch = float(obs[self.mapped_theta_len * 2 + 7])
        terminated = False
        if base_z < self.fall_height or abs(roll) > self.tilt_threshold or abs(pitch) > self.tilt_threshold or not np.isfinite(reward):
            terminated = True

        # truncated by time limit
        truncated = self.step_count >= self.max_episode_steps

        # foot contacts (try to see if either foot in contact with plane)
        in_contact = False
        contacts = p.getContactPoints(bodyA=self.robot_id)
        if len(self.foot_link_indices) > 0:
            for c in contacts:
                link_idx = c[3]  # linkIndexA
                if link_idx in self.foot_link_indices and c[9] > 0:  # normalForce index 9
                    in_contact = True
                    break

        info = {
            "forward_vel": forward_vel,
            "upright": upright_score,
            "energy": energy_cost,
            "action_penalty": action_penalty,
            "step_count": self.step_count,
            "in_contact": in_contact,
        }

        # save last action for logging
        self.last_action = a.copy()
        self.step_count += 1
        return obs, float(reward), bool(terminated), bool(truncated), info
    def close(self):
        try:
            if self.physics_client is not None:
                p.disconnect()
                self.physics_client = None
        except Exception:
            pass

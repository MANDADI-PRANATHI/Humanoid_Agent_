# pose_init/module1.py
"""
Module 1: PoseExtractor for BODY_25 (OpenPose)
- load_image
- extract_keypoints (OpenPose)
- select_main_skeleton (largest bbox)
- draw_skeleton (fixed pairs)
- convert_to_joint_angles (signed 2D angles)
- draw_skeleton_with_angles
- get_initial_pose (end-to-end)

Notes:
- Option B fixes applied: indentation issues fixed, small bugs corrected.
- Wrist "full orientation" (2D pitch + deviation) computed in body-local frame.
- Sign convention: Right-Hand Rule (CCW positive) in the body-local frame (S3).
"""

import os
import logging
from dataclasses import dataclass
import numpy as np
import cv2

# Try import OpenPose Python bindings
try:
    from openpose import pyopenpose as op
except Exception as e:
    op = None
    _OP_ERR = e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pose_init.module1")

# BODY_25 mapping (pairs used for drawing)
BODY_25_PAIRS = [
    (1, 8), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),
    (8, 12), (12, 13), (13, 14),
    (0, 15), (15, 17),
    (0, 16), (16, 18),
    (14, 19), (19, 20), (14, 21),
    (11, 22), (22, 23), (11, 24)
]

@dataclass
class PoseResult:
    image: np.ndarray
    keypoints: np.ndarray | None
    selected_idx: int | None
    selected_skeleton: np.ndarray | None
    bbox: tuple | None
    confidence_sum: float | None
    theta_dict: dict
    theta_init_vector: np.ndarray | None


class PoseExtractor:
    def __init__(self, model_folder="./models/", net_resolution="-1x368", disable_multi_thread=True):
        if op is None:
            raise ImportError(
                "OpenPose Python bindings not available. Import error: "
                f"{_OP_ERR}\nInstall OpenPose and set PYTHONPATH to its python folder."
            )
        params = {
            "model_folder": model_folder,
            "model_pose": "BODY_25",
            "net_resolution": net_resolution,
        }
        if disable_multi_thread:
            params["disable_multi_thread"] = True

        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()
        logger.info("OpenPose wrapper started (model_folder=%s)", model_folder)

    # ---------------------------
    # Image loading
    # ---------------------------
    def load_image(self, image_path: str) -> np.ndarray:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"cv2 failed to read image: {image_path}")
        return img

    # ---------------------------
    # OpenPose extraction
    # ---------------------------
    def extract_keypoints(self, img: np.ndarray) -> np.ndarray | None:
        datum = op.Datum()
        datum.cvInputData = img
        try:
            # Try the preferred API with VectorDatum if available
            datum_vector = op.VectorDatum([datum])
            success = self.opWrapper.emplaceAndPop(datum_vector)
        except Exception:
            try:
                success = self.opWrapper.emplaceAndPop([datum])
            except Exception as e:
                logger.error("OpenPose emplaceAndPop failed: %s", e)
                return None

        if not success:
            logger.warning("OpenPose emplaceAndPop returned False - no keypoints.")
            return None

        # datum_vector may exist (depending on API); prefer returning that if present
        try:
            if 'datum_vector' in locals() and hasattr(datum_vector, '__len__') and len(datum_vector) > 0:
                return datum_vector[0].poseKeypoints
        except Exception:
            pass
        try:
            return datum.poseKeypoints
        except Exception:
            return None

    # ---------------------------
    @staticmethod
    def _bbox_from_kps(kps25: np.ndarray, conf_thresh: float = 0.1):
        valid = kps25[:, 2] > conf_thresh
        if not np.any(valid):
            return None
        xs = kps25[valid, 0]
        ys = kps25[valid, 1]
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    def select_main_skeleton(self, keypoints: np.ndarray | None, conf_thresh: float = 0.1):
        if keypoints is None:
            return None, None, None, None
        best_idx = None
        best_area = -1.0
        best_conf = -1.0
        best_bbox = None
        n = keypoints.shape[0]
        for i in range(n):
            sk = keypoints[i]
            bbox = self._bbox_from_kps(sk, conf_thresh)
            conf_sum = float(np.nansum(sk[:, 2]))
            area = 0.0
            if bbox is not None:
                xmin, ymin, xmax, ymax = bbox
                area = (xmax - xmin) * (ymax - ymin)
            if area > best_area or (area == best_area and conf_sum > best_conf):
                best_area = area
                best_conf = conf_sum
                best_idx = i
                best_bbox = bbox
        if best_idx is None:
            return None, None, None, None
        return int(best_idx), keypoints[best_idx], best_bbox, best_conf

    # ---------------------------
    # Geometry helpers
    # ---------------------------
    @staticmethod
    def signed_angle_2d(v1, v2) -> float:
        """
        Return signed angle (radians) from v1 to v2 in 2D using RHR (CCW positive).
        """
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        return float(np.arctan2(cross, dot))

    @staticmethod
    def _flip_y(v):
        v = np.asarray(v, dtype=float).copy()
        v[1] = -v[1]
        return v

    def convert_to_joint_angles(self, skeleton: np.ndarray | None, conf_thresh: float = 0.1) -> dict:
        if skeleton is None:
            return {}

        def kp(i):
            return skeleton[i, :2]

        def conf(i):
            return float(skeleton[i, 2])

        if conf(8) < conf_thresh or conf(1) < conf_thresh:
            return {}

        midhip = kp(8)
        neck = kp(1)
        torso = neck - midhip
        if np.linalg.norm(torso) < 1e-6:
            return {}
        y_axis = torso / np.linalg.norm(torso)  
        x_axis = np.array([-y_axis[1], y_axis[0]])  
        print("Y_axis:",y_axis)
        print("X_axis:",x_axis)
        def to_local(v):
            return np.array([np.dot(v, x_axis), np.dot(v, y_axis)])

        def vec(a, b):
            return to_local(kp(b) - kp(a))

        def angle_1d(a, b):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            if np.linalg.norm(a) < 1e-8 or np.linalg.norm(b) < 1e-8:
                return 0.0
            cross = a[0] * b[1] - a[1] * b[0]
            dot = a[0] * b[0] + a[1] * b[1]
            return float(np.arctan2(cross, dot))

        angles = {}

        # ========== LEGS ==========
        if conf(9) > conf_thresh and conf(10) > conf_thresh and conf(11) > conf_thresh:
            thigh = vec(9, 10)
            shin = vec(10, 11)
            angles['right_knee'] = angle_1d(thigh, shin)

        if conf(12) > conf_thresh and conf(13) > conf_thresh and conf(14) > conf_thresh:
            thigh = vec(12, 13)
            shin = vec(13, 14)
            angles['left_knee'] = angle_1d(thigh, shin)

        # Hips: 2D (Pitch + Abduction)
        if conf(8) > conf_thresh:
            torso_v = vec(1,8) 
            if conf(9) > conf_thresh and conf(10) > conf_thresh:
                thigh_v = vec(9,10)
                print("Thigh right:",thigh_v)
                hip_pitch = angle_1d(torso_v, thigh_v) 
                lateral = np.dot(thigh_v,x_axis)
                vertical=np.dot(thigh_v,y_axis)
                hip_abd = float(np.arctan2(lateral,vertical))
                print("hip pitch",hip_pitch)
                print("hip abd",hip_abd)
                angles['right_hip_pitch'] = hip_pitch
                angles['right_hip_abd'] = hip_abd
           
            if conf(12) > conf_thresh and conf(13) > conf_thresh:
                thigh_v= vec(12,13)
                print("Thight left:",thigh_v)
                hip_pitch = angle_1d(torso_v, thigh_v)
                lateral = np.dot(thigh_v,x_axis)
                vertical=np.dot(thigh_v,y_axis)
                hip_abd = float(np.arctan2(lateral,vertical))
                angles['left_hip_pitch'] = hip_pitch 
                angles['left_hip_abd'] = hip_abd

        if conf(10) > conf_thresh and conf(11) > conf_thresh:
            shin = vec(10, 11)   # knee -> ankle
            foot_idx = max([(22 if conf(22) > conf_thresh else -1),
                        (23 if conf(23) > conf_thresh else -1),
                        (24 if conf(24) > conf_thresh else -1)])
            if foot_idx >=0:
                foot = vec(11, foot_idx)
            
            angles['right_ankle'] = angle_1d(shin, foot)

        if conf(13) > conf_thresh and conf(14) > conf_thresh:
            shin = vec(13, 14)   # knee -> ankle
            foot_idx = max([(19 if conf(19) > conf_thresh else -1),
                        (20 if conf(20) > conf_thresh else -1),
                        (21 if conf(21) > conf_thresh else -1)])
            if foot_idx >=0:
                foot = vec(14, foot_idx)
            else :
                foot = vec(14,22)
            angles['left_ankle'] = angle_1d(shin, foot)

        if conf(2) > conf_thresh and conf(3) > conf_thresh and conf(4) > conf_thresh:
            ua = vec(2, 3)  # upper arm (shoulder->elbow)
            fa = vec(3, 4)  # forearm (elbow->wrist)
            raw = angle_1d(fa,ua)
            sign = np.sign(raw) if abs(raw) > 1e-6 else 1.0
            flex = np.pi - abs(raw) 
            angles['right_elbow'] = sign* abs(raw)

        if conf(5) > conf_thresh and conf(6) > conf_thresh and conf(7) > conf_thresh:
            ua = vec(5, 6)
            fa = vec(6, 7)
            raw = angle_1d(ua, fa)
            sign = np.sign(raw) if abs(raw) > 1e-6 else 1.0
            flex = np.pi - abs(raw)

            angles['left_elbow'] = sign* abs(raw)

  
        if conf(1) > conf_thresh:
            torso_v_sh = vec(1,8) 
            if conf(2) > conf_thresh and conf(3) > conf_thresh:
                arm = vec(2,3)
                pitch = angle_1d(torso_v_sh, arm)
                lateral = np.dot(arm,x_axis)
                vertical=np.dot(arm,y_axis)
                abd = float(np.arctan2(lateral,vertical))
                angles['right_shoulder_pitch'] = pitch
                angles['right_shoulder_abd'] = abd
            # Left shoulder
            if conf(5) > conf_thresh and conf(6) > conf_thresh:
                arm = vec(5,6)
                pitch = angle_1d(torso_v_sh, arm)
                lateral = np.dot(arm,x_axis)
                vertical=np.dot(arm,y_axis)
                abd = float(np.arctan2(lateral,vertical))
                angles['left_shoulder_pitch'] = pitch
                angles['left_shoulder_abd'] = abd

        if conf(3) > conf_thresh and conf(4) > conf_thresh:
            forearm_r = vec(3, 4)
            
            pitch_r = angle_1d(np.array([0.0, 1.0]), forearm_r)  
            dev_r = float(np.arctan2(forearm_r[0], forearm_r[1])) if np.linalg.norm(forearm_r) > 1e-8 else 0.0
            angles['right_wrist_pitch'] = pitch_r
            

        # Left wrist (elbow=6, wrist=7)
        if conf(6) > conf_thresh and conf(7) > conf_thresh:
            forearm_l = vec(6, 7)
            pitch_l = angle_1d(np.array([0.0, 1.0]), forearm_l)
            dev_l = float(np.arctan2(forearm_l[0], forearm_l[1])) if np.linalg.norm(forearm_l) > 1e-8 else 0.0
            angles['left_wrist_pitch'] = pitch_l
            

        return angles

    def draw_skeleton_with_angles(self, img: np.ndarray, keypoints: np.ndarray, theta_dict: dict, conf_thresh: float = 0.1):
        if keypoints is None or theta_dict is None:
            return img
        # Wrap single skeleton
        if keypoints.ndim == 2 and keypoints.shape[1] == 3:
            keypoints = keypoints[np.newaxis, :, :]
        out = self.draw_skeleton(img, keypoints, conf_thresh=conf_thresh)
        kps = keypoints[0]

        joint_map = {
             'right_hip_abd': (8, 9, 10),'right_hip_pitch': (8, 9, 10),
             'left_hip_abd': (8, 12, 13),'left_hip_pitch': (8, 12, 13),
             'right_knee': (9, 10, 11),'left_knee': (12, 13, 14),
             'right_ankle': (10, 11, 22),'left_ankle': (13, 14, 19),
             'right_shoulder_pitch': (1, 2, 3),'left_shoulder_pitch': (1, 5, 6),
             'right_elbow': (2, 3, 4), 'left_elbow': (5, 6, 7),
             'right_wrist_pitch': (3, 4, 4),
             'left_wrist_pitch': (6, 7, 7)
        }

        for name, angle in theta_dict.items():
            if name not in joint_map:
                continue
            a, b, c = joint_map[name]
            if kps[a, 2] < conf_thresh or kps[b, 2] < conf_thresh or kps[c, 2] < conf_thresh:
                continue
            pt_a = tuple(kps[a, :2].astype(int))
            pt_b = tuple(kps[b, :2].astype(int))
            pt_c = tuple(kps[c, :2].astype(int))
            cv2.line(out, pt_a, pt_b, (255, 0, 0), 2)
            cv2.line(out, pt_b, pt_c, (255, 0, 0), 2)
            deg = int(np.degrees(angle))
            cv2.putText(out, f"{deg}°", (pt_b[0] + 5, pt_b[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        return out

    def draw_skeleton(self, img: np.ndarray, keypoints: np.ndarray, selected_idx: int | None = None, conf_thresh: float = 0.1, dim_others: bool = False):
        """
        Draw skeletons on `img`.
        - If selected_idx is None: draw all detected people.
        - If selected_idx is an int: draw only that person's keypoints & connections.
        - If dim_others is True and selected_idx is not None: draw non-selected people
          in a dimmer color so selected stands out.
        Returns a BGR image copy with drawings.
        """
        out = img.copy()
        if keypoints is None:
            return out

        # clamp selected_idx
        if selected_idx is not None:
            try:
                selected_idx = int(selected_idx)
                if selected_idx < 0 or selected_idx >= keypoints.shape[0]:
                    selected_idx = None
            except Exception:
                selected_idx = None

        for i, person in enumerate(keypoints):
            # decide whether to draw this person
            draw_this = (selected_idx is None) or (i == selected_idx)

            # style: highlighted person green lines, red points
            if draw_this:
                kp_color = (0, 0, 255)   # red keypoints (BGR)
                line_color = (0, 255, 0) # green lines
                line_thickness = 2
                radius = 4
            else:
                if dim_others:
                    # dim: use lighter gray-ish colors
                    kp_color = (50, 50, 50)
                    line_color = (80, 80, 80)
                    line_thickness = 1
                    radius = 3
                else:
                    # skip drawing non-selected when selected_idx provided and dim_others=False
                    if selected_idx is not None:
                        continue
                    kp_color = (0, 0, 255)
                    line_color = (0, 255, 0)
                    line_thickness = 2
                    radius = 4

            # draw keypoints
            for j, kp in enumerate(person):
                x, y, c = kp
                if c > conf_thresh:
                    cv2.circle(out, (int(round(x)), int(round(y))), radius, kp_color, -1)

            # draw connections
            for a, b in BODY_25_PAIRS:
                if person[a, 2] > conf_thresh and person[b, 2] > conf_thresh:
                    pa = (int(round(person[a, 0])), int(round(person[a, 1])))
                    pb = (int(round(person[b, 0])), int(round(person[b, 1])))
                    cv2.line(out, pa, pb, line_color, line_thickness)

        return out

    def get_initial_pose(self, image_path: str, conf_thresh: float = 0.1, return_drawn: bool = False) -> PoseResult:
        img = self.load_image(image_path)
        keypoints = self.extract_keypoints(img)
        sel_idx, sel_skeleton, bbox, conf_sum = self.select_main_skeleton(keypoints, conf_thresh)
        theta_dict = {}
        theta_init_vec = None

        # Debug prints (kept minimal)
        if sel_skeleton is not None:
            logger.info("Selected skeleton found (index=%s).", sel_idx)
        else:
            logger.info("No valid skeleton detected in image.")

        if sel_skeleton is not None:
            theta_dict = self.convert_to_joint_angles(sel_skeleton, conf_thresh)
            theta_init_vec = self.retarget_to_robot(theta_dict)

            logger.info("Computed %d joint angles.", len(theta_dict))

        if return_drawn:
            out_img = self.draw_skeleton(img, keypoints, selected_idx=sel_idx, conf_thresh=conf_thresh, dim_others=True)
            # Overlay angles if available
            out_img = self.draw_skeleton_with_angles(out_img, sel_skeleton, theta_dict, conf_thresh=conf_thresh)
        else:
            out_img = img

        return PoseResult(
            image=out_img,
            keypoints=keypoints,
            selected_idx=sel_idx,
            selected_skeleton=sel_skeleton,
            bbox=bbox,
            confidence_sum=conf_sum,
            theta_dict=theta_dict,
            theta_init_vector=theta_init_vec,
        )

    def retarget_to_robot(self, angles_dict: dict, joint_order: list | None = None, urdf_joint_limits: dict | None = None) -> np.ndarray:
        default_order = [
    "right_knee", "left_knee",
    "right_hip_pitch", "left_hip_pitch",
    "right_ankle", "left_ankle",
    "right_shoulder_pitch", "left_shoulder_pitch",
    "right_elbow", "left_elbow",
    "right_wrist_pitch", "left_wrist_pitch",
    "right_hip_abd", "left_hip_abd",
    "right_shoulder_abd", "left_shoulder_abd"
]


        order = joint_order if joint_order is not None else default_order
        theta = []
        for name in order:
            v = float(angles_dict.get(name, 0.0))
            # clamp if limits provided
            if urdf_joint_limits and name in urdf_joint_limits:
                lo, hi = urdf_joint_limits[name]
                if lo is not None and hi is not None and lo <= hi:
                    v = max(lo, min(hi, v))
            theta.append(v)   # ALWAYS append (fixed bug)
        return np.array(theta, dtype=np.float32)
    

    def print_keypoints(self, keypoints, selected_idx=None):
        if keypoints is None:
            print("\nNo keypoints detected.")
            return

        kps = np.array(keypoints)
        print("\n=== BODY_25 Keypoints ===")

        if kps.ndim == 2 and kps.shape[1] == 3:
            for i, (x, y, c) in enumerate(kps):
                print(f"{i:02d}: ({x:.1f}, {y:.1f}) conf={c:.3f}")
                return

        if kps.ndim == 3 and kps.shape[2] == 3:
            n_people = kps.shape[0]
            print(f"Detected {n_people} people")
            if selected_idx is None:
                selected_idx = 0
                selected_idx = min(max(0, selected_idx), n_people - 1)
                print(f"Printing person index: {selected_idx}\n")
            for i, (x, y, c) in enumerate(kps[selected_idx]):
                print(f"{i:02d}: ({x:.1f}, {y:.1f}) conf={c:.3f}")
            return
        print("Unexpected keypoints shape:", kps.shape)


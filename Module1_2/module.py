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

# BODY_25 mapping
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
    theta_init_vector_3d: np.ndarray | None 


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
    def load_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"cv2 failed to read image: {image_path}")
        return img

    # ---------------------------
    # OpenPose extraction
    # ---------------------------
    def extract_keypoints(self, img):
        datum = op.Datum()
        datum.cvInputData = img
        try:
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
    def _bbox_from_kps(kps25, conf_thresh=0.1):
        valid = kps25[:, 2] > conf_thresh
        if not np.any(valid):
            return None
        xs = kps25[valid, 0]
        ys = kps25[valid, 1]
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    def select_main_skeleton(self, keypoints, conf_thresh=0.1):
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
    def signed_angle_2d(v1, v2):
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
    def convert_to_joint_angles_3d(self, skeleton, conf_thresh=0.1):
        if skeleton is None:
            return {}
        angles_3d = {}
        kp3 = lambda idx: skeleton[idx, :3].copy()
        conf = lambda idx: skeleton[idx, 2]
        def vec(a_idx, b_idx):
            return kp3(b_idx) - kp3(a_idx)

        # Right Knee
        if conf(9) > conf_thresh and conf(10) > conf_thresh and conf(11) > conf_thresh:
            thigh = vec(9,10)
            shank = vec(10,11)
            cos_angle = np.clip(np.dot(thigh, shank) / (np.linalg.norm(thigh)*np.linalg.norm(shank)+1e-8), -1,1)
            angles_3d['right_knee_3d'] = np.arccos(cos_angle)

        # Left Knee
        if conf(12) > conf_thresh and conf(13) > conf_thresh and conf(14) > conf_thresh:
            thigh = vec(12,13)
            shank = vec(13,14)
            cos_angle = np.clip(np.dot(thigh, shank) / (np.linalg.norm(thigh)*np.linalg.norm(shank)+1e-8), -1,1)
            angles_3d['left_knee_3d'] = np.arccos(cos_angle)

        vertical = np.array([1,0,0])  # Z-axis

         # Right Hip
        if conf(8) > conf_thresh and conf(9) > conf_thresh:
            thigh = vec(8, 9)  # vector from hip to right knee
            cos_angle = np.clip(np.dot(thigh, -vertical) / (np.linalg.norm(thigh)+1e-8), -1,1)
            t = np.arccos(cos_angle)
            angles_3d['right_hip_pitch_3d'] = -np.arccos(cos_angle)

# Left Hip
        if conf(8) > conf_thresh and conf(12) > conf_thresh:
            thigh = vec(8, 12)  # vector from hip to left knee
            cos_angle = np.clip(np.dot(thigh, vertical) / (np.linalg.norm(thigh)+1e-8), -1,1)
            angles_3d['left_hip_pitch_3d'] = -np.arccos(cos_angle)

        # Right Ankle
        if conf(10) > conf_thresh and conf(11) > conf_thresh and conf(22) > conf_thresh:  # 22 = right foot index
           shank = vec(10,11)
           foot = vec(11,22)
           cos_angle = np.clip(np.dot(shank, foot) / (np.linalg.norm(shank)*np.linalg.norm(foot)+1e-8), -1,1)
           angles_3d['right_ankle_3d'] = np.arccos(cos_angle)

    # Left Ankle
        if conf(13) > conf_thresh and conf(14) > conf_thresh and conf(19) > conf_thresh:  # 19 = left foot index
           shank = vec(13,14)
           foot = vec(14,19)
           cos_angle = np.clip(np.dot(shank, foot) / (np.linalg.norm(shank)*np.linalg.norm(foot)+1e-8), -1,1)
           angles_3d['left_ankle_3d'] = np.arccos(cos_angle)
        # Right Elbow
        if conf(2) > conf_thresh and conf(3) > conf_thresh and conf(4) > conf_thresh:
            upper = vec(2,3)
            lower = vec(3,4)
            cos_angle = np.clip(np.dot(upper, lower) / (np.linalg.norm(upper)*np.linalg.norm(lower)+1e-8), -1,1)
            angles_3d['right_elbow_3d'] = -np.arccos(cos_angle)

        # Left Elbow
        if conf(5) > conf_thresh and conf(6) > conf_thresh and conf(7) > conf_thresh:
            upper = vec(5,6)
            lower = vec(6,7)
            cos_angle = np.clip(np.dot(upper, lower) / (np.linalg.norm(upper)*np.linalg.norm(lower)+1e-8), -1,1)
            angles_3d['left_elbow_3d'] = -np.arccos(cos_angle)

        # Right Shoulder
        if conf(1) > conf_thresh and conf(2) > conf_thresh and conf(3) > conf_thresh:
            upper = vec(1,2)
            lower = vec(2,3)
            cos_angle = np.clip(np.dot(upper, lower) /(np.linalg.norm(upper)*np.linalg.norm(lower)+1e-8), -1,1)
            angle = -np.arccos(cos_angle)
            hand_pos = skeleton[3]   
            shoulder_pos = skeleton[1]
            if hand_pos[2] > shoulder_pos[2]:
                # Hand is in front
                angles_3d['right_shoulder_pitch_3d'] = -angle
            else:
                angles_3d['right_shoulder_pitch_3d'] = angle

        # Left Shoulder
        if conf(1) > conf_thresh and conf(5) > conf_thresh and conf(6) > conf_thresh:
            upper = vec(1,5)
            lower = vec(5,6)
            cos_angle = np.clip(np.dot(upper, lower) / (np.linalg.norm(upper)*np.linalg.norm(lower)+1e-8), -1,1)
            angle = np.arccos(cos_angle)
            hand_pos = skeleton[6]   # left wrist
            shoulder_pos = skeleton[1]  # torso/shoulder
            if hand_pos[2] > shoulder_pos[2]:
                angles_3d['left_shoulder_pitch_3d'] = angle
            else:
                angles_3d['left_shoulder_pitch_3d'] = -angle
        if conf(3) > conf_thresh and conf(4) > conf_thresh and conf(21) > conf_thresh:
            forearm = vec(3,4)
            hand = vec(4,21)
            cos_angle = np.clip(np.dot(forearm, hand) / (np.linalg.norm(forearm)*np.linalg.norm(hand)+1e-8), -1,1)
            angles_3d['right_wrist_3d'] = -np.arccos(cos_angle)

# Left Wrist
        if conf(6) > conf_thresh and conf(7) > conf_thresh and conf(18) > conf_thresh:
            forearm = vec(6,7)
            hand = vec(7,18)
            cos_angle = np.clip(np.dot(forearm, hand) / (np.linalg.norm(forearm)*np.linalg.norm(hand)+1e-8), -1,1)
            angles_3d['left_wrist_3d'] = -np.arccos(cos_angle)

        return angles_3d
    def convert_to_joint_angles(self, skeleton, conf_thresh=0.1):
        if skeleton is None:
            return {}
        angles = {}
        kp2 = lambda idx: skeleton[idx, :2].copy()
        conf = lambda idx: skeleton[idx, 2]
        def vec(a_idx, b_idx):
            v = kp2(b_idx) - kp2(a_idx)
            return self._flip_y(v)

        # Legs
        if conf(9) > conf_thresh and conf(10) > conf_thresh and conf(11) > conf_thresh:
            angles['right_knee'] = self.signed_angle_2d(vec(9,10), vec(10,11))
        if conf(12) > conf_thresh and conf(13) > conf_thresh and conf(14) > conf_thresh:
            angles['left_knee'] = self.signed_angle_2d(vec(12,13), vec(13,14))
        # Hip pitch
        if conf(8) > conf_thresh:
            if conf(9) > conf_thresh and conf(10) > conf_thresh:
                angles['right_hip_pitch'] = self.signed_angle_2d(vec(8,9), vec(9,10))
            if conf(12) > conf_thresh and conf(13) > conf_thresh:
                angles['left_hip_pitch'] = self.signed_angle_2d(vec(8,12), vec(12,13))
        # Arms
        if conf(2) > conf_thresh and conf(3) > conf_thresh and conf(4) > conf_thresh:
            angles['right_elbow'] = self.signed_angle_2d(vec(2,3), vec(3,4))
            if conf(1) > conf_thresh:
                angles['right_shoulder_pitch'] = self.signed_angle_2d(vec(1,2), vec(2,3))
        if conf(5) > conf_thresh and conf(6) > conf_thresh and conf(7) > conf_thresh:
            angles['left_elbow'] = self.signed_angle_2d(vec(5,6), vec(6,7))
            if conf(1) > conf_thresh:
                angles['left_shoulder_pitch'] = self.signed_angle_2d(vec(1,5), vec(5,6))
        # Wrists
        if conf(3) > conf_thresh and conf(4) > conf_thresh and conf(21) > conf_thresh:
            angles['right_wrist'] = self.signed_angle_2d(vec(3,4), vec(4,21))

        if conf(6) > conf_thresh and conf(7) > conf_thresh and conf(18) > conf_thresh:
            angles['left_wrist'] = self.signed_angle_2d(vec(6,7), vec(7,18))

        return angles


    def draw_skeleton_with_angles(self, img, keypoints, theta_dict, conf_thresh=0.1):
        if keypoints is None or theta_dict is None:
            return img
        # Wrap single skeleton
        if keypoints.ndim==2 and keypoints.shape[1]==3:
            keypoints = keypoints[np.newaxis,:,:]
        out = self.draw_skeleton(img, keypoints, conf_thresh=conf_thresh)
        kps = keypoints[0]

        joint_map = {
            'right_knee': (9,10,11),
            'left_knee': (12,13,14),
            'right_hip_pitch': (8,9,10),
            'left_hip_pitch': (8,12,13),
            'right_elbow': (2,3,4),
            'left_elbow': (5,6,7),
            'right_shoulder_pitch': (1,2,3),
            'left_shoulder_pitch': (1,5,6),
            'right_wrist': (3,4,21),
            'left_wrist': (6,7,18)
        }

        for name, angle in theta_dict.items():
            if name not in joint_map:
                continue
            a,b,c = joint_map[name]
            if kps[a,2]<conf_thresh or kps[b,2]<conf_thresh or kps[c,2]<conf_thresh:
                continue
            pt_a = tuple(kps[a,:2].astype(int))
            pt_b = tuple(kps[b,:2].astype(int))
            pt_c = tuple(kps[c,:2].astype(int))
            cv2.line(out, pt_a, pt_b, (255,0,0),2)
            cv2.line(out, pt_b, pt_c, (255,0,0),2)
            deg = int(np.degrees(angle))
            cv2.putText(out,f"{deg}°",(pt_b[0]+5,pt_b[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1,cv2.LINE_AA)
        return out
    def draw_skeleton(self, img, keypoints, selected_idx=None, conf_thresh=0.1, dim_others=False):
        """
        Draw skeletons on `img`.
        - If selected_idx is None: draw all detected people (original behavior).
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
                kp_color = (0, 0, 255)   # red keypoints
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
    def get_initial_pose(self, image_path, conf_thresh=0.1, return_drawn=False):
        img = self.load_image(image_path)
        keypoints = self.extract_keypoints(img)
        sel_idx, sel_skeleton, bbox, conf_sum = self.select_main_skeleton(keypoints, conf_thresh)
        theta_dict = {}
        theta_dict_3d = {}
        theta_init_vec = None
        theta_init_vec_3d = None
        print("\nDEBUG: retarget_to_robot (2D) len:", None if theta_init_vec is None else len(theta_init_vec))
        print(" theta_init_vector:", theta_init_vec)
        print("\nDEBUG: retarget_to_robot_3d (3D) len:", None if theta_init_vec_3d is None else len(theta_init_vec_3d))
        print(" theta_init_vector_3d:", theta_init_vec_3d)
        if sel_skeleton is not None:
            print("\n=== BODY_25 Keypoints (x, y, confidence) ===")
            for i, (x, y, c) in enumerate(sel_skeleton):
                print(f" {i:02d}: x={x:.1f}, y={y:.1f}, conf={c:.3f}")
        else:
            print("\nNo valid skeleton detected in image.")

        if sel_skeleton is not None:
            theta_dict = self.convert_to_joint_angles(sel_skeleton, conf_thresh)
            theta_init_vec = self.retarget_to_robot(theta_dict)
            theta_dict_3d = self.convert_to_joint_angles_3d(sel_skeleton, conf_thresh)
            theta_init_vec_3d = self.retarget_to_robot_3d(theta_dict_3d)

            print("\n2D Joint Angles (radians and degrees):")
            for k,v in theta_dict.items():
                print(f" {k}: {v:.4f} rad / {v*180/np.pi:.1f} deg")
            print("\n3D Joint Angles (radians and degrees):")
            for k,v in theta_dict_3d.items():
                print(f" {k}: {v:.4f} rad / {v*180/np.pi:.1f} deg")
        if return_drawn:
            out_img = self.draw_skeleton(img, keypoints, selected_idx=sel_idx, conf_thresh=conf_thresh, dim_others=True)
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
            theta_init_vector_3d=theta_init_vec_3d
        )
    def retarget_to_robot(self, angles_dict, joint_order=None, urdf_joint_limits=None):
        """
        Build theta vector in the desired joint_order.
        If urdf_joint_limits is provided (dict of name -> (lo,hi)), clamp to those limits.
        """
        default_order = [
            'right_knee','left_knee','right_hip_pitch','left_hip_pitch',
            'right_ankle','left_ankle','right_shoulder_pitch','left_shoulder_pitch',
            'right_elbow','left_elbow','right_wrist','left_wrist'
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

    def retarget_to_robot_3d(self, angles_dict_3d, joint_order=None):
        """
        Build theta vector from 3D angle dict. Default order must use _3d suffixes consistently.
        """
        default_order = [
            'right_knee_3d','left_knee_3d','right_hip_pitch_3d','left_hip_pitch_3d',
            'right_ankle_3d','left_ankle_3d','right_shoulder_pitch_3d','left_shoulder_pitch_3d',
            'right_elbow_3d','left_elbow_3d','right_wrist_3d','left_wrist_3d'
        ]
        order = joint_order if joint_order is not None else default_order
        theta = [float(angles_dict_3d.get(name, 0.0)) for name in order]
        return np.array(theta, dtype=np.float32)

    




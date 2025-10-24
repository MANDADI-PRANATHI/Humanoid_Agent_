# pose_init/module1.py
"""
Module 1: PoseExtractor for BODY_25 (OpenPose)
Implements:
 - load_image
 - extract_keypoints (OpenPose)
 - select_main_skeleton (largest bbox)
 - draw_skeleton (fixed pairs)
 - convert_to_joint_angles (signed 2D angles)
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

# BODY_25 mapping reference (for comments)
# 0 Nose, 1 Neck,
# 2 RShoulder,3 RElbow,4 RWrist,
# 5 LShoulder,6 LElbow,7 LWrist,
# 8 MidHip, 9 RHip,10 RKnee,11 RAnkle,
# 12 LHip,13 LKnee,14 LAnkle,
# 15 REye,16 LEye,17 REar,18 LEar,
# 19 LBigToe,20 LSmallToe,21 LHeel,
# 22 RBigToe,23 RSmallToe,24 RHeel

# canonical BODY_25 pairs (no duplicates)
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
    image: np.ndarray                 # BGR image (possibly drawn)
    keypoints: np.ndarray | None      # (N,25,3) or None
    selected_idx: int | None
    selected_skeleton: np.ndarray | None  # (25,3) or None
    bbox: tuple | None
    confidence_sum: float | None
    theta_dict: dict
    theta_init_vector: np.ndarray | None


class PoseExtractor:
    def __init__(self, model_folder="/home/prana/openpose/models/", net_resolution="-1x368", disable_multi_thread=True):
        if op is None:
            raise ImportError(
                "OpenPose Python bindings not available. Import error: "
                f"{_OP_ERR}\nInstall OpenPose and set PYTHONPATH to its python folder."
            )
        params = dict()
        params["model_folder"] = model_folder
        params["model_pose"] = "BODY_25"
        params["net_resolution"] = net_resolution
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
        """
        Run OpenPose on the image and return poseKeypoints (N,25,3) or None.
        Input: img - BGR image (as loaded by cv2)
        Note: use op.VectorDatum to ensure correct C++ vector type is passed for builds
        that do not provide automatic list->std::vector conversion.
        """
        datum = op.Datum()
        datum.cvInputData = img

        # First try the explicit VectorDatum wrapper (most robust)
        try:
            datum_vector = op.VectorDatum([datum])
            success = self.opWrapper.emplaceAndPop(datum_vector)
        except Exception:
            # Fallback: try passing a python list (works on some pyopenpose builds)
            try:
                success = self.opWrapper.emplaceAndPop([datum])
            except Exception as e:
                logger.error("OpenPose emplaceAndPop failed with both VectorDatum and list: %s", e)
                return None

        if not success:
            logger.warning("OpenPose emplaceAndPop returned False - no keypoints.")
            return None

        # datum.poseKeypoints is attached to the last Datum in the vector
        # when using VectorDatum, the datum we passed is the first element
        # but pyopenpose exposes result on datum variable as well
        # the safe way: retrieve from the first element of the datum_vector if available
        try:
        # If we used op.VectorDatum object, it contains the Datum(s)
            if 'datum_vector' in locals() and hasattr(datum_vector, '__len__') and len(datum_vector) > 0:
                # datum_vector[0] is a Datum object
                return datum_vector[0].poseKeypoints
        except Exception:
            pass

        # fallback: try reading datum.poseKeypoints (works when list was used or pybind copied it)
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
        """
        Heuristic: largest bbox area across detected people; tie-break by sum(confidence).
        Returns: (selected_idx, skeleton25x3, bbox, confidence_sum) or (None, None, None, None)
        """
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
        """Return signed angle (radians) from v1 to v2 in 2D using atan2(cross, dot)."""
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
        """Flip y coordinate (image y grows downwards). Input is 2D vector."""
        v = np.asarray(v, dtype=float).copy()
        v[1] = -v[1]
        return v

    def convert_to_joint_angles(self, skeleton, conf_thresh=0.1):
        """
        Convert a single 25x3 skeleton to a dictionary of estimated angles (radians).
        This is 2D-only and approximate.
        Returns dict e.g. {'right_knee': x, 'left_knee': y, ...}
        """
        if skeleton is None:
            return {}
        angles = {}
        kp = lambda idx: skeleton[idx, :2].copy()

        # Right leg: RHip(9) -> RKnee(10) -> RAnkle(11)
        if skeleton[9, 2] > conf_thresh and skeleton[10, 2] > conf_thresh and skeleton[11, 2] > conf_thresh:
            thigh = self._flip_y(kp(10) - kp(9))
            shank = self._flip_y(kp(11) - kp(10))
            angles['right_knee'] = self.signed_angle_2d(thigh, shank)

        # Left leg: LHip(12) -> LKnee(13) -> LAnkle(14)
        if skeleton[12, 2] > conf_thresh and skeleton[13, 2] > conf_thresh and skeleton[14, 2] > conf_thresh:
            thigh = self._flip_y(kp(13) - kp(12))
            shank = self._flip_y(kp(14) - kp(13))
            angles['left_knee'] = self.signed_angle_2d(thigh, shank)

        # Hip pitch estimates (midHip(8) -> Hip -> Knee)
        if skeleton[8, 2] > conf_thresh:
            if skeleton[9, 2] > conf_thresh and skeleton[10, 2] > conf_thresh:
                v_mid_rhip = self._flip_y(kp(9) - kp(8))
                v_rhip_rknee = self._flip_y(kp(10) - kp(9))
                angles['right_hip_pitch_est'] = self.signed_angle_2d(v_mid_rhip, v_rhip_rknee)
            if skeleton[12, 2] > conf_thresh and skeleton[13, 2] > conf_thresh:
                v_mid_lhip = self._flip_y(kp(12) - kp(8))
                v_lhip_lknee = self._flip_y(kp(13) - kp(12))
                angles['left_hip_pitch_est'] = self.signed_angle_2d(v_mid_lhip, v_lhip_lknee)

        # simple shoulder spread
        if skeleton[1, 2] > conf_thresh and skeleton[2, 2] > conf_thresh and skeleton[5, 2] > conf_thresh:
            v_rshoulder = self._flip_y(kp(2) - kp(1))
            v_lshoulder = self._flip_y(kp(5) - kp(1))
            angles['shoulder_spread_est'] = self.signed_angle_2d(v_rshoulder, v_lshoulder)

        return angles

    # ---------------------------
    # Retarget placeholder
    # ---------------------------
    def retarget_to_robot(self, angles_dict, joint_order=None):
        """
        Placeholder retarget: produce a fixed-order theta_init numpy vector
        using a deterministic ordering. Replace with URDF-based mapping later.
        """
        default_order = ['right_knee', 'left_knee', 'right_hip_pitch_est', 'left_hip_pitch_est']
        order = joint_order if joint_order is not None else default_order
        vals = []
        for name in order:
            v = angles_dict.get(name, None)
            if v is None:
                vals.append(0.0)
            else:
                vals.append(float(v))
        return np.array(vals, dtype=np.float32)

    # ---------------------------
    # Drawing
    # ---------------------------
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

    # ---------------------------
    # End-to-end
    # ---------------------------
    def get_initial_pose(self, image_path, conf_thresh=0.1, return_drawn=False):
        """
        End-to-end pipeline:
         - load image
         - run openpose
         - select main skeleton
         - compute angles
         - produce theta_init placeholder vector
        Returns PoseResult
        """
        img = self.load_image(image_path)
        keypoints = self.extract_keypoints(img)
        sel_idx, sel_skeleton, bbox, conf_sum = self.select_main_skeleton(keypoints, conf_thresh)
        theta_dict = {}
        theta_init_vec = None
        if sel_skeleton is not None:
            theta_dict = self.convert_to_joint_angles(sel_skeleton, conf_thresh)
            theta_init_vec = self.retarget_to_robot(theta_dict)
        if return_drawn:
            # draw only the selected person (if one exists). If you prefer to dim others instead, set dim_others=True
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
            theta_init_vector=theta_init_vec
        )

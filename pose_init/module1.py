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
        Extended joint-angle estimation from a BODY_25 skeleton (2D-based).
        Returns a dict of named angle estimates (radians).
        Produces estimates usable as initial guesses for the robot.
        """

        if skeleton is None:
            return {}

        angles = {}
        kp2 = lambda idx: skeleton[idx, :2].copy()
        conf = lambda idx: skeleton[idx, 2]

        # helper: safe vector from a->b with y-flip
        def vec(a_idx, b_idx):
            v = kp2(b_idx) - kp2(a_idx)
            return self._flip_y(v)

        # ------- LEGS -------
        # Right leg: RHip(9) -> RKnee(10) -> RAnkle(11)
        if conf(9) > conf_thresh and conf(10) > conf_thresh and conf(11) > conf_thresh:
            thigh = vec(9, 10)
            shank = vec(10, 11)
            angles['right_knee'] = self.signed_angle_2d(thigh, shank)
            # ankle: shank -> foot (use RAnkle->RBigToe if available else use small toe/heel)
            # try RBigToe (22), RSmallToe(23), RHeel(24)
            if conf(22) > conf_thresh:
                footvec = vec(11, 22)
                angles['right_ankle'] = self.signed_angle_2d(shank, footvec)
            elif conf(23) > conf_thresh:
                footvec = vec(11, 23)
                angles['right_ankle'] = self.signed_angle_2d(shank, footvec)
            elif conf(24) > conf_thresh:
                footvec = vec(11, 24)
                angles['right_ankle'] = self.signed_angle_2d(shank, footvec)

        # Left leg: LHip(12) -> LKnee(13) -> LAnkle(14)
        if conf(12) > conf_thresh and conf(13) > conf_thresh and conf(14) > conf_thresh:
            thigh = vec(12, 13)
            shank = vec(13, 14)
            angles['left_knee'] = self.signed_angle_2d(thigh, shank)
            if conf(19) > conf_thresh:
                footvec = vec(14, 19)
                angles['left_ankle'] = self.signed_angle_2d(shank, footvec)
            elif conf(20) > conf_thresh:
                footvec = vec(14, 20)
                angles['left_ankle'] = self.signed_angle_2d(shank, footvec)
            elif conf(21) > conf_thresh:
                footvec = vec(14, 21)
                angles['left_ankle'] = self.signed_angle_2d(shank, footvec)

        # Hip pitch estimates (midHip(8) -> Hip -> Knee)
        if conf(8) > conf_thresh:
            if conf(9) > conf_thresh and conf(10) > conf_thresh:
                v_mid_rhip = vec(8, 9)
                v_rhip_rknee = vec(9, 10)
                angles['right_hip_pitch'] = self.signed_angle_2d(v_mid_rhip, v_rhip_rknee)
            if conf(12) > conf_thresh and conf(13) > conf_thresh:
                v_mid_lhip = vec(8, 12)
                v_lhip_lknee = vec(12, 13)
                angles['left_hip_pitch'] = self.signed_angle_2d(v_mid_lhip, v_lhip_lknee)

        # ------- ARMS -------
        # Right arm: RShoulder(2)->RElbow(3)->RWrist(4)
        if conf(2) > conf_thresh and conf(3) > conf_thresh and conf(4) > conf_thresh:
            upper = vec(2, 3)
            fore = vec(3, 4)
            angles['right_elbow'] = self.signed_angle_2d(upper, fore)
            # approximate shoulder pitch: neck(1) -> rshoulder(2) relative to shoulder->elbow
            if conf(1) > conf_thresh:
                v_neck_rshoulder = vec(1, 2)
                angles['right_shoulder_pitch'] = self.signed_angle_2d(v_neck_rshoulder, upper)

        # Left arm: LShoulder(5)->LElbow(6)->LWrist(7)
        if conf(5) > conf_thresh and conf(6) > conf_thresh and conf(7) > conf_thresh:
            upper = vec(5, 6)
            fore = vec(6, 7)
            angles['left_elbow'] = self.signed_angle_2d(upper, fore)
            if conf(1) > conf_thresh:
                v_neck_lshoulder = vec(1, 5)
                angles['left_shoulder_pitch'] = self.signed_angle_2d(v_neck_lshoulder, upper)

        # Shoulder spread (optional)
        if conf(2) > conf_thresh and conf(5) > conf_thresh and conf(1) > conf_thresh:
            v_rshoulder = vec(1, 2)
            v_lshoulder = vec(1, 5)
            angles['shoulder_spread_est'] = self.signed_angle_2d(v_rshoulder, v_lshoulder)

        return angles


    def retarget_to_robot(self, angles_dict: dict, joint_order: list | None = None, urdf_joint_limits: dict | None = None):
        """
        Retarget named angle estimates to a numeric theta_init vector.
        - Default joint_order yields 10-length vector:
            ['right_knee','left_knee','right_hip_pitch','left_hip_pitch',
             'right_ankle','left_ankle','right_shoulder_pitch','left_shoulder_pitch',
             'right_elbow','left_elbow']
        - urdf_joint_limits: optional dict mapping joint_name_or_index -> (lower, upper) in radians;
          will be used to clamp values if provided.
        Returns numpy.float32 vector length = len(joint_order)
        """
        default_order = [
            'right_knee','left_knee','right_hip_pitch','left_hip_pitch',
            'right_ankle','left_ankle','right_shoulder_pitch','left_shoulder_pitch',
            'right_elbow','left_elbow'
        ]
        order = joint_order if joint_order is not None else default_order

        theta = []
        for name in order:
            v = angles_dict.get(name, None)
            if v is None:
                theta.append(0.0)
            else:
                # clamp if urdf limits provided (name may be mapping)
                if urdf_joint_limits and name in urdf_joint_limits:
                    lo, hi = urdf_joint_limits[name]
                    # only clamp if limits appear valid
                    if lo <= hi:
                        v = max(lo, min(hi, float(v)))
                theta.append(float(v))
        return np.array(theta, dtype=np.float32)


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

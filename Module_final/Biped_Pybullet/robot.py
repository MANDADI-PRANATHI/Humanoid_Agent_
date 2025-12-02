# Biped_Pybullet/robot.py
import os
import pybullet as p
import pybullet_data
import numpy as np
import time
from Biped_Pybullet import transform as tf
import scipy.linalg as la


class Robot:
    def __init__(self, robotPath, startPos, startOrn, maxForce,
                 controlMode=p.POSITION_CONTROL, planePath=None):
        """
        Note: This constructor will NOT forcibly open a new GUI if one already exists.
        """

        # --- connect if not already connected ---
        if not p.isConnected():
            # attempt shared memory first (headless clients), otherwise open GUI
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                p.connect(p.GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # planePath default -> pybullet_data plane if not provided or not absolute
        if planePath is None:
            planePath = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
        elif not os.path.isabs(planePath):
            # try to resolve relative to package
            planePath = os.path.join(os.path.dirname(__file__), planePath)
            if not os.path.exists(planePath):
                planePath = os.path.join(pybullet_data.getDataPath(), "plane.urdf")

        self.incline = p.addUserDebugParameter("Incline", -0.1, 0.1, 0.0)
        incline_val = p.readUserDebugParameter(self.incline)
        self.planeId = p.loadURDF(planePath, [0, 0, 0], p.getQuaternionFromEuler([0, incline_val, 0]))
        p.changeDynamics(self.planeId, -1, lateralFriction=60)

        # robotPath: allow None or relative path; resolve relative to this file
        if robotPath is None or not os.path.isabs(robotPath):
            potential = os.path.join(os.path.dirname(__file__), robotPath if robotPath else "biped.urdf")
            if os.path.exists(potential):
                robotPath = potential

        # finally load robot URDF
        self.robotId = p.loadURDF(robotPath, startPos, p.getQuaternionFromEuler(startOrn))
        self.controlMode = controlMode
        self.numJoint = p.getNumJoints(self.robotId)
        self.jointIdList = list(range(self.numJoint))

        # tint robot for visibility
        for elem in range(-1, self.numJoint):
            p.changeVisualShape(self.robotId, elem, rgbaColor=[0.8, 0.8, 0, 1])

        self.maxForce = maxForce
        self.maxForceList = [maxForce] * max(1, self.numJoint)
        self.stride = p.addUserDebugParameter('Stride', 0, 0.2, 0.1)
        self.stepHeight = p.addUserDebugParameter('Step height', 0.03, 0.1, 0.04)

        self.timeStep = 1. / 240.

    def getEuler(self):
        _, qua = p.getBasePositionAndOrientation(self.robotId)
        return p.getEulerFromQuaternion(qua)

    def getQuaternion(self):
        _, orn = p.getBasePositionAndOrientation(self.robotId)
        return orn

    def getRobotPosition(self):
        pos, _ = p.getBasePositionAndOrientation(self.robotId)
        return pos

    def getIncline(self):
        return p.readUserDebugParameter(self.incline)

    def getStride(self):
        return p.readUserDebugParameter(self.stride)

    def getStepHeight(self):
        return p.readUserDebugParameter(self.stepHeight)

    def resetRobotPositionAndOrientation(self, pos, orn):
        p.resetBasePositionAndOrientation(self.robotId, pos, orn)

    def resetIncline(self, incline):
        p.resetBasePositionAndOrientation(self.planeId, [0, 0, 0],
                                          p.getQuaternionFromEuler([0, incline, 0]))

    def setMotorTorqueByArray(self, targetJointTorqueList):
        if self.controlMode is p.TORQUE_CONTROL:
            p.setJointMotorControlArray(self.robotId, jointIndices=self.jointIdList,
                                        controlMode=p.TORQUE_CONTROL, forces=targetJointTorqueList)
        else:
            print('Error: Mode must be set to TORQUE_CONTROL.')

    def setMotorPositionByArray(self, targetJointPositionList):
        p.setJointMotorControlArray(self.robotId, jointIndices=self.jointIdList, controlMode=self.controlMode,
                                    forces=self.maxForceList, targetPositions=targetJointPositionList)

    def oneStep(self):
        robotPos, _ = p.getBasePositionAndOrientation(self.robotId)
        p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=135, cameraPitch=-10,
                                     cameraTargetPosition=robotPos)
        p.stepSimulation()
        time.sleep(self.timeStep)


class Biped(Robot):
    def __init__(self, startPos=[0, 0, 0.55], startOrn=[0, 0, 0], CoM_pos=np.array([0., 0., -0.02]), maxForce=9.0,
                 controlMode=p.POSITION_CONTROL, robotPath=None, planePath=None):
        # default robotPath resolves to biped.urdf in same folder
        if robotPath is None:
            robotPath = os.path.join(os.path.dirname(__file__), "biped.urdf")

        super().__init__(robotPath, startPos, startOrn, maxForce,
                         controlMode=controlMode, planePath=planePath)
        self._lambda = 1.0
        self.L1 = 0.18
        self.L2 = 0.18
        self.R = np.array([0, -0.065, -0.175]) - CoM_pos
        self.L = np.array([0, 0.065, -0.175]) - CoM_pos
        self.legDoF = 6

        # --- robust joint index lookup by name (avoids hard-coded indices) ---
        joint_name_to_index = {}
        for ji in range(self.numJoint):
            info = p.getJointInfo(self.robotId, ji)
            name = info[1].decode('utf-8')
            joint_name_to_index[name] = ji

        right_leg_names = [
            'r_yaw_hipJoint', 'r_roll_hipJoint', 'r_pitch_hipJoint',
            'r_kneeJoint', 'r_pitch_ankleJoint', 'r_roll_ankleJoint'
        ]
        left_leg_names = [
            'l_yaw_hipJoint', 'l_roll_hipJoint', 'l_pitch_hipJoint',
            'l_kneeJoint', 'l_pitch_ankleJoint', 'l_roll_ankleJoint'
        ]

        try:
            self.jointIdListR = [joint_name_to_index[n] for n in right_leg_names]
            self.jointIdListL = [joint_name_to_index[n] for n in left_leg_names]
        except KeyError as e:
            raise RuntimeError(f"Joint name not found in URDF: {e}. Check joint names in URDF and this list.")

        self.jointIdList = list(range(self.numJoint))
        self.maxForceList = [maxForce] * self.numJoint
        self.maxForceListForLeg = [maxForce] * self.legDoF

        self.a = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0],
                           [0, 1, 0], [0, 1, 0], [1, 0, 0]])
        self.E = np.eye(3)

    def setRightLegJointPositions(self, targetJointPositions):
        p.setJointMotorControlArray(self.robotId, jointIndices=self.jointIdListR, controlMode=self.controlMode,
                                   forces=self.maxForceListForLeg, targetPositions=targetJointPositions)

    def setLeftLegJointPositions(self, targetJointPositions):
        p.setJointMotorControlArray(self.robotId, jointIndices=self.jointIdListL, controlMode=self.controlMode,
                                   forces=self.maxForceListForLeg, targetPositions=targetJointPositions)

    def setLegPositions(self, targetPosL, targetPosR, targetRPY):
        posL = self.inverseKinematics(targetPosL, targetRPY, self.L)
        posR = self.inverseKinematics(targetPosR, targetRPY, self.R)
        self.setLeftLegJointPositions(posL)
        self.setRightLegJointPositions(posR)

    def torqueControlModeEnableForAll(self):
        p.setJointMotorControlArray(self.robotId, jointIndices=self.jointIdList, controlMode=p.VELOCITY_CONTROL,
                                    forces=[0] * self.numJoint)
        self.controlMode = p.TORQUE_CONTROL

    def getLegTrans(self, jointPositions, leg):
        hipyaw = jointPositions[0]
        hiproll = jointPositions[1]
        hippitch = jointPositions[2]
        knee = jointPositions[3]
        anklepitch = jointPositions[4]
        ankleroll = jointPositions[5]
        zero_v = np.zeros(3)

        T_0_1 = tf.getTransFromRp(tf.rodriguesEquation(self.E, self.a[0], hipyaw), leg)
        T_0_2 = T_0_1.dot(tf.getTransFromRp(tf.rodriguesEquation(self.E, self.a[1], hiproll), zero_v))
        T_0_3 = T_0_2.dot(tf.getTransFromRp(tf.rodriguesEquation(self.E, self.a[2], hippitch), zero_v))
        T_0_4 = T_0_3.dot(tf.getTransFromRp(tf.rodriguesEquation(self.E, self.a[3], knee), [0, 0, -self.L1]))
        T_0_5 = T_0_4.dot(tf.getTransFromRp(tf.rodriguesEquation(self.E, self.a[4], anklepitch), [0, 0, -self.L2]))
        T_0_6 = T_0_5.dot(tf.getTransFromRp(tf.rodriguesEquation(self.E, self.a[5], ankleroll), zero_v))

        return T_0_1, T_0_2, T_0_3, T_0_4, T_0_5, T_0_6

    def forwardKinematics(self, jointPositions, leg):
        T_0_6 = self.getLegTrans(jointPositions, leg)[5]

        return tf.getRotationAndPositionFromT(T_0_6)

    def inverseKinematics(self, p_ref, omega_ref, leg):
        q = np.array(self.getJointPositions(leg), dtype=float)
        R, p_pos = self.forwardKinematics(q, leg)
        omega = np.array(tf.getRollPitchYawFromR(R))

        dp = p_ref - p_pos
        domega = omega_ref - omega
        dp_omega = np.append(dp, domega)

        J = self.jacobian(q, leg)
        dq = self._lambda * la.inv(J).dot(dp_omega)
        return q + dq

    def jacobian(self, q, leg):
        T0 = self.getLegTrans(q, leg)
        zero_v = np.zeros(3)

        R = [tf.getRotationFromT(T0[i]) for i in range(len(T0))]
        p = [tf.getPositionFromT(T0[i]) for i in range(len(T0))]

        wa = [R[i].dot(self.a[i]) for i in range(len(R))]

        rows = []
        for i in range(len(wa) - 1):
            lin = np.cross(wa[i], (p[5] - p[i]))
            ang = wa[i]
            rows.append(np.hstack((lin, ang)))

        Jp = np.vstack(rows)
        J = np.vstack((Jp, np.hstack((zero_v, wa[5])))).T
        return J

    def getJointPositions(self, leg):
        if np.sum(leg == self.R) == len(leg):
            jointStates = p.getJointStates(self.robotId, jointIndices=self.jointIdListR)
            jointPositions = [jointStates[i][0] for i in range(len(jointStates))]
        elif np.sum(leg == self.L) == len(leg):
            jointStates = p.getJointStates(self.robotId, jointIndices=self.jointIdListL)
            jointPositions = [jointStates[i][0] for i in range(len(jointStates))]
        else:
            raise ValueError('Invalid parameter.')

        return jointPositions

    def positionInitialize(self, startCoM_height=0.45, initialLegRPY=[0, 0, 0],
                           initializeTime=1.0, initialJointPosRL=[0.0, 0.0, -0.44, 0.88, -0.44, 0.0]):
        initializeStep = np.arange(0, initializeTime / self.timeStep, 1)
        initialLegPosR = [0, self.R[1], -startCoM_height]
        initialLegPosL = [0, self.L[1], -startCoM_height]

        for _ in initializeStep:
            self.setLeftLegJointPositions(initialJointPosRL)
            self.setRightLegJointPositions(initialJointPosRL)
            self.resetRobotPositionAndOrientation(pos=[0, 0, startCoM_height + 0.02], orn=[0, 0, 0, 1])
            self.oneStep()

        for _ in initializeStep:
            posR = self.inverseKinematics(initialLegPosR, initialLegRPY, self.R)
            posL = self.inverseKinematics(initialLegPosL, initialLegRPY, self.L)
            self.setRightLegJointPositions(posR)
            self.setLeftLegJointPositions(posL)
            self.oneStep()

    def disconnect(self):
        p.disconnect()

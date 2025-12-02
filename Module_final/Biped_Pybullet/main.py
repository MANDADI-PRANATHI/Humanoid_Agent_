# Biped_Pybullet/main.py
import numpy as np
import argparse
import pybullet as p

from Biped_Pybullet.robot import Biped
from Biped_Pybullet.walking import PreviewControl

def ensure_connection_and_reset():
    """
    Ensure there's a PyBullet connection (open GUI if none) and reset the world.
    This allows the biped to load into the same GUI that may already be open.
    """
    if not p.isConnected():
        p.connect(p.GUI)
    # Reset the simulation (clear fallen humanoid, debug lines, etc) but retain the GUI window
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)


def stand():
    ensure_connection_and_reset()
    biped = Biped()
    CoM_height = 0.45

    targetRPY = [0.0, 0.0, 0.0]
    targetPosL = [0.0, 0.065, -CoM_height]
    targetPosR = [0.0, -0.065, -CoM_height]
    biped.positionInitialize(initializeTime=0.2)

    while True:
        incline = biped.getIncline()
        biped.resetIncline(incline)
        targetRPY[1] = incline

        biped.setLegPositions(targetPosL, targetPosR, targetRPY)
        biped.oneStep()


def squat():
    ensure_connection_and_reset()
    biped = Biped()
    CoM_height = 0.45

    targetRPY = [0.0, 0.0, 0.0]
    targetPosL = [0.0, 0.065, -CoM_height]
    targetPosR = [0.0, -0.065, -CoM_height]
    biped.positionInitialize(initializeTime=0.1)

    dp = 0.002
    while True:
        incline = biped.getIncline()
        biped.resetIncline(incline)
        targetRPY[1] = incline

        for _ in range(100):
            biped.setLegPositions(targetPosL, targetPosR, targetRPY)
            biped.oneStep()
            targetPosL[2] += dp
            targetPosR[2] += dp

        for _ in range(100):
            biped.setLegPositions(targetPosL, targetPosR, targetRPY)
            biped.oneStep()
            targetPosL[2] -= dp
            targetPosR[2] -= dp


def jump(withTorsoTwist=False):
    ensure_connection_and_reset()
    biped = Biped()
    CoM_height = 0.45

    targetRPY = [0.0, 0.0, 0.0]
    targetPosL = [0.0, 0.065, -CoM_height]
    targetPosR = [0.0, -0.065, -CoM_height]
    biped.positionInitialize(initializeTime=0.1)

    dp = 0.0025
    dRPY = 0.0065
    while True:
        incline = biped.getIncline()
        biped.resetIncline(incline)
        targetRPY[1] = incline

        for _ in range(60):
            biped.setLegPositions(targetPosL, targetPosR, targetRPY)
            biped.oneStep()
            targetPosL[2] += dp
            targetPosR[2] += dp
            if withTorsoTwist:
                targetRPY[2] += dRPY

        for _ in range(60):
            biped.setLegPositions(targetPosL, targetPosR, targetRPY)
            biped.oneStep()
            targetPosL[2] -= dp
            targetPosR[2] -= dp
            if withTorsoTwist:
                targetRPY[2] -= dRPY


def torsoTwist():
    ensure_connection_and_reset()
    biped = Biped()
    CoM_height = 0.45

    targetRPY = [0.0, 0.0, -0.25]
    targetPosL = [0.0, 0.065, -CoM_height]
    targetPosR = [0.0, -0.065, -CoM_height]
    biped.positionInitialize(initializeTime=0.1)

    dp = 0.005
    while True:
        incline = biped.getIncline()
        biped.resetIncline(incline)
        targetRPY[1] = incline

        for _ in range(100):
            biped.setLegPositions(targetPosL, targetPosR, targetRPY)
            biped.oneStep()
            targetRPY[2] += dp

        for _ in range(100):
            biped.setLegPositions(targetPosL, targetPosR, targetRPY)
            biped.oneStep()
            targetRPY[2] -= dp


def walk():
    """
    Full walking controller using PreviewControl (your original implementation).
    This function resets the simulation (keeps the GUI) and then runs the walk loop.
    """
    ensure_connection_and_reset()
    biped = Biped()
    # CoM_height = 0.45
    # CoM_to_body = np.array([0.0, 0.0, 0.0])

    targetRPY = [0.0, 0.0, 0.0]
    pre = PreviewControl(dt=1./240., Tsup_time=0.3, Tdl_time=0.1, previewStepNum=190)
    biped.positionInitialize(initializeTime=0.2)
    CoM_trajectory = np.empty((0, 3), float)

    trjR_log = np.empty((0, 3), float)
    trjL_log = np.empty((0, 3), float)
    supPoint = np.array([0., 0.065])

    while True:
        incline = biped.getIncline()
        biped.resetIncline(incline)
        targetRPY[1] = incline

        stepHeight = biped.getStepHeight()

        # Generates one cycle trajectory
        CoM_trj, footTrjL, footTrjR = pre.footPrintAndCoM_trajectoryGenerator(inputTargetZMP=supPoint,
                                                                              inputFootPrint=supPoint,
                                                                              stepHeight=stepHeight)
        CoM_trajectory = np.vstack((CoM_trajectory, CoM_trj))
        trjR_log = np.vstack((trjR_log, footTrjR))
        trjL_log = np.vstack((trjL_log, footTrjL))

        for j in range(len(CoM_trj)):
            targetPosR = footTrjR[j] - CoM_trj[j]
            targetPosL = footTrjL[j] - CoM_trj[j]

            biped.setLegPositions(targetPosL, targetPosR, targetRPY)
            biped.oneStep()

        supPoint[0] += biped.getStride()
        supPoint[1] = -supPoint[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--action', help='Choose from walk, stand, squat, jump_w_Twist, jump_wo_Twist, torsoTwist', type=str, default='walk')
    args = parser.parse_args()

    if args.action == 'walk':
        walk()
    elif args.action == 'jump_w_Twist':
        jump(withTorsoTwist=True)
    elif args.action == 'jump_wo_Twist':
        jump(withTorsoTwist=False)
    elif args.action == 'squat':
        squat()
    elif args.action == 'torsoTwist':
        torsoTwist()
    else:
        stand()

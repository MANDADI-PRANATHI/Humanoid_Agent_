# debug_load_urdf.py
import pybullet as p, pybullet_data, os, sys
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
urdf = os.path.join("assets","human_test.urdf")
print("Loading:", urdf)
try:
    uid = p.loadURDF(urdf, [0,0,1.0], p.getQuaternionFromEuler([0,0,0]), useFixedBase=False, flags=p.URDF_USE_SELF_COLLISION)
except Exception as e:
    print("loadURDF failed:", e)
    sys.exit(1)
print("Loaded uid:", uid)
print("Num joints:", p.getNumJoints(uid))
for i in range(p.getNumJoints(uid)):
    info = p.getJointInfo(uid, i)
    print(i, info[1].decode('utf-8'), "type", info[2], "lower", info[8], "upper", info[9])
p.disconnect()

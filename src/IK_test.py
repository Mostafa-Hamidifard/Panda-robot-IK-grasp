import mujoco
import numpy as np
from mujoco import viewer
from IK import qpos_from_site_pose


def create_scene():
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body()
    light = spec.worldbody.add_light()
    light.pos = [0, 0, 1]
    jnt = body.add_joint()
    jnt.type = mujoco.mjtJoint.mjJNT_SLIDE
    jnt.name = "sjx"
    jnt.axis = [1, 0, 0]

    jnt1 = body.add_joint()
    jnt1.type = mujoco.mjtJoint.mjJNT_SLIDE
    jnt1.name = "sjy"
    jnt1.axis = [0, 1, 0]

    kp = 5000
    kd = 450
    act = spec.add_actuator()
    act.target = jnt.name

    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.dyntype = mujoco.mjtDyn.mjDYN_FILTEREXACT
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    act.dynprm[0] = 0
    act.gainprm[0] = kp
    act.biasprm[0] = 0
    act.biasprm[1] = -kp
    act.biasprm[2] = -kd

    geom = body.add_geom()
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.rgba = [1, 0, 1, 1]
    geom.size = [0.2, 0.1, 0.1]

    site = body.add_site()
    site.name = "ee_site"

    site2 = body.add_site()
    site2.pos = [1, 0, 0]
    site2.name = "ee_site1"
    model = spec.compile()
    return model, site.name, jnt.name


def main():

    model, site_name, joint_name = create_scene()
    data = mujoco.MjData(model)
    print(site_name)
    print(joint_name)
    IKResult = qpos_from_site_pose(model, data, site_name, target_pos=[1, 0.5, 0])
    print(IKResult)


if __name__ == "__main__":
    main()

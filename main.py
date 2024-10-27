# Don't use random initials. it results in a strange behaviour in the EE


# %% loading general modules
import numpy as np
import mujoco
from mujoco import viewer
import src.IK as IK
from typing import Dict, Tuple, List

# %% Loading custom modules
from src import loader
from src import planner

# %% Defining PATH
GRASP_SITE_NAME = "grasp_site"
ROBOT_PATH = "asset/franka_emika_panda/scene.xml"
CUBE_SPEC = {"position": [0.5, 0, 0.2], "scale": [0.025, 0.025, 0.2]}
# ROBOT_INIT = np.array([[0.2, 0], [0.2, 0], [0.2, 0], [0.2, 0], [0.2, 0], [0.2, 0], [0.2, 0], [0.2, 0], [0.2, 0]])


def get_actuator_id(model, data, joint_name):
    for i in range(model.nu):
        if data.joint(joint_name).id == model.actuator_trnid[i, 0]:
            return i
    return -1


def set_init_condition(model, data, init_condition: Dict[str, tuple[float, float]]):
    print("set_init_condition assumes that if a joint is actuated, then its actuator is defined as a postion one")
    for joint_name, init_cond in init_condition.items():
        actuator_id = get_actuator_id(model, data, joint_name)
        if actuator_id == -1:
            print(f"actuator not found for joint with name: {joint_name}")
        else:
            data.actuator(actuator_id).ctrl = init_cond[0]

        data.joint(joint_name).qpos = init_cond[0]
        data.joint(joint_name).qvel = init_cond[1]
        mujoco.mj_forward(model, data)


def set_init_cond(model, data, robot_init, robot_info):
    init_condition = {joint_name: robot_init[idx, :] for idx, joint_name in enumerate(robot_info["joints_names"])}
    init_condition.pop("finger_joint1")
    init_condition.pop("finger_joint2")
    set_init_condition(model, data, init_condition)


def create_scene(cube_spec, robot_path):
    spec = mujoco.MjSpec()
    robotloader = loader.RobotLoader(spec)
    robot_info = robotloader.add_robot(robot_path)

    objloader = loader.ObjectLoader(spec)
    objloader.add_cube(cube_spec["position"], cube_spec["scale"], np.array([144, 238, 144, 255]) / 255)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, robot_info


def get_trajectories(q0, qt, init, duration):
    coef = planner.cubic_polynomial_trajectory_coefs(q0, qt, 0, 0, duration)
    traj = planner.create_trajectory_func(coef, init, duration)
    return traj


def compute_currents(model, data):
    curr_pos = data.site(GRASP_SITE_NAME).xpos.copy()
    curr_xmat = data.site(GRASP_SITE_NAME).xmat.copy()
    curr_quat = np.zeros((4, 1))
    mujoco.mju_mat2Quat(curr_quat, curr_xmat)
    return curr_pos, curr_quat


def compute_desired():
    desire_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]).reshape((-1, 1))
    desired_quat = np.zeros((4, 1))
    mujoco.mju_mat2Quat(desired_quat, desire_mat)
    desired_pos = np.array(CUBE_SPEC["position"])
    desired_pos[2] += CUBE_SPEC["scale"][2]
    return desired_pos, desired_quat


def main():
    model, data, robot_info = create_scene(CUBE_SPEC, ROBOT_PATH)
    des_pos, des_quat = compute_desired()
    curr_pos, curr_quat = compute_currents(model, data)

    grasp_planner = planner.Planner()
    reach_initial_time = 0
    reach_duration = 2.5

    open_start_time = 0
    t_open = 2
    delay_time = 0.5
    t_close = 1

    grasp_traj = grasp_planner.create_grasp_trajectory(open_start_time, t_open, delay_time, t_close, np.array([[255]]), np.array([[90]]))

    reach_traj = grasp_planner.create_reach_trajectory(curr_pos, curr_quat, des_pos, des_quat, reach_initial_time, reach_duration)
    lift_pos = des_pos.copy()
    lift_pos[2] += 0.2
    lift_init_time = reach_initial_time + reach_duration + t_close
    lift_traj = grasp_planner.create_reach_trajectory(des_pos, des_quat, lift_pos, des_quat, lift_init_time, 1)

    def control_call(model, data):
        t = data.time
        desired_arr = None
        if t < lift_init_time:
            desired_arr = reach_traj(t)
        else:
            desired_arr = lift_traj(t)
        des_pp = desired_arr[:3].reshape((-1,))
        des_qq = desired_arr[3:].reshape((-1,))
        IKresult = IK.qpos_from_site_pose(model=model, data=data, site_name=GRASP_SITE_NAME, target_pos=des_pp, target_quat=des_qq, joint_names=None)
        print(IKresult)
        data.ctrl[0:7] = IKresult.qpos[0:7].copy()
        grasp_act_val = grasp_traj(t)
        data.ctrl[7] = grasp_act_val.copy()

    mujoco.set_mjcb_control(control_call)
    viewer.launch(model, data)


if __name__ == "__main__":
    main()

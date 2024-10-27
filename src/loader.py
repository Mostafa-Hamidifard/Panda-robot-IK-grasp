"""
This module is responsible for loading objects in a MuJoCo model
The syntax:
    Inputs: gets the model, gets AbstractOBJSpec containing obj properties when loading
    Outputs:  A new model containing previous and the loaded object.
"""

# %% importing necessary modules
import mujoco
import numpy as np
from mujoco import viewer
from typing import Dict, Tuple, List
import os


# %% custom exceptions
class InvalidDirectoryError(Exception):
    pass


# %% defining classes
def create_scene(spec: mujoco.MjSpec, rgba):
    ground_geom = spec.worldbody.add_geom()
    ground_geom.size = [0, 0, 0.01]
    ground_geom.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground_geom.rgba = rgba
    return None


def add_light(spec, light_pos=[0, 0, 5]):
    light = spec.worldbody.add_light()
    light.pos = light_pos


class RobotLoader:

    def __init__(self, spec: mujoco.MjSpec):
        assert isinstance(spec, mujoco.MjSpec)
        self.spec = spec

    def add_robot(self, path: str):
        if not os.path.exists(path):
            raise InvalidDirectoryError(f"The directory '{path}' is not valid.")
        self.spec.from_file(path)
        dummy_model = self.compile()
        joint_names = [mujoco.mj_id2name(dummy_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(dummy_model.nq)]
        actuator_names = [mujoco.mj_id2name(dummy_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(dummy_model.nu)]
        sensor_names = [mujoco.mj_id2name(dummy_model, mujoco.mjtSensor, i) for i in range(dummy_model.nsensor)]
        return {"actuators_names": actuator_names, "sensors_names": sensor_names, "joints_names": joint_names}

    def compile(self):
        return self.spec.compile()


class ObjectLoader:

    def __init__(self, spec: mujoco.MjSpec):
        self.spec = spec
        assert self.spec != None
        self.current_objects = []
        self.counter = 0

    def add_cube(self, pos, size, rgba):
        cube_body = self.add_free_body(body_position=pos)
        geom = cube_body.add_geom()
        geom.mass = 0.01
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = size
        geom.rgba = rgba
        return None

    def add_free_body(self, body_name=None, body_position=[0, 0, 0], body_quat=[1, 0, 0, 0]):
        body = self.spec.worldbody.add_body()
        body.name = body_name if isinstance(body_name, str) else ("AddedBody_" + str(self.counter))
        body.pos = body_position
        body.quat = body_quat
        jnt = body.add_joint()
        jnt.type = mujoco.mjtJoint.mjJNT_FREE
        self.counter += 1
        return body

    def compile(self):
        return self.spec.compile()


def _test_robot_loader_add_robot():
    ROBOT_PATH = r"D:/RAIIS lab/research/Shared control/gripper_grasp_cube/asset/franka_emika_panda/scene.xml"
    spec = mujoco.MjSpec()
    robot_loader = RobotLoader(spec)
    test_info = robot_loader.add_robot(ROBOT_PATH)
    print(test_info)
    model = robot_loader.compile()
    data = mujoco.MjData(model)
    viewer.launch(model, data)


def _test_robot_loader_init_cond():
    ROBOT_PATH = r"D:/RAIIS lab/research/Shared control/gripper_grasp_cube/asset/franka_emika_panda/scene.xml"
    spec = mujoco.MjSpec()
    robot_loader = RobotLoader(spec)
    test_info = robot_loader.add_robot(ROBOT_PATH)
    model = robot_loader.compile()
    data = mujoco.MjData(model)
    init_condition = {joint_name: [0.2, 0] for joint_name in test_info["joints_names"]}
    robot_loader.set_init_condition(model, data, init_condition)
    viewer.launch(model, data)


def _test_object_loader():
    spec = mujoco.MjSpec()
    add_light(spec)
    create_scene(spec, rgba=[0.7, 0.7, 0.7, 1])
    objloader = ObjectLoader(spec)
    objloader.add_cube([0, 0, 0.5], 3 * [0.2], rgba=np.array([24, 135, 181, 255]) / 255)
    model = objloader.compile()
    data = mujoco.MjData(model)
    viewer.launch(model, data)


if __name__ == "__main__":
    _test_robot_loader_init_cond()
    # _test_robot_loader_add_robot()
    # _test_object_loader()

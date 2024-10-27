"""
This module defines a sequential planner for robotic apps
"""

import numpy as np


class Planner:
    # def __init__(self, grasp_jnts_names, endeffector_jnts_names, endeffector_act_ids, grasp_act_ids):
    #     self.grasp_jnts = grasp_jnts_names
    #     self.ee_jnts = endeffector_jnts_names
    #     self.ee_act = endeffector_act_ids
    #     self.grasp_act = grasp_act_ids

    def create_grasp_trajectory(self, open_start_time, t_open, delay_time, t_close, act_open, act_close):
        zero = np.array([[0]])
        coef_open = Planner.cubic_polynomial_trajectory_coefs(zero, act_open, zero, zero, t_open)
        coef_close = Planner.cubic_polynomial_trajectory_coefs(act_open, act_close, zero, zero, t_close)
        traj_open = Planner.create_trajectory_func(coef_open, open_start_time, t_open)
        traj_close = Planner.create_trajectory_func(coef_close, open_start_time + t_open + delay_time, t_close)

        def grasp_traj(time):
            if time < open_start_time + t_open + delay_time:
                return traj_open(time)
            else:
                return traj_close(time)

        return grasp_traj

    @staticmethod
    def create_reach_trajectory(curr_pos, curr_quat, des_pos, des_quat, start_time, duration):
        p0 = np.concatenate((curr_pos.reshape((-1, 1)), curr_quat.reshape((-1, 1))), axis=0)
        pT = np.concatenate((des_pos.reshape((-1, 1)), des_quat.reshape((-1, 1))), axis=0)
        v0 = np.zeros_like(pT)
        vT = np.zeros_like(pT)
        coefs = Planner.cubic_polynomial_trajectory_coefs(p0, pT, v0, vT, duration)
        traj = Planner.create_trajectory_func(coefs, start_time, duration)
        return traj

    @staticmethod
    def create_trajectory_func(cubic_coef, initial_time, duration):
        if duration < 0:
            raise ValueError("duration must be positive.")

        def traj(time):
            t = time - initial_time
            if t < 0:
                t = 0
            if duration < t:
                t = duration
            return (np.array([[t**3, t**2, t, 1]]) @ cubic_coef).T

        return traj

    @staticmethod
    def cubic_polynomial_trajectory_coefs(p0: np.array, pT: np.array, v0: np.array, vT: np.array, T: float):
        assert p0.shape == pT.shape and v0.shape == pT.shape and v0.shape == vT.shape and (vT.shape[1] == 1)
        # Solve for cubic coefficients [a3, a2, a1, a0]
        point_stacked = np.stack((p0, pT, v0, vT), axis=0).squeeze()
        M = np.array(
            [
                [0, 0, 0, 1],
                [T**3, T**2, T, 1],
                [0, 0, 1, 0],
                [3 * T**2, 2 * T, 1, 0],
            ]
        )  # p(0) = p0  # p(T) = pT  # v(0) = 0  # v(T) = 0

        # Solve for coefficients [a3, a2, a1, a0]
        coeffs = np.linalg.inv(M) @ point_stacked
        return coeffs


# def trajectory_func_test():
#     import matplotlib.pyplot as plt

#     initial = 5
#     duration = 1
#     p0 = np.array([1, 10]).reshape((-1, 1))
#     pT = np.array([2, 5]).reshape((-1, 1))
#     v0 = np.array([0, 0]).reshape((-1, 1))
#     vT = np.array([0, 0]).reshape((-1, 1))
#     coef = Planner.cubic_polynomial_trajectory_coefs(p0, pT, v0, vT, duration)
#     traj = Planner.create_trajectory_func(coef, initial, duration)
#     t = np.linspace(0, 10, 100)
#     y = [traj(i) for i in t]
#     y = np.array(y).squeeze()
#     plt.plot(t, y[:, 0])
#     plt.plot(t, y[:, 1])
#     plt.show()


def main():
    # trajectory_func_test()
    pass


if __name__ == "__main__":
    main()

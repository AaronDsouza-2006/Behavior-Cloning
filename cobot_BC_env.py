import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import mujoco
import mujoco.viewer
from cobot_env import CobotEnv

class CobotBcEnv(CobotEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)
    
    def matrix_to_euler(self, R):
        # Check for gimbal lock
        if abs(R[2, 0]) < 1 - 1e-7:
            y = -np.arcsin(R[2, 0])
            z = np.arctan2(R[1, 0] / np.cos(y), R[0, 0] / np.cos(y))
            x = np.arctan2(R[2, 1] / np.cos(y), R[2, 2] / np.cos(y))
        else:
            # Gimbal lock case
            z = 0.0
            if R[2, 0] == -1:
                y = np.pi / 2
                x = z + np.arctan2(R[0, 1], R[0, 2])
            else:
                y = -np.pi / 2
                x = -z + np.arctan2(-R[0, 1], -R[0, 2])
                
        return np.degrees([x, y, z])

    def is_contact_present(self):
        left_touch = False
        right_touch = False
        cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
        left_gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "fr_link7_pad")
        right_gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "fr_link8_pad")

        for c in self.data.contact:
            g1, g2 = c.geom1, c.geom2

            if ((g1 == cube_geom_id and g2 == left_gripper_id) or
                (g2 == cube_geom_id and g1 == left_gripper_id)):
                left_touch = True

            if ((g1 == cube_geom_id and g2 == right_gripper_id) or
                (g2 == cube_geom_id and g1 == right_gripper_id)):
                right_touch = True

        return left_touch and right_touch
    
    def get_observation(self):
        motor_qpos = self.sim2real(self.data.qpos[-8:])
        motor_qvel = self.sim2real(self.data.qvel[-8:])
        cube_pos = self.data.xpos[1]
        disk_pos = self.data.xpos[2]
        gripper_pos = self.data.site_xpos[1]
        gripper_rot = self.matrix_to_euler(
            self.data.site_xmat[1].reshape(3, 3)
        )
        is_contact = [1.0 if self.is_contact_present() else 0.0,]
        all_observations = np.concatenate((motor_qpos, motor_qvel, cube_pos, disk_pos, 
                                           gripper_pos, gripper_rot, is_contact), dtype=np.float32)
        
        return all_observations
import mujoco
import mujoco.viewer
import sys
import numpy as np
from cobot_env import CobotEnv

class IKExpert():
    def __init__(self, env = CobotEnv(render_mode="human")):
        self.env = env
        self.gripper_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SITE, "fr_gripper_center")
        self.env.reset()

    def damped_pseudoinverse(self, J, damping=0.01):
        I = np.eye(J.shape[0])
        return J.T @ np.linalg.solve(
            J @ J.T + damping**2 * I,
            I
        )
    
    def get_jacobian(self):
        jacp = np.zeros((3, self.env.model.nv))
        jacr = np.zeros((3, self.env.model.nv))
        mujoco.mj_jacSite(self.env.model, self.env.data, jacp, jacr, self.gripper_id)
        jac = np.vstack((jacp, jacr))
        J_arm = jac[:, -8:-2]
        return J_arm

    def euler_to_matrix(self, euler):
        z, y, x = euler
        c1, c2, c3 = np.cos([z, y, x])
        s1, s2, s3 = np.sin([z, y, x])
        return np.array([[c1*c2, c1*s2*s3 - s1*c3, c1*s2*c3 + s1*s3],
                        [s1*c2, s1*s2*s3 + c1*c3, s1*s2*c3 - c1*s3],
                        [-s2,   c2*s3,            c2*c3]])


    def get_orientation_error(self, target_rot_mat):
        current_rot = self.env.data.site_xmat[self.gripper_id].reshape(3, 3)
        R_error = target_rot_mat @ current_rot

        return 0.5 * np.array([
            R_error[2,1] - R_error[1,2],
            R_error[0, 2] - R_error[2, 0],
            R_error[1, 0] - R_error[0, 1]
        ])

    def move_to_step(self, target_pos, target_rot_mat=None, max_speed = 0.2,
                         tolerance_pos=0.01, tolerance_rot = 0.2):

        current_pos = self.env.data.site_xpos[self.gripper_id]

        pos_error = target_pos - current_pos
        pos_err_norm = np.linalg.norm(pos_error)

        if target_rot_mat is not None:
            rot_error = self.get_orientation_error(target_rot_mat)
            rot_err_norm = np.linalg.norm(rot_error)
        else: 
            rot_err_norm = 0         
        
        if (pos_err_norm < tolerance_pos and rot_err_norm < tolerance_rot):
            return np.zeros(7), True

        J= self.get_jacobian()

        Jp = J[:3, :]
        Jp_pinv = self.damped_pseudoinverse(Jp)
        dq_pos = Jp_pinv @ pos_error

        if target_rot_mat is not None:
            Jo = J[3:, :]

            N = np.eye(6) - Jp_pinv @ Jp
            Jo_proj = Jo @ N
            Jo_proj_pinv = self.damped_pseudoinverse(Jo_proj)
            dq_ori = Jo_proj_pinv @ rot_error
        else:
            dq_ori = np.zeros(6)

        if (pos_err_norm < tolerance_pos):
            dq = dq_ori
        else:
            dq = dq_pos + 0.15 * dq_ori

        speed = np.linalg.norm(dq)
        # print(speed)
        if speed > max_speed:
            dq = dq * (max_speed/speed)

        return dq, False

    def move_to(self, target_pos, target_rot=None, grasp = True, max_itr=500, max_speed = 0.2,
                         tolerance_pos=0.01, tolerance_rot = 0.2):
        if target_rot is not None:
            target_rot_mat = self.euler_to_matrix(target_rot)
        else:
            target_rot_mat = None
        for itr in range(max_itr):
            joint_dq, is_end = self.move_to_step(target_pos, target_rot_mat, max_speed, tolerance_pos, tolerance_rot)
            if(is_end):
                print(f"Reached target in {itr} iterations")
                break

            gripper_dq = [0.5, 0.5] if not grasp else [-0.5, -0.5]
            dq = np.concatenate([joint_dq, gripper_dq])
            action = self.env.sim2real(dq)
            self.env.apply_action(action)
            self.env.render()
        else:
            print(f"Max iters")

    def perform_grasp(self, close, itr=5):
        gripper_dq = [0.5, 0.5] if not close else [-0.5, -0.5]
        dq = np.concatenate([np.zeros(6), gripper_dq])
        action = self.env.sim2real(dq)
        for _ in range(itr):
            expert.env.apply_action(action)

    def is_contact_present(self):
        left_touch = False
        right_touch = False
        cube_geom_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
        left_gripper_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "fr_link7_pad")
        right_gripper_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "fr_link8_pad")

        for c in self.env.data.contact:
            g1, g2 = c.geom1, c.geom2

            if ((g1 == cube_geom_id and g2 == left_gripper_id) or
                (g2 == cube_geom_id and g1 == left_gripper_id)):
                left_touch = True

            if ((g1 == cube_geom_id and g2 == right_gripper_id) or
                (g2 == cube_geom_id and g1 == right_gripper_id)):
                right_touch = True

        return left_touch and right_touch

if __name__ == "__main__":

    expert = IKExpert()
    try:
        cube_position = expert.env.data.xpos[1] + np.array([0.0, 0.0, 0.015])
        cube_approach_position = cube_position + np.array([0.0, 0.0, 0.10])

        disk_position = expert.env.data.xpos[2] + np.array([0.0, 0.0, 0.05])
        disk_approach_position = disk_position + np.array([0.0, 0.0, 0.10])
        target_rot = np.array([0, 1.57, 0])

        #pick
        expert.move_to(cube_approach_position, target_rot=target_rot, grasp=True, max_itr=300)
        expert.move_to(cube_position,  target_rot=None, tolerance_pos=0.005, grasp=False, max_itr=50, max_speed=0.05)
        expert.perform_grasp(True, itr=5)
        # expert.grasp()
        expert.move_to(cube_approach_position, target_rot=None, grasp=True, max_itr=50, max_speed=0.05)

        #place
        expert.move_to(disk_approach_position, target_rot=None, grasp=True, max_itr=100)
        expert.move_to(disk_position, target_rot=None, grasp=False, max_itr=50, max_speed=0.02)
        expert.move_to(disk_approach_position, target_rot=None, grasp = False, max_itr=50, max_speed=0.05)

    except KeyboardInterrupt:
        sys.exit(0)
from IK_expert import IKExpert
import mujoco
import mujoco.viewer
import sys
import numpy as np
import datasets
from cobot_BC_env import CobotBcEnv

from lerobot.datasets import LeRobotDataset

FPS = 1/(0.002*20) # 20 mujoco steps
NUM_EPISODES = 100

class RecordExpert(IKExpert):
    def __init__(self, env = CobotBcEnv(render_mode="human"),
                 name = 'pick_place_ik_test'
                 ):
        super().__init__(env)

        dataset_path = "./datasets/pick_place_ik_test"

        features = {
            "observation.state" : {
                "dtype": "float32",
                "shape" : (27,),
                "names": [
                    "joint_1", "joint_2", "joint_3", "joint_4",
                    "joint_5", "joint_6", "joint_7",
                    "vel_1", "vel_2", "vel_3", "vel_4",
                    "vel_5", "vel_6", "vel_7",
                    "cube_x", "cube_y", "cube_z",
                    "disk_x", "disk_y", "disk_z",
                    "ee_x", "ee_y", "ee_z",
                    "ee_rx", "ee_ry", "ee_rz",
                    "contact",
                ],
                "type": "STATE",
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "type": "ACTION",
            },
        }

        try:
            self.dataset = LeRobotDataset.create(
                repo_id = "aaron/" + name,
                fps=FPS,
                features = features,
                robot_type = "cobot_sim",
                use_videos=False,
                root= './datasets/' + name,
            )
        except:
            self.dataset = LeRobotDataset.resume(
                repo_id = "aaron/" + name,
                root= './datasets/' + name,
            )

    def record_move_to(self, target_pos, target_rot=None, grasp = True, max_itr=500, max_speed = 0.2,
                         tolerance_pos=0.01, tolerance_rot = 0.2, viewer=None):
        if target_rot is not None:
            target_rot_mat = self.euler_to_matrix(target_rot)
        else:
            target_rot_mat = None
        for itr in range(max_itr):
            observation = self.env.get_observation()
            joint_dq, is_end = self.move_to_step(target_pos, target_rot_mat, max_speed, tolerance_pos, tolerance_rot)
            
            if(is_end):
                # print(f"Reached target in {itr} iterations")
                break

            gripper_dq = [0.5, 0.5] if not grasp else [-0.5, -0.5]
            dq = np.concatenate([joint_dq, gripper_dq], dtype=np.float32)
            action = self.env.sim2real(dq)
            self.env.apply_action(action)
            if self.env.render_mode == 'human':
                self.env.render()

            self.dataset.add_frame({
                "observation.state": observation,
                "action": action,
                "task": "pick up the cube and place it on the disk",
            })
        else:
            # print(f"Max iters")
            pass

    def perform_grasp(self, close, itr=5):
        gripper_dq = [0.5, 0.5] if not close else [-0.5, -0.5]
        dq = np.concatenate([np.zeros(6), gripper_dq], dtype=np.float32)
        action = self.env.sim2real(dq)
        for _ in range(itr):
            observation = self.env.get_observation()
            expert.env.apply_action(action)
            self.dataset.add_frame({
                "observation.state": observation,
                "action": action,
                "task": "pick up the cube and place it on the disk",
            })

    def record_episode(self):
        observation, _ = self.env.reset()
        cube_position = np.array(observation[14:17]) + np.array([0.0, 0.0, 0.015])
        disk_position = np.array(observation[17:20]) + np.array([0.0, 0.0, 0.05])
        
        cube_approach_position = cube_position + np.array([0.0, 0.0, 0.10])
        
        disk_approach_position = disk_position + np.array([0.0, 0.0, 0.10])
        target_rot = np.array([0, 1.57, 0])

        #pick
        self.record_move_to(cube_approach_position, target_rot=target_rot, grasp=True, max_itr=300)
        # self.grasp(False)
        self.record_move_to(cube_position, tolerance_pos=0.005, max_speed=0.02, grasp=False, max_itr=50)
        expert.perform_grasp(True, itr=5)

        #place
        self.record_move_to(disk_approach_position, grasp=True, max_itr=100)
        self.record_move_to(disk_position, grasp=False, max_itr=50, max_speed=0.02)
        self.record_move_to(disk_approach_position, grasp=True, max_itr=50)

        final_obs = self.env.get_observation()
        cube_final_pos = np.array(final_obs[14:17]) 
        disk_final_pos = np.array(final_obs[17:20]) 
        cube_disk_distance = np.linalg.norm(cube_final_pos-disk_final_pos)
        if(cube_disk_distance<0.1):
            print('saving episode')
            self.dataset.save_episode()
        else:
            print('discarding episode')
            self.dataset.clear_episode_buffer()

if __name__=="__main__":
    datasets.logging.disable_progress_bar()
    expert = RecordExpert(name='pick_place_200')
    n_episodes = 2000
    while expert.dataset.num_episodes < n_episodes:
        expert.record_episode()
        print(str(expert.dataset.num_episodes))
    expert.dataset.finalize()
    print('dataset created')
    expert.env.close()
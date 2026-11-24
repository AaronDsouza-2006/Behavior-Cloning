from IK_expert import IKExpert
import numpy as np


class PickPlaceExpert(IKExpert):

    APPROACH_CUBE = "approach_cube"
    DESCEND_CUBE = "descend_cube"
    GRASP = "grasp"
    APPROACH_DISK = "approach_disk"
    DESCEND_DISK = "descend_disk"
    RETREAT = "retreat"
    DONE = "done"

    def __init__(self, env):
        super().__init__(env)
        self.reset_task()

    def reset_task(self):
        """Reset expert's internal task state. Does not reset MuJoCo."""
        self.phase = self.APPROACH_CUBE
        self.lost_contact_steps = 0

    def _get_action(
        self,
        target_pos,
        target_rot=None,
        gripper_closed=True,
        max_speed=0.2,
        tolerance_pos=0.01,
        tolerance_rot=0.2
    ):
        target_rot_mat = (
            self.euler_to_matrix(target_rot)
            if target_rot is not None else None
        )

        joint_dq, reached = self.move_to_step(
            target_pos,
            target_rot_mat,
            max_speed,
            tolerance_pos,
            tolerance_rot
        )

        # Same convention as your RecordExpert
        gripper_dq = (
            [-0.5, -0.5] if gripper_closed else [0.5, 0.5]
        )

        dq = np.concatenate([joint_dq, gripper_dq]).astype(np.float32)
        action = self.env.sim2real(dq)

        return action, reached

    def _gripper_action(self, closed):
        gripper_dq = [-0.5, -0.5] if closed else [0.5, 0.5]
        dq = np.concatenate([
            np.zeros(6),
            gripper_dq
        ]).astype(np.float32)

        return self.env.sim2real(dq)

    def __call__(self, state):

        if hasattr(state, "detach"):
            state = state.detach().cpu().numpy()

        state = np.asarray(state, dtype=np.float32).reshape(-1)

        if state.shape != (27,):
            raise ValueError(f"Expected state dimension 27, got {state.shape}")

        cube_pos = state[14:17] + np.array([0.0, 0.0, 0.015])
        disk_pos = state[17:20] + np.array([0.0, 0.0, 0.05])

        cube_approach = cube_pos + np.array([0.0, 0.0, 0.15])
        disk_approach = disk_pos + np.array([0.0, 0.0, 0.15])

        cube_disk_distance = np.linalg.norm(cube_pos - disk_pos)
        contact = self.is_contact_present()

        # -----------------------------------------
        # 1. APPROACH CUBE
        # -----------------------------------------
        if self.phase == self.APPROACH_CUBE:

            action, reached = self._get_action(
                cube_approach,
                target_rot=np.array([0, 1.57, 0]),
                gripper_closed=True
            )

            if reached:
                self.phase = self.DESCEND_CUBE

            return action

        # -----------------------------------------
        # 2. DESCEND CUBE
        # -----------------------------------------
        elif self.phase == self.DESCEND_CUBE:

            action, reached = self._get_action(
                cube_pos,
                gripper_closed=False,
                max_speed=0.02,
                tolerance_pos=0.005
            )

            if reached:
                self.phase = self.GRASP

            return action

        # -----------------------------------------
        # 3. GRASP
        # -----------------------------------------
        elif self.phase == self.GRASP:

            action, _ = self._get_action(
                cube_pos,
                gripper_closed=True,
                max_speed=0.02,
                tolerance_pos=0.005
            )

            if contact:
                self.phase = self.APPROACH_DISK

            return action

        # -----------------------------------------
        # 4. APPROACH DISK
        # -----------------------------------------
        elif self.phase == self.APPROACH_DISK:

            action, reached = self._get_action(
                disk_approach,
                gripper_closed=True,
                max_speed=0.1
            )

            if not contact:
                self.lost_contact_steps += 1
            else:
                self.lost_contact_steps = 0

            # # Object was dropped: attempt to reacquire it
            # if self.lost_contact_steps >= 5 and cube_disk_distance > 0.1:
            #     self.phase = self.APPROACH_CUBE
            #     self.lost_contact_steps = 0

            if reached:
                self.phase = self.DESCEND_DISK

            return action

        # -----------------------------------------
        # 5. DESCEND DISK / RELEASE
        # -----------------------------------------
        elif self.phase == self.DESCEND_DISK:

            action, reached = self._get_action(
                disk_pos,
                gripper_closed=True,
                max_speed=0.01,
                tolerance_pos=0.005
            )

            if cube_disk_distance < 0.05:
                self.phase = self.RETREAT

            # elif not contact:
            #     self.lost_contact_steps += 1

            #     if self.lost_contact_steps >= 5:
            #         self.phase = self.APPROACH_CUBE
            #         self.lost_contact_steps = 0

            # else:
            #     self.lost_contact_steps = 0

            return action

        # -----------------------------------------
        # 6. RETREAT
        # -----------------------------------------
        elif self.phase == self.RETREAT:

            action, reached = self._get_action(
                disk_approach,
                gripper_closed=False,
                max_speed=0.05
            )

            if reached:
                self.phase = self.DONE

            return action

        # -----------------------------------------
        # 7. DONE
        # -----------------------------------------
        elif self.phase == self.DONE:

            return None

        raise RuntimeError(f"Unknown expert phase: {self.phase}")

    @property
    def done(self):
        return self.phase == self.DONE


if __name__ == "__main__":
    from cobot_BC_env import CobotBcEnv

    env = CobotBcEnv(render_mode="human")
    expert = PickPlaceExpert(env)

    for _ in range(10):
        obs, _ = env.reset()
        expert.reset_task()
        for step in range(500):

            state = env.get_observation()

            expert_action = expert(state)

            # For testing, execute expert action directly
            env.apply_action(expert_action)
            env.render()

            if expert.done:
                print("Expert completed task")
                break
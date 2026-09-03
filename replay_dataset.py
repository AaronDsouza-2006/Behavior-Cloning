import numpy as np
import time
from lerobot.datasets import LeRobotDataset
from cobot_BC_env import CobotBcEnv

DATASET_PATH = "./datasets/pick_place_ik_test"
EPISODE = 0

dataset = LeRobotDataset(
    repo_id="aaron/pick_place_ik_test",
    root=DATASET_PATH,
    download_videos=False,
)

env = CobotBcEnv(render_mode="human")
env.reset()

episode_indices = dataset.hf_dataset["episode_index"]
frames = np.where(np.array(episode_indices) == EPISODE)[0]

print(f"Replaying episode {EPISODE}")
print(f"Frames: {len(frames)}")

for idx in frames:
    frame = dataset[int(idx)]
    action = frame["action"].numpy()

    env.apply_action(action)
    env.render()
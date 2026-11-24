import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import mujoco

from cobot_BC_env import CobotBcEnv

from torch.utils.tensorboard import SummaryWriter
import time
from lerobot.datasets import LeRobotDataset
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.processor.normalize_processor import UnnormalizerProcessorStep
from torch.utils.data import DataLoader, Subset

class MlpNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        
        layers = []
        in_dim = state_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, action_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        action = self.network(state)
        return action


class BcDataset:

    def __init__(self, name, val_split=0.1):
        
        full_dataset = LeRobotDataset(
            root="./datasets/" + name,
            repo_id="aaron/" + name,
            #download_videos=False,
        )

        episode_indices = np.array(full_dataset.hf_dataset["episode_index"])
        episodes = np.unique(episode_indices)

        rng = np.random.default_rng(42)
        rng.shuffle(episodes)

        n_val = max(1, int(len(episodes) * val_split))

        val_episodes = episodes[:n_val]
        train_episodes = episodes[n_val:]

        train_indices = np.where(np.isin(episode_indices, train_episodes))[0]
        val_indices = np.where(np.isin(episode_indices, val_episodes))[0]

        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)

        # 5. Create native PyTorch Dataloaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=32, 
            shuffle=True
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=32, 
            shuffle=False  
        )

        self.normalizer =  NormalizerProcessorStep.from_lerobot_dataset(
            dataset=full_dataset,
            features={
                "observation.state": full_dataset.features["observation.state"],
                "action": full_dataset.features["action"],
            },
            norm_map={
                "ACTION": "MEAN_STD",          
                "STATE": "MEAN_STD" 
            },
            device = "cpu"
        )

        self.unnormalizer =  UnnormalizerProcessorStep.from_lerobot_dataset(
            dataset=full_dataset,
            features={
                "observation.state": full_dataset.features["observation.state"],
                "action": full_dataset.features["action"],
            },
            norm_map={
                "ACTION": "MEAN_STD",          
                "STATE": "MEAN_STD" 
            },
            device = "cpu"
        )


class BehaviorCloning:
    def __init__(self, CobotCubeEnv, dataset, hidden_dims=(256, 256)):
        self.env = CobotCubeEnv

        state_dim = 7+7+3+3+3+3+1 #27
        action_dim = 7

        self.policy = MlpNetwork(state_dim, action_dim, hidden_dims)

        self.writer = SummaryWriter(f"run/{int(time.time())}")

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.dataset = dataset

    def get_val_loss(self):
        self.policy.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in self.dataset.val_loader:
                batch = self.dataset.normalizer(batch)
                obs = batch['observation.state']
                action = batch['action']

                pred_action = self.policy(obs)
                action_loss = self.criterion(pred_action, action)

                running_loss += action_loss.item()

        avg_val_loss = running_loss / len(self.dataset.val_loader)
        return avg_val_loss

    def train(self, epochs, add_noise = True):
        self.policy.to('cpu')
        obs_noise_std = 0.015
        for epoch in range(epochs):
            self.policy.train()
            running_loss = 0.0
            for batch in self.dataset.train_loader:
                batch = self.dataset.normalizer(batch)

                obs = batch['observation.state']
                action = batch['action']

                # if add_noise:
                #     noise = torch.randn_like(obs) * obs_noise_std
                #     obs = obs+noise

                pred_action = self.policy(obs)

                action_loss = self.criterion(pred_action, action)
                self.policy_optimizer.zero_grad()
                action_loss.backward()
                self.policy_optimizer.step()
                running_loss += action_loss.item()

            train_loss = running_loss / len(self.dataset.train_loader)
            val_loss = self.get_val_loss()

            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

        self.writer.close()

    @torch.no_grad()
    def rollout(self, n_steps=300):
        self.env.reset()
        self.policy.eval()
        for i in range(n_steps):
            raw_state = self.env.get_observation()
            state_dict = {"observation.state": torch.from_numpy(raw_state).float().to('cpu')}
            state = self.dataset.normalizer(state_dict)["observation.state"]

            normalized_action = self.policy(state)
            
            action_dict = {"action": normalized_action}
            action = self.dataset.unnormalizer(action_dict)["action"].cpu().numpy()

            self.env.apply_action(action)
            self.env.render()

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim: int, data_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, data_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, data_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
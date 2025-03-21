import torch
import torch.nn as nn
import torch.nn.functional as F

class SFMNNEncoder(nn.Module):
    def __init__(self, input_channels=3623, num_variables=9, latent_dim=3620):
        super(SFMNNEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_norm = nn.BatchNorm1d(input_channels)
        self.fc_layers = nn.Sequential(
            nn.Linear(input_channels, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8192, 16384),
            nn.BatchNorm1d(16384),
            nn.ReLU(),
            nn.Linear(16384, 16384),
            nn.BatchNorm1d(16384),
            nn.ReLU()
        )
        self.latent_proj = nn.Linear(16384, num_variables * latent_dim)
    
    def forward(self, x):
        """
        x: [B, C, H, W] with C = num_wavelengths+3.
        Output: [B, H, W, 9, latent_dim].
        """
        B, C, H, W = x.shape
        x_flat = x.view(B * H * W, C)
        x_norm = self.input_norm(x_flat)
        x_fc = self.fc_layers(x_norm)
        latent = self.latent_proj(x_fc)
        latent = latent.view(B, H, W, -1, self.latent_dim)
        return latent
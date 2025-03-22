import torch
import torch.nn as nn
import torch.nn.functional as F

class SFMNNEncoder(nn.Module):
    def __init__(self, input_channels=3623, num_variables=9, latent_dim=256, out_dim=3620):
        """
        Args:
          input_channels: number of input channels (num_wavelengths + 3 extras).
          num_variables: number of latent streams to be generated.
          latent_dim: compact latent dimensionality (e.g. 256).
          out_dim: desired output dimensionality per stream (e.g. number of wavelengths, 3620).
        """
        super(SFMNNEncoder, self).__init__()
        self.num_variables = num_variables
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        
        # Input normalization over channels.
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
            nn.Linear(8192, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
        )
        # Project flattened features to a compact latent code.
        self.latent_proj = nn.Linear(8192, num_variables * latent_dim)
        # Now project the compact latent vector to the full high-res space.
        self.project = nn.Linear(latent_dim, out_dim)
    
    def forward(self, x):
        """
        Args:
          x: [B, C, H, W] with C = num_wavelengths + 3.
        Returns:
          latent_projected: [B, H, W, num_variables, out_dim]
        """
        B, C, H, W = x.shape
        x_flat = x.view(B * H * W, C)
        x_norm = self.input_norm(x_flat)
        x_fc = self.fc_layers(x_norm)
        # latent has shape [B*H*W, num_variables * latent_dim]
        latent = self.latent_proj(x_fc)
        # Reshape to [B, H, W, num_variables, latent_dim]
        latent = latent.view(B, H, W, self.num_variables, self.latent_dim)
        # Project each latent vector to the full spectral dimension.
        latent_projected = self.project(latent)  # shape: [B, H, W, num_variables, out_dim]
        return latent_projected

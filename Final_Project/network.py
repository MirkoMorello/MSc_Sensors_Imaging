import torch
import torch.nn as nn
import torch.nn.functional as F

class SFMNNEncoder(nn.Module):
    def __init__(self,
                 input_channels=3623,  # 3620 spectral + 3 metadata channels
                 num_variables=9,      # Number of output variables
                 latent_dim=64):       # Latent dimension per variable
        super(SFMNNEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        # BatchNorm1d normalizes over the feature dimension (C)
        self.input_norm = nn.BatchNorm1d(input_channels)
        
        # Fully connected layers
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
        
        # Final projection to multi-variable latent space:
        # Output dimension = num_variables * latent_dim (i.e. 9*64 = 576)
        self.latent_proj = nn.Linear(16384, num_variables * latent_dim)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, C, H, W] with C=3623.
        Returns:
            latent: Tensor of shape [B, H, W, num_variables, latent_dim],
                    i.e. [B, H, W, 9, 64].
        """
        B, C, H, W = x.shape
        # Flatten spatial dimensions; shape becomes [B*H*W, C]
        x_flat = x.view(B * H * W, C)
        # Apply normalization
        x_norm = self.input_norm(x_flat)
        # Process through fully connected layers
        x_fc = self.fc_layers(x_norm)
        # Project to latent space; output shape [B*H*W, 9*latent_dim]
        latent = self.latent_proj(x_fc)
        # Reshape to [B, H, W, 9, latent_dim]
        latent = latent.view(B, H, W, -1, self.latent_dim)
        return latent

import torch
import torch.nn as nn
import torch.nn.functional as F

class SFMNNEncoder(nn.Module):
    def __init__(self,
                 input_channels=3623,  # 3620 spectral + 3 metadata
                 num_variables=9,     # Number of output variables
                 latent_dim=3623):
        super(SFMNNEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_channels)
        
        # Fully connected sequential layers (extended depth)
        self.fc_layers = nn.Sequential(
            # First FC block
            nn.Linear(input_channels, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Second FC block
            nn.Linear(4096, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Third FC block
            nn.Linear(8192, 16384),
            nn.BatchNorm1d(16384),
            nn.ReLU(),
            
            # Fourth FC block
            nn.Linear(16384, 16384),
            nn.BatchNorm1d(16384),
            nn.ReLU()
        )
        
        # Final projection to multi-variable latent space
        self.latent_proj = nn.Linear(16384, num_variables * latent_dim) # 9*64 = 576
    
    def forward(self, x):
        """
        Input: x with shape [B, H, W, 3620]
        Output: latent with shape [B, H, W, 9, 64]
        """
        B, H, W, C = x.shape
        # Flatten spatial dimensions so each pixel is processed independently
        x_flat = x.view(-1, C)  # shape: [B*H*W, 3620]
        
        # Normalize
        x_norm = self.input_norm(x_flat)
        
        # Process through the sequential fully connected layers
        x_fc = self.fc_layers(x_norm)  # shape: [B*H*W, 50]
        
        # Project to the multi-variable latent space
        latent = self.latent_proj(x_fc)  # shape: [B*H*W, 9*64]
        
        # Reshape to [B, H, W, 9, 64]
        latent = latent.view(B, H, W, -1, self.latent_dim)
        
        
        return latent
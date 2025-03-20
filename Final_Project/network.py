import torch
import torch.nn as nn
import torch.nn.functional as F

class SFMNNEncoder(nn.Module):
    def __init__(self,
                 input_channels=244,  # 241 spectral + 3 metadata
                 num_variables=9,     # Number of output variables
                 latent_dim=64):
        super().__init__()
        
        # Architectural parameters from paper's Table 4
        self.layer_dims = [100, 50, 50]
        self.layer_repeats = [3, 3, 1]
        self.dropout_rates = [0.1, 0.1, 0]
        self.num_variables = num_variables
        self.latent_dim = latent_dim

        # Input normalization and reduction
        self.input_norm = nn.BatchNorm1d(input_channels)
        self.dim_reduction = nn.Linear(input_channels, self.layer_dims[0])
        
        # Shared residual blocks
        self.res_blocks = nn.ModuleList()
        current_dim = self.layer_dims[0]
        
        for dim, repeats, dropout in zip(self.layer_dims, self.layer_repeats, self.dropout_rates):
            for _ in range(repeats):
                self.res_blocks.append(
                    ResidualBlock(current_dim, dim, dropout)
                )
                current_dim = dim

        # Final projection to multi-variable latent space
        self.latent_proj = nn.Linear(current_dim, num_variables * latent_dim)

    def forward(self, x):
        """
        Input: [B, H, W, 244] 
        Output: [B, H, W, 9, 64] where 9 variables each get 64D latent
        """
        B, H, W, C = x.shape
        
        # Process each pixel independently
        x_flat = x.reshape(-1, C)  # [B*H*W, 244]
        
        # Normalization and reduction
        x = self.input_norm(x_flat)
        x = F.relu(self.dim_reduction(x))  # [B*H*W, 100]
        
        # Shared residual processing
        for block in self.res_blocks:
            x = block(x)  # Final shape [B*H*W, 50]
            
        # Project to variable-specific latent space
        latent = self.latent_proj(x)  # [B*H*W, 9*64]
        
        # Reshape to [B, H, W, 9, 64]
        latent = latent.view(B, H, W, self.num_variables, self.latent_dim)
        
        return latent

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_p=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x):
        return F.relu(self.block(x) + self.shortcut(x))
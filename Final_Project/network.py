import torch
import torch.nn as nn

class ResidualMLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Residual connection
        self.residual = nn.Identity()
        if in_dim != out_dim:
            self.residual = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
        
    def forward(self, x):
        identity = self.residual(x)
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out + identity

class SFMNNEncoder(nn.Module):
    def __init__(self, spectral_bands):
        super().__init__()
        
        # Input parameters from paper
        self.spectral_bands = spectral_bands  # FLUO module channels
        self.additional_vars = 3    # Flight height, SZA, across-track position
        self.total_input_dim = self.spectral_bands + self.additional_vars
        
        # Architecture parameters from Table 4
        self.dims = [100, 50, 50]
        self.repeats = [3, 3, 1]
        self.dropout_rates = [0.1, 0.0, 0.0]
        
        # Input normalization (per-pixel BN)
        self.input_norm = nn.BatchNorm1d(self.total_input_dim)
        
        # Dimensionality reduction (k_in)
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.total_input_dim, self.dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(self.dims[0])
        )
        
        # Build encoder blocks (h modules)
        self.blocks = nn.ModuleList()
        current_dim = self.dims[0]
        
        for i, (dim, n_repeat) in enumerate(zip(self.dims, self.repeats)):
            for _ in range(n_repeat):
                block = nn.Sequential(
                    ResidualMLPBlock(current_dim, dim, self.dropout_rates[i]),
                    nn.BatchNorm1d(dim),
                    nn.ReLU()
                )
                self.blocks.append(block)
                current_dim = dim
        
        # Final projection
        self.final_proj = nn.Linear(current_dim, 50)  # Output latent dimension
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W] where:
                - C = 1024 spectral bands + 3 additional variables
                - H, W = spatial dimensions (e.g., 17x17 patch)
        
        Returns:
            p_xy: Encoded features [B, latent_dim, H, W]
        """
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions
        x = x.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        # Input normalization
        x = self.input_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Dimensionality reduction
        x = self.dim_reduction(x)
        
        # Process through encoder blocks
        for block in self.blocks:
            x = block(x)
            
        # Final projection
        p_xy = self.final_proj(x)
        
        # Reshape back to spatial dimensions
        p_xy = p_xy.permute(0, 2, 1).view(B, -1, H, W)
        
        return p_xy
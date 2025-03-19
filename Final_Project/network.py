import torch
import torch.nn as nn
import torch.nn.functional as F

class DimReduction(nn.Module):
    """
    Reduces the input dimensionality from (C) to reduced_dim.
    Expects input shape (N, C) where N = B * H * W.
    """
    def __init__(self, input_dim=1027, reduced_dim=100):
        super(DimReduction, self).__init__()
        self.fc = nn.Linear(input_dim, reduced_dim)
        self.bn = nn.BatchNorm1d(reduced_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [N, input_dim]
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualMLPBlock(nn.Module):
    """
    A residual block that applies a linear layer with batch normalization,
    ReLU activation, optional dropout, and a skip connection.
    """
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super(ResidualMLPBlock, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.relu = nn.ReLU()
        # Adjust the skip connection if the dimensions differ.
        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = out + residual
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    """
    Encoder module composed of a sequence of residual MLP blocks.
    According to the paper's Table 4, the encoder (e_in) has:
      - Stage 1: Three blocks that transform from 100 to 50 channels (dropout=0.1)
      - Stage 2: Three blocks keeping dimension 50 (dropout=0.0)
      - Stage 3: One final residual block (dropout=0.0)
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.stage1 = nn.Sequential(
            ResidualMLPBlock(100, 50, dropout=0.1),
            ResidualMLPBlock(50, 50, dropout=0.1),
            ResidualMLPBlock(50, 50, dropout=0.1)
        )
        self.stage2 = nn.Sequential(
            ResidualMLPBlock(50, 50, dropout=0.0),
            ResidualMLPBlock(50, 50, dropout=0.0),
            ResidualMLPBlock(50, 50, dropout=0.0)
        )
        self.stage3 = ResidualMLPBlock(50, 50, dropout=0.0)

    def forward(self, x):
        # x shape: [N, 100]
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x

class SFMNNEncoder(nn.Module):
    """
    Complete encoder that works with input patches of shape [B, H, W, C]
    where C = 1024 (spectral channels) + 3 (extra variables).
    The network processes each pixel (flattening the H and W dimensions),
    applies dimensionality reduction and the encoder, and then reshapes
    the output back to [B, H, W, latent_dim] (latent_dim=50).
    """
    def __init__(self, input_dim=1027, reduced_dim=100):
        super(SFMNNEncoder, self).__init__()
        self.dim_red = DimReduction(input_dim, reduced_dim)
        self.encoder = Encoder()

    def forward(self, x):
        # x shape: [B, H, W, C]
        B, H, W, C = x.shape
        # Flatten spatial dimensions: [B*H*W, C]
        x = x.reshape(B * H * W, C)
        x = self.dim_red(x)  # Shape: [B*H*W, reduced_dim] (e.g., [B*H*W, 100])
        x = self.encoder(x)  # Shape: [B*H*W, latent_dim] (latent_dim is 50)
        # Reshape back to spatial layout: [B, H, W, latent_dim]
        x = x.reshape(B, H, W, -1)
        return x

# Example usage:
if __name__ == "__main__":
    # Suppose we have a batch of patches with each patch of shape [H, W, C],
    # e.g., H = W = 17 and C = 1027 (1024 spectral + 3 extra variables)
    dummy_input = torch.randn(8, 17, 17, 1027)  # Batch size = 8
    model = SFMNNEncoder()
    latent = model(dummy_input)
    print("Latent representation shape:", latent.shape)  # Expected: [8, 17, 17, 50]

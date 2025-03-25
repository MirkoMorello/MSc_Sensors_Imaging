import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class SFMNNEncoder(nn.Module):
    def __init__(self, input_channels=3623, num_variables=11, out_dim=3620):
        """
        Args:
          input_channels: Number of input channels (num_wavelengths + 3 extras).
          num_variables: Number of latent streams (variables) to be generated.
          out_dim: Desired output dimensionality per stream (e.g., number of wavelengths, typically 3620).
          
        The network projects each pixel's input directly into a space of dimension 
        (num_variables * out_dim), which is reshaped to [B, H, W, num_variables, out_dim].
        """
        super(SFMNNEncoder, self).__init__()
        self.num_variables = num_variables
        self.out_dim = out_dim
        
        # Normalize each pixel's input features.
        self.input_norm = nn.BatchNorm1d(input_channels)
        
        # Fully-connected (FC) layers for feature extraction.
        # These layers process the flattened pixel vector and extract a high-dimensional feature.
        self.fc_layers = nn.Sequential(
            nn.Linear(input_channels, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8192, 8192),
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
        
        # Project extracted features directly to the final spectral output.
        # The projection outputs num_variables * out_dim values for each pixel.
        self.latent_proj = nn.Linear(8192, num_variables * out_dim)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
          x: Input tensor of shape [B, C, H, W] with C = num_wavelengths + 3.
          
        Returns:
          out: Tensor of shape [B, H, W, num_variables, out_dim] where each latent stream
               directly outputs a spectral vector of dimension out_dim.
               All values are enforced to be non-negative.
        """
        B, C, H, W = x.shape
        # Flatten spatial dimensions: treat each pixel (H x W) as an independent sample.
        x_flat = x.view(B * H * W, C)
        
        # Normalize the inputs.
        x_norm = self.input_norm(x_flat)
        
        # Extract features using the fully-connected layers.
        x_fc = self.fc_layers(x_norm)
        
        # Project features directly into the final output space.
        # This yields a tensor of shape [B*H*W, num_variables * out_dim].
        latent = self.latent_proj(x_fc)
        
        # Reshape the projection into the desired output shape:
        # [B, H, W, num_variables, out_dim]
        out = latent.view(B, H, W, self.num_variables, self.out_dim)
        
        # Enforce non-negativity (e.g., reflectance values cannot be negative).
        out = F.relu(out)
        
        return out



class SFMNNEncoderWithHeads(nn.Module):
    def __init__(self, input_channels=3623, num_variables=11, encoded_dim=8192, out_dim=3620):
        """
        Args:
          input_channels: Number of input channels (num_wavelengths + 3 extras).
          num_variables: Number of separate variables/outputs to predict.
          encoded_dim: Dimension of the shared feature space.
          out_dim: Desired output dimension per head (e.g., number of wavelengths, typically 3620).
          
        The network first normalizes and processes each pixel’s input vector through a shared
        backbone (fully connected layers) and then feeds the resulting features into separate
        output heads (one per variable) that directly predict the final spectral vector.
        """
        super(SFMNNEncoderWithHeads, self).__init__()
        self.num_variables = num_variables
        self.out_dim = out_dim
        
        # Normalize each pixel's input features.
        self.input_norm = nn.BatchNorm1d(input_channels)
        
        # Shared backbone: fully connected layers that extract a high-dimensional feature vector.
        self.shared_layers = nn.Sequential(
            nn.Linear(input_channels, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8192, encoded_dim),
            nn.BatchNorm1d(encoded_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Create a separate head for each variable.
        # Each head is a linear layer mapping the shared feature (encoded_dim) to the desired output (out_dim).
        self.heads = nn.ModuleList([nn.Linear(encoded_dim, out_dim) for _ in range(num_variables)])
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
          x: Tensor of shape [B, C, H, W] where C = num_wavelengths + 3.
          
        Returns:
          outputs: Tensor of shape [B, H, W, num_variables, out_dim] where each head predicts a 
                   spectral vector of dimension out_dim for each pixel. A ReLU is applied so that outputs are non-negative.
        """
        B, C, H, W = x.shape
        # Flatten spatial dimensions: each pixel becomes an independent sample.
        x_flat = x.view(B * H * W, C)  # Shape: [B*H*W, C]
        
        # Normalize the input features.
        x_norm = self.input_norm(x_flat)
        
        # Pass through the shared backbone.
        features = self.shared_layers(x_norm)  # Shape: [B*H*W, encoded_dim]
        
        # Apply each head separately.
        head_outputs = []
        for head in self.heads:
            # Each head produces output of shape [B*H*W, out_dim]
            out_head = head(features)
            # Enforce non-negativity.
            out_head = F.relu(out_head)
            head_outputs.append(out_head)
        
        # Stack the outputs from all heads.
        # This creates a tensor of shape [num_variables, B*H*W, out_dim]
        outputs = torch.stack(head_outputs, dim=0)
        
        # Permute to shape [B*H*W, num_variables, out_dim]
        outputs = outputs.permute(1, 0, 2)
        
        # Finally, reshape to [B, H, W, num_variables, out_dim]
        outputs = outputs.view(B, H, W, self.num_variables, self.out_dim)
        return outputs

class SFMNNEncoderWithHeadsSingle(nn.Module):
    def __init__(self, input_channels=3623, num_variables=11, encoded_dim=8192, out_dim=3620):
        """
        Args:
          input_channels: Number of input channels (num_wavelengths + 3 extras).
          num_variables: Number of separate variables/outputs to predict.
          encoded_dim: Dimension of the shared feature space.
          out_dim: Desired output dimension per head (e.g., number of wavelengths, typically 3620).
          
        The network first normalizes each pixel's input vector through a shared
        backbone (fully connected layers) and then feeds the resulting features into separate
        output heads (one per variable) that directly predict the final spectral vector.
        """
        super(SFMNNEncoderWithHeadsSingle, self).__init__()
        self.num_variables = num_variables
        self.out_dim = out_dim
        
        # Normalize each pixel's input features.
        self.input_norm = nn.BatchNorm1d(input_channels)
        
        # Shared backbone: fully connected layers that extract a high-dimensional feature vector.
        self.shared_layers = nn.Sequential(
            nn.Linear(input_channels, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8192, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8192, encoded_dim),
            nn.BatchNorm1d(encoded_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Create a separate head for each variable.
        # Each head is a linear layer mapping the shared feature (encoded_dim) to the desired output (out_dim).
        self.heads = nn.ModuleList([nn.Linear(encoded_dim, out_dim) for _ in range(num_variables)])
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
          x: Tensor of shape [B, input_channels], where each row corresponds to a single pixel.
          
        Returns:
          outputs: Tensor of shape [B, num_variables, out_dim] where each head predicts a 
                   spectral vector of dimension out_dim for each pixel. A ReLU is applied so that outputs are non-negative.
        """
        # Normalize the input features.
        x_norm = self.input_norm(x)  # shape: [B, input_channels]
        
        # Pass through the shared backbone.
        features = self.shared_layers(x_norm)  # shape: [B, encoded_dim]
        
        # Apply each head separately.
        head_outputs = []
        for head in self.heads:
            # Each head produces output of shape [B, out_dim]
            out_head = head(features)
            # Enforce non-negativity.
            out_head = F.relu(out_head)
            head_outputs.append(out_head)
        
        # Stack the outputs from all heads to get shape [B, num_variables, out_dim]
        outputs = torch.stack(head_outputs, dim=1)
        return outputs
      


class ImprovedSFMNNEncoderWithHeadsResidual(nn.Module):
    def __init__(self, input_channels=3623, num_variables=11, encoded_dim=8192, out_dim=3620,
                 hidden_dims=[8192, 4096, 2048], dropout_rate=0.1, use_layer_norm=False):
        """
        Encoder with residual connections.
        """
        super().__init__()
        self.num_variables = num_variables
        self.use_layer_norm = use_layer_norm

        # Input normalization
        self.input_norm = nn.LayerNorm(input_channels) if use_layer_norm else nn.BatchNorm1d(input_channels)

        # Shared backbone with residual blocks
        self.shared_layers = nn.ModuleList()
        prev_dim = input_channels
        for hidden_dim in hidden_dims:
            self.shared_layers.append(ResidualBlock(prev_dim, hidden_dim, use_layer_norm, dropout_rate))
            prev_dim = hidden_dim
        
        self.final_linear = nn.Linear(prev_dim, encoded_dim)
        self.final_norm = nn.LayerNorm(encoded_dim) if use_layer_norm else nn.BatchNorm1d(encoded_dim)
        self.final_activation = nn.ReLU()
        self.final_dropout = nn.Dropout(dropout_rate)

        # Initialize weights
        self._initialize_weights()

        # Heads
        self.heads = nn.ModuleList([nn.Linear(encoded_dim, out_dim) for _ in range(num_variables)])
        for head in self.heads:
            nn.init.xavier_normal_(head.weight)
            if head.bias is not None:
                nn.init.constant_(head.bias, 0)

    def _initialize_weights(self):
        for m in self.shared_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.kaiming_normal_(self.final_linear.weight, mode='fan_out', nonlinearity='relu')
        if self.final_linear.bias is not None:
            nn.init.constant_(self.final_linear.bias, 0)

    def forward(self, x):
        x_norm = self.input_norm(x)
        
        for layer in self.shared_layers:
            x_norm = layer(x_norm)

        features = self.final_dropout(self.final_activation(self.final_norm(self.final_linear(x_norm))))
        
        head_outputs = [F.relu(head(features)) for head in self.heads]  # Apply heads and activation
        outputs = torch.stack(head_outputs, dim=1)
        return outputs

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_layer_norm=False, dropout_rate=0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim) if use_layer_norm else nn.BatchNorm1d(out_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim) if use_layer_norm else nn.BatchNorm1d(out_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Shortcut connection: project input if dimensions don't match
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.dropout1(self.relu1(self.norm1(self.linear1(x))))
        out = self.dropout2(self.relu2(self.norm2(self.linear2(out))))
        out += residual  # Add the residual
        return out
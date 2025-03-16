import torch
import torch.nn as nn

class SFMNNLoss(nn.Module):
    def __init__(self, w_f, gamma_f=5, gamma_N=10, gamma_a=1, ndvi_threshold=0.15):
        super().__init__()
        # Precomputed SNR-based weights for fluorescence bands
        self.register_buffer('w_f', w_f)  # Shape: [n_spectral_bands]
        
        # Loss weights from the paper
        self.gamma_f = gamma_f
        self.gamma_N = gamma_N
        self.gamma_a = gamma_a
        
        # NDVI threshold for physiological constraint
        self.ndvi_threshold = ndvi_threshold
        
        # Red and NIR band indices for NDVI calculation
        self.red_idx = 680   # Example indices - adjust based on sensor
        self.nir_idx = 800   # Example indices - adjust based on sensor

    def forward(self, pred, target, outputs):
        """
        Args:
            pred: Reconstructed radiance [B, C, H, W]
            target: Ground truth radiance [B, C, H, W]
            outputs: Dictionary containing:
                - 'sif'        : Predicted SIF [B, H, W]
                - 'reflectance': Predicted reflectance [B, C, H, W]
                - 't_tot'      : Atmospheric transfer function [B, C, H, W]
                - 'ndvi'       : Computed NDVI [B, H, W] (or raw spectra)
                
        Returns:
            loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Main spectral reconstruction loss (L_w)
        recon_loss = torch.mean((pred - target)**2)
        
        # Signal regularization (L_f)
        weighted_diff = (pred - target)**2 * self.w_f.view(1, -1, 1, 1)
        signal_loss = torch.mean(weighted_diff)
        
        # Physiological constraint (L_NDVI)
        if 'ndvi' not in outputs:
            # Compute NDVI from raw spectra if not provided
            red = outputs['reflectance'][:, self.red_idx]
            nir = outputs['reflectance'][:, self.nir_idx]
            ndvi = (nir - red) / (nir + red + 1e-6)
        else:
            ndvi = outputs['ndvi']
            
        veg_mask = (ndvi <= self.ndvi_threshold).float()
        ndvi_loss = torch.mean(outputs['sif'] * veg_mask)
        
        # Physical regularization (L_atm)
        atm_penalty = torch.relu(outputs['t_tot'] - 1)
        atm_loss = torch.mean(atm_penalty)
        
        # Combine losses
        total_loss = (recon_loss + 
                     self.gamma_f * signal_loss +
                     self.gamma_N * ndvi_loss +
                     self.gamma_a * atm_loss)
        
        loss_dict = {
            'total': total_loss,
            'recon': recon_loss,
            'signal': signal_loss,
            'ndvi': ndvi_loss,
            'atm': atm_loss
        }
        
        return total_loss, loss_dict

    def compute_t_tot(self, L_atm_R, E_s, cos_theta_s, R):
        """
        Compute total atmospheric transfer function
        Args:
            L_atm_R: Reflectance contribution to at-sensor radiance
            E_s: Solar irradiance
            cos_theta_s: Cosine of solar zenith angle
            R: Predicted reflectance
        """
        t_tot = (L_atm_R * R.mean()) / (E_s * cos_theta_s)
        return t_tot
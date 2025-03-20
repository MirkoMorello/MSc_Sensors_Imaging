import torch
import torch.nn as nn

class SFMNNLoss(nn.Module):
    def __init__(self, 
                 gamma_f=5, 
                 gamma_N=10, 
                 gamma_a=1,
                 ndvi_threshold=0.15,
                 red_idx=680,    # Should match sensor bands
                 nir_idx=800     # Should match sensor bands
                ):
        super().__init__()
        
        # Loss weights from the paper
        self.gamma_f = gamma_f
        self.gamma_N = gamma_N
        self.gamma_a = gamma_a
        
        # NDVI calculation parameters
        self.ndvi_threshold = ndvi_threshold
        self.red_idx = red_idx
        self.nir_idx = nir_idx

    def forward(self, pred, target, outputs, E_s, cos_theta_s):
        """
        Args:
            pred: Reconstructed radiance [B, C, H, W]
            target: Ground truth radiance [B, C, H, W]
            outputs: Dictionary containing:
                - 'sif'        : Predicted SIF [B, H, W]
                - 'reflectance': Predicted reflectance [B, C, H, W]
            E_s: Solar irradiance [B, C] (from Kurucz model)
            cos_theta_s: Cosine of solar zenith angle [B] (from acquisition metadata)
        """
        # Compute NDVI from reflectance
        red = outputs['reflectance'][:, self.red_idx]
        nir = outputs['reflectance'][:, self.nir_idx]
        ndvi = (nir - red) / (nir + red + 1e-6)
        
        # Compute t_tot from reflectance contribution
        R_mean = outputs['reflectance'].mean(dim=(2,3), keepdim=True)  # Spatial mean
        L_atm_R = pred - outputs['sif'].unsqueeze(1)  # Estimate reflectance contribution
        t_tot = (L_atm_R * R_mean) / (E_s * cos_theta_s.unsqueeze(-1).unsqueeze(-1) + 1e-6)

        # Compute fluorescence weight function
        w_f = self.compute_w_f(target, outputs['sif'])

        # Main spectral reconstruction loss (L_w)
        recon_loss = torch.mean((pred - target)**2)
        
        # Signal regularization (L_f)
        weighted_diff = (pred - target)**2 * w_f.view(1, -1, 1, 1)
        signal_loss = torch.mean(weighted_diff)
        
        # Physiological constraint (L_NDVI)
        veg_mask = (ndvi <= self.ndvi_threshold).float()
        ndvi_loss = torch.mean(outputs['sif'] * veg_mask)
        
        # Physical regularization (L_atm)
        atm_penalty = torch.relu(t_tot - 1)
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

    def compute_w_f(self, L_HyP, sif_pred, epsilon=1e-6):
        """
        Improved weight function based on paper's equation (16)
        Args:
            L_HyP: Measured radiance [B, C, H, W]
            sif_pred: Predicted SIF [B, H, W]
        """
        # Compute fluorescence magnitude term (across spatial dimensions)
        f_sq = sif_pred.unsqueeze(1)**2  # [B, 1, H, W]
        sum_f_sq = torch.mean(f_sq, dim=(0, 2, 3))  # [C]
        
        # Estimate sensor noise variance (per wavelength)
        sigma_L2 = torch.var(L_HyP, dim=(0, 2, 3), unbiased=False)  # [C]
        
        # Compute weighting components
        numerator = sum_f_sq
        denominator = sum_f_sq / (sigma_L2 + epsilon)
        
        # Final weight calculation
        w_f = (1.0 / (sigma_L2 + epsilon)) * (numerator / (denominator.sum() + epsilon))
        
        return w_f
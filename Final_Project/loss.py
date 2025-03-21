import torch
import torch.nn as nn

class SFMNNLoss(nn.Module):
    def __init__(self, 
                 gamma_f=5, 
                 gamma_N=10, 
                 gamma_a=1,
                 ndvi_threshold=0.15,
                 red_idx=680,    # These indices refer to wavelengths (0-indexed) in a 3620-channel spectrum.
                 nir_idx=800
                ):
        super().__init__()
        self.gamma_f = gamma_f
        self.gamma_N = gamma_N
        self.gamma_a = gamma_a
        self.ndvi_threshold = ndvi_threshold
        self.red_idx = red_idx
        self.nir_idx = nir_idx

    def forward(self, pred, target, outputs, E_s, cos_theta_s):
        # Compute NDVI from reflectance
        red = outputs['reflectance'][:, self.red_idx]  # pred reflectance should have 3620 channels
        nir = outputs['reflectance'][:, self.nir_idx]
        ndvi = (nir - red) / (nir + red + 1e-6)
        
        R_mean = outputs['reflectance'].mean(dim=(2,3), keepdim=True)
        L_atm_R = pred - outputs['sif'].unsqueeze(1)
        t_tot = (L_atm_R * R_mean) / (E_s * cos_theta_s.unsqueeze(-1).unsqueeze(-1) + 1e-6)

        w_f = self.compute_w_f(target, outputs['sif'])
        recon_loss = torch.mean((pred - target)**2)
        weighted_diff = (pred - target)**2 * w_f.view(1, -1, 1, 1)
        signal_loss = torch.mean(weighted_diff)
        veg_mask = (ndvi <= self.ndvi_threshold).float()
        ndvi_loss = torch.mean(outputs['sif'] * veg_mask)
        atm_penalty = torch.relu(t_tot - 1)
        atm_loss = torch.mean(atm_penalty)
        
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
        f_sq = sif_pred.unsqueeze(1)**2  # [B, 1, H, W]
        sum_f_sq = torch.mean(f_sq, dim=(0, 2, 3))
        sigma_L2 = torch.var(L_HyP, dim=(0, 2, 3), unbiased=False)
        numerator = sum_f_sq
        denominator = sum_f_sq / (sigma_L2 + epsilon)
        w_f = (1.0 / (sigma_L2 + epsilon)) * (numerator / (denominator.sum() + epsilon))
        return w_f
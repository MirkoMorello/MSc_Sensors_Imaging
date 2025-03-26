import torch
import torch.nn as nn
from torch.nn import functional as F

class SFMNNLoss(nn.Module):
    def __init__(self, gamma_f=5, gamma_N=10, ndvi_threshold=0.15,
                 red_idx=760, nir_idx=780, delta=1.0):
        """
        Args:
          gamma_f: weight for the signal/fluorescence regularization
          gamma_N: weight for the NDVI-based physiological constraint
          ndvi_threshold: threshold for NDVI below which fluorescence is suppressed
          red_idx: index corresponding to a red wavelength (e.g., 760 nm)
          nir_idx: index corresponding to a NIR wavelength (e.g., 780 nm)
          delta: parameter for Huber loss (unused in this example)
        """
        super().__init__()
        self.gamma_f = gamma_f
        self.gamma_N = gamma_N
        self.ndvi_threshold = ndvi_threshold
        self.red_idx = red_idx
        self.nir_idx = nir_idx
        self.delta = delta

    def forward(self, predicted_reflectance, target_reflectance, predicted_sif, target_sif, outputs, loss_indices):
        """
        Args:
          predicted_reflectance: [B, window_size, H, W] – predicted LTOA over the spectral window
          target_reflectance: [B, window_size, H, W] – measured LTOA over the same window
          predicted_sif: [B, full_spectrum, H, W] – predicted fluorescence (SIF)
          target_sif: [B, full_spectrum, H, W] – measured fluorescence (SIF)
          outputs: dict containing:
            'reflectance': [B, full_spectrum, H, W]
            'sif': [B, full_spectrum, H, W] (predicted fluorescence; assumed to be full-spectrum)
          loss_indices: tensor of indices selecting the spectral window used for loss computation
        Returns:
          total_loss, loss_dict
        """
        # 1) Reconstruction loss (L_w)
        recon_loss = F.mse_loss(predicted_reflectance, target_reflectance)
        
        # 2) NDVI-based physiological loss (L_ndvi)
        # Extract red and NIR channels from the full reflectance.
        red = outputs['reflectance'][:, self.red_idx]  # [B, H, W]
        nir = outputs['reflectance'][:, self.nir_idx]    # [B, H, W]
        red = red.clamp(min=1e-3)
        nir = nir.clamp(min=1e-3)
        ndvi = (nir - red) / (nir + red + 1e-6)
        ndvi_mask = (ndvi <= self.ndvi_threshold).float().unsqueeze(1)  # [B, 1, H, W]
        ndvi_loss = torch.mean(F.relu(outputs['sif']) * ndvi_mask)
        
        # 3) Regularization loss for SIF (L_reg)
        reg_loss = torch.mean(torch.abs(predicted_sif))
        
        # 4) Signal/fluorescence regularization (L_f)
        # Select the corresponding spectral window for SIF.
        sif_window = outputs['sif'][:, loss_indices, :, :]  # shape: [B, window_size, H, W]
        w_f = self.compute_w_f(target_reflectance, sif_window)  # shape: [window_size]
        signal_loss = torch.mean((predicted_reflectance - target_reflectance)**2 * w_f.view(1, -1, 1, 1))
        
        # 5) New term: MSE loss over the predicted SIF (L_sif)
        mse_sif = F.mse_loss(predicted_sif, target_sif)
        
        # Set loss term weights (adjust as necessary)
        self.lambda_recon = 0.01
        self.lambda_ndvi = 1
        self.lambda_reg = 1
        self.lambda_signal = 1
        self.lambda_sif = 1
        
        total_loss = (self.lambda_recon * recon_loss +
                      self.lambda_ndvi * ndvi_loss +
                      self.lambda_reg * reg_loss +
                      self.lambda_signal * signal_loss +
                      self.lambda_sif * mse_sif)
        
        loss_dict = {
            'recon': recon_loss,
            'ndvi': ndvi_loss,
            'reg': reg_loss,
            'signal': signal_loss,
            'sif_mse': mse_sif
        }
        return total_loss, loss_dict

    def compute_w_f(self, L_windowed, sif_pred, epsilon=1e-6):
        """
        Compute per-wavelength weights based on the predicted fluorescence.
        Args:
          L_windowed: [B, window_size, H, W] – target reflectance over the spectral window
          sif_pred: [B, window_size, H, W] – predicted fluorescence over the same window
        Returns:
          w_f: tensor of shape [window_size] (normalized weights)
        """
        # Compute squared fluorescence and average over batch and spatial dimensions.
        f_sq = sif_pred.unsqueeze(1) ** 2  # shape: [B, 1, window_size, H, W]
        sum_f_sq = torch.mean(f_sq, dim=(0, 3, 4))  # shape: [1, window_size]
        # Compute per-wavelength variance of L_windowed over B, H, and W.
        sigma_L2 = torch.var(L_windowed, dim=(0, 2, 3)) + 1e-3  # shape: [window_size]
        denominator = sum_f_sq / (sigma_L2 + epsilon)  # shape: [1, window_size]
        w_f = (1.0 / sigma_L2) * (sum_f_sq.squeeze(0) / (denominator.sum() + 1e-6))
        w_f = w_f / (w_f.sum() + 1e-6)
        return w_f


class SFMNNLossSingle(nn.Module):
    def __init__(self, gamma_f=5, gamma_N=10, ndvi_threshold=0.15,
                 red_idx=None, nir_idx=None, delta=1.0):
        """
        Original SFMNN loss adapted for single pixel processing.
        Requires red_idx and nir_idx corresponding to the reflectance/LTOA channels.
        """
        super().__init__()
        self.gamma_f = gamma_f
        self.gamma_N = gamma_N
        self.ndvi_threshold = ndvi_threshold
        if red_idx is None or nir_idx is None:
             raise ValueError("red_idx and nir_idx must be provided.")
        self.red_idx = red_idx
        self.nir_idx = nir_idx
        self.delta = delta # Currently unused

    def forward(self, predicted_ltoa_window, target_ltoa_window,
                predicted_sif_full, target_sif_full, outputs_dict, loss_indices):
        """
        Calculates the original SFMNN loss components.

        Args:
          predicted_ltoa_window: Tensor [B, window_size] – Predicted LTOA (reconstructed) over spectral window.
          target_ltoa_window: Tensor [B, window_size] – Target LTOA over the same window.
          predicted_sif_full: Tensor [B, full_spectrum] – Predicted fluorescence (SIF).
          target_sif_full: Tensor [B, full_spectrum] – Target fluorescence (SIF).
          outputs_dict: dict containing 'reflectance' [B, full_spectrum] (predicted R)
                         and 'sif' [B, full_spectrum] (predicted F/SIF).
          loss_indices: tensor/list of indices selecting the spectral window.

        Returns:
          total_loss, loss_dict
        """
        # Ensure inputs have batch dimension
        B = predicted_ltoa_window.shape[0] # Get batch size early
        if predicted_ltoa_window.ndim == 1: predicted_ltoa_window = predicted_ltoa_window.unsqueeze(0)
        if target_ltoa_window.ndim == 1: target_ltoa_window = target_ltoa_window.unsqueeze(0)
        if predicted_sif_full.ndim == 1: predicted_sif_full = predicted_sif_full.unsqueeze(0)
        if target_sif_full.ndim == 1: target_sif_full = target_sif_full.unsqueeze(0)
        # Ensure outputs_dict values also have batch dim if needed (should come from model with batch dim)
        if 'reflectance' not in outputs_dict or 'sif' not in outputs_dict:
            raise KeyError("outputs_dict must contain 'reflectance' and 'sif' keys.")
        if outputs_dict['reflectance'].ndim == 1: outputs_dict['reflectance'] = outputs_dict['reflectance'].unsqueeze(0)
        if outputs_dict['sif'].ndim == 1: outputs_dict['sif'] = outputs_dict['sif'].unsqueeze(0)

        # 1) Reconstruction loss (L_w) - MSE on the LTOA window
        recon_loss = F.mse_loss(predicted_ltoa_window, target_ltoa_window)

        # 2) NDVI-based physiological loss (L_ndvi)
        pred_R_full = outputs_dict['reflectance']
        # Check indices are valid *before* trying to access them
        if not (0 <= self.red_idx < pred_R_full.shape[1]):
             raise IndexError(f"red_idx ({self.red_idx}) out of bounds for predicted R shape {pred_R_full.shape}")
        if not (0 <= self.nir_idx < pred_R_full.shape[1]):
             raise IndexError(f"nir_idx ({self.nir_idx}) out of bounds for predicted R shape {pred_R_full.shape}")

        red = pred_R_full[:, self.red_idx].clamp(min=1e-4)
        nir = pred_R_full[:, self.nir_idx].clamp(min=1e-4)
        ndvi = (nir - red) / (nir + red + 1e-6)
        ndvi_mask = (ndvi <= self.ndvi_threshold).float().unsqueeze(1) # [B, 1]
        pred_sif_full_for_ndvi = outputs_dict['sif']
        ndvi_loss = torch.mean(F.relu(pred_sif_full_for_ndvi) * ndvi_mask)

        # 3) Regularization loss for SIF (L_reg) - L1 norm
        reg_loss = torch.mean(torch.abs(predicted_sif_full))

        # 4) Signal/fluorescence regularization (L_f)
        # Ensure loss_indices are valid for the predicted SIF shape
        if not all(0 <= i < outputs_dict['sif'].shape[1] for i in loss_indices):
             raise IndexError(f"loss_indices contain values out of bounds for predicted SIF shape {outputs_dict['sif'].shape}")
        sif_window = outputs_dict['sif'][:, loss_indices]
        signal_loss = torch.tensor(0.0, device=predicted_ltoa_window.device) # Default value
        try:
             # Check if target_ltoa_window is valid before computing w_f
             if target_ltoa_window.numel() > 0 and sif_window.numel() > 0: # Ensure tensors are not empty
                 w_f = self.compute_w_f(target_ltoa_window, sif_window)
                 signal_loss = torch.mean((predicted_ltoa_window - target_ltoa_window)**2 * w_f.view(1, -1))
             else:
                 logger.debug("Skipping signal_loss calculation due to empty window tensors.")
        except Exception as e:
            logger.debug(f"Could not compute w_f or signal_loss: {e}. Setting signal_loss to 0.")
            # Keep signal_loss as 0

        # 5) MSE loss over the full predicted SIF vs target SIF (L_sif)
        mse_sif = F.mse_loss(predicted_sif_full, target_sif_full)

        # --- Tunable Loss Weights ---
        lambda_recon = 0.1
        lambda_ndvi = 1.0
        lambda_reg = 0.0 # Often set to 0
        lambda_signal = 0.0 # Often set to 0 if unstable
        lambda_sif = 1.0
        # --------------------------

        total_loss = (lambda_recon * recon_loss +
                      lambda_ndvi * ndvi_loss +
                      lambda_reg * reg_loss +
                      lambda_signal * signal_loss +
                      lambda_sif * mse_sif)

        loss_dict = {
            'recon': recon_loss.item(), 'ndvi': ndvi_loss.item(), 'reg': reg_loss.item(),
            'signal': signal_loss.item(), 'sif_mse': mse_sif.item()
        }
        return total_loss, loss_dict

    def compute_w_f(self, L_windowed, sif_pred_windowed, epsilon=1e-6):
        """Compute per-wavelength weights based on predicted fluorescence contribution."""
        # Add defensive checks for tensor shapes and values
        if L_windowed.shape != sif_pred_windowed.shape:
            raise ValueError(f"Shape mismatch in compute_w_f: L {L_windowed.shape}, SIF {sif_pred_windowed.shape}")
        if L_windowed.ndim == 1: L_windowed = L_windowed.unsqueeze(0)
        if sif_pred_windowed.ndim == 1: sif_pred_windowed = sif_pred_windowed.unsqueeze(0)
        B = L_windowed.shape[0]

        sum_f_sq = torch.mean(sif_pred_windowed ** 2, dim=0) # [window_size]

        if B > 1:
             sigma_L2 = torch.var(L_windowed, dim=0, unbiased=False)
        else: # Handle batch size 1 case for variance
             sigma_L2 = torch.zeros_like(sum_f_sq) # Variance is 0 for B=1

        # Use clamp *after* calculation, but ensure epsilon is added before division potentially happens
        sigma_L2_reg = torch.clamp(sigma_L2, min=epsilon)

        denominator = sum_f_sq / sigma_L2_reg # Element-wise division
        sum_denominator = denominator.sum()

        if sum_denominator < epsilon:
            # If SIF contribution or variance is negligible, use uniform weights
            w_f = torch.ones_like(sum_f_sq) / max(len(sum_f_sq), 1) # Avoid division by zero if len is 0
        else:
            # Weight by SIF contribution relative to total, scaled by inverse variance
            w_f = (1.0 / sigma_L2_reg) * (sum_f_sq / sum_denominator)

        # Normalize weights to sum to 1
        w_f_sum = w_f.sum()
        if w_f_sum < epsilon:
             return torch.ones_like(w_f) / max(len(w_f), 1) # Return uniform if sum is near zero
        else:
             w_f = w_f / w_f_sum

        return w_f

# New Physics-Regularized Loss
class PhysicsRegularizedLoss(nn.Module):
    def __init__(self, lambda_t=1.0, lambda_r=1.0, lambda_f=1.0, lambda_phys_ltoa=1.0):
        """
        Loss combining MSE on predicted t, R, F and physics-based LTOA reconstruction.
        Assumes model predicts 9 t-values (t1,t2,t3,t6-t11), R, F.
        Targets include 11 t-values (t1-t11).

        Args:
            lambda_t: Weight for the 9 t-values MSE loss.
            lambda_r: Weight for the R MSE loss.
            lambda_f: Weight for the F MSE loss.
            lambda_phys_ltoa: Weight for the LTOA reconstruction MSE loss.
        """
        super().__init__()
        self.lambda_t = lambda_t
        self.lambda_r = lambda_r
        self.lambda_f = lambda_f
        self.lambda_phys_ltoa = lambda_phys_ltoa

        # 9 t-keys predicted by the model (indices 0-8)
        self.t_keys_predicted = ['t1', 't2', 't3', 't6', 't7', 't8', 't9', 't10', 't11']
        # Mapping from these keys to model output indices (0-8)
        self.model_t_indices_map = {key: i for i, key in enumerate(self.t_keys_predicted)}

        # Indices within the *target* tensor (0-10 for t1-t11) corresponding to the 9 predicted t's
        self.target_indices_for_9t_compare = [0, 1, 2, 5, 6, 7, 8, 9, 10]

    def _calculate_ltoa(self, pred_9t, pred_r, pred_f):
        """
        Calculates LTOA using the 9 predicted t-values, R, and F.
        Assumes pred_9t shape [B, 9, n_spec], pred_r/pred_f shape [B, n_spec].
        """
        # Extract t values using the map to indices 0-8
        t1  = pred_9t[:, self.model_t_indices_map['t1'], :]
        t2  = pred_9t[:, self.model_t_indices_map['t2'], :]
        t3  = pred_9t[:, self.model_t_indices_map['t3'], :]
        t6  = pred_9t[:, self.model_t_indices_map['t6'], :]
        t7  = pred_9t[:, self.model_t_indices_map['t7'], :]
        t8  = pred_9t[:, self.model_t_indices_map['t8'], :]
        t9  = pred_9t[:, self.model_t_indices_map['t9'], :]
        t10 = pred_9t[:, self.model_t_indices_map['t10'], :]
        t11 = pred_9t[:, self.model_t_indices_map['t11'], :]

        R_clamped = torch.clamp(pred_r, 0.0, 0.999) # Avoid R*t3 == 1
        epsilon = 1e-6

        term1 = t1 * t2
        
        numerator = t1 * (t8 * pred_r + t9 * pred_r + t10 * pred_r + t11 * pred_r) + t6 * pred_f + t7 * pred_f
        denominator = 1.0 - t3 * R_clamped + epsilon
        term2 = numerator / denominator

        ltoa_pred = term1 + term2
        return ltoa_pred

    def forward(self, predicted_9t, predicted_r, predicted_f, target_11t, target_r, target_f, target_ltoa):
        """
        Calculate the combined physics loss. All inputs are UNNORMALIZED.

        Args:
            predicted_9t (Tensor): Predicted 9 t-values [B, 9, n_spec].
            predicted_r (Tensor): Predicted R [B, n_spec].
            predicted_f (Tensor): Predicted F [B, n_spec].
            target_11t (Tensor): Target 11 t-values [B, 11, n_spec].
            target_r (Tensor): Target R [B, n_spec].
            target_f (Tensor): Target F [B, n_spec].
            target_ltoa (Tensor): Target LTOA [B, n_spec].

        Returns:
            torch.Tensor: Total combined physics loss.
            dict: Dictionary containing individual physics loss components.
        """
        # Ensure inputs have batch dimension if missing
        if predicted_9t.ndim == 2: predicted_9t = predicted_9t.unsqueeze(0)
        if predicted_r.ndim == 1: predicted_r = predicted_r.unsqueeze(0)
        if predicted_f.ndim == 1: predicted_f = predicted_f.unsqueeze(0)
        if target_11t.ndim == 2: target_11t = target_11t.unsqueeze(0)
        if target_r.ndim == 1: target_r = target_r.unsqueeze(0)
        if target_f.ndim == 1: target_f = target_f.unsqueeze(0)
        if target_ltoa.ndim == 1: target_ltoa = target_ltoa.unsqueeze(0)

        # 1. MSE Loss for the 9 predicted t-values
        if self.lambda_t > 0:
            target_9t = target_11t[:, self.target_indices_for_9t_compare, :]
            loss_t9 = F.mse_loss(predicted_9t, target_9t)
        else:
            loss_t9 = torch.tensor(0.0, device=predicted_9t.device)

        # 2. MSE Loss for R
        loss_r = F.mse_loss(predicted_r, target_r) if self.lambda_r > 0 else torch.tensor(0.0, device=predicted_r.device)

        # 3. MSE Loss for F (SIF)
        loss_f = F.mse_loss(predicted_f, target_f) if self.lambda_f > 0 else torch.tensor(0.0, device=predicted_f.device)

        # 4. Physics-based LTOA Reconstruction Loss
        if self.lambda_phys_ltoa > 0:
            ltoa_reconstructed = self._calculate_ltoa(predicted_9t, predicted_r, predicted_f)
            loss_phys_ltoa = F.mse_loss(ltoa_reconstructed, target_ltoa)
        else:
            loss_phys_ltoa = torch.tensor(0.0, device=predicted_9t.device)
            ltoa_reconstructed = None # Not calculated if lambda is 0

        # Total Physics Loss
        total_physics_loss = (self.lambda_t * loss_t9 +
                              self.lambda_r * loss_r +
                              self.lambda_f * loss_f +
                              self.lambda_phys_ltoa * loss_phys_ltoa)

        loss_dict = {
            'mse_t9': loss_t9.item() if self.lambda_t > 0 else 0.0,
            'mse_r': loss_r.item() if self.lambda_r > 0 else 0.0,
            'mse_f': loss_f.item() if self.lambda_f > 0 else 0.0,
            'phys_ltoa': loss_phys_ltoa.item() if self.lambda_phys_ltoa > 0 else 0.0,
        }

        # Return reconstructed LTOA as well, useful for original loss calculation
        return total_physics_loss, loss_dict, ltoa_reconstructed



class FourStreamSimulator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t1, t2, t3, t6, t7, t8, t9, t10, t11, R, F):
        # Simplified toy model for LTOA.
        LTOA = t1 * t2 + ( t1  * (t8 * R + t9 * R + t10 * R + t11 * R) + t6 * F + t7 * F) / (1 - t3 * R + 1e-6)
        return LTOA

# --- HyPlantSensorSimulator ---
class HyPlantSensorSimulator(nn.Module):
    def __init__(self, sensor_wavelengths):
        super().__init__()
        self.register_buffer('sensor_wavelengths', sensor_wavelengths)
        # Compute high_res as the spacing
        self.high_res = (sensor_wavelengths.max() - sensor_wavelengths.min()) / len(sensor_wavelengths)
        self.register_buffer('wl_range', torch.tensor([sensor_wavelengths.min(), sensor_wavelengths.max()]))
        # Define a high-resolution wavelength grid matching the sensor wavelengths length.
        self.register_buffer('lambda_hr', torch.linspace(self.wl_range[0].item(),
                                                           self.wl_range[1].item(),
                                                           len(sensor_wavelengths),
                                                           device=sensor_wavelengths.device))
        self.register_buffer('sensor_wavelengths', sensor_wavelengths.view(-1))



        # Fixed sensor miscalibration (in nm)
        self.fixed_delta_lambda = 0.04226162119141463

    def forward(self, L_hr, delta_lambda, delta_sigma):
        # L_hr is assumed to be of shape [B, H, W, spec] where spec ~241
        B, H, W, spec = L_hr.shape
        # Bring spectral axis to channel dimension: [B, spec, H, W]
        L_hr = L_hr.permute(0, 3, 1, 2)
        
        # --- 1. Convolve high-res spectrum with Gaussian SRF ---
        sigma = (0.27 + delta_sigma) * 2.3548  # Convert FWHM to sigma
        sigma_val = sigma.item() if torch.is_tensor(sigma) else sigma
        kernel_size = int(6 * sigma_val / self.high_res)
        if kernel_size % 2 == 0:
            kernel_size += 1
        x = torch.linspace(-3 * sigma_val, 3 * sigma_val, kernel_size, device=L_hr.device)
        kernel = torch.exp(-0.5 * (x / sigma_val)**2)
        kernel = kernel / kernel.sum()
        # Reshape for conv1d
        L_hr_reshaped = L_hr.permute(0, 2, 3, 1).reshape(B*H*W, 1, spec)
        L_blur = F.conv1d(L_hr_reshaped, kernel.view(1,1,kernel_size), padding=kernel_size//2)
        L_blur = L_blur.reshape(B, H, W, spec).permute(0, 3, 1, 2)
        
        # --- 2. Integrate (sample) the blurred spectrum at sensor wavelengths ---
        # Effective sensor wavelengths are shifted by fixed_delta_lambda.
        effective_sensor_wl = self.sensor_wavelengths + self.fixed_delta_lambda

        # Build Gaussian response matrix using lambda_hr:
        diff = self.lambda_hr.unsqueeze(0) - effective_sensor_wl.unsqueeze(1)  # [num_sensor, spec]
        sigma_prime = sigma_val  # Using the same sigma for response
        g = torch.exp(-0.5 * (diff / sigma_prime)**2)
        g = g / (g.sum(dim=1, keepdim=True) + 1e-6)  # [num_sensor, spec]
        
        # Integrate: result shape [B, num_sensor, H, W]
        L_hyp = torch.einsum('ij,bjhw->bihw', g, L_hr) * self.high_res
        return L_hyp

# --- SFMNNSimulation ---
class SFMNNSimulation(nn.Module):
    def __init__(self, sensor_wavelengths):
        super().__init__()
        self.four_stream = FourStreamSimulator()
        self.sensor_sim = HyPlantSensorSimulator(sensor_wavelengths)

    def forward(self, t1, t2, t3, t6, t7, t8, t9, t10, t11, R, F):
        # Compute auxiliary variables as before.
        L_hr = self.four_stream(t1, t2, t3, t6, t7, t8, t9, t10, t11, R, F)
        #L_hr = L_hr.permute(0, 2, 3, 1) # [B, H, W, spec]
        #L_hyp = self.sensor_sim(L_hr, delta_lambda, delta_sigma)
        # Optionally incorporate E_s here if needed.
        #return L_hyp
        return L_hr
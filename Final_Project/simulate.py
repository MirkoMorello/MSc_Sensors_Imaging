import torch
import torch.nn as nn
import torch.nn.functional as F

class FourStreamSimulator(nn.Module):
    def __init__(self, spectral_window=(750, 770), high_res=0.0055):
        super().__init__()
        self.register_buffer('lambda_hr', torch.arange(*spectral_window, high_res))
        self.mu_f = 737.0  # Fluorescence peak wavelength

    def forward(self, t1, t2, t3, t4, t5, t6, R, F, E_s, cos_theta_s):
        """
        Paper Eq.2 implementation with explicit parameters
        Input shapes:
        - t1-t6: [B, H, W] 
        - R: [B, C_hr, H, W] (reflectance spectrum)
        - F: [B, C_hr, H, W] (fluorescence spectrum)
        - E_s: [B, C_hr] (solar irradiance)
        - cos_theta_s: [B]
        """

        print(f"t1: {t1.shape}")
        print(f"t2: {t2.shape}")
        print(f"t3: {t3.shape}")
        print(f"t4: {t4.shape}")
        print(f"t5: {t5.shape}")import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# --- FourStreamSimulator ---
class FourStreamSimulator(nn.Module):
    def __init__(self, spectral_window=(750, 770), high_res=0.0055):
        super().__init__()
        self.register_buffer('lambda_hr', torch.arange(*spectral_window, high_res))
        self.mu_f = 737.0  # Fluorescence peak wavelength

    def forward(self, t1, t2, t3, t4, t5, t6, R, F, E_s, cos_theta_s):
        print(f"t1: {t1.shape}")
        print(f"t2: {t2.shape}")
        print(f"t3: {t3.shape}")
        print(f"t4: {t4.shape}")
        print(f"t5: {t5.shape}")
        print(f"t6: {t6.shape}")
        print(f"R: {R.shape}")
        print(f"F: {F.shape}")

        # Compute product terms (following Table 3 from the paper)
        t7 = t3 * t4
        t8 = t3 * t6
        t9 = t4 * t5
        t10 = t4 * t2
        t11 = t3 * t2

        LTOA = t1 * t2 + (t1 * t8 * R + t9 * R + t10 * R + t11 * R + t6 * F + t7 * F) / (1 - t3 * R)
        return LTOA

# --- HyPlantSensorSimulator ---
class HyPlantSensorSimulator(nn.Module):
    def __init__(self, sensor_wavelengths, high_res=0.0055):
        super().__init__()
        self.register_buffer('sensor_wavelengths', sensor_wavelengths)
        self.high_res = high_res
        self.register_buffer('wl_range', torch.tensor([sensor_wavelengths.min(), sensor_wavelengths.max()]))

    def forward(self, L_hr, delta_lambda, delta_sigma):
        B, C_hr, H, W = L_hr.shape

        # 1. Create Gaussian SRF kernel
        sigma = (0.27 + delta_sigma) * 2.3548  # convert FWHM to sigma
        kernel_size = int(6 * sigma / self.high_res)
        x = torch.linspace(-3 * sigma, 3 * sigma, kernel_size, device=L_hr.device)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()

        # 2. Spectral convolution (reshape for 1D conv)
        L_blur = F.conv1d(
            L_hr.view(B * H * W, 1, -1),
            kernel.view(1, 1, -1),
            padding=kernel_size // 2
        ).view(B, H, W, -1).permute(0, 3, 1, 2)

        # 3. Handle wavelength shift parameter (delta_lambda)
        # If delta_lambda is [B, N] (per-image), average over the second dim.
        if delta_lambda.ndim == 2:
            avg_delta = delta_lambda.mean(dim=1, keepdim=True)  # shape: [B, 1]
        elif delta_lambda.ndim >= 3:
            avg_delta = delta_lambda.mean(dim=(1, 2)).view(B, 1)
        else:
            avg_delta = delta_lambda.view(B, 1)

        # Shift sensor wavelengths by avg_delta (broadcasted addition)
        shifted_wl = self.sensor_wavelengths + avg_delta
        normalized_wl = 2 * (shifted_wl - self.wl_range[0]) / (self.wl_range[1] - self.wl_range[0]) - 1

        # Create grid for sampling: expand to spatial dimensions
        grid = normalized_wl.view(B, 1, 1, -1).expand(B, H, W, -1)
        # Use grid_sample to simulate sensor response
        return F.grid_sample(L_blur, grid.unsqueeze(1), mode='bilinear', align_corners=False).squeeze(2)

# --- SFMNNSimulation ---
class SFMNNSimulation(nn.Module):
    def __init__(self, sensor_wavelengths):
        super().__init__()
        self.four_stream = FourStreamSimulator()
        self.sensor_sim = HyPlantSensorSimulator(sensor_wavelengths)

    def forward(self, t1, t2, t3, t4, t5, t6, R, F, delta_lambda, delta_sigma, E_s, cos_theta_s):
        # Four-stream radiance calculation
        L_hr = self.four_stream(t1, t2, t3, t4, t5, t6, R, F, E_s, cos_theta_s)
        print("L_hr shape:", L_hr.shape)
        # Apply sensor simulation
        L_hyp = self.sensor_sim(L_hr, delta_lambda, delta_sigma)
        return L_hyp
        print(f"t6: {t6.shape}")
        print(f"R: {R.shape}")
        print(f"F: {F.shape}")

        # Compute product terms from Table 3
        t7 = t3 * t4
        t8 = t3 * t6
        t9 = t4 * t5
        t10 = t4 * t2
        t11 = t3 * t2
        
        LTOA = t1 * t2 + (t1 * t8 * R + t9 * R + t10 * R + t11 * R + t6 * F + t7 * F) / (1 - t3 * R)

        return LTOA

class HyPlantSensorSimulator(nn.Module):
    def __init__(self, sensor_wavelengths, high_res=0.0055):
        super().__init__()
        self.register_buffer('sensor_wavelengths', sensor_wavelengths)
        self.high_res = high_res
        self.register_buffer('wl_range', torch.tensor([sensor_wavelengths.min(), 
                                                     sensor_wavelengths.max()]))

    def forward(self, L_hr, delta_lambda, delta_sigma):
        """
        Sensor model implementation
        Input shapes:
        - L_hr: [B, C_hr, H, W]
        - delta_lambda: [B, H, W]
        - delta_sigma: [1]
        """
        B, C_hr, H, W = L_hr.shape
        
        # 1. Create Gaussian SRF kernel
        sigma = (0.27 + delta_sigma) * 2.3548  # FWHM to sigma
        kernel_size = int(6*sigma/self.high_res)
        x = torch.linspace(-3*sigma, 3*sigma, kernel_size, device=L_hr.device)
        kernel = torch.exp(-0.5*(x/sigma)**2)
        kernel /= kernel.sum()

        # 2. Spectral convolution
        L_blur = F.conv1d(
            L_hr.view(B*H*W, 1, -1), 
            kernel.view(1, 1, -1), 
            padding=kernel_size//2
        ).view(B, H, W, -1).permute(0, 3, 1, 2)

        # 3. Wavelength shift and interpolation
        shifted_wl = self.sensor_wavelengths + delta_lambda.mean(dim=(1,2)).view(B,1)
        normalized_wl = 2*(shifted_wl - self.wl_range[0])/(self.wl_range[1]-self.wl_range[0]) - 1
        
        # Create grid sample coordinates
        grid = normalized_wl.view(B, 1, 1, -1).expand(B, H, W, -1)
        
        return F.grid_sample(L_blur, grid.unsqueeze(1), 
                           mode='bilinear', align_corners=False).squeeze(2)

class SFMNNSimulation(nn.Module):
    def __init__(self, sensor_wavelengths):
        super().__init__()
        self.four_stream = FourStreamSimulator()
        self.sensor_sim = HyPlantSensorSimulator(sensor_wavelengths)

    def forward(self, t1, t2, t3, t4, t5, t6, R, F, delta_lambda, delta_sigma, E_s, cos_theta_s):
        # 1. Four-stream radiance calculation
        L_hr = self.four_stream(t1, t2, t3, t4, t5, t6, R, F, E_s, cos_theta_s)
        
        print(L_hr.shape)
        # 2. Apply sensor characteristics
        L_hyp = self.sensor_sim(L_hr, delta_lambda, delta_sigma)
        
        return L_hyp
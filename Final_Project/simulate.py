import torch
import torch.nn as nn
import torch.nn.functional as F

class FourStreamSimulator(nn.Module):
    def __init__(self, spectral_window=(750, 770), high_res=0.0055):
        super().__init__()
        # Spectral configuration
        self.register_buffer('lambda_hr', torch.arange(*spectral_window, high_res))
        self.n_bands_hr = len(self.lambda_hr)
        
        # Constants from paper
        self.lambda0 = 740.0  # Reflectance model parameter
        self.lambda1 = 780.0  # Reflectance model parameter
        self.mu_f = 737.0     # Fluorescence emission peak

    def forward(self, t_params, R_params, f_params, E_s, cos_theta_s):
        """
        Simulates high-resolution at-sensor radiance using 4-stream model
        Args:
            t_params: Atmospheric parameters [B, 6, H, W]
            R_params: Reflectance parameters (ρ, s_ρ, e) [B, 3, H, W]
            f_params: Fluorescence parameters (A_f, σ_f) [B, 2, H, W]
            E_s: Solar irradiance spectrum [B, n_bands_hr]
            cos_theta_s: Cosine of solar zenith angle [B, 1, 1, 1]
        
        Returns:
            L_at_s_hr: High-res simulated radiance [B, n_bands_hr, H, W]
        """
        B, _, H, W = t_params.shape
        
        # Expand to high-res spectral dimension
        lambda_hr = self.lambda_hr.view(1, -1, 1, 1)  # [1, n_bands_hr, 1, 1]
        
        # 1. Compute reflectance spectrum (Eq.9)
        rho = R_params[:, 0]  # [B, H, W]
        s_rho = R_params[:, 1]
        e = R_params[:, 2]
        
        # Reflectance polynomial [B, H, W] -> [B, n_bands_hr, H, W]
        R = rho.unsqueeze(1) + s_rho.unsqueeze(1)*(lambda_hr - self.lambda0) + \
            (s_rho.unsqueeze(1)*(e.unsqueeze(1)-1)*(lambda_hr - self.lambda0)**2) / \
            (2*(self.lambda1 - self.lambda0))
        
        # 2. Compute fluorescence spectrum (Eq.8)
        A_f = f_params[:, 0]  # [B, H, W]
        sigma_f = f_params[:, 1]
        f = A_f.unsqueeze(1) * torch.exp(-0.5*(lambda_hr - self.mu_f)**2 / sigma_f.unsqueeze(1)**2)
        
        # 3. Compute atmospheric terms (t1-t12)
        t = {
            't1': t_params[:, 0],  # ρ_so
            't2': t_params[:, 1],  # ρ_dd
            't3': t_params[:, 2],  # τ_so
            't4': t_params[:, 3],  # τ_sl
            't5': t_params[:, 4],  # τ_do
            't6': t_params[:, 5],  # τ_sd
        }
        
        # Compute product terms (Table 3)
        t['t7'] = t['t3'] * t['t4']    # τ_so*τ_sl
        t['t8'] = t['t3'] * t['t6']    # τ_so*τ_sd
        t['t9'] = t['t4'] * t['t5']    # τ_sl*τ_do
        t['t10'] = t['t4'] * t['t2']   # τ_sl*ρ_dd
        t['t11'] = t['t3'] * t['t2']   # τ_so*ρ_dd
        t['t12'] = t['t3'] * t['t10']  # τ_so*τ_sl*ρ_dd
        
        # 4. Compute radiance components (Eq.2)
        numerator = (t['t8'].unsqueeze(1)*R + t['t12'].unsqueeze(1)*R + 
                    t['t9'].unsqueeze(1) + t['t10'].unsqueeze(1))
        denominator = 1 - t['t2'].unsqueeze(1)*R
        
        L_at_s_R = E_s.unsqueeze(-1).unsqueeze(-1) * cos_theta_s * (
            t['t1'].unsqueeze(1) + t['t7'].unsqueeze(1)*R + numerator/denominator
        )
        
        L_at_s_f = t['t5'].unsqueeze(1)*f + (
            t['t6'].unsqueeze(1) + t['t11'].unsqueeze(1)*R
        ) / denominator
        
        return L_at_s_R + L_at_s_f

class HyPlantSensorSimulator(nn.Module):
    def __init__(self, sensor_wavelengths, high_res=0.0055):
        super().__init__()
        self.register_buffer('sensor_wavelengths', sensor_wavelengths)
        self.high_res = high_res
        
    def forward(self, L_at_s_hr, delta_lambda, delta_sigma):
        """
        Simulates HyPlant sensor measurements
        Args:
            L_at_s_hr: High-res radiance [B, n_hr, H, W]
            delta_lambda: Wavelength shift [B, H, W]
            delta_sigma: SRF width shift [1]
        
        Returns:
            L_hyp: Sensor-measured radiance [B, n_sensor_bands, H, W]
        """
        B, C, H, W = L_at_s_hr.shape
        
        # 1. Apply wavelength shift
        shifted_wavelengths = self.sensor_wavelengths.view(1, -1, 1, 1) + \
                             delta_lambda.unsqueeze(1)
        
        # 2. Create Gaussian SRF
        sigma = 0.27 + delta_sigma  # Base FWHM=0.27nm from paper
        sigma = sigma * 2.3548  # Convert FWHM to sigma
        
        # Create Gaussian kernel
        x = torch.arange(-3*sigma, 3*sigma, self.high_res)
        kernel = torch.exp(-0.5*(x/sigma)**2)
        kernel /= kernel.sum()
        
        # 3. Apply spectral convolution
        L_blur = F.conv1d(L_at_s_hr.view(B*H*W, 1, -1), 
                         kernel.view(1, 1, -1), 
                         padding=kernel.size(0)//2)
        L_blur = L_blur.view(B, H, W, -1).permute(0, 3, 1, 2)
        
        # 4. Interpolate to sensor wavelengths
        x_hr = torch.arange(750, 770, self.high_res).expand(B, -1)
        L_hyp = F.grid_sample(L_blur, 
                             shifted_wavelengths.permute(0,2,3,1).unsqueeze(1),
                             align_corners=False)
        
        return L_hyp

class SFMNNSimulation(nn.Module):
    def __init__(self, sensor_wavelengths):
        super().__init__()
        self.four_stream = FourStreamSimulator()
        self.sensor_sim = HyPlantSensorSimulator(sensor_wavelengths)
        
    def forward(self, t_params, R_params, f_params, E_s, cos_theta_s, 
               delta_lambda, delta_sigma):
        # 1. Four-stream simulation
        L_hr = self.four_stream(t_params, R_params, f_params, E_s, cos_theta_s)
        
        # 2. Sensor characteristics application
        L_hyp = self.sensor_sim(L_hr, delta_lambda, delta_sigma)
        
        return L_hyp
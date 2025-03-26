import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import os
import logging
import json  # In case fallback to JSON is needed

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

class SFMNNDataset(Dataset):
    def __init__(self, lookup_table, data_folder, patch_size=5, out_dim=3620):
        """
        Loads simulation files from the given folder and constructs a dataframe.
        Each simulation file (e.g. simulation_sim*.parquet) is assumed to contain:
          - 'LTOA'
          - 'F' (target fluorescence/SIF) as a spectral vector of length out_dim.
          - 'MODTRAN_settings' with keys 'ATM' (containing 'SZA', 'GNDALT', 'VZA', etc.)
          - Optionally 'Esun'
        Only the first n_spectral channels (LTOA) and three extra channels (XTE, SZA, GNDALT)
        are used as network input.
        """
        self.patch_size = patch_size  # Each patch is patch_size x patch_size
        self.out_dim = out_dim

        # --- Load lookup table ---
        if lookup_table.endswith('.parquet'):
            lookup_table_data = pd.read_parquet(lookup_table).to_dict(orient='records')[0]
        else:
            with open(lookup_table) as f:
                lookup_table_data = json.load(f)
        self.wl = lookup_table_data["modtran_wavelength"]
        if len(self.wl) < 100:
            logger.warning("Lookup table wavelengths appear to be a range. Generating full wavelength grid.")
            self.wl = np.linspace(self.wl[0], self.wl[1], 3620)
        else:
            self.wl = np.array(self.wl)
        logger.info(f"Using {len(self.wl)} wavelength channels.")

        # --- Load simulation files ---
        sim_files = glob.glob(os.path.join(data_folder, "simulation_sim*.parquet"))
        logger.info(f"Found {len(sim_files)} simulation Parquet files.")
        LTOA_list = []
        XTE_list = []
        SZA_list = []
        GNDALT_list = []
        F_list = []  # For target SIF.
        AMBC_list = []   
        for file_path in tqdm(sim_files, desc="Loading simulation files"):
            if file_path.endswith('.parquet'):
                data = pd.read_parquet(file_path).to_dict(orient='records')[0]
            else:
                with open(file_path, "r") as f:
                    data = json.load(f)
            filename = os.path.basename(file_path)
            try:
                _, _, sim_n, _, amb_c = filename.split("_")
            except Exception as e:
                logger.error(f"Filename {filename} parsing error: {e}")
                continue

            LTOA_value = data['LTOA']  
            F_value = data['F']
            SZA_value = data['MODTRAN_settings']['ATM']['SZA']
            GNDALT_value = data['MODTRAN_settings']['ATM']['GNDALT']
            VZA_value = data['MODTRAN_settings']['ATM']['VZA']
            XTE_value = np.tan(np.deg2rad(VZA_value)) * GNDALT_value 

            LTOA_list.append(LTOA_value)
            SZA_list.append(SZA_value)
            GNDALT_list.append(GNDALT_value)
            XTE_list.append(XTE_value)
            F_list.append(F_value)
            try:
                amb_val = int(amb_c.split(".")[0])
            except:
                amb_val = 0
            AMBC_list.append(amb_val)

        # Build a DataFrame from the loaded simulation values.
        self.dataset = pd.DataFrame(LTOA_list, columns=self.wl)
        self.dataset['XTE'] = XTE_list
        self.dataset['SZA'] = SZA_list
        self.dataset['GNDALT'] = GNDALT_list
        self.dataset['F'] = F_list
        self.dataset['AMBC'] = AMBC_list

        # Compute normalization parameters for LTOA.
        self.ltoa_mean = self.dataset[self.wl].mean(axis=0).values
        self.ltoa_std = self.dataset[self.wl].std(axis=0).values

        # Compute normalization parameters for extra channels.
        self.xte_mean = self.dataset['XTE'].mean()
        self.xte_std = self.dataset['XTE'].std()
        self.sza_mean = self.dataset['SZA'].mean()
        self.sza_std = self.dataset['SZA'].std()
        self.gndalt_mean = self.dataset['GNDALT'].mean()
        self.gndalt_std = self.dataset['GNDALT'].std()

        # Compute normalization parameters for target SIF (F).
        F_stack = np.stack(self.dataset['F'].values).astype(np.float32)  # shape: (N, out_dim)
        self.f_mean = F_stack.mean(axis=0)  # shape: (out_dim,)
        self.f_std = F_stack.std(axis=0)    # shape: (out_dim,)

        # Build patches for training.
        self._resample_patches()
        self._reset_retrieval_markers()
        
        logger.info(f"Total simulations loaded: {len(self.dataset)}")
        logger.info(f"Total patches loaded: {len(self.patches)}")

    def _compute_tensor_patch(self, patch, num_wl):
        # Reshape LTOA values: shape (patch_size, patch_size, num_wl)
        ltoa = patch[self.wl].values.reshape(self.patch_size, self.patch_size, num_wl).astype(np.float32)
        # Reshape extra parameters: XTE, SZA, GNDALT each as (patch_size, patch_size, 1)
        xte = patch['XTE'].values.reshape(self.patch_size, self.patch_size, 1).astype(np.float32)
        sza = patch['SZA'].values.reshape(self.patch_size, self.patch_size, 1).astype(np.float32)
        gndalt = patch['GNDALT'].values.reshape(self.patch_size, self.patch_size, 1).astype(np.float32)
        # Concatenate along channel dimension -> shape: (patch_size, patch_size, num_wl+3)
        patch_tensor = np.concatenate([ltoa, xte, sza, gndalt], axis=-1)
        patch_tensor = torch.tensor(patch_tensor, dtype=torch.float)
        patch_tensor = self.normalize_patch(patch_tensor, num_wl)
        return patch_tensor

    def normalize_patch(self, patch_tensor, num_wl):
        # Normalize LTOA channels.
        ltoa_mean = torch.tensor(self.ltoa_mean, dtype=patch_tensor.dtype, device=patch_tensor.device)
        ltoa_std = torch.tensor(self.ltoa_std, dtype=patch_tensor.dtype, device=patch_tensor.device)
        patch_tensor[..., :num_wl] = (patch_tensor[..., :num_wl] - ltoa_mean) / (ltoa_std + 1e-6)
        # Normalize extra channels.
        xte_mean = torch.tensor(self.xte_mean, dtype=patch_tensor.dtype, device=patch_tensor.device)
        xte_std = torch.tensor(self.xte_std, dtype=patch_tensor.dtype, device=patch_tensor.device)
        patch_tensor[..., num_wl] = (patch_tensor[..., num_wl] - xte_mean) / (xte_std + 1e-6)
        sza_mean = torch.tensor(self.sza_mean, dtype=patch_tensor.dtype, device=patch_tensor.device)
        sza_std = torch.tensor(self.sza_std, dtype=patch_tensor.dtype, device=patch_tensor.device)
        patch_tensor[..., num_wl+1] = (patch_tensor[..., num_wl+1] - sza_mean) / (sza_std + 1e-6)
        gndalt_mean = torch.tensor(self.gndalt_mean, dtype=patch_tensor.dtype, device=patch_tensor.device)
        gndalt_std = torch.tensor(self.gndalt_std, dtype=patch_tensor.dtype, device=patch_tensor.device)
        patch_tensor[..., num_wl+2] = (patch_tensor[..., num_wl+2] - gndalt_mean) / (gndalt_std + 1e-6)
        return patch_tensor

    def unnormalize_patch(self, patch_tensor, num_wl):
        patch_tensor = patch_tensor.clone()
        ltoa_mean = torch.tensor(self.ltoa_mean, dtype=patch_tensor.dtype, device=patch_tensor.device)
        ltoa_std = torch.tensor(self.ltoa_std, dtype=patch_tensor.dtype, device=patch_tensor.device)
        patch_tensor[..., :num_wl] = patch_tensor[..., :num_wl] * (ltoa_std + 1e-6) + ltoa_mean
        xte_mean = torch.tensor(self.xte_mean, dtype=patch_tensor.dtype, device=patch_tensor.device)
        xte_std = torch.tensor(self.xte_std, dtype=patch_tensor.dtype, device=patch_tensor.device)
        patch_tensor[..., num_wl] = patch_tensor[..., num_wl] * (xte_std + 1e-6) + xte_mean
        sza_mean = torch.tensor(self.sza_mean, dtype=patch_tensor.dtype, device=patch_tensor.device)
        sza_std = torch.tensor(self.sza_std, dtype=patch_tensor.dtype, device=patch_tensor.device)
        patch_tensor[..., num_wl+1] = patch_tensor[..., num_wl+1] * (sza_std + 1e-6) + sza_mean
        gndalt_mean = torch.tensor(self.gndalt_mean, dtype=patch_tensor.dtype, device=patch_tensor.device)
        gndalt_std = torch.tensor(self.gndalt_std, dtype=patch_tensor.dtype, device=patch_tensor.device)
        patch_tensor[..., num_wl+2] = patch_tensor[..., num_wl+2] * (gndalt_std + 1e-6) + gndalt_mean
        return patch_tensor

    def unnormalize_sif(self, sif_tensor):
        """
        Reverses the normalization applied to SIF. Expects sif_tensor of shape [out_dim, H, W] or [H, W, out_dim].
        """
        # Assume sif_tensor is of shape [H, W, out_dim] and permute it to [out_dim, H, W] if necessary.
        if sif_tensor.ndim == 3 and sif_tensor.shape[-1] == self.out_dim:
            sif_tensor = sif_tensor.permute(2, 0, 1)
        f_mean = torch.tensor(self.f_mean, dtype=sif_tensor.dtype, device=sif_tensor.device)
        f_std = torch.tensor(self.f_std, dtype=sif_tensor.dtype, device=sif_tensor.device)
        return sif_tensor * (f_std + 1e-6) + f_mean

    def _resample_patches(self):
        self.patches = []
        self.sif_patches = []
        patch_elem_count = self.patch_size ** 2
        num_wl = len(self.wl)
        for ambc, group in self.dataset.groupby('AMBC'):
            group = group.sample(frac=1).reset_index(drop=True)
            n_rows = len(group)
            num_full_patches = n_rows // patch_elem_count
            remainder = n_rows % patch_elem_count
            for i in range(num_full_patches):
                patch = group.iloc[i * patch_elem_count : (i + 1) * patch_elem_count]
                tensor_patch = self._compute_tensor_patch(patch, num_wl)
                sif_patch = self._compute_sif_patch(patch)  # shape: [patch_size, patch_size, out_dim]
                self.patches.append(tensor_patch)
                self.sif_patches.append(sif_patch)
            if remainder > 0:
                patch = group.iloc[num_full_patches * patch_elem_count :]
                pad_needed = patch_elem_count - remainder
                replace_flag = pad_needed > len(group)
                pad = group.sample(n=pad_needed, replace=replace_flag)
                patch = pd.concat([patch, pad], ignore_index=True)
                tensor_patch = self._compute_tensor_patch(patch, num_wl)
                sif_patch = self._compute_sif_patch(patch)
                self.patches.append(tensor_patch)
                self.sif_patches.append(sif_patch)

    def _reset_retrieval_markers(self):
        self._retrieved = [False] * len(self.patches)
    
    def get_wl(self):
        return self.wl
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.patches):
            raise IndexError("Index out of range")
        self._retrieved[idx] = True
        # The input patch is of shape [patch_size, patch_size, num_wl+3]. Permute to [C, H, W]
        tensor_patch = self.patches[idx].permute(2, 0, 1)
        # Get target SIF patch and permute to [out_dim, H, W]
        target_sif = self.sif_patches[idx].permute(2, 0, 1)
        if all(self._retrieved):
            self._resample_patches()
            self._reset_retrieval_markers()
        return tensor_patch, target_sif
    
    


class SFMNNDatasetSingleWithTargets(Dataset):
    def __init__(self, lookup_table, data_folder, out_dim_hint=3620):
        """
        Loads simulation files, including LTOA, t_vals (t1-t11), R, and F.
        Returns normalized input vector and unnormalized target t, R, F, LTOA tensors.
        Normalization parameters are computed and stored.

        Args:
            lookup_table (str): Path to the Parquet or JSON lookup table file.
            data_folder (str): Path to the directory containing simulation_sim*.parquet files.
            out_dim_hint (int): Expected number of spectral bands, used if lookup table fails.
        """
        self.out_dim = out_dim_hint
        self.n_spectral_bands = out_dim_hint
        self.t_keys_all = [f't{i}' for i in range(1, 12)] # t1 to t11
        # Define the 9 t-keys relevant for LTOA calc and model prediction order
        self.t_keys_model_output = ['t1', 't2', 't3', 't6', 't7', 't8', 't9', 't10', 't11']
        # Define indices corresponding to the 9 model t-keys within the full target_11t tensor
        self.target_indices_for_9t_compare = [
            0, # t1
            1, # t2
            2, # t3
            5, # t6
            6, # t7
            7, # t8
            8, # t9
            9, # t10
            10 # t11
        ]
        assert len(self.t_keys_model_output) == 9
        assert len(self.target_indices_for_9t_compare) == 9


        # --- Load lookup table ---
        try:
            if lookup_table.endswith('.parquet'):
                lookup_table_data = pd.read_parquet(lookup_table).to_dict(orient='records')[0]
            elif lookup_table.endswith('.json'):
                with open(lookup_table) as f:
                    lookup_table_data = json.load(f)
            else:
                 raise ValueError(f"Unsupported lookup table format: {lookup_table}. Use .parquet or .json")

            self.wl = lookup_table_data.get("modtran_wavelength")
            if self.wl is None or not isinstance(self.wl, (list, np.ndarray)) or len(self.wl) < 100:
                logger.warning(f"MODTRAN wavelengths invalid/missing in {lookup_table}. Falling back to out_dim_hint ({self.out_dim}).")
                # Create placeholder wavelengths if needed for indexing, but rely on data dimension
                min_wl, max_wl = 650, 850 # Example range, adjust if known
                self.wl = np.linspace(min_wl, max_wl, self.out_dim)
                self.n_spectral_bands = self.out_dim
            else:
                 self.wl = np.array(self.wl)
                 self.n_spectral_bands = len(self.wl)
                 if self.n_spectral_bands != self.out_dim:
                     logger.info(f"Adjusting out_dim from hint ({self.out_dim}) to actual wavelengths found ({self.n_spectral_bands}).")
                     self.out_dim = self.n_spectral_bands

            logger.info(f"Using {self.n_spectral_bands} wavelength channels.")

        except Exception as e:
            logger.error(f"Error loading lookup table {lookup_table}: {e}. Exiting.")
            raise

        # --- Load simulation files ---
        sim_files = glob.glob(os.path.join(data_folder, "simulation_sim*.parquet"))
        if not sim_files:
            raise FileNotFoundError(f"No simulation parquet files found in {data_folder}")
        logger.info(f"Found {len(sim_files)} simulation Parquet files.")

        all_data = []
        required_sim_keys = ['LTOA', 'F', 'R', 't_vals', 'MODTRAN_settings']
        required_modtran_keys = ['ATM']
        required_atm_keys = ['SZA', 'GNDALT', 'VZA']
        skipped_files = 0

        for file_path in tqdm(sim_files, desc="Loading simulation files"):
            try:
                data = pd.read_parquet(file_path).to_dict(orient='records')[0]

                # --- Data Validation ---
                if not all(key in data for key in required_sim_keys):
                    skipped_files += 1
                    continue
                if not isinstance(data['t_vals'], dict) or not all(tk in data['t_vals'] for tk in self.t_keys_all):
                    skipped_files += 1
                    continue
                if not isinstance(data['MODTRAN_settings'], dict) or not all(key in data['MODTRAN_settings'] for key in required_modtran_keys):
                     skipped_files += 1
                     continue
                if not isinstance(data['MODTRAN_settings']['ATM'], dict) or not all(key in data['MODTRAN_settings']['ATM'] for key in required_atm_keys):
                     skipped_files += 1
                     continue

                # Extract spectral data first to check dimensions
                ltoa = data['LTOA']
                f = data['F']
                r = data['R']
                t_vals_dict = data['t_vals']

                if len(ltoa) != self.n_spectral_bands or len(f) != self.n_spectral_bands or len(r) != self.n_spectral_bands:
                    skipped_files += 1
                    continue
                if any(len(t_vals_dict[tk]) != self.n_spectral_bands for tk in self.t_keys_all):
                    skipped_files += 1
                    continue
                # --- End Validation ---

                # Extract geometry
                sza = data['MODTRAN_settings']['ATM']['SZA']
                gndalt = data['MODTRAN_settings']['ATM']['GNDALT'] # Assume km
                vza = data['MODTRAN_settings']['ATM']['VZA']
                xte = np.tan(np.deg2rad(vza)) * gndalt # Stays in km if gndalt is km

                # Store extracted data (keep as numpy arrays for efficient stacking)
                all_data.append({
                    'ltoa': np.array(ltoa, dtype=np.float32),
                    'f': np.array(f, dtype=np.float32),
                    'r': np.array(r, dtype=np.float32),
                    # Stack t_vals in the defined order t1..t11
                    't_vals': np.stack([np.array(t_vals_dict[tk], dtype=np.float32) for tk in self.t_keys_all]), # Shape [11, n_spec]
                    'xte': np.float32(xte),
                    'sza': np.float32(sza),
                    'gndalt': np.float32(gndalt)
                })

            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
                skipped_files += 1

        if skipped_files > 0:
             logger.warning(f"Skipped {skipped_files} files due to missing keys, dimension mismatches, or errors.")
        if not all_data:
             raise ValueError("No valid simulation data could be loaded. Check simulation files and paths.")

        # Convert list of dicts to a dict of stacked numpy arrays
        self.data_store = {
            'ltoa': np.stack([d['ltoa'] for d in all_data]),
            'f': np.stack([d['f'] for d in all_data]),
            'r': np.stack([d['r'] for d in all_data]),
            't_vals': np.stack([d['t_vals'] for d in all_data]), # Shape: [N, 11, n_spectral_bands]
            'xte': np.array([d['xte'] for d in all_data], dtype=np.float32),
            'sza': np.array([d['sza'] for d in all_data], dtype=np.float32),
            'gndalt': np.array([d['gndalt'] for d in all_data], dtype=np.float32),
        }

        logger.info(f"Successfully loaded {len(self)} valid simulations.")

        # --- Compute normalization parameters ---
        # Input features (LTOA + extras)
        self.ltoa_mean = self.data_store['ltoa'].mean(axis=0)
        self.ltoa_std = self.data_store['ltoa'].std(axis=0)
        self.xte_mean = self.data_store['xte'].mean()
        self.xte_std = self.data_store['xte'].std()
        self.sza_mean = self.data_store['sza'].mean()
        self.sza_std = self.data_store['sza'].std()
        self.gndalt_mean = self.data_store['gndalt'].mean()
        self.gndalt_std = self.data_store['gndalt'].std()

        # Target values (t, R, F) - For unnormalizing model output
        # Normalize each t-variable (t1..t11) individually across all samples and wavelengths
        self.t_means = self.data_store['t_vals'].mean(axis=(0, 2)) # Shape: [11]
        self.t_stds = self.data_store['t_vals'].std(axis=(0, 2))   # Shape: [11]

        self.r_mean = self.data_store['r'].mean(axis=0) # Shape: [n_spectral_bands]
        self.r_std = self.data_store['r'].std(axis=0)   # Shape: [n_spectral_bands]
        self.f_mean = self.data_store['f'].mean(axis=0) # Shape: [n_spectral_bands]
        self.f_std = self.data_store['f'].std(axis=0)   # Shape: [n_spectral_bands]

        # Add epsilon to std devs
        self.epsilon = 1e-6
        self.ltoa_std += self.epsilon
        self.xte_std += self.epsilon
        self.sza_std += self.epsilon
        self.gndalt_std += self.epsilon
        self.t_stds += self.epsilon
        self.r_std += self.epsilon
        self.f_std += self.epsilon

        logger.info("Normalization parameters computed.")
        logger.info(f"Dataset initialized with {len(self)} samples.")

    def __len__(self):
        return len(self.data_store['ltoa'])

    def get_wl(self):
        return self.wl

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        # --- Input Vector (Normalized) ---
        ltoa_norm = (self.data_store['ltoa'][idx] - self.ltoa_mean) / self.ltoa_std
        # Ensure extras are treated as scalars before normalization
        xte_norm = (self.data_store['xte'][idx] - self.xte_mean) / self.xte_std
        sza_norm = (self.data_store['sza'][idx] - self.sza_mean) / self.sza_std
        gndalt_norm = (self.data_store['gndalt'][idx] - self.gndalt_mean) / self.gndalt_std

        input_vec = np.concatenate([
            ltoa_norm,
            np.array([xte_norm], dtype=np.float32),
            np.array([sza_norm], dtype=np.float32),
            np.array([gndalt_norm], dtype=np.float32)
        ])
        input_tensor = torch.tensor(input_vec, dtype=torch.float)

        # --- Target Tensors (Unnormalized) ---
        # Return all 11 target t's
        target_11t_tensor = torch.tensor(self.data_store['t_vals'][idx], dtype=torch.float) # Shape: [11, n_spec]
        target_r_tensor = torch.tensor(self.data_store['r'][idx], dtype=torch.float)      # Shape: [n_spec]
        target_f_tensor = torch.tensor(self.data_store['f'][idx], dtype=torch.float)      # Shape: [n_spec]
        target_ltoa_tensor = torch.tensor(self.data_store['ltoa'][idx], dtype=torch.float) # Shape: [n_spec]

        return input_tensor, target_11t_tensor, target_r_tensor, target_f_tensor, target_ltoa_tensor

    # --- Unnormalization methods for model outputs ---
    def unnormalize_t(self, norm_t_tensor):
        """
        Unnormalizes the 9 predicted t-values (t1,t2,t3,t6-t11) using the
        means/stds calculated for t1..t11.
        Expects norm_t_tensor shape [B, 9, n_spec] or [9, n_spec].
        """
        num_vars_in_tensor = norm_t_tensor.shape[-2]
        if num_vars_in_tensor != 9:
             raise ValueError(f"unnormalize_t expects 9 t-variables, but got {num_vars_in_tensor}")

        # Select the means/stds for t1,t2,t3,t6,t7,t8,t9,t10,t11 using their indices [0,1,2,5,6,7,8,9,10]
        means_for_9t = torch.tensor(self.t_means[self.target_indices_for_9t_compare], dtype=norm_t_tensor.dtype, device=norm_t_tensor.device) # Shape [9]
        stds_for_9t = torch.tensor(self.t_stds[self.target_indices_for_9t_compare], dtype=norm_t_tensor.dtype, device=norm_t_tensor.device)   # Shape [9]

        # Reshape means/stds for broadcasting
        if norm_t_tensor.ndim == 3: # Batch dimension present [B, 9, n_spec]
             means_for_9t = means_for_9t.view(1, -1, 1)
             stds_for_9t = stds_for_9t.view(1, -1, 1)
        elif norm_t_tensor.ndim == 2: # No batch dimension [9, n_spec]
             means_for_9t = means_for_9t.view(-1, 1)
             stds_for_9t = stds_for_9t.view(-1, 1)
        else:
            raise ValueError("Unsupported tensor dimension for unnormalize_t")

        return norm_t_tensor * stds_for_9t + means_for_9t

    def unnormalize_r(self, norm_r_tensor):
        """Unnormalizes predicted R. Expects [B, n_spec] or [n_spec]."""
        r_mean_t = torch.tensor(self.r_mean, dtype=norm_r_tensor.dtype, device=norm_r_tensor.device)
        r_std_t = torch.tensor(self.r_std, dtype=norm_r_tensor.dtype, device=norm_r_tensor.device)
        if norm_r_tensor.ndim == 2: # Batch dimension present
            r_mean_t = r_mean_t.unsqueeze(0)
            r_std_t = r_std_t.unsqueeze(0)
        return norm_r_tensor * r_std_t + r_mean_t

    def unnormalize_f(self, norm_f_tensor):
        """Unnormalizes predicted F. Expects [B, n_spec] or [n_spec]."""
        f_mean_t = torch.tensor(self.f_mean, dtype=norm_f_tensor.dtype, device=norm_f_tensor.device)
        f_std_t = torch.tensor(self.f_std, dtype=norm_f_tensor.dtype, device=norm_f_tensor.device)
        if norm_f_tensor.ndim == 2: # Batch dimension present
            f_mean_t = f_mean_t.unsqueeze(0)
            f_std_t = f_std_t.unsqueeze(0)
        return norm_f_tensor * f_std_t + f_mean_t

    def unnormalize_ltoa(self, norm_ltoa_tensor):
        """Unnormalizes predicted LTOA. Expects [B, n_spec] or [n_spec]."""
        ltoa_mean_t = torch.tensor(self.ltoa_mean, dtype=norm_ltoa_tensor.dtype, device=norm_ltoa_tensor.device)
        ltoa_std_t = torch.tensor(self.ltoa_std, dtype=norm_ltoa_tensor.dtype, device=norm_ltoa_tensor.device)
        if norm_ltoa_tensor.ndim == 2: # Batch dimension present
            ltoa_mean_t = ltoa_mean_t.unsqueeze(0)
            ltoa_std_t = ltoa_std_t.unsqueeze(0)
        return norm_ltoa_tensor * ltoa_std_t + ltoa_mean_t
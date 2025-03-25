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


class SFMNNDatasetSingle(Dataset):
    def __init__(self, lookup_table, data_folder, out_dim=3620):
        """
        Loads simulation files from the given folder and constructs a DataFrame.
        Each simulation file (e.g. simulation_sim*.parquet) is assumed to contain:
          - 'LTOA': A spectral vector.
          - 'F' (target fluorescence/SIF) as a spectral vector of length out_dim.
          - 'MODTRAN_settings' with keys 'ATM' (containing 'SZA', 'GNDALT', 'VZA', etc.)
          - Optionally 'Esun'
        The network input is built from the LTOA channels and three extra channels (XTE, SZA, GNDALT).
        Each sample corresponds to a single simulation (a “pixel”) without normalization applied.
        Normalization can be applied later using the provided normalize_F/unnormalize_F functions.
        """
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

        logger.info(f"Total simulations loaded: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)
    
    def get_wl(self):
        return self.wl

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.dataset):
            raise IndexError("Index out of range")
            
        row = self.dataset.iloc[idx]
        num_wl = len(self.wl)
        # Build input vector by concatenating LTOA and extra channels (XTE, SZA, GNDALT)
        ltoa = np.array([row[w] for w in self.wl], dtype=np.float32)  # shape: (num_wl,)
        xte = np.array([row['XTE']], dtype=np.float32)
        sza = np.array([row['SZA']], dtype=np.float32)
        gndalt = np.array([row['GNDALT']], dtype=np.float32)
        input_vec = np.concatenate([ltoa, xte, sza, gndalt], axis=0)  # shape: (num_wl+3,)
        input_tensor = torch.tensor(input_vec, dtype=torch.float)
        
        # Get target SIF (F) as a raw (unnormalized) tensor.
        F_val = np.array(row['F'], dtype=np.float32)  # shape: (out_dim,)
        F_tensor = torch.tensor(F_val, dtype=torch.float)
        
        return input_tensor, F_tensor

    def normalize_F(self, F_tensor):
        """
        Normalizes the SIF (F) tensor using the dataset's precomputed mean and std.
        
        Args:
            F_tensor (torch.Tensor): Tensor containing SIF values. Expected shape can be (out_dim,)
                                     or have extra dimensions.
        
        Returns:
            torch.Tensor: Normalized SIF tensor.
        """
        f_mean = torch.tensor(self.f_mean, dtype=F_tensor.dtype, device=F_tensor.device)
        f_std = torch.tensor(self.f_std, dtype=F_tensor.dtype, device=F_tensor.device)
        return (F_tensor - f_mean) / (f_std + 1e-6)

    def unnormalize_F(self, norm_F_tensor):
        """
        Reverses the normalization of the SIF (F) tensor using the dataset's precomputed mean and std.
        
        Args:
            norm_F_tensor (torch.Tensor): Normalized SIF tensor. Expected shape can be (out_dim,)
                                          or have extra dimensions.
        
        Returns:
            torch.Tensor: Unnormalized SIF tensor.
        """
        f_mean = torch.tensor(self.f_mean, dtype=norm_F_tensor.dtype, device=norm_F_tensor.device)
        f_std = torch.tensor(self.f_std, dtype=norm_F_tensor.dtype, device=norm_F_tensor.device)
        return norm_F_tensor * (f_std + 1e-6) + f_mean
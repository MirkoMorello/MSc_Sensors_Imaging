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
    def __init__(self, lookup_table, data_folder, patch_size=5):
        """
        Loads simulation files from the given folder and constructs a dataframe.
        Each simulation file (e.g. simulation_sim*.parquet) is assumed to contain:
          - 'LTOA'
          - 'MODTRAN_settings' with keys 'ATM' (containing 'SZA', 'GNDALT', 'VZA', etc.)
          - Optionally 'Esun'
        Only the first n_spectral channels (LTOA) and three extra channels (XTE, SZA, GNDALT)
        are used as network input. Esun is used only to compute a fixed Esun.
        """
        self.patch_size = patch_size  # Each patch is patch_size x patch_size

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
        # Assume simulation files are stored as Parquet files in the data folder.
        sim_files = glob.glob(os.path.join(data_folder, "simulation_sim*.parquet"))
        logger.info(f"Found {len(sim_files)} simulation Parquet files.")
        LTOA_list = []
        XTE_list = []
        SZA_list = []
        GNDALT_list = []
        F_list = []  # For computing fixed Esun later.
        AMBC_list = []   
        for file_path in tqdm(sim_files, desc="Loading simulation files"):
            if file_path.endswith('.parquet'):
                # Load as a DataFrame and convert first row to dict.
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

            # Extract required values.
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

        # Compute normalization parameters over the dataset.
        self.ltoa_mean = self.dataset[self.wl].mean(axis=0).values
        self.ltoa_std = self.dataset[self.wl].std(axis=0).values

        self.xte_mean = self.dataset['XTE'].mean()
        self.xte_std = self.dataset['XTE'].std()
        self.sza_mean = self.dataset['SZA'].mean()
        self.sza_std = self.dataset['SZA'].std()
        self.gndalt_mean = self.dataset['GNDALT'].mean()
        self.gndalt_std = self.dataset['GNDALT'].std()

        self._resample_patches()
        self._reset_retrieval_markers()
        
        logger.info(f"Total simulations loaded: {len(self.dataset)}")
        logger.info(f"Total patches loaded: {len(self.patches)}")

    def _compute_tensor_patch(self, patch, num_wl):
        patch_elem_count = self.patch_size ** 2
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

    def _resample_patches(self):
        self.patches = []
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
                self.patches.append(tensor_patch)
            if remainder > 0:
                patch = group.iloc[num_full_patches * patch_elem_count :]
                pad_needed = patch_elem_count - remainder
                replace_flag = pad_needed > len(group)
                pad = group.sample(n=pad_needed, replace=replace_flag)
                patch = pd.concat([patch, pad], ignore_index=True)
                tensor_patch = self._compute_tensor_patch(patch, num_wl)
                self.patches.append(tensor_patch)

    def _reset_retrieval_markers(self):
        self._retrieved = [False] * len(self.patches)
    
    def get_wl(self):
        return self.wl
    
    def _get_sif(self, idx):
        return self.dataset['F'].iloc[idx]
        
    def get_wl(self):
        return self.wl
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.patches):
            raise IndexError("Index out of range")
        self._retrieved[idx] = True
        tensor_patch = self.patches[idx].permute(2, 0, 1)
        # Retrieve the target fluorescence using the helper function.
        target_sif = self._get_sif(idx)  
        if all(self._retrieved):
            self._resample_patches()
            self._reset_retrieval_markers()
        return tensor_patch, target_sif


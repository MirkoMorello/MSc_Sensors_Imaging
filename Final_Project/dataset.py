import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import glob
from tqdm import tqdm
import os

import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

class SFMNNDataset(Dataset):
    def __init__(self, lookup_table, data_folder, patch_size=5):
        """
        Loads simulation JSON files and constructs a dataframe.
        For each simulation, we load LTOA, XTE, SZA, GNDALT, and Esun.
        Only LTOA, XTE, SZA, and GNDALT (i.e. num_wavelengths+3 channels)
        are used as network input. Esun is used only to compute a fixed Esun.
        """
        self.patch_size = patch_size  # Each patch is patch_size x patch_size

        # Temporary lists to collect data from JSON files
        LTOA_list = []  
        XTE_list = []   
        SZA_list = []   
        GNDALT_list = [] 
        Esun_list = []  # For later computing fixed Esun
        AMBC_list = []   

        # Load lookup table to get wavelengths (used as column names)
        with open(lookup_table) as f:
            lookup_table_data = json.load(f)
            self.wl = lookup_table_data["modtran_wavelength"]
        # If the lookup table contains only a range, generate full grid with 3620 points.
        if len(self.wl) < 100:
            logger.warning("Lookup table wavelengths appear to be a range. Generating full wavelength grid.")
            self.wl = np.linspace(self.wl[0], self.wl[1], 3620)
        else:
            self.wl = np.array(self.wl)
        logger.info(f"Using {len(self.wl)} wavelength channels.")

        sim_files = glob.glob(os.path.join(data_folder, "simulation_sim*.json"))
        logger.info(f"Found {len(sim_files)} simulation JSON files.")
        for file_path in tqdm(sim_files, desc="Loading JSON files"):
            with open(file_path, "r") as f:
                data = json.load(f)

            filename = os.path.basename(file_path)
            # Assuming filename format: simulation_sim_1_amb_0.json
            try:
                _, _, sim_n, _, amb_c = filename.split("_")
            except Exception as e:
                logger.error(f"Filename {filename} parsing error: {e}")
                continue

            # Extract values from JSON structure.
            LTOA_value = data['LTOA']  
            SZA_value = data['MODTRAN_settings']['ATM']['SZA']
            GNDALT_value = data['MODTRAN_settings']['ATM']['GNDALT']
            VZA_value = data['MODTRAN_settings']['ATM']['VZA']
            # Compute horizontal extent.
            XTE_value = np.tan(np.deg2rad(VZA_value)) * GNDALT_value
            # Extract Esun; if missing, default to 1.0.
            Esun_value = data.get("Esun", None)
            if Esun_value is None:
                logger.warning(f"File {filename} missing 'Esun'; using default value 1.0.")
                Esun_value = 1.0

            LTOA_list.append(LTOA_value)
            SZA_list.append(SZA_value)
            GNDALT_list.append(GNDALT_value)
            XTE_list.append(XTE_value)
            Esun_list.append(Esun_value)
            try:
                amb_val = int(amb_c.split(".")[0])
            except:
                amb_val = 0
            AMBC_list.append(amb_val)

        # Build dataframe.
        self.dataset = pd.DataFrame(LTOA_list, columns=self.wl)
        self.dataset['XTE'] = XTE_list
        self.dataset['SZA'] = SZA_list
        self.dataset['GNDALT'] = GNDALT_list
        self.dataset['Esun'] = Esun_list
        self.dataset['AMBC'] = AMBC_list

        # Compute fixed Esun as the mean over all Esun values.
        self.fixed_esun = np.mean(Esun_list)
        logger.info(f"Fixed Esun computed as: {self.fixed_esun:.4f}")

        self._resample_patches()
        self._reset_retrieval_markers()
        
        logger.info(f"Total simulations loaded: {len(self.dataset)}")
        logger.info(f"Total patches loaded: {len(self.patches)}")

    def _compute_tensor_patch(self, patch, num_wl):
        """
        Convert a patch (DataFrame) into a tensor of shape [patch_size, patch_size, num_wl+3],
        where:
          - The first num_wl channels are the LTOA spectral signal.
          - The next three channels are XTE, SZA, and GNDALT.
        Esun is not included.
        """
        patch_elem_count = self.patch_size ** 2
        ltoa = patch[self.wl].values.reshape(self.patch_size, self.patch_size, num_wl)
        xte = patch['XTE'].values.reshape(self.patch_size, self.patch_size, 1)
        sza = patch['SZA'].values.reshape(self.patch_size, self.patch_size, 1)
        gndalt = patch['GNDALT'].values.reshape(self.patch_size, self.patch_size, 1)
        patch_tensor = np.concatenate([ltoa, xte, sza, gndalt], axis=-1)
        return torch.tensor(patch_tensor, dtype=torch.float)

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
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.patches):
            raise IndexError("Index out of range")
        self._retrieved[idx] = True
        # Permute to [C, H, W]
        tensor_patch = self.patches[idx].permute(2, 0, 1)
        if all(self._retrieved):
            self._resample_patches()
            self._reset_retrieval_markers()
        return tensor_patch
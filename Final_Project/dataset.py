import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import glob
from tqdm import tqdm

# -------------------------
# Modified Dataset
# -------------------------
class SFMNNDataset(Dataset):
    def __init__(self, lookup_table, data_folder, patch_size=5):
        self.patch_size = patch_size  # Each patch is patch_size x patch_size
        # Temporary lists to collect data from JSON files
        LTOA_list = []  
        XTE_list = []   
        SZA_list = []   
        GNDALT_list = [] 
        AMBC_list = []   

        # Load lookup table to get wavelengths (used as column names)
        with open(lookup_table) as f:
            lookup_table_data = json.load(f)
            self.wl = lookup_table_data["modtran_wavelength"]  # Expect length = 3620

        # Load and process each JSON file
        for file_path in tqdm(glob.glob(data_folder + "simulation_sim*.json"), desc="Loading JSON files"):
            with open(file_path, "r") as f:
                data = json.load(f)

            filename = file_path.split("/")[-1]
            # Assuming filename format: simulation_sim_1_amb_0.json
            _, _, sim_n, _, amb_c = filename.split("_")
            # Extract values from JSON structure
            LTOA_value = data['LTOA']  
            SZA_value = data['MODTRAN_settings']['ATM']['SZA']
            GNDALT_value = data['MODTRAN_settings']['ATM']['GNDALT']
            VZA_value = data['MODTRAN_settings']['ATM']['VZA']
            XTE_value = np.tan(VZA_value) * GNDALT_value

            LTOA_list.append(LTOA_value)
            SZA_list.append(SZA_value)
            GNDALT_list.append(GNDALT_value)
            XTE_list.append(XTE_value)
            AMBC_list.append(int(amb_c[:-5]))  # remove ".json" and convert to int

        # Build the dataset DataFrame from LTOA values and add extra columns.
        self.dataset = pd.DataFrame(LTOA_list, columns=self.wl)
        self.dataset['XTE'] = XTE_list
        self.dataset['SZA'] = SZA_list
        self.dataset['GNDALT'] = GNDALT_list
        self.dataset['AMBC'] = AMBC_list

        self._resample_patches()
        self._reset_retrieval_markers()
        
        print(f"Total number of elements loaded: {len(self.dataset)}")
        print(f"Total number of patches loaded: {len(self.patches)}")

    def _compute_tensor_patch(self, patch, num_wl):
        """
        Convert a patch (DataFrame) into a tensor of shape [patch_size, patch_size, num_wl+3],
        where:
          - The first num_wl channels correspond to the LTOA spectral signal.
          - The last three channels correspond to XTE, SZA, and GNDALT.
        """
        patch_elem_count = self.patch_size ** 2
        # LTOA: shape (patch_elem_count, num_wl)
        ltoa = patch[self.wl].values  # shape: (patch_elem_count, num_wl)
        ltoa = ltoa.reshape(self.patch_size, self.patch_size, num_wl)

        xte = patch['XTE'].values.reshape(self.patch_size, self.patch_size, 1)
        sza = patch['SZA'].values.reshape(self.patch_size, self.patch_size, 1)
        gndalt = patch['GNDALT'].values.reshape(self.patch_size, self.patch_size, 1)

        patch_tensor = np.concatenate([ltoa, xte, sza, gndalt], axis=-1)
        # Return as a tensor of shape [H, W, C]
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
        if idx >= len(self.patches) or idx < 0:
            raise IndexError("Index out of range")
        self._retrieved[idx] = True
        tensor_patch = self.patches[idx]  # shape: [patch_size, patch_size, num_wl+3]
        # Permute to [C, H, W] for easier use later.
        tensor_patch = tensor_patch.permute(2, 0, 1)
        if all(self._retrieved):
            self._resample_patches()
            self._reset_retrieval_markers()
        return tensor_patch
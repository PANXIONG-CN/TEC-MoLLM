import torch
import numpy as np
import logging
import os
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SlidingWindowSamplerDataset(Dataset):
    """
    A PyTorch Dataset that uses pre-processed, pre-aligned data chunks and
    implements sliding window sampling.
    It loads an entire data split (e.g., 'train') into memory.
    """
    def __init__(self, data_path: str, mode: str, L_in: int = 336, L_out: int = 12):
        """
        Args:
            data_path (str): Path to the directory containing the processed .pt files.
            mode (str): The dataset split to load, one of ['train', 'val', 'test'].
            L_in (int): Input sequence length (window size).
            L_out (int): Output sequence length (prediction horizon).
        """
        super().__init__()
        assert mode in ['train', 'val', 'test'], "Mode must be one of 'train', 'val', or 'test'"
        
        self.L_in = L_in
        self.L_out = L_out
        
        # --- Load pre-processed data from .pt file ---
        file_path = os.path.join(data_path, f"{mode}_set.pt")
        logging.info(f"Attempting to load pre-processed data from: {file_path}")
        try:
            data = torch.load(file_path, map_location='cpu')
            self.X = data['X']
            self.Y = data['Y']
            self.time_features = data['time_features']
            logging.info(f"Successfully loaded '{mode}' data. Shapes: X={self.X.shape}, Y={self.Y.shape}, time_features={self.time_features.shape}")
        except FileNotFoundError:
            logging.error(f"FATAL: Pre-processed data file not found at {file_path}. Please run the preprocessing script first.")
            raise
        
        # The number of samples is determined by how many full windows can be created.
        # Since data is pre-aligned, the logic is simplified.
        # --- START MODIFICATION 1.2.1 ---
        # REASON: The original calculation did not account for the output window length (L_out),
        #         leading to index errors or using incomplete targets at the end of the dataset.
        # OLD: self.num_samples = max(0, len(self.X) - self.L_in + 1)
        # NEW:
        self.num_samples = max(0, len(self.X) - self.L_in - self.L_out + 1)
        # --- END MODIFICATION 1.2.1 ---
        
        if self.num_samples <= 0:
            logging.warning(f"Insufficient data for windowing: Total length={len(self.X)}, L_in={self.L_in}. Resulting samples: {self.num_samples}")
            self.num_samples = 0
            
        logging.info(f"Dataset '{mode}' initialized. Window config: L_in={self.L_in}, L_out={self.L_out}")
        logging.info(f"Total available samples: {self.num_samples}")

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single sample from the dataset. This is now a lightweight slicing operation.
        
        Args:
            idx (int): The index of the sample to retrieve.
            
        Returns:
            dict: A dictionary containing the input window, target window, and time features as tensors.
        """
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} is out of bounds for a dataset of size {self.num_samples}")

        # Determine the slice for the input window
        x_start = idx
        x_end = idx + self.L_in
        
        # Slice the data tensors directly
        x_window = self.X[x_start:x_end]
        time_features_window = self.time_features[x_start:x_end]
        
        # The target `Y` was pre-calculated and aligned. Y[t] contains future values for X[t].
        # Our input window ends at time `t = idx + L_in - 1`.
        # The corresponding target is at this same index.
        target_y = self.Y[idx + self.L_in - 1]

        # 直接返回，不再有任何scaler操作
        return {
            'x': x_window,      # 已是tensor且已标准化
            'y': target_y,      # 已是tensor且已标准化
            'x_time_features': time_features_window  # 已是tensor
        } 
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
    def __init__(self, data_path: str, mode: str, L_in: int = 336, L_out: int = 12, stride: int = 1):
        """
        Args:
            data_path (str): Path to the directory containing the processed .pt files.
            mode (str): The dataset split to load, one of ['train', 'val', 'test'].
            L_in (int): Input sequence length (window size).
            L_out (int): Output sequence length (prediction horizon).
            stride (int): Stride for sampling windows. Default 1 means no skipping.
        """
        super().__init__()
        assert mode in ['train', 'val', 'test'], "Mode must be one of 'train', 'val', or 'test'"
        
        self.L_in = L_in
        self.L_out = L_out
        self.stride = stride
        
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
        
        # Calculate number of samples with stride support
        max_start_idx = len(self.X) - self.L_in - self.L_out + 1
        if max_start_idx <= 0:
            self.num_samples = 0
            self.sample_indices = []
        else:
            # Generate sample indices with stride
            self.sample_indices = list(range(0, max_start_idx, self.stride))
            self.num_samples = len(self.sample_indices)
        
        if self.num_samples <= 0:
            logging.warning(f"Insufficient data for windowing: Total length={len(self.X)}, L_in={self.L_in}, L_out={self.L_out}, stride={self.stride}. Resulting samples: {self.num_samples}")
            
        logging.info(f"Dataset '{mode}' initialized. Window config: L_in={self.L_in}, L_out={self.L_out}, stride={self.stride}")
        logging.info(f"Total available samples: {self.num_samples}")

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single sample from the dataset using stride-based indexing.
        
        Args:
            idx (int): The sample index (not the raw data index).
            
        Returns:
            dict: A dictionary containing the input window, target window, and time features as tensors.
        """
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} is out of bounds for a dataset of size {self.num_samples}")

        # Get the actual data index using stride
        actual_idx = self.sample_indices[idx]
        
        # Determine the slice for the input window
        x_start = actual_idx
        x_end = actual_idx + self.L_in
        
        # Slice the data tensors directly
        x_window = self.X[x_start:x_end]
        time_features_window = self.time_features[x_start:x_end]
        
        # The target `Y` was pre-calculated and aligned. Y[t] contains future values for X[t].
        # Our input window ends at time `t = actual_idx + L_in - 1`.
        # The corresponding target is at this same index.
        target_y = self.Y[actual_idx + self.L_in - 1]

        # 直接返回，不再有任何scaler操作
        return {
            'x': x_window,      # 已是tensor且已标准化
            'y': target_y,      # 已是tensor且已标准化
            'x_time_features': time_features_window  # 已是tensor
        } 
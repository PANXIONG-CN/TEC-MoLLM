import torch
import numpy as np
import logging
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SlidingWindowSamplerDataset(Dataset):
    """
    A PyTorch Dataset that implements sliding window sampling for time-series data.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, time_features: np.ndarray, 
                 L_in: int = 336, L_out: int = 12):
        """
        Args:
            X (np.ndarray): Input feature tensor of shape (N, H, W, C).
            Y (np.ndarray): Target tensor of shape (N, H, W, L_out).
            time_features (np.ndarray): Time-related features of shape (N, num_time_features).
            L_in (int): Input sequence length (window size).
            L_out (int): Output sequence length (prediction horizon).
        """
        super().__init__()
        self.X = X
        self.Y = Y
        self.time_features = time_features
        self.L_in = L_in
        self.L_out = L_out
        
        # The number of samples is determined by how many full windows can be created.
        # For X and time_features: we need L_in consecutive samples for input
        # For Y: we already have the targets prepared in the right format
        self.num_samples = max(0, len(X) - self.L_in + 1)
        
        # Validate that we have enough data
        if self.num_samples <= 0:
            logging.warning(f"Insufficient data: X length={len(X)}, L_in={L_in}, resulting in {self.num_samples} samples")
            self.num_samples = 0
        
        # Ensure Y and time_features have sufficient length
        if len(Y) < self.num_samples:
            logging.warning(f"Y tensor too short: {len(Y)} < {self.num_samples}, truncating samples")
            self.num_samples = len(Y)
            
        if len(time_features) < len(X):
            logging.warning(f"Time features too short: {len(time_features)} < {len(X)}")
            self.num_samples = 0
        
        logging.info(f"Dataset initialized. Available data: X={len(X)}, Y={len(Y)}, time_features={len(time_features)}")
        logging.info(f"Window config: L_in={L_in}, L_out={L_out}")
        logging.info(f"Final dataset samples: {self.num_samples}")

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single sample from the dataset.
        
        Args:
            idx (int): The index of the sample to retrieve.
            
        Returns:
            dict: A dictionary containing the input window, target window, and time features.
        """
        if idx >= self.num_samples or self.num_samples <= 0:
            raise IndexError(f"Index {idx} out of bounds for dataset with {self.num_samples} samples")

        # Determine the slice for the input window
        x_start = idx
        x_end = idx + self.L_in
        
        # Slice the data arrays
        x_window = self.X[x_start:x_end]  # Shape: (L_in, H, W, C)
        time_features_window = self.time_features[x_start:x_end]  # Shape: (L_in, 2)
        
        # For the target, Y[idx] already contains the correct 12-step ahead prediction
        # that corresponds to the input window ending at idx + L_in - 1
        target_y = self.Y[idx]  # Shape: (H, W, L_out)

        return {
            'x': torch.tensor(x_window, dtype=torch.float32),
            'y': torch.tensor(target_y, dtype=torch.float32),
            'x_time_features': torch.tensor(time_features_window, dtype=torch.float32)
        } 
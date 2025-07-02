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
        self.num_samples = len(X) - self.L_in - self.L_out + 1
        
        logging.info(f"Dataset initialized. Number of samples: {self.num_samples}")

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
        if idx >= self.num_samples:
            raise IndexError("Index out of bounds")

        # Determine the slice for the input window
        x_start = idx
        x_end = idx + self.L_in
        
        # The target window starts right after the input window
        y_start = x_end
        y_end = y_start + self.L_out
        
        # Slice the data arrays
        x_window = self.X[x_start:x_end]
        y_window = self.Y[y_start:y_end] # This is a window of future targets
        time_features_window = self.time_features[x_start:x_end]
        
        # For the target, we need a single target that corresponds to the input window.
        # This will be the Y value at the START of the prediction window.
        # The shape of Y is (N, H, W, L_out), so Y[t] already contains the 12 future steps.
        # Therefore, we take the target corresponding to the END of the input window.
        target_y = self.Y[x_end - 1] 

        return {
            'x': torch.tensor(x_window, dtype=torch.float32),
            'y': torch.tensor(target_y, dtype=torch.float32),
            'x_time_features': torch.tensor(time_features_window, dtype=torch.float32)
        } 
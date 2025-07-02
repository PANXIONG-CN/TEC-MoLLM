import torch
import pytest
import numpy as np
from src.data.dataset import SlidingWindowSamplerDataset

@pytest.fixture
def mock_data():
    """Creates mock data for testing the dataset."""
    N, H, W, C_in, L_out, C_time = 500, 41, 71, 6, 12, 2
    X = np.arange(N * H * W * C_in).reshape(N, H, W, C_in)
    Y = np.arange(N * H * W * L_out).reshape(N, H, W, L_out)
    time_features = np.arange(N * C_time).reshape(N, C_time)
    return {
        "X": X,
        "Y": Y,
        "time_features": time_features,
        "L_in": 336,
        "L_out": 12
    }

def test_dataset_len(mock_data):
    """Tests the __len__ method."""
    dataset = SlidingWindowSamplerDataset(**mock_data)
    expected_len = 500 - 336 - 12 + 1
    assert len(dataset) == expected_len

def test_dataset_getitem_shapes(mock_data):
    """Tests the shapes of the tensors returned by __getitem__."""
    dataset = SlidingWindowSamplerDataset(**mock_data)
    sample = dataset[0]
    
    assert isinstance(sample, dict)
    assert 'x' in sample and 'y' in sample and 'x_time_features' in sample
    
    assert sample['x'].shape == (mock_data['L_in'], 41, 71, 6)
    assert sample['y'].shape == (41, 71, mock_data['L_out'])
    assert sample['x_time_features'].shape == (mock_data['L_in'], 2)
    
    assert sample['x'].dtype == torch.float32
    assert sample['y'].dtype == torch.float32
    assert sample['x_time_features'].dtype == torch.float32

def test_dataset_getitem_content(mock_data):
    """Tests the content and window sliding logic of __getitem__."""
    dataset = SlidingWindowSamplerDataset(**mock_data)
    
    # Check sample at index 10
    idx = 10
    sample = dataset[idx]
    
    # The input window should start at index 10 of the original X
    expected_x_start = mock_data["X"][idx]
    assert np.array_equal(sample['x'][0].numpy(), expected_x_start)
    
    # The target y should be the Y data at index (idx + L_in - 1)
    expected_y = mock_data["Y"][idx + mock_data["L_in"] - 1]
    assert np.array_equal(sample['y'].numpy(), expected_y)

def test_dataset_index_out_of_bounds(mock_data):
    """Tests that accessing an invalid index raises an IndexError."""
    dataset = SlidingWindowSamplerDataset(**mock_data)
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)] 
import torch
import pytest
from src.model.tec_mollm import TEC_MoLLM
import numpy as np

@pytest.fixture
def model_config():
    """Provides a default model configuration for testing."""
    return {
        "num_nodes": 2911,
        "d_emb": 16,
        "spatial_in_channels_base": 6, # Base channels before embedding
        "spatial_out_channels": 32,
        "spatial_heads": 2,
        "temporal_in_channels": 64, # This will be spatial_out * heads
        "temporal_channel_list": [64, 128],
        "temporal_strides": [2, 2],
        "temporal_seq_len": 84, # Derived from data loader
        "patch_len": 4,
        "d_llm": 768,
        "llm_layers": 3,
        "prediction_horizon": 12
    }

@pytest.fixture
def dummy_model_input():
    """Provides dummy input tensors for the model."""
    B, L_in, N, C_in = 2, 336, 2911, 6
    x = torch.randn(B, L_in, N, C_in)
    time_features = torch.randint(0, 24, (B, L_in, N, 2))
    graph_data = torch.load('data/processed/graph_A.pt')
    edge_index = graph_data['edge_index']
    return x, time_features, edge_index

def test_tec_mollm_forward_pass(model_config, dummy_model_input):
    """
    Tests the full forward pass of the TEC_MoLLM model.
    """
    # No longer need to adjust spatial_in_channels here, it's handled in the model's __init__
    model_config['temporal_seq_len'] = (dummy_model_input[0].shape[1] // 
                                        np.prod(model_config['temporal_strides']))
    
    model = TEC_MoLLM(model_config)
    model.eval()
    
    x, time_features, edge_index = dummy_model_input
    
    with torch.no_grad():
        output = model(x, time_features, edge_index)
        
    B, _, N, _ = x.shape
    L_out = model_config['prediction_horizon']
    expected_shape = (B, L_out, N, 1)
    
    assert output.shape == expected_shape, f"Final output shape mismatch: expected {expected_shape}, got {output.shape}" 
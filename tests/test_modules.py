import torch
import pytest
import numpy as np
from src.model.modules import TemporalEncoder, LLMBackbone, SpatioTemporalEmbedding, PredictionHead, SpatialEncoder
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from peft import PeftModel
from peft.tuners.lora import Linear as LoraLinear
import logging

# Fixture to provide parameters for the encoder
@pytest.fixture
def encoder_params():
    return {
        "in_channels": 7,
        "channel_list": [16, 32, 64, 128],
        "strides": [2, 2, 1, 1],
        "patch_len": 4,
        "d_llm": 768
    }

# Fixture to provide a dummy input tensor
@pytest.fixture
def dummy_input(encoder_params):
    batch_size = 4
    seq_length = 336
    return torch.randn(batch_size, seq_length, encoder_params["in_channels"])

def test_temporal_encoder_output_shape(encoder_params, dummy_input):
    """
    Tests if the TemporalEncoder produces an output with the correct shape.
    """
    encoder = TemporalEncoder(**encoder_params)
    encoder.eval() # Set to evaluation mode for testing
    
    with torch.no_grad(): # No need to track gradients for shape test
        output = encoder(dummy_input)

    batch_size = dummy_input.shape[0]
    seq_length = dummy_input.shape[1]
    
    total_stride = np.prod(encoder_params["strides"])
    num_patches = (seq_length // total_stride) // encoder_params["patch_len"]
    
    expected_shape = (batch_size, num_patches, encoder_params["d_llm"])
    
    assert output.shape == expected_shape, f"Output shape mismatch: expected {expected_shape}, got {output.shape}"

def test_temporal_encoder_backward_pass(encoder_params, dummy_input):
    """
    Tests if the backward pass runs successfully without errors.
    """
    encoder = TemporalEncoder(**encoder_params)
    encoder.train() # Set to training mode for backward pass

    output = encoder(dummy_input)
    
    # The actual loss value is not important, just that it's a scalar.
    loss = output.sum()
    
    try:
        loss.backward()
    except Exception as e:
        pytest.fail(f"Backward pass failed with an exception: {e}")

    # Check if gradients are present in some parameters
    assert encoder.conv_embedder.embedder[0].final_conv.weight.grad is not None, "Gradients are missing after backward pass."
    assert encoder.patcher.projection.weight.grad is not None, "Gradients are missing after backward pass."

def test_llm_backbone_initialization_and_truncation():
    """
    Tests if the LLMBackbone correctly loads and truncates the GPT-2 model.
    """
    num_layers = 3
    backbone = LLMBackbone(num_layers_to_keep=num_layers)
    
    # Test 9.1: Check if model is loaded
    assert isinstance(backbone.model, GPT2Model), "Model is not a GPT2Model instance."
    
    # Test 9.2: Check if model is truncated
    assert len(backbone.model.h) == num_layers, f"Model not truncated correctly. Expected {num_layers} layers, got {len(backbone.model.h)}."

def test_llm_backbone_lora_application():
    """
    Tests if LoRA is correctly applied to the LLMBackbone's model.
    """
    backbone = LLMBackbone()
    
    # Test 9.3: Check if model is wrapped with PeftModel
    assert isinstance(backbone.model, PeftModel), "Model is not a PeftModel instance."
    
    # Check if the target module has been replaced with a LoRA layer
    # We inspect the first attention block in the first transformer layer
    first_attn_block = backbone.model.h[0].attn
    assert isinstance(first_attn_block.c_attn, LoraLinear), "c_attn module was not replaced by a LoRA Linear layer."

def test_llm_backbone_parameter_freezing():
    """
    Tests the selective parameter freezing logic.
    """
    backbone = LLMBackbone()
    
    # Test 9.4: Verify which parameters are trainable
    trainable_params = []
    for name, param in backbone.model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
    
    # Check that there are some trainable parameters
    assert len(trainable_params) > 0, "No parameters are trainable."
    
    # Check that all trainable parameters are of the allowed types
    for name in trainable_params:
        assert 'lora_' in name or 'ln_' in name or 'wpe' in name, f"Parameter '{name}' should be frozen but is trainable."

def test_llm_backbone_forward_pass():
    """
    Tests the forward pass of the LLMBackbone module.
    """
    backbone = LLMBackbone()
    backbone.eval() # Set to evaluation mode

    batch_size = 2
    seq_length = 21 # Number of patches from TemporalEncoder
    hidden_size = 768 # GPT-2's hidden size
    
    # Create dummy inputs
    dummy_embeds = torch.randn(batch_size, seq_length, hidden_size)
    dummy_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    
    with torch.no_grad():
        output = backbone(inputs_embeds=dummy_embeds, attention_mask=dummy_mask)
        
    # Test 9.5: Check output shape
    expected_shape = (batch_size, seq_length, hidden_size)
    assert output.shape == expected_shape, f"Output shape mismatch: expected {expected_shape}, got {output.shape}" 

def test_spatio_temporal_embedding():
    """
    Tests the SpatioTemporalEmbedding module.
    """
    d_emb = 64
    num_nodes = 2911
    batch_size = 4
    seq_len = 10
    
    # The input feature dimension must be >= d_emb
    in_channels = 128 
    
    # Subtask 6.5: Test the full module
    embedding_layer = SpatioTemporalEmbedding(d_emb=d_emb, num_nodes=num_nodes)
    
    # Create dummy data
    x = torch.randn(batch_size, seq_len, num_nodes, in_channels)
    # time_features: hour (0-23), day of year (0-365)
    tod = torch.randint(0, 24, (batch_size, seq_len, num_nodes, 1))
    doy = torch.randint(0, 366, (batch_size, seq_len, num_nodes, 1))
    time_features = torch.cat([tod, doy], dim=-1)
    
    x_original = x.clone()
    
    # Forward pass
    output = embedding_layer(x, time_features)
    
    # Check output shape
    assert output.shape == x_original.shape, "Output shape does not match input shape."
    
    # Check that the embeddings were added (i.e., output is different from input)
    assert not torch.allclose(output, x_original), "Output is identical to input; embeddings were not added."
    
    # Check that the part of x beyond d_emb is unchanged
    assert torch.allclose(output[..., d_emb:], x_original[..., d_emb:]), "Features beyond d_emb were altered."
    
    logging.info("Test PASSED: SpatioTemporalEmbedding module works as expected.") 

def test_prediction_head():
    """
    Tests the PredictionHead module.
    """
    batch_size = 10
    seq_len = 21
    hidden_size = 768
    output_dim = 12
    
    input_dim = seq_len * hidden_size
    
    head = PredictionHead(input_dim=input_dim, output_dim=output_dim)
    head.eval()
    
    dummy_input = torch.randn(batch_size, seq_len, hidden_size)
    
    with torch.no_grad():
        output = head(dummy_input)
        
    expected_shape = (batch_size, output_dim)
    assert output.shape == expected_shape, f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
    logging.info("Test PASSED: PredictionHead module works as expected.") 

def test_spatial_encoder():
    """
    Tests the SpatialEncoder module.
    """
    in_channels = 64
    out_channels = 128
    heads = 2
    num_nodes = 2911
    batch_dim = 10 # Represents B * L_in

    # Load the pre-computed graph
    graph_data = torch.load('data/processed/graph_A.pt')
    edge_index = graph_data['edge_index']

    encoder = SpatialEncoder(in_channels=in_channels, out_channels=out_channels, heads=heads)
    encoder.eval()
    
    # Create dummy input
    dummy_x = torch.randn(batch_dim, num_nodes, in_channels)
    
    with torch.no_grad():
        output = encoder(dummy_x, edge_index)
        
    expected_out_channels = out_channels * heads
    expected_shape = (batch_dim, num_nodes, expected_out_channels)
    
    assert output.shape == expected_shape, f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
    logging.info("Test PASSED: SpatialEncoder module works as expected.") 
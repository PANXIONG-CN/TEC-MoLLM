import torch
import torch.nn as nn
import logging
from .modules import (
    SpatioTemporalEmbedding, 
    SpatialEncoder, 
    TemporalEncoder, 
    LLMBackbone, 
    PredictionHead
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TEC_MoLLM(nn.Module):
    """
    The main TEC-MoLLM model that assembles all sub-modules.
    """
    def __init__(self, model_config: dict):
        super().__init__()
        
        self.num_nodes = model_config['num_nodes']
        
        # --- START MODIFICATION 1 ---
        self.spatio_temporal_embedding = SpatioTemporalEmbedding(
            d_emb=model_config['d_emb'],
            num_nodes=self.num_nodes,
            num_years=model_config.get('num_years', 13) # 从配置中获取年份数
        )
        # --- END MODIFICATION 1 ---
        # The input to the spatial encoder is now the original channels + embedding dim
        spatial_in_channels = model_config['spatial_in_channels_base'] + model_config['d_emb']
        self.spatial_encoder = SpatialEncoder(
            in_channels=spatial_in_channels,
            out_channels=model_config['spatial_out_channels'],
            heads=model_config['spatial_heads']
        )
        # The input to the temporal encoder is the output of the spatial encoder
        temporal_in_channels = model_config['spatial_out_channels'] * model_config['spatial_heads']
        self.temporal_encoder = TemporalEncoder(
            in_channels=temporal_in_channels,
            channel_list=model_config['temporal_channel_list'],
            strides=model_config['temporal_strides'],
            patch_len=model_config['patch_len'],
            d_llm=model_config['d_llm']
        )
        self.llm_backbone = LLMBackbone(
            num_layers_to_keep=model_config['llm_layers']
        )
        # Calculate the actual sequence length after convolutions: 24 -> 12 -> 6 (stride 2, stride 2)
        conv_output_len = model_config['temporal_seq_len'] // (model_config['temporal_strides'][0] * model_config['temporal_strides'][1])
        num_patches = conv_output_len // model_config['patch_len']
        self.prediction_head = PredictionHead(
            input_dim=model_config['d_llm'] * num_patches,
            output_dim=model_config['prediction_horizon']
        )
        logging.info("TEC_MoLLM model initialized with all sub-modules.")

    def forward(self, x: torch.Tensor, time_features: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Defines the complete forward pass from input data to prediction.
        
        Args:
            x (torch.Tensor): Input features of shape (B, L_in, N, C_in).
            time_features (torch.Tensor): Time features for embedding, shape (B, L, N, 4).
            edge_index (torch.Tensor): Graph connectivity.
            edge_weight (torch.Tensor): Graph edge weights.
            
        Returns:
            torch.Tensor: Final prediction of shape (B, L_out, N, 1).
        """
        B, L_in, N, C_in = x.shape
        
        # 1. SpatioTemporalEmbedding
        x = self.spatio_temporal_embedding(x, time_features)
        
        # 2. SpatialEncoder
        # Reshape for SpatialEncoder: (B, L, N, C) -> (B*L, N, C)
        C_in_with_emb = x.shape[-1]
        x_spatial = x.reshape(-1, N, C_in_with_emb)
        
        # --- START MODIFICATION 2 ---
        x_spatial = self.spatial_encoder(x_spatial, edge_index, edge_weight)
        # --- END MODIFICATION 2 ---
        
        # 3. TemporalEncoder
        # Reshape for TemporalEncoder: (B*L, N, C_out_spatial) -> (B*N, L, C_out_spatial)
        C_out_spatial = x_spatial.shape[-1]
        x_temporal = x_spatial.reshape(B, L_in, N, C_out_spatial).permute(0, 2, 1, 3)
        x_temporal = x_temporal.reshape(-1, L_in, C_out_spatial)
        x_temporal = self.temporal_encoder(x_temporal) # Output: (B*N, num_patches, d_llm)
        
        # 4. LLMBackbone
        # The LLM doesn't need an attention mask if we pass embeddings directly
        attention_mask = torch.ones(x_temporal.shape[:-1], device=x.device, dtype=torch.long)
        x_llm = self.llm_backbone(inputs_embeds=x_temporal, attention_mask=attention_mask)
        
        # --- START MODIFICATION 3 ---
        x_llm = torch.nn.functional.dropout(x_llm, p=0.1, training=self.training)
        # --- END MODIFICATION 3 ---
        
        # 5. PredictionHead
        predictions = self.prediction_head(x_llm) # Output: (B*N, L_out)
        
        # 6. Final Reshape
        L_out = predictions.shape[-1]
        final_output = predictions.view(B, N, L_out).permute(0, 2, 1).unsqueeze(-1)
        
        return final_output 
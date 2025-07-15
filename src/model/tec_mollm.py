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
        
        # 1. SpatioTemporalEmbedding - (B, L, N, C_in_emb)
        x = self.spatio_temporal_embedding(x, time_features)
        C_in_with_emb = x.shape[-1]

        # --- START MODIFICATION 1.1 (CRITICAL ARCHITECTURE FIX) ---
        # REASON: The original reshape broke graph connectivity for batched processing.
        #         This new approach processes the spatial dimension correctly for each time step.

        # 2. Reshape for GNN: 将时间步和批次合并，但保持空间维度独立
        # (B, L, N, C) -> (L, B, N, C) -> (L*B, N, C)
        x_for_gnn = x.permute(1, 0, 2, 3).reshape(-1, N, C_in_with_emb)

        # 3. SpatialEncoder (GATv2) with Residual Connection - 现在它处理的是一个批量的图
        # GATv2Conv内部会自动处理批处理图的邻接关系，只要输入是(Batch_of_graphs * N, C)
        # 我们的输入 (L*B, N, C) 正好符合这个期望，每个时间步的图被正确处理
        x_spatial_processed = self.spatial_encoder(x_for_gnn, edge_index, edge_weight)
        
        # --- START MODIFICATION: Add Residual Connection ---
        # 残差连接：由于已确保GNN输入输出维度一致（22维），可以直接相加
        # 这能保证原始的时空特征能够"绕过"GNN直接流向后续模块，防止信息损耗
        x_spatial = x_for_gnn + x_spatial_processed
        # --- END MODIFICATION ---

        # 4. Reshape back for Temporal Processing
        # (L*B, N, C_spatial) -> (L, B, N, C_spatial) -> (B, N, L, C_spatial)
        C_spatial = x_spatial.shape[-1]
        x_reshaped = x_spatial.view(L_in, B, N, C_spatial).permute(1, 2, 0, 3)
        
        # --- END MODIFICATION 1.1 ---

        # 5. Reshape for TemporalEncoder
        # (B, N, L, C_spatial) -> (B*N, L, C_spatial)
        x_temporal = x_reshaped.reshape(-1, L_in, C_spatial)
        x_temporal = self.temporal_encoder(x_temporal) # Output: (B*N, num_patches, d_llm)
        
        # 6. LLMBackbone
        # The LLM doesn't need an attention mask if we pass embeddings directly
        attention_mask = torch.ones(x_temporal.shape[:-1], device=x.device, dtype=torch.long)
        x_llm = self.llm_backbone(inputs_embeds=x_temporal, attention_mask=attention_mask)
        
        # --- START MODIFICATION 3 ---
        x_llm = torch.nn.functional.dropout(x_llm, p=0.1, training=self.training)
        # --- END MODIFICATION 3 ---
        
        # 7. PredictionHead
        predictions = self.prediction_head(x_llm) # Output: (B*N, L_out)
        
        # 8. Final Reshape
        L_out = predictions.shape[-1]
        final_output = predictions.view(B, N, L_out).permute(0, 2, 1).unsqueeze(-1)
        
        return final_output
        
    # **这个修改是本次修复的重中之重。** 它修正了信息流，让GNN能在每个时间步上正确地进行空间消息传递。
import torch
import torch.nn as nn
import logging
import numpy as np
from einops import rearrange
from transformers import AutoModel
from peft import get_peft_model, LoraConfig
from torch_geometric.nn import GATv2Conv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Multi_Scale_Conv_Block(nn.Module):
    """
    A building block for multi-scale 1D convolutions.
    It applies multiple parallel convolutions with different kernel sizes,
    concatenates their outputs, and then passes them through a final convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int, kernel_sizes: list = [3, 5, 7]):
        super().__init__()
        
        # --- START MODIFICATION: Replace BatchNorm with GroupNorm ---
        # REASON: GroupNorm is batch-size independent and more stable for time-series
        #         and distributed training scenarios. GroupNorm with 1 group is equivalent to InstanceNorm.
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2),
                nn.GroupNorm(1, out_channels),  # GroupNorm with 1 group = InstanceNorm
                nn.GELU()
            ) for k in kernel_sizes
        ])
        # --- END MODIFICATION ---
        
        # The final convolution takes the concatenated output of all parallel convolutions
        # and applies the specified stride.
        self.final_conv = nn.Conv1d(
            in_channels=out_channels * len(kernel_sizes), 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, L_in)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, L_out)
        """
        # Apply parallel convolutions
        conv_outputs = [conv(x) for conv in self.convs]
        
        # Concatenate along the channel dimension
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # Apply final convolution
        output = self.final_conv(concatenated)
        
        return output

class MultiScaleConvEmbedder(nn.Module):
    """
    A multi-scale convolutional embedder that stacks multiple Multi_Scale_Conv_Block
    to downsample a time series.
    """
    def __init__(self, in_channels: int, channel_list: list, strides: list):
        super().__init__()
        
        assert len(channel_list) == len(strides), "Channel list and strides list must have the same length."
        
        layers = []
        current_channels = in_channels
        for i, (out_channels, stride) in enumerate(zip(channel_list, strides)):
            layers.append(Multi_Scale_Conv_Block(current_channels, out_channels, stride))
            current_channels = out_channels
        
        self.embedder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, L_in)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, L_out)
        """
        return self.embedder(x)

class LatentPatchingProjection(nn.Module):
    """
    Reshapes the latent sequence from the convolutional embedder into patches
    and projects them to the LLM's hidden dimension.
    """
    def __init__(self, latent_dim: int, patch_len: int, d_llm: int):
        super().__init__()
        self.patch_len = patch_len
        self.projection = nn.Linear(patch_len * latent_dim, d_llm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D_latent)
                              Note: The input from conv embedder is (B, D_latent, L),
                              so it must be permuted before calling this module.
        
        Returns:
            torch.Tensor: Output tensor of shape (B, num_patches, d_llm)
        """
        # b: batch size, p: number of patches, l: patch length, d: latent dimension
        # The rearrange pattern 'b (p l) d -> b p (l d)' does the following:
        # 1. Divides the sequence of length L into p patches of length l.
        # 2. For each patch, it flattens the patch length and latent dimensions together.
        x = rearrange(x, 'b (p l) d -> b p (l d)', l=self.patch_len)
        
        # Project the flattened patch dimension to the LLM's dimension
        x = self.projection(x)
        
        return x

class TemporalEncoder(nn.Module):
    """
    The complete TemporalEncoder module which combines multi-scale convolutions
    and latent patching to prepare time-series data for an LLM.
    """
    def __init__(self, in_channels: int, channel_list: list, strides: list, patch_len: int, d_llm: int):
        super().__init__()
        self.conv_embedder = MultiScaleConvEmbedder(in_channels, channel_list, strides)
        
        # The latent dimension is the output channel count of the last conv block
        latent_dim = channel_list[-1]
        self.patcher = LatentPatchingProjection(latent_dim, patch_len, d_llm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, L_in, C_in)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, num_patches, d_llm)
        """
        # Permute from (B, L, C) to (B, C, L) for Conv1d
        x = x.permute(0, 2, 1)
        
        # Pass through the convolutional embedder
        x = self.conv_embedder(x)
        
        # Permute back from (B, C_out, L_out) to (B, L_out, C_out) for patching
        x = x.permute(0, 2, 1)
        
        # Pass through the patching and projection layer
        x = self.patcher(x)
        
        return x

class LLMBackbone(nn.Module):
    """
    An LLM backbone using a pre-trained GPT-2, truncated and adapted with LoRA.
    """
    def __init__(self, num_layers_to_keep: int = 3):
        super().__init__()
        
        # Subtask 9.1: Load pre-trained GPT-2 model
        logging.info("Loading pre-trained GPT-2 model...")
        self.model = AutoModel.from_pretrained('gpt2')
        logging.info("GPT-2 model loaded.")

        # Subtask 9.2: Truncate the model
        logging.info(f"Truncating model to keep only the first {num_layers_to_keep} layers.")
        self.model.h = self.model.h[:num_layers_to_keep]
        logging.info(f"Model truncated. Number of layers: {len(self.model.h)}")

        # Subtask 9.3: Define and apply LoRA config
        logging.info("Defining LoRA configuration...")
        # --- START MODIFICATION 1.4.1 ---
        # REASON: 增加 LoRA rank 提升模型容量以学习更复杂的模式。
        lora_config = LoraConfig(
            r=32,             # OLD: 16
            lora_alpha=64,    # OLD: 32
            target_modules=["c_attn"], # For GPT-2, attention weights are in 'c_attn'
            lora_dropout=0.1,
            bias="none"
        )
        # --- END MODIFICATION 1.4.1 ---
        logging.info("Applying LoRA to the model...")
        self.model = get_peft_model(self.model, lora_config)
        logging.info("LoRA applied successfully.")
        
        # Subtask 9.4: Freeze parameters selectively
        self._freeze_parameters()
        
        logging.info("Trainable parameters after freezing:")
        self.model.print_trainable_parameters()

    def _freeze_parameters(self):
        logging.info("Freezing all parameters initially...")
        for param in self.model.parameters():
            param.requires_grad = False
        
        logging.info("Unfreezing LoRA, LayerNorm, and positional encoding parameters...")
        for name, param in self.model.named_parameters():
            if 'lora_' in name or 'ln_' in name or 'wpe' in name:
                param.requires_grad = True

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # This will be fully implemented in a later subtask
        # Note: We now use inputs_embeds instead of input_ids
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs.last_hidden_state

class SpatioTemporalEmbedding(nn.Module):
    """
    Creates learnable embeddings for nodes and multiple time features.
    """
    def __init__(self, d_emb: int, num_nodes: int = 2911, num_years: int = 13):
        super().__init__()
        self.d_emb = d_emb
        
        self.node_embedding = nn.Embedding(num_embeddings=num_nodes, embedding_dim=d_emb)
        self.tod_embedding = nn.Embedding(num_embeddings=12, embedding_dim=d_emb)
        self.doy_embedding = nn.Embedding(num_embeddings=366, embedding_dim=d_emb)
        
        # --- START MODIFICATION ---
        self.year_embedding = nn.Embedding(num_embeddings=num_years, embedding_dim=d_emb)
        self.season_embedding = nn.Embedding(num_embeddings=4, embedding_dim=d_emb)
        # --- END MODIFICATION ---
        
        logging.info("SpatioTemporalEmbedding module initialized with year and season embeddings.")

    def forward(self, x: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        """
        Adds spatio-temporal embeddings to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, N, C_in).
            time_features (torch.Tensor): Time features of shape (B, L, N, 4) 
                                          containing (hour_of_day, day_of_year, year_index, season_index).
                                          
        Returns:
            torch.Tensor: Output tensor with embeddings added, shape (B, L, N, C_in + d_emb).
        """
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Node embeddings: one for each node, repeat across batch and time
        node_ids = torch.arange(num_nodes, device=x.device)
        node_emb = self.node_embedding(node_ids).view(1, 1, num_nodes, self.d_emb)
        
        # --- START MODIFICATION ---
        # time_features is now (B, L, N, 4)
        tod_indices = time_features[..., 0].long()
        doy_indices = time_features[..., 1].long()
        year_indices = time_features[..., 2].long()
        season_indices = time_features[..., 3].long()

        tod_emb = self.tod_embedding(tod_indices)
        doy_emb = self.doy_embedding(doy_indices)
        year_emb = self.year_embedding(year_indices)
        season_emb = self.season_embedding(season_indices)
        
        temporal_emb = tod_emb + doy_emb + year_emb + season_emb
        combined_emb = node_emb + temporal_emb
        # --- END MODIFICATION ---
        
        output = torch.cat([x, combined_emb], dim=-1)
        
        return output

class PredictionHead(nn.Module):
    """
    Maps features from the LLM backbone to the final prediction sequence.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim_ratio: int = 4, dropout_rate: float = 0.1):
        """
        Args:
            input_dim (int): The flattened input dimension from the LLM (e.g., 21 * 768).
            output_dim (int): The output prediction dimension (e.g., 12).
            hidden_dim_ratio (int): Ratio to reduce hidden dimension.
            dropout_rate (float): Dropout rate for regularization.
        """
        super().__init__()
        # --- START MODIFICATION 1.3.1 (Architecture Enhancement) ---
        # REASON: A two-layer MLP provides more expressive power than a single
        #         linear layer, allowing the model to learn more complex
        #         mappings from latent features to the final predictions.
        hidden_dim = input_dim // hidden_dim_ratio
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        # --- END MODIFICATION 1.3.1 ---
        logging.info(f"PredictionHead initialized with MLP structure.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input tensor to produce the final output.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, seq_len, hidden_size).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, output_dim).
        """
        # Flatten the sequence and hidden dimensions
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        # --- START MODIFICATION 1.3.2 ---
        output = self.mlp(x)
        # --- END MODIFICATION 1.3.2 ---
        
        return output

class SpatialEncoder(nn.Module):
    """
    Captures spatial dependencies using a GATv2 layer.
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int = 2, dropout: float = 0.1):
        """
        Args:
            in_channels (int): Number of input features for each node.
            out_channels (int): Number of output features for each node.
            heads (int): Number of multi-head attentions.
            dropout (float): Dropout rate.
        """
        super().__init__()
        # Subtask 7.2: Instantiate GATv2Conv Layer
        self.gat_conv = GATv2Conv(
            in_channels, 
            out_channels, 
            heads=heads, 
            dropout=dropout, 
            concat=True, # Concatenates the multi-head attentions
            add_self_loops=True # Recommended for GAT
        )
        self.output_channels = out_channels * heads
        logging.info(f"SpatialEncoder initialized with out_channels={self.output_channels}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Processes the batched graph data.
        
        Args:
            x (torch.Tensor): Batched node features of shape (B * L, N, C_in).
            edge_index (torch.Tensor): Graph connectivity in COO format.
            edge_weight (torch.Tensor, optional): 兼容性参数，被忽略。
            
        Returns:
            torch.Tensor: Updated node features.
        """
        batch_size, num_nodes, in_channels = x.shape
        x_reshaped = x.reshape(-1, in_channels)
        
        # GATv2Conv不支持edge_weight参数，GAT通过注意力机制自动学习边权重
        gat_output = self.gat_conv(x_reshaped, edge_index)
        
        output = gat_output.view(batch_size, num_nodes, self.output_channels)
        return output 
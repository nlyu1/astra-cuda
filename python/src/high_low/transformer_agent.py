import torch
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)  # Memory-efficient attention
torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import sys 
sys.path.append('../')
from model_components import layer_init, OptimizedResidualBlock
from typing import Dict
import warnings

class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings for transformer."""
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.register_buffer('position_ids', torch.arange(max_len).expand((1, -1)))

    def forward(self, x):
        # x: [B, T, D]
        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.pos_embedding(position_ids)
        return x + position_embeddings


class HighLowTransformerModel(nn.Module):
    """
    Transformer-based model for High-Low Trading that replaces GRU with parallel attention.
    
    Key parameters:
    - n_hidden: Embedding dimension (replaces hidden_size)
    - n_head: Number of attention heads
    - n_layer: Number of transformer blocks
    - dropout: Dropout rate for regularization
    - pre_encoder_blocks: Number of residual blocks before transformer
    - post_decoder_blocks: Number of residual blocks after transformer
    """
    def __init__(self, args, env):
        super().__init__()
        self.B, self.F = env.observation_shape()
        self.T = args.steps_per_player 
        self.P = args.players
        self.M = args.max_contract_value
        self.S = args.max_contracts_per_trade
        self.device = torch.device(f'cuda:{args.device_id}')
        
        # Transformer hyperparameters
        self.n_hidden = args.n_hidden
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.n_layer = args.n_layer
        
        # Pre/post processing blocks
        pre_blocks = args.pre_encoder_blocks
        post_blocks = args.post_decoder_blocks
        
        # Input encoder
        assert pre_blocks >= 2, "Pre-encoder blocks must be at least 2"
        self.encoder = nn.Sequential(
            OptimizedResidualBlock(self.F, self.n_hidden),
            *[OptimizedResidualBlock(self.n_hidden, self.n_hidden) for _ in range(pre_blocks - 2)],
            OptimizedResidualBlock(self.n_hidden, self.n_embd))
        self.pos_encoding = LearnedPositionalEncoding(self.n_embd, max_len=self.T)
        
        # Transformer core using memory-efficient configurations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.n_embd,
            nhead=self.n_head,
            dim_feedforward=self.n_embd * 2,
            dropout=0,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        
        # Configure encoder for memory efficiency
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layer,
            enable_nested_tensor=False  # Disable for SDPA efficiency
        )
        
        # Output decoder
        assert post_blocks >= 1, "Post-decoder blocks must be at least 1"
        self.decoder = nn.Sequential(
            OptimizedResidualBlock(self.n_embd, self.n_hidden),
            *[OptimizedResidualBlock(self.n_hidden, self.n_hidden) for _ in range(post_blocks - 1)])
        
        # Action heads
        self.actors = nn.ModuleDict({
            "bid_price": layer_init(nn.Linear(self.n_hidden, self.M)),
            "ask_price": layer_init(nn.Linear(self.n_hidden, self.M)),
            "bid_size": layer_init(nn.Linear(self.n_hidden, 1 + self.S)),
            "ask_size": layer_init(nn.Linear(self.n_hidden, 1 + self.S))
        })
        
        # Private information prediction heads
        self.pinfo_model = nn.ModuleDict({
            'settle_price': nn.Linear(self.n_hidden, 1),
            'private_roles': nn.Linear(self.n_hidden, self.P * 3)
        })
        
        # Value head
        self.critic = nn.Linear(self.n_hidden, 1)
        
        # For compilation
        self._compiled_batch_forward = None
    
    def _batch_forward(self, x, actions):
        """
        Fully parallel forward pass across all timesteps.
        
        x: [B, T, F]
        actions: [B, T, 4] type long
        
        Returns:
            values: [B, T]
            logprobs: [B, T]
            entropy: [B, T]
            pinfo_preds: dict
        """
        B, T, F = x.shape
        bid_px, ask_px, bid_sz, ask_sz = actions.unbind(dim=-1)
        
        # Encode all timesteps at once
        x_flat = x.reshape(B * T, F)
        encoded = self.encoder(x_flat).view(B, T, self.n_embd)
        
        # Add positional encoding
        encoded = self.pos_encoding(encoded)
        
        # Apply transformer with causal mask
        # Use generate_square_subsequent_mask for compatibility
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.transformer(encoded, mask=mask, is_causal=True)
        
        # Decode features
        h_flat = h.reshape(B * T, self.n_embd)
        features = self.decoder(h_flat)
        
        # Compute action distributions
        dists = {k: Categorical(logits=self.actors[k](features)) for k in self.actors}
        
        # Compute log probabilities
        actions_for_logprobs = {
            'bid_price': bid_px - 1,  # Zero-indexed
            'ask_price': ask_px - 1,
            'bid_size': bid_sz,
            'ask_size': ask_sz
        }
        
        logprobs = sum(
            dists[k].log_prob(actions_for_logprobs[k].reshape(B * T)) 
            for k in dists
        ).reshape(B, T)
        
        entropy = sum(d.entropy() for d in dists.values()).reshape(B, T)
        
        # Value and private info predictions
        values = self.critic(features).reshape(B, T)
        pinfo_preds = {k: self.pinfo_model[k](features) for k in self.pinfo_model}
        
        return {
            'values': values,
            'logprobs': logprobs,
            'entropy': entropy,
            'pinfo_preds': pinfo_preds
        }

    def initial_belief(self):
        """Returns initial hidden state (empty tensor for compatibility)."""
        return torch.zeros(0, device=self.device)

    @torch.jit.export
    @torch.inference_mode()
    def sample_actions(self, x: torch.Tensor, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Sample actions for a single timestep with causal context.
        
        x: [B, F] - Current observation
        hidden: [B, T_prev, n_hidden] - Previous timestep embeddings
        
        Returns:
            action: [B, 4]
            hidden: [B, T_prev+1, n_hidden] - Updated context
        """
        B, F = x.shape
        
        # Encode current observation
        encoded = self.encoder(x).view(B, 1, self.n_embd)
        
        if hidden.numel() == 0:
            # First timestep
            context = encoded
        else:
            # Concatenate with previous context
            context = torch.cat([hidden, encoded], dim=1)
        
        # Add positional encoding
        context = self.pos_encoding(context)
        
        # Apply transformer with causal mask
        T_ctx = context.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T_ctx, device=x.device)
        h = self.transformer(context, mask=mask, is_causal=True)
        
        # Get features for current timestep
        current_features = h[:, -1]  # [B, n_hidden // 4]
        features = self.decoder(current_features)
        
        # Sample actions
        dists = {k: Categorical(logits=self.actors[k](features)) for k in self.actors}
        actions = torch.stack([
            dists['bid_price'].sample() + 1,
            dists['ask_price'].sample() + 1,
            dists['bid_size'].sample(),
            dists['ask_size'].sample(),
        ], dim=-1)
        
        # Update hidden state (keep last T-1 timesteps for context window)
        new_hidden = h if h.size(1) < self.T else h[:, -(self.T-1):]
        
        return {'action': actions.int(), 'hidden': new_hidden}
    
    def compile(self, mode: str = "max-autotune", fullgraph: bool = True):
        """Compile the batch forward method for faster execution."""
        self._compiled_batch_forward = torch.compile(
            self._batch_forward, 
            mode=mode, 
            fullgraph=fullgraph
        )
        return self

    def forward(self, x, actions=None):
        """Forward pass using compiled method if available."""
        if self._compiled_batch_forward is not None:
            return self._compiled_batch_forward(x, actions)
        else:
            return self._batch_forward(x, actions)
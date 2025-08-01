import torch
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)  # Memory-efficient attention
torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import sys 
sys.path.append('../')
from model_components import layer_init, OptimizedResidualBlock, ResidualBlock
from typing import Dict
import warnings

class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings for transformer."""
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.register_buffer('position_ids', torch.arange(max_len))

    def forward(self, x):
        seq_length = x.size(0) # x: [T, B, D]
        # Get embeddings [T, 1, D]
        position_embeddings = self.pos_embedding(self.position_ids[:seq_length]).unsqueeze(1)
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
    def __init__(self, args, env, verbose=True):
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
            ResidualBlock(self.F, self.n_hidden),
            *[ResidualBlock(self.n_hidden, self.n_hidden) for _ in range(pre_blocks - 2)],
            ResidualBlock(self.n_hidden, self.n_embd))
        self.pos_encoding = LearnedPositionalEncoding(self.n_embd, max_len=self.T)
        
        # Transformer core using memory-efficient configurations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.n_embd,
            nhead=self.n_head,
            dim_feedforward=self.n_embd * 4,
            dropout=0,
            activation="gelu",
            batch_first=False,
            norm_first=True)  # Pre-norm architecture
        
        # Configure encoder for memory efficiency
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layer,
            enable_nested_tensor=False)  # Disable for SDPA efficiency
        
        # Output decoder
        assert post_blocks >= 1, "Post-decoder blocks must be at least 1"
        self.decoder = nn.Sequential(
            ResidualBlock(self.n_embd, self.n_hidden),
            *[ResidualBlock(self.n_hidden, self.n_hidden) for _ in range(post_blocks - 1)])
        
        # Action heads
        self.actors = nn.ModuleDict({
            "bid_price": layer_init(nn.Linear(self.n_hidden, self.M)),
            "ask_price": layer_init(nn.Linear(self.n_hidden, self.M)),
            "bid_size": layer_init(nn.Linear(self.n_hidden, 1 + self.S)),
            "ask_size": layer_init(nn.Linear(self.n_hidden, 1 + self.S))})
        
        # Private information prediction heads
        self.pinfo_model = nn.ModuleDict({
            'settle_price': nn.Linear(self.n_hidden, 1),
            'private_roles': nn.Linear(self.n_hidden, self.P * 3)})
        
        # Value head
        self.critic = nn.Linear(self.n_hidden, 1)
        
        # For compilation
        self._compiled_batch_forward = None
        # Pre-generate causal masks for different sequence lengths to avoid dynamic allocation
        self.register_buffer('causal_mask', nn.Transformer.generate_square_subsequent_mask(self.T, device=self.device))

        if verbose:
            B, F = env.observation_shape()
            T = args.steps_per_player
            print(f"Batch size: {B}, Sequence length: {T}, Feature dim: {F}")
            print(f"Transformer config: {args.n_hidden}d hidden, {args.n_head} heads, {args.n_layer} layers")

            # Count parameters
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (assuming float32)")
    
    def _batch_forward(self, x, actions):
        """
        Fully parallel forward pass across all timesteps.
        
        x: [T, B, F]
        actions: [T, B, 4] type long
        
        Returns:
            values: [T, B]
            logprobs: [T, B]
            entropy: [T, B]
            pinfo_preds: dict
        """
        T, B, F = x.shape
        bid_px, ask_px, bid_sz, ask_sz = actions.unbind(dim=-1) #[T, B] each
        encoded = self.encoder(x.view(T*B, F)).view(T, B, self.n_embd)
        encoded = self.pos_encoding(encoded) # [T, B, D]
        # [T, B, D]. causal_mask shape [T, T] with -inf on strict upper-diagonal
        h = self.transformer(encoded, mask=self.causal_mask, is_causal=True) 
        features = self.decoder(h.view(T * B, self.n_embd)) # [T * B, D]
        dists = {k: Categorical(logits=self.actors[k](features)) for k in self.actors}

        # Compute log probabilities
        actions_for_logprobs = {
            'bid_price': bid_px - 1,  # Zero-indexed
            'ask_price': ask_px - 1,
            'bid_size': bid_sz,
            'ask_size': ask_sz}

        logprobs = sum(
            dists[k].log_prob(actions_for_logprobs[k].reshape(T * B)) 
            for k in dists
        ).reshape(T, B)
        entropy = sum(d.entropy() for d in dists.values()).reshape(T, B)
        values = self.critic(features).reshape(T, B)
        pinfo_preds = {k: self.pinfo_model[k](features) for k in self.pinfo_model}
        
        return {
            'values': values,
            'logprobs': logprobs,
            'entropy': entropy,
            'pinfo_preds': pinfo_preds
        }

    @torch.inference_mode()
    def incremental_forward(self, x, step):
        if step == 0:
            assert self.context.numel() == 0, f"Context must be empty for first step, but got {self.context.shape}"
        else:
            assert self.context.shape[0] == step, f"Context must be of length {step}, but got {self.context.shape[0]}"
        outputs = self.incremental_forward_with_context(x, self.context)
        self.context = outputs['context']
        return {
            'action': outputs['action'], 
            'logprobs': outputs['logprobs'], 
            'pinfo_preds': outputs['pinfo_preds']}

    @torch.inference_mode()
    def incremental_forward_with_context(self, x, prev_context):
        """
        x: [B, F]
        prev_context: [T_sofar, B, D]. Post-encoder, pre-posEncoding

        Returns:
            action: [B, 4]
            logprobs: [B]
            context: [T_sofar+1, B, D]
        """
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            assert x.shape[1] == self.F, f"Expected observation feature dim {self.F}, got {x.shape[1]}"
            assert prev_context.numel() == 0 or prev_context.shape[2] == self.n_embd, f"Expected context embedding dim {self.n_embd}, got {prev_context.shape[2]}"
            
            B, F = x.shape
            encoded = self.encoder(x).view(1, B, self.n_embd)
            
            if prev_context.numel() == 0: # First timestep
                context = encoded
            else: # Concatenate with previous context
                context = torch.cat([prev_context, encoded], dim=0)
            features = self._compiled_incremental_core(context)
            dists = {k: Categorical(logits=self.actors[k](features)) for k in self.actors}
            actions = {k: dists[k].sample() for k in dists}
            action_tensor = torch.stack([
                actions['bid_price'] + 1,
                actions['ask_price'] + 1,
                actions['bid_size'],
                actions['ask_size'],
            ], dim=-1)
            logprobs = sum(dists[k].log_prob(actions[k]) for k in dists) 
            pinfo_preds = {k: self.pinfo_model[k](features) for k in self.pinfo_model}

            return {
                'action': action_tensor.int(), 
                'logprobs': logprobs, 
                'pinfo_preds': pinfo_preds,
                'context': context}

    def initial_context(self):
        """Returns initial context (empty tensor for compatibility)."""
        return torch.zeros(0, device=self.device)

    def reset_context(self):
        self.context = self.initial_context()

    def _incremental_core(self, context: torch.Tensor) -> torch.Tensor:
        """
        Sample actions for a single timestep given augmented context. 
        context: [T_sofar, B, D]
        """
        context = self.pos_encoding(context)
        T_ctx = context.size(0)
        mask = nn.Transformer.generate_square_subsequent_mask(T_ctx, device=context.device)
        h = self.transformer(context, mask=mask, is_causal=True)[-1] # [B, D]
        features = self.decoder(h)
        return features # [B, D]

    def forward(self, x, actions=None):
        if self._compiled_batch_forward is not None:
            assert x.shape[0] == self.T and x.shape[2] == self.F, f"Expected observation shape[0, 2] {self.T, self.F}, got {x.shape}"
            assert actions.shape[0] == self.T and actions.shape[2] == 4, f"Expected action shape[0, 2] {self.T, 4}, got {actions.shape}"
            return self._compiled_batch_forward(x, actions)
        else:
            raise RuntimeError("Model not compiled. Call compile() first.")

    def compile(self, mode: str = "max-autotune", fullgraph: bool = True):
        """Compile the batch forward method for faster execution."""
        self._compiled_batch_forward = torch.compile(
            self._batch_forward, 
            mode=mode, 
            fullgraph=fullgraph)
        
        # Compile sample_actions with dynamic shapes support
        self._compiled_incremental_core = torch.compile(
            self._incremental_core,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=True  # Enable dynamic shape support
        )
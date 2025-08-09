import torch
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)  # Memory-efficient attention
torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner

import torch.nn as nn
import torch.nn.functional as F
import sys 
sys.path.append('../')
from model_components import ResidualBlock, LearnedPositionalEncoding
from discrete_actor import DiscreteActor

class HighLowTransformerModel(nn.Module):
    """
    Transformer-based model for High-Low Trading that replaces GRU with parallel attention.
    
    Key parameters:
    - n_hidden: Embedding dimension (replaces hidden_size)
    - n_head: Number of attention heads
    - n_layer: Number of transformer blocks
    - dropout: Dropout rate for regularization
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
        
        # Input encoder
        self.encoder = ResidualBlock(self.F, self.n_embd)
        self.pos_encoding = LearnedPositionalEncoding(self.n_embd, max_len=max(self.T, 512))
        self.pinfo_numfeatures = 2 + 1 + self.P # see pinfo_tensor method in `env.py`
        
        # Transformer core using memory-efficient configurations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.n_embd,
            nhead=self.n_head,
            dim_feedforward=self.n_embd * 4,
            dropout=0,
            activation="gelu",
            batch_first=False,
            norm_first=True)  # Pre-norm architecture
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layer,
            enable_nested_tensor=False)
        self.transformer_norm = nn.LayerNorm(self.n_embd)
        
        self.actors = DiscreteActor(
            self.n_embd, 4, 
            min_values=torch.tensor([1, 1, 0, 0], device=self.device),
            max_values=torch.tensor([self.M, self.M, self.S, self.S], device=self.device))
        
        # Private information prediction heads
        self.pinfo_model = nn.ModuleDict({
            'settle_price': nn.Linear(self.n_embd, 1),
            'private_roles': nn.Linear(self.n_embd, self.P * 3)})
        
        # Value head
        self.critic = nn.Sequential(
            ResidualBlock(self.n_embd + self.pinfo_numfeatures, self.n_hidden),
            ResidualBlock(self.n_hidden, self.n_hidden),
            ResidualBlock(self.n_hidden, self.n_hidden),
            nn.Linear(self.n_hidden, 1))
        
        # Pre-generate causal masks for different sequence lengths to avoid dynamic allocation
        self.register_buffer(
            'causal_mask', nn.Transformer.generate_square_subsequent_mask(self.T, device=self.device), persistent=False) 

        self.register_buffer('uniform_buffer', torch.empty(0, device=self.device), persistent=False)
        self.context = self.initial_context()

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

    def _populate_uniform_buffer(self, shape):
        """Resize buffer if needed"""
        if self.uniform_buffer.shape != shape:
            self.uniform_buffer = torch.empty(shape, device=self.device)
        self.uniform_buffer.uniform_()
        return self.uniform_buffer
    
    def _batch_forward(self, x, pinfo_tensor, actions):
        """
        Fully parallel forward pass across all timesteps.
        
        x: [T, B, F]
        pinfo_tensor: [B, Pinfo_numfeatures]. Automatically expanded to all timesteps. Used for decentralized critic only. 
        actions: [T, B, 4] type long
        
        Returns:
            values: [T, B]
            logprobs: [T, B]
            entropy: [T, B]
            pinfo_preds / private_roles: [T, B, num_players, 3]
            pinfo_preds / settle_price: [T, B]
        """
        T, B, F = x.shape
        encoded = self.encoder(x.view(T*B, F)).view(T, B, self.n_embd)
        encoded = self.pos_encoding(encoded) # [T, B, D]
        # [T, B, D]. causal_mask shape [T, T] with -inf on strict upper-diagonal
        features = self.transformer(encoded, mask=self.causal_mask, is_causal=True).view(T * B, self.n_embd) # [T * B, D]
        features = self.transformer_norm(features)
        critic_features = torch.cat([
            features.view(T, B, self.n_embd),
            pinfo_tensor.expand(T, B, self.pinfo_numfeatures)
        ], dim=-1).reshape(T*B, self.n_embd + self.pinfo_numfeatures)
        values = self.critic(critic_features).reshape(T, B)

        logp_entropy = self.actors.logp_entropy(features, actions.view(T*B, 4))
        logprobs = logp_entropy['logprobs'].sum(-1).reshape(T, B) # Sum over action types [bid_px, ask_px, bid_sz, ask_sz]
        entropy = logp_entropy['entropy'].sum(-1).reshape(T, B)

        pinfo_preds = {k: self.pinfo_model[k](features) for k in self.pinfo_model}
        pinfo_preds['private_roles'] = pinfo_preds['private_roles'].reshape(T, B, self.P, 3)
        pinfo_preds['settle_price'] = pinfo_preds['settle_price'].reshape(T, B)
        
        return {
            'values': values,
            'logprobs': logprobs.reshape(T, B),
            'entropy': entropy.reshape(T, B),
            'pinfo_preds': pinfo_preds}

    @torch.inference_mode()
    def incremental_forward(self, x, step):
        if step == 0:
            assert self.context.numel() == 0, f"Context must be empty for first step, but got {self.context.shape}"
        else:
            assert self.context.shape[0] == step, f"Context must be of length {step}, but got {self.context.shape[0]}"
        # 4 action types
        uniform_samples = self._populate_uniform_buffer((x.shape[0], 4)) 
        outputs = self.incremental_forward_with_context(x, self.context, uniform_samples)
        self.context = outputs['context'].clone()
        return outputs

    @torch.inference_mode()
    @torch.compile(fullgraph=False, mode="max-autotune")
    def incremental_forward_with_context(self, x, prev_context, uniform_samples):
        """
        x: [B, F]
        uniform_samples: [B, 3] of torch.rand()
        prev_context: [T_sofar, B, D]. Post-encoder, pre-posEncoding

        Returns:
            action: [B, 4]
            logprobs: [B]
            context: [T_sofar+1, B, D]
        """
        assert x.shape[1] == self.F, f"Expected observation feature dim {self.F}, got {x.shape[1]}"
        assert prev_context.numel() == 0 or prev_context.shape[2] == self.n_embd, f"Expected context embedding dim {self.n_embd}, got {prev_context.shape[2]}"
        
        B, _F = x.shape
        encoded = self.encoder(x).view(1, B, self.n_embd)
        
        if prev_context.numel() == 0: # First timestep
            context = encoded
        else: # Concatenate with previous context
            context = torch.cat([prev_context, encoded], dim=0)
        features = self._incremental_core(context)
        action_outputs = self.actors.logp_entropy_and_sample(
            features, uniform_samples)

        pinfo_preds = {k: self.pinfo_model[k](features) for k in self.pinfo_model}
        pinfo_preds['private_roles'] = pinfo_preds['private_roles'].reshape(B, self.P, 3)
        pinfo_preds['settle_price'] = pinfo_preds['settle_price'].reshape(B)
        return {
            'action': action_outputs['samples'],
            'action_params': action_outputs['dist_params'],
            'logprobs': action_outputs['logprobs'].sum(-1),
            'logprobs_by_type': action_outputs['logprobs'],
            'entropy_by_type': action_outputs['entropy'],
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
        features = self.transformer(context, mask=mask, is_causal=True)[-1] # [B, D]
        features = self.transformer_norm(features)
        return features # [B, D]

    def forward(self, x, pinfo_tensor, actions=None):
        assert x.shape[0] == self.T and x.shape[2] == self.F, f"Expected observation shape[0, 2] {self.T, self.F}, got {x.shape}"
        assert actions.shape[0] == self.T and actions.shape[2] == 4, f"Expected action shape[0, 2] {self.T, 4}, got {actions.shape}"
        return self._batch_forward(x, pinfo_tensor, actions)
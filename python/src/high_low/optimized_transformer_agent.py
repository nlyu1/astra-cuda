import torch
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import sys 
sys.path.append('../')
from model_components import layer_init
from typing import Dict


# JIT-compiled fused operations for speed
@torch.jit.script
def fused_gelu_residual(x: torch.Tensor, linear_weight: torch.Tensor, 
                        linear_bias: torch.Tensor, ln_weight: torch.Tensor, 
                        ln_bias: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Fused Linear -> LayerNorm -> GELU -> Residual operation."""
    out = F.linear(x, linear_weight, linear_bias)
    out = F.layer_norm(out, (out.shape[-1],), ln_weight, ln_bias)
    out = F.gelu(out, approximate='tanh')  # Faster approximation
    return out + residual


class OptimizedResidualBlock(nn.Module):
    """Optimized residual block with fused operations."""
    def __init__(self, features: int, output_size: int):
        super().__init__()
        self.fc = nn.Linear(features, output_size)
        self.ln = nn.LayerNorm(output_size)
        
        # Pre-compute residual projection if needed
        self.need_projection = features != output_size
        if self.need_projection:
            self.residual_proj = nn.Linear(features, output_size, bias=False)
        
        # Initialize weights
        nn.init.orthogonal_(self.fc.weight, 2. ** .5)
        nn.init.constant_(self.fc.bias, 0.0)
        
    def forward(self, x):
        residual = self.residual_proj(x) if self.need_projection else x
        return fused_gelu_residual(x, self.fc.weight, self.fc.bias,
                                  self.ln.weight, self.ln.bias, residual)


class OptimizedTransformerModel(nn.Module):
    """
    Optimized transformer model targeting <100ms training steps.
    
    Key optimizations:
    1. Fused operations to reduce memory bandwidth
    2. Pre-computed masks and position embeddings
    3. Shared computations for actor heads
    4. Reduced reshaping operations
    5. JIT compilation for critical paths
    """
    def __init__(self, args, env):
        super().__init__()
        self.B, self.F = env.observation_shape()
        self.T = args.steps_per_player 
        self.P = args.players
        self.M = args.max_contract_value
        self.S = args.max_contracts_per_trade
        self.device = torch.device(f'cuda:{args.device_id}')
        
        # Model dimensions
        self.n_hidden = args.n_hidden
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.n_layer = args.n_layer
        
        # Optimized encoder - reduce to minimum necessary blocks
        self.encoder = nn.Sequential(
            OptimizedResidualBlock(self.F, self.n_hidden),
            OptimizedResidualBlock(self.n_hidden, self.n_embd)
        )
        
        # Pre-computed positional embeddings (faster than embedding lookup)
        self.register_buffer('pos_embeddings', 
                           torch.randn(1, self.T, self.n_embd) * 0.02)
        
        # Single optimized transformer layer (often sufficient)
        if self.n_layer == 1:
            self.transformer = nn.TransformerEncoderLayer(
                d_model=self.n_embd,
                nhead=self.n_head,
                dim_feedforward=self.n_embd * 2,  # Smaller FFN for speed
                dropout=0,
                activation=F.gelu,  # Use function directly
                batch_first=True,
                norm_first=True
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.n_embd,
                nhead=self.n_head,
                dim_feedforward=self.n_embd * 2,
                dropout=0,
                activation=F.gelu,
                batch_first=True,
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=self.n_layer
            )
        
        # Pre-generate causal mask for the full sequence
        self.register_buffer('causal_mask', 
                           nn.Transformer.generate_square_subsequent_mask(self.T))
        
        # Optimized decoder - single projection
        self.decoder = nn.Sequential(
            nn.Linear(self.n_embd, self.n_hidden),
            nn.GELU(approximate='tanh')
        )
        
        # Shared actor computation
        self.actor_shared = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(inplace=True)
        )
        
        # Fused price heads (both bid and ask)
        self.price_logits = nn.Linear(self.n_hidden, 2 * self.M)
        # Fused size heads (both bid and ask)
        self.size_logits = nn.Linear(self.n_hidden, 2 * (1 + self.S))
        
        # Value and private info heads
        self.critic = nn.Linear(self.n_hidden, 1)
        self.pinfo_model = nn.ModuleDict({
            'settle_price': nn.Linear(self.n_hidden, 1),
            'private_roles': nn.Linear(self.n_hidden, self.P * 3)
        })
        
        # Initialize output heads
        for m in [self.price_logits, self.size_logits]:
            nn.init.orthogonal_(m.weight, 2. ** .5)
            nn.init.constant_(m.bias, 0.0)
        
        self._compiled_batch_forward = None
    
    def _batch_forward(self, x, actions):
        """Optimized forward pass with minimal operations."""
        B, T, feat_dim = x.shape
        bid_px, ask_px, bid_sz, ask_sz = actions.unbind(dim=-1)
        
        # Single reshape at the beginning
        x_flat = x.view(-1, feat_dim)  # More efficient than reshape
        
        # Encode and add positional encoding in one go
        encoded = self.encoder(x_flat).view(B, T, self.n_embd)
        encoded = encoded + self.pos_embeddings
        
        # Apply transformer with pre-computed mask
        # Use is_causal=True to avoid mask detection issues with torch.compile
        if self.n_layer == 1:
            h = self.transformer(encoded, src_mask=self.causal_mask, is_causal=True)
        else:
            h = self.transformer(encoded, mask=self.causal_mask, is_causal=True)
        
        # Decode all at once
        h_flat = h.reshape(-1, self.n_embd)
        features = self.decoder(h_flat)
        
        # Shared actor computation
        actor_features = self.actor_shared(features)
        
        # Get all logits in one forward pass
        all_price_logits = self.price_logits(actor_features)
        all_size_logits = self.size_logits(actor_features)
        
        # Split logits efficiently
        bid_price_logits = all_price_logits[:, :self.M]
        ask_price_logits = all_price_logits[:, self.M:]
        bid_size_logits = all_size_logits[:, :(1 + self.S)]
        ask_size_logits = all_size_logits[:, (1 + self.S):]
        
        # Debug info
        # print(f"all_price_logits shape: {all_price_logits.shape}")
        # print(f"bid_price_logits type: {type(bid_price_logits)}")
        
        # Vectorized log probability computation
        actions_flat = actions.view(-1, 4)
        bid_px_idx = actions_flat[:, 0] - 1
        ask_px_idx = actions_flat[:, 1] - 1
        bid_sz_idx = actions_flat[:, 2]
        ask_sz_idx = actions_flat[:, 3]
        
        # Compute log probabilities for each action
        bid_px_logprobs = F.log_softmax(bid_price_logits, dim=-1).gather(1, bid_px_idx.unsqueeze(1)).squeeze(1)
        ask_px_logprobs = F.log_softmax(ask_price_logits, dim=-1).gather(1, ask_px_idx.unsqueeze(1)).squeeze(1)
        bid_sz_logprobs = F.log_softmax(bid_size_logits, dim=-1).gather(1, bid_sz_idx.unsqueeze(1)).squeeze(1)
        ask_sz_logprobs = F.log_softmax(ask_size_logits, dim=-1).gather(1, ask_sz_idx.unsqueeze(1)).squeeze(1)
        
        # Sum log probabilities and reshape
        logprobs = (bid_px_logprobs + ask_px_logprobs + bid_sz_logprobs + ask_sz_logprobs).view(B, T)
        
        # Compute entropy using negative expected log probability
        bid_px_entropy = -(F.softmax(bid_price_logits, dim=-1) * F.log_softmax(bid_price_logits, dim=-1)).sum(dim=-1)
        ask_px_entropy = -(F.softmax(ask_price_logits, dim=-1) * F.log_softmax(ask_price_logits, dim=-1)).sum(dim=-1)
        bid_sz_entropy = -(F.softmax(bid_size_logits, dim=-1) * F.log_softmax(bid_size_logits, dim=-1)).sum(dim=-1)
        ask_sz_entropy = -(F.softmax(ask_size_logits, dim=-1) * F.log_softmax(ask_size_logits, dim=-1)).sum(dim=-1)
        
        entropy = (bid_px_entropy + ask_px_entropy + bid_sz_entropy + ask_sz_entropy).view(B, T)
        
        # Value and private info predictions
        values = self.critic(features).view(B, T)
        pinfo_preds = {k: v(features) for k, v in self.pinfo_model.items()}
        
        return {
            'values': values,
            'logprobs': logprobs,
            'entropy': entropy,
            'pinfo_preds': pinfo_preds
        }
    
    def initial_belief(self):
        """Returns initial hidden state."""
        return torch.zeros(0, device=self.device)
    
    @torch.inference_mode()
    def sample_actions(self, x: torch.Tensor, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Optimized action sampling for inference."""
        B = x.shape[0]
        
        # Encode current observation
        encoded = self.encoder(x).view(B, 1, self.n_embd)
        
        # Handle context
        if hidden.numel() == 0:
            context = encoded
            pos_end = 1
        else:
            context = torch.cat([hidden, encoded], dim=1)
            pos_end = context.size(1)
        
        # Add positional encoding
        context = context + self.pos_embeddings[:, :pos_end]
        
        # Apply transformer
        if pos_end <= self.T:
            mask = self.causal_mask[:pos_end, :pos_end]
        else:
            mask = nn.Transformer.generate_square_subsequent_mask(pos_end, device=x.device)
        
        # Use is_causal=True to avoid mask detection issues
        if self.n_layer == 1:
            h = self.transformer(context, src_mask=mask, is_causal=True)
        else:
            h = self.transformer(context, mask=mask, is_causal=True)
        
        # Get current features
        current_features = self.decoder(h[:, -1])
        actor_features = self.actor_shared(current_features)
        
        # Get all logits
        all_price_logits = self.price_logits(actor_features)
        all_size_logits = self.size_logits(actor_features)
        
        # Split and sample
        bid_price_logits = all_price_logits[:, :self.M]
        ask_price_logits = all_price_logits[:, self.M:]
        bid_size_logits = all_size_logits[:, :(1 + self.S)]
        ask_size_logits = all_size_logits[:, (1 + self.S):]
        
        actions = torch.stack([
            torch.multinomial(F.softmax(bid_price_logits, dim=-1), 1).squeeze(-1) + 1,
            torch.multinomial(F.softmax(ask_price_logits, dim=-1), 1).squeeze(-1) + 1,
            torch.multinomial(F.softmax(bid_size_logits, dim=-1), 1).squeeze(-1),
            torch.multinomial(F.softmax(ask_size_logits, dim=-1), 1).squeeze(-1),
        ], dim=-1)
        
        # Update hidden state
        new_hidden = h if h.size(1) < self.T else h[:, -(self.T-1):]
        
        return {'action': actions.int(), 'hidden': new_hidden}
    
    def compile(self, mode: str = "reduce-overhead", fullgraph: bool = True):
        """Compile for maximum speed."""
        self._compiled_batch_forward = torch.compile(
            self._batch_forward, 
            mode=mode,  # reduce-overhead is best for latency
            fullgraph=fullgraph
        )
        return self
    
    def forward(self, x, actions=None):
        """Forward pass using compiled method if available."""
        if self._compiled_batch_forward is not None:
            return self._compiled_batch_forward(x, actions)
        else:
            return self._batch_forward(x, actions)
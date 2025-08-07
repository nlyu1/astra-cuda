import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from contextlib import contextmanager
import sys
sys.path.append('./src')
from high_low.config import Args
from model_components import ResidualBlock, LearnedPositionalEncoding

@contextmanager
def cuda_timer(name):
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"{name}: {(end - start) * 1000:.2f}ms")

class KVCacheTransformerLayer(nn.Module):
    """Single transformer layer with KV caching support"""
    def __init__(self, d_model, n_head, dim_feedforward, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, k_cache=None, v_cache=None, mask=None):
        """
        x: [seq_len, batch, d_model] - current input
        k_cache: [past_len, batch, n_head, head_dim] - cached keys
        v_cache: [past_len, batch, n_head, head_dim] - cached values
        """
        seq_len, batch_size, _ = x.shape
        
        # Pre-norm
        x_norm = self.norm1(x)
        
        # Compute Q, K, V
        q = self.q_proj(x_norm).view(seq_len, batch_size, self.n_head, self.head_dim)
        k = self.k_proj(x_norm).view(seq_len, batch_size, self.n_head, self.head_dim)
        v = self.v_proj(x_norm).view(seq_len, batch_size, self.n_head, self.head_dim)
        
        # Concatenate with cache if provided
        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=0)
            v = torch.cat([v_cache, v], dim=0)
        
        # Transpose for attention: [batch, n_head, seq_len, head_dim]
        q = q.transpose(0, 1).transpose(1, 2)
        k = k.transpose(0, 1).transpose(1, 2)
        v = v.transpose(0, 1).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores + mask
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, v)
        
        # Reshape and project
        out = out.transpose(1, 2).transpose(0, 1).contiguous()
        out = out.view(seq_len, batch_size, self.d_model)
        out = self.o_proj(out)
        out = self.dropout(out)
        
        # Residual connection
        x = x + out
        
        # MLP block with pre-norm
        x = x + self.mlp(self.norm2(x))
        
        return x, k, v

def benchmark_attention_methods():
    # Setup
    args = Args()
    args.fill_runtime_args()
    device = torch.device(f'cuda:{args.device_id}')
    
    B = args.num_envs  # 4096
    T = args.steps_per_player  # 16
    D = args.n_embd  # 256
    H = args.n_head  # 8
    L = args.n_layer  # 5
    F = 512  # Feature dim (approximate from your code)
    
    print(f"Benchmarking with B={B}, T={T}, D={D}, H={H}, L={L}")
    print(f"Batch size: {B}")
    print("-" * 60)
    
    # Create models
    encoder = ResidualBlock(F, D).to(device)
    pos_encoding = LearnedPositionalEncoding(D, max_len=T).to(device)
    
    # Standard transformer
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=D, nhead=H, dim_feedforward=D*4, dropout=0,
        activation="gelu", batch_first=False, norm_first=True
    )
    standard_transformer = nn.TransformerEncoder(encoder_layer, num_layers=L).to(device)
    
    # KV-cache transformer
    kv_layers = nn.ModuleList([
        KVCacheTransformerLayer(D, H, D*4) for _ in range(L)
    ]).to(device)
    
    # Generate random input
    full_sequence = torch.randn(T, B, F, device=device)
    
    # Warmup
    for _ in range(5):
        _ = encoder(full_sequence[0])
        torch.cuda.synchronize()
    
    print("\n1. Current approach (recompute attention over full sequence):")
    total_time = 0
    for t in range(T):
        with cuda_timer(f"  Step {t}"):
            # Encode all previous + current
            context = full_sequence[:t+1]
            encoded = encoder(context.reshape(-1, F)).view(t+1, B, D)
            encoded = pos_encoding(encoded)
            
            # Generate mask
            mask = nn.Transformer.generate_square_subsequent_mask(t+1, device=device)
            
            # Run transformer
            _ = standard_transformer(encoded, mask=mask, is_causal=True)[-1]
    
    print("\n2. KV-caching approach:")
    k_caches = [None] * L
    v_caches = [None] * L
    total_time = 0
    
    for t in range(T):
        with cuda_timer(f"  Step {t}"):
            # Encode only current token
            x = encoder(full_sequence[t]).view(1, B, D)
            x = pos_encoding(x) if t == 0 else x  # Only add positional encoding once
            
            # Run through KV-cache layers
            for i, layer in enumerate(kv_layers):
                x, k, v = layer(x, k_caches[i], v_caches[i])
                # Update caches
                k_caches[i] = k
                v_caches[i] = v
    
    print("\n3. Component breakdown (single forward pass):")
    
    # Test individual components
    x = torch.randn(B, F, device=device)
    
    with cuda_timer("  Encoder"):
        _ = encoder(x)
    
    # Full sequence through transformer
    full_encoded = encoder(full_sequence.reshape(-1, F)).view(T, B, D)
    full_encoded = pos_encoding(full_encoded)
    mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
    
    with cuda_timer("  Transformer (full sequence)"):
        _ = standard_transformer(full_encoded, mask=mask, is_causal=True)
    
    # Dummy actor/critic operations
    critic = nn.Linear(D, 1).to(device)
    actor = nn.Linear(D, 4).to(device)
    
    features = torch.randn(B, D, device=device)
    with cuda_timer("  Critic"):
        _ = critic(features)
    
    with cuda_timer("  Actor"):
        _ = actor(features)
    
    print("\n4. Memory usage:")
    print(f"  KV cache per step: {2 * L * H * (D//H) * B * 2 / 1024 / 1024:.2f} MB")
    print(f"  Total KV cache (full sequence): {2 * L * H * (D//H) * B * T * 2 / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    # Compile models for fair comparison
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    
    benchmark_attention_methods()
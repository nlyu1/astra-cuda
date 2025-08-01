import torch
import torch.nn as nn 
import torch.nn.functional as F

# -------------------------------- utilities -------------------------------- #
def layer_init(layer: nn.Module, std: float = 2. ** .5, bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, features: int, output_size: int):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(features, output_size))
        self.ln = nn.LayerNorm(output_size)
        self.act = nn.GELU()

        self.output_size = output_size
        self.features = features

    def forward(self, x):
        if self.output_size > self.features:
            residual_x = torch.cat([x, torch.zeros(x.shape[0], self.output_size - self.features).type_as(x)], dim=-1)
        else:
            residual_x = x[:, :self.output_size]
        return residual_x + self.act(self.ln(self.fc1(x)))

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
        self.fc = layer_init(nn.Linear(features, output_size))
        self.ln = nn.LayerNorm(output_size)
        
        # Pre-compute residual projection if needed
        self.need_projection = features != output_size
        if self.need_projection:
            self.residual_proj = nn.Linear(features, output_size, bias=False)
        
    def forward(self, x):
        residual = self.residual_proj(x) if self.need_projection else x
        return fused_gelu_residual(x, self.fc.weight, self.fc.bias,
                                  self.ln.weight, self.ln.bias, residual)
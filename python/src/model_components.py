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
        
        # Pre-compute if we need padding
        self.is_residual = output_size == features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_residual:
            return x + self.act(self.ln(self.fc1(x)))
        else:
            return self.act(self.ln(self.fc1(x)))

class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings for transformer."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.register_buffer('position_ids', torch.arange(max_len))

    def forward(self, x):
        seq_length = x.size(0) # x: [T, B, D]
        # Get embeddings [T, 1, D]
        position_embeddings = self.pos_embedding(self.position_ids[:seq_length]).unsqueeze(1)
        return x + position_embeddings
    

from torch.distributions.beta import Beta
"""
Ingredients for beta-ST (straight through) estimation. 

Model outputs continuous parameters of the beta distribution. 
Supports discrete, bounded sampling. 
"""

class BetaActor(nn.Module):
    def __init__(self, n_hidden, n_actors, max_kappa=20):
        """
        Model outputs the location and dispersion parameters of the beta distribution. 
        Beta(alpha, beta) where alpha = k*m, beta=k*(1-m)
        k: dispersion parameter unbounded in [0, inf]. Parameterized by softplus(k_hidden)
        m: location parameter bounded in [0, 1]. Parameterized by sigmoid(m_hidden)
        """
        super().__init__()
        self.n_hidden = n_hidden
        self.n_actors = n_actors
        self.actor = nn.Linear(n_hidden, n_actors * 2)
        self.max_kappa = max_kappa
    
    # @torch.compile(fullgraph=True, mode="max-autotune")
    def forward(self, x):
        output = self.actor(x)
        m_hidden, kappa_hidden = output[:, :self.n_actors], output[:, self.n_actors:]
        # print('m_hidden:', m_hidden.min().item(), m_hidden.max().item())
        # print('kappa_hidden:', kappa_hidden.min().item(), kappa_hidden.max().item())
        kappa = self.max_kappa - nn.functional.softplus(
            self.max_kappa - nn.functional.softplus(kappa_hidden))
        m = nn.functional.sigmoid(torch.tanh(m_hidden / 6.9) * 6.9) # Determines the mean of the beta distribution. sigmoid(6.9) ~ 0.999
        alpha = kappa * m
        beta = kappa * (1 - m)
        # print('m:', m.min().item(), m.max().item())
        # print('kappa:', kappa.min().item(), kappa.max().item())
        # print('alpha:', alpha.min().item(), alpha.max().item())
        # print('beta:', beta.min().item(), beta.max().item())
        return alpha, beta
    
    @classmethod 
    def sample(cls, alpha, beta, min_val, max_val, eps=1e-3):
        """
        alpha, beta: [batch_size, n_actors]
        min_val, max_val: float. 
        eps: float. Prevents rounded result exceeding [min_val, max_val]

        Samples from the (alpha-beta) parameterized beta distribution, then 
        rescales to ~[min_val - 0.5, max_val + 0.5] and rounds to int. 
        """
        dist = Beta(alpha, beta)
        result = (dist.sample() * (max_val - min_val + (1 - eps)) + (min_val - 0.5 + eps)).round().int()
        return result 
    
    @classmethod
    def logp_entropy(cls, x, alpha, beta, min_val, max_val):
        """
        x: [batch_size, n_actors] resulting from `sample`. Between [min_val, max_val]
        alpha, beta: [batch_size, n_actors]
        min_val, max_val: float. 

        Returns: logprobs, entropy of shape [batch_size] 

        Renormalizes x to (0, 1) and computes logprobs and entropy. 
        """
        dist = Beta(alpha, beta)
        normalized_x = (x - (min_val - 0.5)) / ((max_val - min_val + 1))
        logprobs = dist.log_prob(normalized_x)
        return logprobs, dist.entropy()
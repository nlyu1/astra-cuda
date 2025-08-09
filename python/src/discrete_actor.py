"""
Truncated Gaussian-based discrete action sampler for RL.

Use case: Discrete action spaces that are discretizations of continuous intervals
(e.g. bid/ask prices, order sizes in trading environments).

Key components:
- GaussianActionDistribution: Truncated Normal on (0,1) parameterized by center and precision
- DiscreteActor: Neural network that outputs distribution parameters and samples discrete actions

The approach:
1. Sample from truncated Gaussian in [0,1]
2. Scale to action range and round to nearest integer
3. Compute log probabilities over discrete intervals
"""

# %%

import torch.nn as nn 
import torch, math
from torch import Tensor
from typing import Tuple

_TWO_PI = 2.0 * math.pi
_LOG_SQRT_2PI = 0.5 * math.log(_TWO_PI)

def _log_normal_pdf_prec(x: Tensor, loc: Tensor, prec: Tensor) -> Tensor:
    z = (x - loc) * prec
    return -0.5 * z.square() - _LOG_SQRT_2PI + torch.log(prec)

def _log_ndtr(z: Tensor) -> Tensor:
    return torch.special.log_ndtr(z)

def _logsubexp(a: Tensor, b: Tensor) -> Tensor:
    """
    Stable log(exp(a) - exp(b))   with   a ≥ b.
    (Analogous to torch.logaddexp for the *sum* case.)
    """
    return a + torch.log1p(-torch.exp(b - a))

class GaussianActionDistribution:
    """
    Truncated Normal N(center, σ²) on the open interval (0, 1),
    **constructed with precision** (prec = 1 / std).
    """

    def __init__(self, center: Tensor, precision: Tensor):
        self.center = center
        self.prec   = precision           # 1 / σ
        alpha = (0.0 - center) * precision
        beta  = (1.0 - center) * precision
        self._log_F_alpha = _log_ndtr(alpha)
        self._log_F_beta  = _log_ndtr(beta)
        # Total weight contained within the truncated distribution 

        # Plain CDF values (needed only for sampling)
        self._F_alpha = torch.exp(self._log_F_alpha)
        self._F_beta  = torch.exp(self._log_F_beta)
        self._log_Z = _logsubexp(self._log_F_beta, self._log_F_alpha)

    @torch.no_grad()
    def sample(self, uniform_samples: Tensor) -> Tensor:
        u = uniform_samples.clamp(1e-6, 1.0 - 1e-6)
        p = u * (self._F_beta - self._F_alpha) + self._F_alpha
        z = torch.special.ndtri(p)               # z-scores
        return z / self.prec + self.center       # x = z * std + mean

    def log_prob(self, x: Tensor) -> Tensor:
        return _log_normal_pdf_prec(x, self.center, self.prec) - self._log_Z

    def log_cdf(self, x: Tensor) -> Tensor:
        z = (x - self.center) * self.prec
        return _logsubexp(_log_ndtr(z), self._log_F_alpha) - self._log_Z

    def logp_interval(self, lo: Tensor, hi: Tensor) -> Tensor:
        """log P(lo ≤ X ≤ hi)."""
        z_low = (lo - self.center) * self.prec
        z_high = (hi - self.center) * self.prec
        return _logsubexp(_log_ndtr(z_high), _log_ndtr(z_low)) - self._log_Z

    def entropy(self) -> Tensor:
        alpha = (0.0 - self.center) * self.prec
        beta  = (1.0 - self.center) * self.prec
        phi   = lambda t: torch.exp(-0.5 * t.square()) / math.sqrt(_TWO_PI)

        num = alpha * phi(alpha) - beta * phi(beta)
        return (-torch.log(self.prec) + 0.5 * math.log(_TWO_PI * math.e) + 
                num / torch.exp(self._log_Z) - self._log_Z)


class DiscreteActor(nn.Module):
    """
    Neural network for discrete action sampling using truncated Gaussians.
    
    Architecture:
    - Input: hidden state [B, n_hidden]
    - Linear layer outputs 2*n_actors values
    - Splits into: center (sigmoid to [0,1]), precision (softplus with bias)
    - Creates truncated Gaussian distributions and samples discrete actions
    
    Important: Use learning rate warmup to prevent early collapse where
    the model outputs bad actions and gets stuck in high-entropy mode.
    """
    def __init__(self, n_hidden, n_actors, 
        min_values: torch.Tensor, max_values: torch.Tensor, eps=1e-4):
        """
        Args:
            n_hidden: Input feature dimension
            n_actors: Number of parallel action distributions
            min_values: Minimum action values per actor [n_actors]
            max_values: Maximum action values per actor [n_actors]
            eps: Small constant to prevent boundary issues
        """
        super().__init__()
        self.n_hidden = n_hidden
        self.n_actors = n_actors
        self.actor = nn.Linear(n_hidden, n_actors * 2)
        assert min_values.shape == max_values.shape, f"min_values and max_values must have the same shape, but got {min_values.shape} and {max_values.shape}"
        assert min_values.ndim == 1, f"min_values and max_values must be 1D, but got {min_values.ndim} and {max_values.ndim}"
        assert min_values.shape[0] == n_actors, f"min_values and max_values must have length {n_actors}, but got {min_values.shape[0]} and {max_values.shape[0]}"
        self.eps = eps 
        self.register_buffer('min_values', min_values, persistent=False)
        self.register_buffer('max_values', max_values, persistent=False)
        self.register_buffer('rangeP1', max_values - min_values + 1, persistent=False)
        
    #@#torch.compile(fullgraph=True, mode="max-autotune")
    def forward(self, x):
        output = self.actor(x)
        mean, precision = output[:, :self.n_actors], output[:, self.n_actors:]
        mean = torch.sigmoid(mean)
        precision = nn.functional.softplus(
            nn.functional.leaky_relu(precision, negative_slope=0.1) + 1.5) + 0.5
        return mean, precision
    
    def _integer_samples_from_unit_samples(self, unit_samples):
        unit_samples = unit_samples.clamp(self.eps, 1.0 - self.eps)
        integer_samples = (unit_samples * self.rangeP1 + self.min_values - 0.5).clamp(self.min_values, self.max_values).round().int()
        return integer_samples
    
    def _unit_interval_of_integer_samples(self, integer_samples):
        unit_samples_ub = ((integer_samples + 0.5) + 0.5 - self.min_values) / self.rangeP1
        unit_samples_lb = ((integer_samples - 0.5) + 0.5 - self.min_values) / self.rangeP1
        return unit_samples_lb, unit_samples_ub
    
    def logp_entropy_and_sample(self, x, uniform_samples):
        """
        Sample actions and compute log probabilities.
        
        Args:
            x: Hidden states [B, n_hidden]
            uniform_samples: Random samples [B, n_actors]
            
        Returns:
            Dict with samples, logprobs, entropy, and distribution parameters
        """
        center, prec = self(x)
        dist = GaussianActionDistribution(center, prec)
        unit_samples = dist.sample(uniform_samples) # between [0, 1]
        integer_samples = self._integer_samples_from_unit_samples(unit_samples)
        unit_lb, unit_ub = self._unit_interval_of_integer_samples(integer_samples)
        # Logprobs needs to be appropriately scaled
        logprobs = dist.logp_interval(unit_lb, unit_ub) - self.rangeP1.log() 

        # Approximating discrete entropy by continuous differential entropy, so we need to scale by the range
        entropy = dist.entropy() + self.rangeP1.log()
        return {
            'samples': integer_samples, # [B, n_actors] int. A batch of samples
            'logprobs': logprobs, # [B, n_actors] float. Logprobs of the samples
            'entropy': entropy, # [B] float. Entropy of the computed distribution
            'dist_params': { # Distribution parameters
                'center': center, # [B, n_actors] float
                'precision': prec, # [B, n_actors] float
            }}
    
    def logp_entropy(self, x, integer_samples):
        """
        Compute log probabilities and entropy for given actions.
        
        Args:
            x: Hidden states [B, n_hidden]
            integer_samples: Integer actions [B, n_actors]
            
        Returns:
            Dict with logprobs, entropy, and distribution parameters
        """
        center, prec = self(x)
        dist = GaussianActionDistribution(center, prec)

        unit_lb, unit_ub = self._unit_interval_of_integer_samples(integer_samples)
        logprobs = dist.logp_interval(unit_lb, unit_ub) - self.rangeP1.log() 
        entropy = dist.entropy() + self.rangeP1.log()
        return {
            'logprobs': logprobs, # [B, n_actors] float. Logprobs of the samples
            'entropy': entropy, # [B] float. Entropy of the computed distribution
            'dist_params': { # Distribution parameters
                'center': center, # [B, n_actors] float
                'precision': prec, # [B, n_actors] float
            }}
    
if __name__ == '__main__':
    # %%
    import matplotlib.pyplot as plt

    num_samples = 1024000
    center = torch.zeros((num_samples,), device='cuda:0') + 0.99
    prec = torch.ones((num_samples,), device='cuda:0') * 1000

    uniform_samples = torch.rand(num_samples, device='cuda:0')

    dist = GaussianActionDistribution(center, prec)
    # dist = TrapezoidFullSupportDistribution(center, epsilon_uniform)
    samples = dist.sample(uniform_samples)
    plt.hist(samples.cpu().numpy(), bins=100, density=True)
    x = torch.linspace(0, 1, num_samples, device='cuda:0')
    pdf_value = dist.log_prob(x).exp()
    plt.plot(x.cpu().numpy(), pdf_value.cpu().numpy())
    plt.show()

    cdf = dist.log_cdf(x).exp()
    numerical_difference = (cdf - pdf_value.cumsum(dim=0) / num_samples).abs()
    plt.plot(cdf.cpu().numpy() + 0.01, label='cdf')
    plt.plot(pdf_value.cumsum(dim=0).cpu().numpy() / num_samples, label='cdf approx')
    plt.plot(x.cpu().numpy(), numerical_difference.cpu().numpy(), label='difference')
    plt.title(f'max difference: {numerical_difference.max().item():.6f}')
    plt.legend()
    plt.show()

    # %%
    batch_size = 102400
    num_actors = 10
    center = torch.zeros((batch_size, num_actors), device='cuda:0') + 0.9
    prec = torch.ones((batch_size, num_actors), device='cuda:0') * 10
    uniform_samples = torch.rand(batch_size, num_actors, device='cuda:0')

    min_values = torch.zeros((num_actors,), device='cuda:0')
    max_values = torch.zeros((num_actors,), device='cuda:0') + 100
    rangeP1 = max_values - min_values + 1

    dist = GaussianActionDistribution(center, prec)
    unit_samples = dist.sample(uniform_samples) # between [0, 1]
    integer_samples = (unit_samples * rangeP1 + min_values - 0.5).round().int().clamp(min=min_values, max=max_values)
    unit_samples_ub = ((integer_samples + 0.5) + 0.5 - min_values) / rangeP1
    unit_samples_lb = ((integer_samples - 0.5) + 0.5 - min_values) / rangeP1

    # Logprobs needs to be appropriately scaled
    logprobs = dist.logp_interval(unit_samples_lb, unit_samples_ub) - rangeP1.log() 
    # Approximating discrete entropy by continuous differential entropy, so we need to scale by the range
    entropy = dist.entropy() + rangeP1.log()
    output = {
        'samples': integer_samples, # [B, n_actors] int. A batch of samples
        'logprobs': logprobs, # [B, n_actors] float. Logprobs of the samples
        'entropy': entropy, # [B] float. Entropy of the computed distribution
        'dist_params': { # Distribution parameters
            'center': center, # [B, n_actors] float
            'precision': prec, # [B, n_actors] float
        }}

    output_samples = output['samples']
    plt.hist(output_samples.cpu().numpy().flatten(), bins=100, density=True)
    discrete_probs = torch.tensor([
        (output_samples == i).float().mean().item() 
        for i in range(min_values[0].long().item(), max_values[0].long().item() + 1)])

    entropy = -torch.xlogy(discrete_probs, discrete_probs).sum()
    # Roughly accurate 
    print('Entropy:', entropy.item(), 'guess:', output['entropy'][0, 0].item())
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

Use arithmetic masking instead of torch.where. The *~@*$# torch.where propagates nan gradients on paths which are not used. 
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
    Stable log(exp(a) - exp(b)) with a ≥ b.
    
    Key insight: log(exp(a) - exp(b)) = a + log(1 - exp(b-a))
    Since b ≤ a, we have b-a ≤ 0, so exp(b-a) ≤ 1.
    """
    diff = b - a
    return a + torch.log1p(-torch.exp(diff))

def _compute_truncation_params(center: Tensor, precision: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute truncation parameters for the distribution."""
    alpha = (0.0 - center) * precision
    beta = (1.0 - center) * precision
    log_F_alpha = _log_ndtr(alpha)
    log_F_beta = _log_ndtr(beta)
    log_Z = _logsubexp(log_F_beta, log_F_alpha)
    return alpha, beta, log_F_alpha, log_F_beta, log_Z

def _sample_truncated_gaussian(center: Tensor, precision: Tensor, uniform_samples: Tensor,
                             log_F_alpha: Tensor, log_F_beta: Tensor) -> Tensor:
    """Sample from truncated Gaussian distribution."""
    u = uniform_samples.clamp(1e-6, 1.0 - 1e-6)
    F_alpha = torch.exp(log_F_alpha)
    F_beta = torch.exp(log_F_beta)
    p = u * (F_beta - F_alpha) + F_alpha
    z = torch.special.ndtri(p)  # z-scores
    return z / precision + center  # x = z * std + mean

def _log_prob_truncated(x: Tensor, center: Tensor, precision: Tensor, log_Z: Tensor) -> Tensor:
    """Compute log probability for truncated Gaussian."""
    return _log_normal_pdf_prec(x, center, precision) - log_Z

def _log_cdf_truncated(x: Tensor, center: Tensor, precision: Tensor, 
                      log_F_alpha: Tensor, log_Z: Tensor) -> Tensor:
    """Compute log CDF for truncated Gaussian."""
    z = (x - center) * precision
    return _logsubexp(_log_ndtr(z), log_F_alpha) - log_Z

def _logp_interval_truncated(lo: Tensor, hi: Tensor, center: Tensor, precision: Tensor,
                           log_Z: Tensor) -> Tensor:
    """
    log P(lo ≤ X ≤ hi) using arithmetic masking for torch.compile compatibility.
    """
    z_low = (lo - center) * precision
    z_high = (hi - center) * precision
    
    # Detect extreme cases (convert to float for arithmetic)
    is_extreme = (((z_low > 4.0) & (z_high > 4.0)) | 
                ((z_low < -4.0) & (z_high < -4.0))).float()
    
    # Detect which formulation to use for non-extreme cases
    use_complement = ((z_low > -z_high) & (z_low > 0)).float()
    
    # === Compute all three branches with safe fallbacks ===
    
    # 1. Extreme case approximation
    z_mid = 0.5 * (z_low + z_high)
    log_pdf_mid = -0.5 * z_mid**2 - _LOG_SQRT_2PI
    width = hi - lo
    approx_logp = log_pdf_mid + torch.log(width + 1e-30) + torch.log(precision)
    
    # 2. Standard computation: log(Φ(z_high) - Φ(z_low))
    # Mask out extreme cases by replacing with safe dummy values
    safe_z_high_std = z_high * (1.0 - is_extreme) + is_extreme * 1.0
    safe_z_low_std = z_low * (1.0 - is_extreme) + is_extreme * 0.0
    log_cdf_high = _log_ndtr(safe_z_high_std)
    log_cdf_low = _log_ndtr(safe_z_low_std)
    standard = _logsubexp(log_cdf_high, log_cdf_low) - log_Z
    
    # 3. Complement computation: log(Φ(-z_low) - Φ(-z_high))
    # Mask out extreme cases by replacing with safe dummy values
    safe_z_low_comp = z_low * (1.0 - is_extreme) + is_extreme * 0.0
    safe_z_high_comp = z_high * (1.0 - is_extreme) + is_extreme * 1.0
    log_sf_low = _log_ndtr(-safe_z_low_comp)
    log_sf_high = _log_ndtr(-safe_z_high_comp)
    complement = _logsubexp(log_sf_low, log_sf_high) - log_Z
    
    # === Arithmetic combination of all three branches ===
    # First combine standard and complement for non-extreme cases
    non_extreme_value = complement * use_complement + standard * (1.0 - use_complement)
    
    # Then combine with extreme approximation
    result = approx_logp * is_extreme + non_extreme_value * (1.0 - is_extreme)
    
    return result

def _entropy_truncated(center: Tensor, precision: Tensor, alpha: Tensor, beta: Tensor,
                      log_Z: Tensor) -> Tensor:
    """Compute entropy for truncated Gaussian."""
    phi = lambda t: torch.exp(-0.5 * t.square()) / math.sqrt(_TWO_PI)
    num = alpha * phi(alpha) - beta * phi(beta)
    return (-torch.log(precision) + 0.5 * math.log(_TWO_PI * math.e) + 
            num / torch.exp(log_Z) - log_Z)

# def _logsubexp(a: Tensor, b: Tensor) -> Tensor:
#     """
#     Stable log(exp(a) - exp(b)) with a ≥ b.
#     Uses arithmetic operations to avoid torch.where gradient issues.
#     """
#     a32, b32 = a.to(torch.float32), b.to(torch.float32)
#     diff = b32 - a32  # This is ≤ 0
    
#     eps = torch.finfo(torch.bfloat16).eps
#     use_series = (diff.abs() < eps).float()  # Convert to float for arithmetic
    
#     # Compute both branches with safe fallbacks
#     # For series: when |diff| < eps, use approximation
#     # Add small epsilon to avoid log(0) issues
#     safe_diff_series = diff - (1.0 - use_series) * 1.0  # Make diff = diff - 1 when not using series
#     series_result = a32 + torch.log(-safe_diff_series + 1e-30)
    
#     # For standard: when |diff| >= eps
#     # Clamp diff to avoid exp overflow in unused branch
#     safe_diff_standard = diff * (1.0 - use_series) + use_series * (-10.0)
#     standard_result = a32 + torch.log1p(-torch.exp(safe_diff_standard))
    
#     # Combine using arithmetic (not torch.where)
#     out32 = series_result * use_series + standard_result * (1.0 - use_series)
#     return out32.to(a.dtype)

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
        """
        log P(lo ≤ X ≤ hi) using arithmetic masking for torch.compile compatibility.
        """
        z_low = (lo - self.center) * self.prec
        z_high = (hi - self.center) * self.prec
        
        # Detect extreme cases (convert to float for arithmetic)
        is_extreme = (((z_low > 4.0) & (z_high > 4.0)) | 
                    ((z_low < -4.0) & (z_high < -4.0))).float()
        
        # Detect which formulation to use for non-extreme cases
        use_complement = ((z_low > -z_high) & (z_low > 0)).float()
        
        # === Compute all three branches with safe fallbacks ===
        
        # 1. Extreme case approximation
        z_mid = 0.5 * (z_low + z_high)
        log_pdf_mid = -0.5 * z_mid**2 - _LOG_SQRT_2PI
        width = hi - lo
        approx_logp = log_pdf_mid + torch.log(width + 1e-30) + torch.log(self.prec)
        
        # 2. Standard computation: log(Φ(z_high) - Φ(z_low))
        # Mask out extreme cases by replacing with safe dummy values
        safe_z_high_std = z_high * (1.0 - is_extreme) + is_extreme * 1.0
        safe_z_low_std = z_low * (1.0 - is_extreme) + is_extreme * 0.0
        log_cdf_high = _log_ndtr(safe_z_high_std)
        log_cdf_low = _log_ndtr(safe_z_low_std)
        standard = _logsubexp(log_cdf_high, log_cdf_low) - self._log_Z
        
        # 3. Complement computation: log(Φ(-z_low) - Φ(-z_high))
        # Mask out extreme cases by replacing with safe dummy values
        safe_z_low_comp = z_low * (1.0 - is_extreme) + is_extreme * 0.0
        safe_z_high_comp = z_high * (1.0 - is_extreme) + is_extreme * 1.0
        log_sf_low = _log_ndtr(-safe_z_low_comp)
        log_sf_high = _log_ndtr(-safe_z_high_comp)
        complement = _logsubexp(log_sf_low, log_sf_high) - self._log_Z
        
        # === Arithmetic combination of all three branches ===
        # First combine standard and complement for non-extreme cases
        non_extreme_value = complement * use_complement + standard * (1.0 - use_complement)
        
        # Then combine with extreme approximation
        result = approx_logp * is_extreme + non_extreme_value * (1.0 - is_extreme)
        
        return result

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
        num_distinct_values = (max_values - min_values + 1)
        self.register_buffer('precision_ceiling', num_distinct_values * 2) # Corresponds to integer bin containing \pm 2-sigma
        self.register_buffer('rangeP1', max_values - min_values + 1, persistent=False)
        
    # @torch.compile(fullgraph=True, mode="max-autotune")
    def forward(self, x):
        output = self.actor(x)
        mean, precision = output[:, :self.n_actors], output[:, self.n_actors:]
        mean = torch.sigmoid(mean)
        precision = nn.functional.softplus(
            nn.functional.leaky_relu(precision, negative_slope=0.1) + 1.5) + 0.5 
        # Soft-cap at precision ceiling. 
        precision = -nn.functional.softplus(-precision + self.precision_ceiling) + self.precision_ceiling
        return mean, precision
    
    def _integer_samples_from_unit_samples(self, unit_samples):
        unit_samples = unit_samples.clamp(self.eps, 1.0 - self.eps)
        integer_samples = (unit_samples * self.rangeP1 + self.min_values - 0.5).clamp(self.min_values, self.max_values).round().int()
        return integer_samples
    
    def _unit_interval_of_integer_samples(self, integer_samples):
        unit_samples_ub = ((integer_samples + 0.5) + 0.5 - self.min_values) / self.rangeP1
        unit_samples_lb = ((integer_samples - 0.5) + 0.5 - self.min_values) / self.rangeP1
        return unit_samples_lb, unit_samples_ub
    
    # @torch.compile(fullgraph=False, mode="max-autotune-no-cudagraphs")
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
        # Compute truncation parameters
        alpha, beta, log_F_alpha, log_F_beta, log_Z = _compute_truncation_params(center, prec)
        # Sample from truncated Gaussian
        unit_samples = _sample_truncated_gaussian(center, prec, uniform_samples, log_F_alpha, log_F_beta)
        integer_samples = self._integer_samples_from_unit_samples(unit_samples)
        unit_lb, unit_ub = self._unit_interval_of_integer_samples(integer_samples)
        # Logprobs needs to be appropriately scaled
        logprobs = _logp_interval_truncated(unit_lb, unit_ub, center, prec, log_Z) - self.rangeP1.log() 

        # Approximating discrete entropy by continuous differential entropy, so we need to scale by the range
        entropy = _entropy_truncated(center, prec, alpha, beta, log_Z) + self.rangeP1.log()
        return {
            'samples': integer_samples, # [B, n_actors] int. A batch of samples
            'logprobs': logprobs, # [B, n_actors] float. Logprobs of the samples
            'entropy': entropy, # [B] float. Entropy of the computed distribution
            'dist_params': { # Distribution parameters
                'center': center, # [B, n_actors] float
                'precision': prec, # [B, n_actors] float
            }}
    
    # @torch.compile(fullgraph=True, mode="max-autotune")
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
        # Compute truncation parameters
        alpha, beta, log_F_alpha, log_F_beta, log_Z = _compute_truncation_params(center, prec)

        unit_lb, unit_ub = self._unit_interval_of_integer_samples(integer_samples)
        logprobs = _logp_interval_truncated(unit_lb, unit_ub, center, prec, log_Z) - self.rangeP1.log() 
        entropy = _entropy_truncated(center, prec, alpha, beta, log_Z) + self.rangeP1.log()
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
"""
Parabolic (Epanechnikov) distribution-based discrete action sampler for RL with continuous parameterization. 

Use case: action space is a discretization of continuous interval (e.g. bid and ask prices / size). 

Key components:
- ParabolicVariableWidthDistribution: Epanechnikov distribution with adjustable width and center
- ParabolicFullSupportDistribution: Parabolic covering full [0,1] range  
- ParabolicFullSupportUniformMixture: Mixture of parabolic and uniform for exploration
- ParabolicActionDistribution: Hierarchical mixture of above distributions
- DiscreteActor: NN module that outputs distribution parameters and samples discrete actions

The approach maps continuous distributions to discrete actions via:
1. Sample from continuous distribution in [0,1]
2. Scale to action range and round to nearest integer
3. Compute log probabilities over discrete intervals

Mathematical advantages over triangular:
- Epanechnikov kernel is theoretically optimal for density estimation
- Smooth gradients at boundaries (no singularities)
- Better numerical stability during training
"""
# %%

import torch 
import math 
import torch.nn as nn 

# Parabolic distribution constants
_PARABOLIC_EPS = 1.0e-12        # protects log/√ near the boundary
_LOG_3_OVER_4 = math.log(3.0/4.0)
# ∫_{-1}^{1} ¾(1-z²)·ln[¾(1-z²)] dz  ≈ -0.567996…  (derived analytically)
_PARABOLIC_ENTROPY_CONST = 0.567996

def inv_sigmoid(x):
    return math.log(x) - math.log(1 - x)

class ParabolicVariableWidthDistribution:
    r"""
    Epanechnikov kernel centred at *center* with half-width *h*:

        f(x) = ¾ · (1 - ((x-c)/h)²) / h,      |x-c| ≤ h
              = 0                             otherwise

    The pdf integrates to one, is C¹ at the peak and at the two edges, and
    vanishes outside the support - exactly mirroring the triangle helper that
    it replaces.
    """
    def __init__(self, center: torch.Tensor, half_width: torch.Tensor):
        # clamp to keep gradients finite
        self.center     = torch.clamp(center,     0.0, 1.0)
        half_width      = torch.clamp(half_width, 1.0e-6, 0.5)
        # also forbid the support from leaving [0,1]
        self.half_width = torch.min(
            half_width,
            torch.min(self.center, 1.0 - self.center) - 1.0e-6
        )
        self.min_val = self.center - self.half_width
        self.max_val = self.center + self.half_width
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0.0, 1.0)                       # clip to unit box
        z = (x - self.center) / self.half_width            # ∈ (-1,1)
        inside = (z.abs() <= 1.0)
        log_pdf = (_LOG_3_OVER_4 - torch.log(self.half_width)
            + torch.log1p(-z.pow(2) + _PARABOLIC_EPS))
        return torch.where(inside, log_pdf, torch.full_like(log_pdf, -100.))
    
    @torch.no_grad()
    def sample(self, uniform_u: torch.Tensor) -> torch.Tensor:
        """
        Inverse-transform sampling.

        For |z| ≤ 1, CDF(z) = (2 + 3z - z³) / 4       (Devroye & Györfi 1985).
        Solve the depressed cubic  z³ - 3z + (4u-2)=0  via the trigonometric
        form that is numerically stable for all u∈(0,1):

            z = 2 cos( (arccos(1-2u)/3) - 2π/3 ).
        """
        u = torch.clamp(uniform_u, _PARABOLIC_EPS, 1.0 - _PARABOLIC_EPS)
        theta = (torch.arccos(1.0 - 2.0 * u) / 3.0) - (2.0 * math.pi / 3.0)
        z = 2.0 * torch.cos(theta)                         # ∈ (-1,1)
        return self.center + self.half_width * z
    
    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.clamp((x - self.center) / self.half_width, -1.0, 1.0)
        Fz = (2.0 + 3.0*z - z.pow(3)) / 4.0                # CDF on (-1,1)
        return torch.where(
            x < self.min_val, torch.zeros_like(x),
            torch.where(x > self.max_val, torch.ones_like(x), Fz))
    
    def entropy(self) -> torch.Tensor:
        #  H[X] = K + ln(h),   K ≈ 0.567996…  (see derivation below)
        return _PARABOLIC_ENTROPY_CONST + torch.log(self.half_width)
    
class ParabolicFullSupportDistribution:
    r"""
    Quadratic (Epanechnikov-type) kernel that *always* covers [0,1] yet still
    peaks at a movable *center* c:

        f(x) = ³⁄₂ · (1 - ((c-x)/c)²)       if x ≤ c            (left branch)
             = ³⁄₂ · (1 - ((x-c)/(1-c))²)   if x ≥ c            (right branch)

    It integrates to one because ∫₀ᶜ … dx = c and ∫ᶜ¹ … dx = 1-c.
    """
    def __init__(self, center: torch.Tensor):
        self.center = torch.clamp(center, 1.0e-6, 1.0 - 1.0e-6)
        # entropy is constant (support length fixed); pre-store for speed
        self.entropy_value = (
            _PARABOLIC_ENTROPY_CONST + math.log(0.5)       # h = 0.5
        )
        placeholder = torch.ones_like(center) * 0.5
        self.fullwidth_equivalent = ParabolicVariableWidthDistribution(placeholder, placeholder)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, _PARABOLIC_EPS, 1.0 - _PARABOLIC_EPS)
        c  = self.center
        left  = x <= c
        scale = torch.where(left, c, 1.0 - c)
        diff  = torch.where(left, c - x, x - c)
        log_pdf = (math.log(1.5)                                  # ln(3/2)
            + torch.log1p(- (diff / scale).pow(2) + _PARABOLIC_EPS))
        return log_pdf
    
    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Exact CDF for the full-support Epanechnikov kernel.

        • left branch  (x ≤ c):
            F = 1.5 c (z² − z³/3),     z = x / c
        • right branch (x ≥ c):
            F = c + 1.5 (1−c) (t − t³/3),  t = (x−c)/(1−c)
        """
        c   = self.center
        x   = torch.clamp(x, 0.0, 1.0)

        left_mask = x <= c

        # ----- left side -------------------------------------------------------
        z   = x / c.clamp_min(_PARABOLIC_EPS)          # safe divide
        F_l = 1.5 * c * (z*z - z*z*z / 3.0)

        # ----- right side ------------------------------------------------------
        t   = (x - c) / (1.0 - c).clamp_min(_PARABOLIC_EPS)
        F_r = c + 1.5 * (1.0 - c) * (t - t*t*t / 3.0)

        return torch.where(left_mask, F_l, F_r)

    def entropy(self) -> torch.Tensor:
        # constant tensor, broadcast to match input batch if necessary
        return torch.zeros_like(self.center) + self.entropy_value
    
    @torch.no_grad()
    def sample(self, uniform_u: torch.Tensor) -> torch.Tensor:
        """
        Inverse CDF for the piece‑wise Epanechnikov PDF on [0,1].

        ─ Left branch (u ≤ c) ──────────────────────────────────────────────
        write  k = u / c  ∈ [0,1]   and  z = x / c  ∈ [0,1]

              k = 1.5 z² − 0.5 z³                    (†)

        Put v = z − 1   ⇒   v³ − 3 v + 2 (1−k) = 0.  
        The root we need is

            v = 2 cos( arccos(1−k)/3 − 2π/3 ),      v ∈ [‑1,0]

        then   z = v + 1  and   x = c z.

        ─ Right branch (u ≥ c) ─────────────────────────────────────────────
        Symmetric:  let  k = (u−c)/(1−c),  solve the same cubic,
        finally  x = c + (1−c) z.

        This yields a strictly monotone mapping u ↦ x and guarantees
        x ∈ [0,1] for every u ∈ (0,1).
        """
        c  = self.center
        u  = torch.clamp(uniform_u, _PARABOLIC_EPS, 1.0 - _PARABOLIC_EPS)

        # -------- left part -------------------------------------------------
        mask_left = u <= c
        k_left    = torch.where(mask_left, u / c, torch.zeros_like(u))
        m_left    = 1.0 - k_left                                           #  m = 1 − k
        theta_l   = torch.arccos(m_left) / 3.0 - 2.0 * math.pi / 3.0
        v_left    = 2.0 * torch.cos(theta_l)                               #  v ∈ [‑1,0]
        z_left    = 1.0 + v_left                                           #  z ∈ [0,1]
        x_left    = c * z_left

        # -------- right part ------------------------------------------------
        k_right   = torch.where(mask_left, torch.zeros_like(u),(u - c) / (1.0 - c))
        m_right   = 1.0 - k_right
        theta_r   = torch.arccos(m_right) / 3.0 - 2.0 * math.pi / 3.0
        v_right   = 2.0 * torch.cos(theta_r)
        z_right   = 1.0 + v_right
        x_right   = c + (1.0 - c) * z_right
        x_right = -(x_right - (1 + c) / 2.) + (1 + c) / 2.

        return torch.where(mask_left, x_left, x_right)
    
def binary_entropy(x):
    """Binary entropy H(p) = -p*log(p) - (1-p)*log(1-p)"""
    return -torch.xlogy(x, x) - torch.xlogy(1 - x, 1 - x)

class ParabolicFullSupportUniformMixture:
    """
    Mixture of parabolic and uniform distributions.
    
    With probability uniform_epsilon: sample uniformly from [0,1]
    With probability 1-uniform_epsilon: sample from ParabolicFullSupport
    
    Creates a smooth mixture that interpolates between parabolic and uniform.
    """
    def __init__(self, center, uniform_epsilon):
        """
        center: Peak location for parabolic component
        uniform_epsilon: Mixing weight for uniform component
        """
        self.parabolic = ParabolicFullSupportDistribution(center)
        self.uniform_epsilon = uniform_epsilon

    def log_prob(self, x):
        return torch.logaddexp(
            self.parabolic.log_prob(x) + torch.log1p(-self.uniform_epsilon),
            torch.log(self.uniform_epsilon))
    
    def cdf(self, x):
        return self.uniform_epsilon * x + (1 - self.uniform_epsilon) * self.parabolic.cdf(x)
    
    def sample(self, uniform_samples):
        return torch.where(
            uniform_samples[..., 0] < self.uniform_epsilon,
            uniform_samples[..., 1],
            self.parabolic.sample(uniform_samples[..., 1]))
    
    def entropy(self):
        """
        Returns an estimate of entropy as mixture of entropy + 0.5 * binary entropy of epsilon-Bernoulli
        """
        return ((1 - self.uniform_epsilon) * self.parabolic.entropy()
            + 0.5 * binary_entropy(self.uniform_epsilon))
    
class ParabolicActionDistribution:
    """
    Hierarchical mixture model for action sampling using parabolic distributions.
    
    Two-level mixture:
    1. With probability (1-epsilon_fullsupport): sample from concentrated ParabolicVariableWidth
    2. With probability epsilon_fullsupport: sample from ParabolicFullSupportUniformMixture
    
    The ParabolicFullSupportUniformMixture itself mixes parabolic and uniform based on epsilon_uniform.
    This creates a flexible distribution with smoother gradients than triangular.
    """
    def __init__(self, center, half_width, epsilon_fullsupport, epsilon_uniform):
        """
        center: Mode of distributions, shape [...] in (0, 1)
        half_width: Width of concentrated distribution, shape [...] in (0, 0.5)
        epsilon_fullsupport: Prob of using full-support distribution, shape [...] in (0, 1)
        epsilon_uniform: Uniform mixing weight in full-support dist, shape [...] in (0, 1)
        """
        self.main = ParabolicVariableWidthDistribution(center, half_width)
        self.support = ParabolicFullSupportUniformMixture(center, epsilon_uniform)
        self.epsilon_fullsupport = epsilon_fullsupport

    def sample(self, uniform_samples):
        """
        Note that uniform_samples needs shape [..., 3]
        """
        return torch.where(
            uniform_samples[..., 0] < self.epsilon_fullsupport, 
            self.support.sample(uniform_samples[..., 1:]),
            self.main.sample(uniform_samples[..., 1])) # Sample from the main distribution
        
    def log_prob(self, x):
        return torch.logaddexp(
            self.main.log_prob(x) + torch.log1p(-self.epsilon_fullsupport),
            self.support.log_prob(x) + torch.log(self.epsilon_fullsupport))
    
    def logp_interval(self, lower_bound, upper_bound):
        return torch.log(self.cdf(upper_bound) - self.cdf(lower_bound)).clamp(min=-100.)
    
    def cdf(self, x):
        return (
            self.epsilon_fullsupport * self.support.cdf(x) 
            + (1 - self.epsilon_fullsupport) * self.main.cdf(x))
    
    def entropy(self):
        """
        Returns an approximation of the entropy. 
        
        Note: Parabolic distributions have higher base entropy than triangular,
        so you may need to reduce entropy_coef by 20-30% when switching.
        """
        return (
            self.epsilon_fullsupport * self.support.entropy() 
            + (1 - self.epsilon_fullsupport) * self.main.entropy()
            + 0.75 * binary_entropy(self.epsilon_fullsupport))

class DiscreteActor(nn.Module):
    """
    Neural network module for discrete action sampling using parabolic distributions.
    
    Architecture:
    - Input: hidden state [B, n_hidden]
    - Linear layer outputs 4*n_actors values through squishing to [0, 1]
    - Splits into: center, half_width, epsilon_fullsupport, epsilon_uniform
    - Creates ParabolicActionDistribution and samples discrete actions
    
    Handles discrete action spaces by:
    1. Sampling continuous values in [0,1]
    2. Scaling to [min_values, max_values] and rounding
    3. Computing log-probs over discrete intervals
    
    Benefits over triangular:
    - Smoother gradients during training
    - Better numerical stability
    - Theoretically optimal kernel for density estimation
    
    **Important!**
    Ideally pair with learning rate warmup and smaller LR!!, else it's very easy to observe:
    1. Models output bad actions which receive negative gradients. 
    2. Gradients quickly push epsilon_fullsupport and epsilon_uniform to 1, which makes the model output uniform actions. 
    3. Value network receives totally useless signals, and model collapses. 
    
    Note: When switching from triangular, reduce entropy_coef by 20-30% due to higher base entropy.
    """
    def __init__(self, n_hidden, n_actors, 
        min_values: torch.Tensor, max_values: torch.Tensor, 
        eps=1e-4, eps_logic_bias=6., eps_logic_inv_scale=25.):
        """
        n_hidden: Input feature dimension
        n_actors: Number of parallel action distributions 
        min_values, max_values: Inclusive bounds per actor, shape [n_actors]
        eps: float. Prevents rounded result exceeding [min_values, max_values]. Introduces small bias to the rounded result. 
        eps_logic_bias: float. Bias for epsilon_fullsupport and epsilon_uniform. 
        eps_logic_inv_scale: float. Inverse scale for epsilon_fullsupport and epsilon_uniform. 
        """
        super().__init__()
        self.n_hidden = n_hidden
        self.n_actors = n_actors
        self.actor = nn.Linear(n_hidden, n_actors * 4)
        assert min_values.shape == max_values.shape, f"min_values and max_values must have the same shape, but got {min_values.shape} and {max_values.shape}"
        assert min_values.ndim == 1, f"min_values and max_values must be 1D, but got {min_values.ndim} and {max_values.ndim}"
        assert min_values.shape[0] == n_actors, f"min_values and max_values must have length {n_actors}, but got {min_values.shape[0]} and {max_values.shape[0]}"
        self.eps = eps 
        self.register_buffer('min_values', min_values, persistent=False)
        self.register_buffer('max_values', max_values, persistent=False)
        self.register_buffer('rangeP1', max_values - min_values + 1, persistent=False)
        self.eps_logic_bias = eps_logic_bias
        self.eps_logic_inv_scale = eps_logic_inv_scale
        
    #@#torch.compile(fullgraph=True, mode="max-autotune")
    def forward(self, x):
        output = self.actor(x)
        center, half_width, epsilon_fullsupport, epsilon_uniform = (
            output[:, :self.n_actors], 
            output[:, self.n_actors:2 * self.n_actors],  
            output[:, 2 * self.n_actors:3 * self.n_actors],
            output[:, 3 * self.n_actors:])
        center = torch.sigmoid(center)
        half_width = torch.sigmoid(half_width) * 0.5
        # Very important! Systemically bias weights to supporting mixtures at the beginning, and make it learn slower. 
        # Helps value-net learn more reliably. 
        # Cannot replace with initializing with different initializations (!) since gradient calculations are different (biases are easy to change)
        epsilon_fullsupport = torch.sigmoid(epsilon_fullsupport / self.eps_logic_inv_scale - self.eps_logic_bias)
        epsilon_uniform = torch.sigmoid(epsilon_uniform / self.eps_logic_inv_scale - self.eps_logic_bias)
        return center, half_width, epsilon_fullsupport, epsilon_uniform
    
    def _integer_samples_from_unit_samples(self, unit_samples):
        integer_samples = (unit_samples * (
            self.rangeP1 - 2 * self.eps) + self.min_values - 0.5 + self.eps
        ).round().clamp(min=self.min_values, max=self.max_values).int()
        return integer_samples
    
    def _unit_interval_of_integer_samples(self, integer_samples):
        unit_samples_ub = ((integer_samples + 0.5) + 0.5 - self.min_values) / self.rangeP1
        unit_samples_lb = ((integer_samples - 0.5) + 0.5 - self.min_values) / self.rangeP1
        return unit_samples_lb, unit_samples_ub
    
    def logp_entropy_and_sample(self, x, uniform_samples):
        """
        x: [B, n_hidden]
        uniform_samples: [B, n_actors, 3] float

        Returns: 
        """
        center, half_width, epsilon_fs, epsilon_uniform = self(x)
        dist = ParabolicActionDistribution(center, half_width, epsilon_fs, epsilon_uniform)
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
                'half_width': half_width, # [B, n_actors] float
                'epsilon_fullsupport': epsilon_fs, # [B, n_actors] float
                'epsilon_uniform': epsilon_uniform, # [B, n_actors] float
            }}
    
    def logp_entropy(self, x, integer_samples):
        """
        x: [B, n_hidden]
        samples: [B, n_actors] float in (0, 1)

        Returns: 
        - logprobs: [B, n_actors] float. Logprobs of the samples
        - entropy: [B] float. Entropy of the computed distribution
        """
        center, half_width, epsilon_fs, epsilon_uniform = self(x)
        dist = ParabolicActionDistribution(center, half_width, epsilon_fs, epsilon_uniform)

        unit_lb, unit_ub = self._unit_interval_of_integer_samples(integer_samples)
        logprobs = dist.logp_interval(unit_lb, unit_ub) - self.rangeP1.log() 
        entropy = dist.entropy() + self.rangeP1.log()
        return {
            'logprobs': logprobs, # [B, n_actors] float. Logprobs of the samples
            'entropy': entropy, # [B] float. Entropy of the computed distribution
        }
    
if __name__ == '__main__':
    # %%
    import matplotlib.pyplot as plt

    num_samples = 1024000
    center = torch.zeros((num_samples,), device='cuda:0') + 0.25
    half_width = torch.ones((num_samples,), device='cuda:0') * 0.5
    epsilon_fullsupport = torch.ones((num_samples,), device='cuda:0') * 0.6
    epsilon_uniform = torch.ones((num_samples,), device='cuda:0') * 1.

    uniform_samples = torch.rand(num_samples, 3, device='cuda:0')

    # dist = ParabolicActionDistribution(center, half_width, epsilon_fullsupport, epsilon_uniform)
    # dist = ParabolicVariableWidthDistribution(center, half_width)
    dist = ParabolicFullSupportDistribution(center)
    samples = dist.sample(uniform_samples[..., 0])
    # dist = ParabolicFullSupportUniformMixture(center, epsilon_uniform)
    plt.hist(samples.cpu().numpy(), bins=100, density=True)
    x = torch.linspace(0, 1, num_samples, device='cuda:0')
    pdf_value = dist.log_prob(x).exp()
    plt.plot(x.cpu().numpy(), pdf_value.cpu().numpy())
    plt.show()

    numerical_difference = (dist.cdf(x) - pdf_value.cumsum(dim=0) / num_samples).abs()
    plt.plot(dist.cdf(x).cpu().numpy() + 0.01, label='cdf')
    plt.plot(pdf_value.cumsum(dim=0).cpu().numpy() / num_samples, label='cdf approx')
    plt.plot(x.cpu().numpy(), numerical_difference.cpu().numpy(), label='difference')
    plt.title(f'max difference: {numerical_difference.max().item():.6f}')
    plt.legend()
    plt.show()

    # %%

    batch_size = 102400
    num_actors = 10
    center = torch.zeros((batch_size, num_actors), device='cuda:0') + 0.5
    half_width = torch.ones((batch_size, num_actors), device='cuda:0') * 0.1
    epsilon_fullsupport = torch.ones((batch_size, num_actors), device='cuda:0') * 0.3
    epsilon_uniform = torch.ones((batch_size, num_actors), device='cuda:0') * 0.4
    uniform_samples = torch.rand(batch_size, num_actors, 3, device='cuda:0')

    min_values = torch.zeros((num_actors,), device='cuda:0')
    max_values = torch.zeros((num_actors,), device='cuda:0') + 2000
    rangeP1 = max_values - min_values + 1

    dist = ParabolicActionDistribution(center, half_width, epsilon_fullsupport, epsilon_uniform)
    # dist = ParabolicVariableWidthDistribution(center, half_width)
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
            'half_width': half_width, # [B, n_actors] float
            'epsilon_fullsupport': epsilon_fullsupport, # [B, n_actors] float
            'epsilon_uniform': epsilon_uniform, # [B, n_actors] float
        }}

    output_samples = output['samples']
    plt.hist(output_samples.cpu().numpy().flatten(), bins=100, density=True)
    discrete_probs = torch.tensor([
        (output_samples == i).float().mean().item() 
        for i in range(min_values[0].long().item(), max_values[0].long().item() + 1)])

    entropy = -torch.xlogy(discrete_probs, discrete_probs).sum()
    # Roughly accurate 
    print('Entropy:', entropy.item(), 'guess:', output['entropy'][0, 0].item())
    # %%

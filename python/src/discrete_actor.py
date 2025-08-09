"""
Truncated Gaussian-based discrete action sampler for reinforcement learning.

This module provides a differentiable approach to sampling from discrete action spaces
that represent discretizations of continuous intervals (e.g., bid/ask prices, order 
sizes in trading environments).

Key Components:
--------------
1. GaussianActionDistribution: 
   - Implements a truncated Normal distribution on the open interval (0, 1)
   - Parameterized by center (mean) and precision (1/std)
   - Provides exact log probabilities for discrete intervals
   
2. DiscreteActor:
   - Neural network module that outputs distribution parameters
   - Samples discrete actions from the truncated Gaussian
   - Computes log probabilities and entropy for policy gradient methods

The Sampling Process:
--------------------
1. Neural network outputs center ∈ (0,1) and precision > 0
2. Sample from truncated Gaussian N(center, 1/precision²) on (0,1)
3. Scale sample to action range [min_val, max_val]
4. Round to nearest integer to get discrete action
5. Compute exact log probability over the discrete interval

Implementation Notes:
--------------------
- Uses arithmetic masking instead of torch.where to avoid NaN gradient propagation
- All computations are numerically stable using log-space operations
- Supports batched operations and multiple parallel actors
- Compatible with torch.compile for performance optimization

Example Usage:
-------------
>>> actor = DiscreteActor(n_hidden=256, n_actors=2, 
...                      min_values=torch.tensor([0, 10]),
...                      max_values=torch.tensor([100, 50]))
>>> hidden_state = torch.randn(32, 256)  # batch_size=32
>>> uniform_samples = torch.rand(32, 2)
>>> result = actor.logp_entropy_and_sample(hidden_state, uniform_samples)
>>> actions = result['samples']  # shape: [32, 2], discrete actions
>>> log_probs = result['logprobs']  # shape: [32, 2], log probabilities
"""

# %%

import torch.nn as nn 
import torch, math
from torch import Tensor
from typing import Tuple
from model_components import layer_init 

_TWO_PI = 2.0 * math.pi
_LOG_SQRT_2PI = 0.5 * math.log(_TWO_PI)

def _log_normal_pdf_prec(x: Tensor, loc: Tensor, prec: Tensor) -> Tensor:
    """
    Compute log PDF of Normal distribution parameterized by precision.
    
    Args:
        x: Points at which to evaluate the PDF
        loc: Mean/center of the distribution
        prec: Precision (1/std) of the distribution
        
    Returns:
        Log probability density at x
        
    Note:
        Using precision parameterization: N(x | μ, σ²) where prec = 1/σ
        log p(x) = -0.5 * ((x-μ)/σ)² - log(σ√(2π))
                 = -0.5 * ((x-μ)*prec)² - log(√(2π)) + log(prec)
    """
    z = (x - loc) * prec
    return -0.5 * z.square() - _LOG_SQRT_2PI + torch.log(prec)

def _log_ndtr(z: Tensor) -> Tensor:
    """
    Compute log of the standard normal cumulative distribution function.
    
    Args:
        z: Standardized values (z-scores)
        
    Returns:
        log(Φ(z)) where Φ is the standard normal CDF
        
    Note:
        This is a numerically stable version that avoids underflow for large negative z.
    """
    return torch.special.log_ndtr(z)

def _logsubexp(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute log(exp(a) - exp(b)) in a numerically stable way.
    
    Args:
        a: First log value (must satisfy a ≥ b)
        b: Second log value
        
    Returns:
        log(exp(a) - exp(b))
        
    Mathematical Derivation:
        log(exp(a) - exp(b)) = log(exp(a) * (1 - exp(b-a)))
                             = a + log(1 - exp(b-a))
                             
    Stability:
        Since b ≤ a, we have b-a ≤ 0, so exp(b-a) ∈ (0, 1]
        This ensures log1p(-exp(diff)) is well-defined and stable.
    """
    diff = b - a
    return a + torch.log1p(-torch.exp(diff))

class GaussianActionDistribution:
    """
    Truncated Normal distribution on the open interval (0, 1).
    
    This class implements a Normal distribution N(center, σ²) truncated to (0, 1),
    parameterized by precision (prec = 1/σ) instead of standard deviation.
    
    The truncated distribution has PDF:
        p(x) = φ((x-μ)/σ) / (σ * Z)  for x ∈ (0, 1)
        p(x) = 0                      otherwise
        
    where:
        - φ is the standard normal PDF
        - Z = Φ((1-μ)/σ) - Φ((0-μ)/σ) is the normalization constant
        - Φ is the standard normal CDF
        
    Attributes:
        center: Mean parameter μ of the underlying normal distribution
        prec: Precision parameter (1/σ) of the underlying normal distribution
        _log_Z: Log of the normalization constant
        _F_alpha, _F_beta: CDF values at truncation boundaries
        _log_F_alpha, _log_F_beta: Log CDF values (for numerical stability)
    """

    def __init__(self, center: Tensor, precision: Tensor):
        """
        Initialize truncated normal distribution.
        
        Args:
            center: Mean parameter μ ∈ ℝ of the underlying normal
            precision: Precision parameter (1/σ) > 0 of the underlying normal
            
        The distribution is truncated to the interval (0, 1).
        """
        self.center = center
        self.prec   = precision  # 1 / σ
        
        # Compute standardized truncation boundaries
        # alpha = (0 - μ) / σ = (0 - μ) * precision
        # beta = (1 - μ) / σ = (1 - μ) * precision
        alpha = (0.0 - center) * precision
        beta  = (1.0 - center) * precision
        
        # Log CDF values at truncation points (for numerical stability)
        self._log_F_alpha = _log_ndtr(alpha)
        self._log_F_beta  = _log_ndtr(beta)
        
        # Plain CDF values (needed only for sampling via inverse transform)
        self._F_alpha = torch.exp(self._log_F_alpha)
        self._F_beta  = torch.exp(self._log_F_beta)
        
        # Log normalization constant: log(Φ(beta) - Φ(alpha))
        self._log_Z = _logsubexp(self._log_F_beta, self._log_F_alpha)

    @torch.no_grad()
    def sample(self, uniform_samples: Tensor) -> Tensor:
        """
        Sample from truncated normal using inverse transform method.
        
        Args:
            uniform_samples: Uniform random samples in [0, 1]
            
        Returns:
            Samples from the truncated normal distribution in (0, 1)
            
        Algorithm:
            1. Map uniform sample u ∈ [0,1] to CDF range [F(0), F(1)]
            2. Apply inverse normal CDF to get z-score
            3. Transform back to original scale: x = μ + σ*z = μ + z/prec
            
        Note:
            Clamping prevents numerical issues at boundaries.
        """
        u = uniform_samples.clamp(1e-6, 1.0 - 1e-6)
        # Map u to the CDF range of the truncated distribution
        p = u * (self._F_beta - self._F_alpha) + self._F_alpha
        # Inverse standard normal CDF to get z-scores
        z = torch.special.ndtri(p)
        # Transform to original scale: x = z * std + mean = z / prec + center
        return z / self.prec + self.center

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Compute log probability density at x.
        
        Args:
            x: Points at which to evaluate the log PDF
            
        Returns:
            Log probability density log p(x)
            
        Note:
            For truncated distribution:
            log p(x) = log φ((x-μ)/σ) - log(σ) - log(Z)
                     = log_normal_pdf(x) - log(Z)
        """
        return _log_normal_pdf_prec(x, self.center, self.prec) - self._log_Z

    def log_cdf(self, x: Tensor) -> Tensor:
        """
        Compute log cumulative distribution function at x.
        
        Args:
            x: Points at which to evaluate the log CDF
            
        Returns:
            Log cumulative probability log P(X ≤ x)
            
        Formula:
            P(X ≤ x | X ∈ (0,1)) = [Φ((x-μ)/σ) - Φ((0-μ)/σ)] / Z
            
        where Z = Φ((1-μ)/σ) - Φ((0-μ)/σ)
        """
        z = (x - self.center) * self.prec
        return _logsubexp(_log_ndtr(z), self._log_F_alpha) - self._log_Z

    def logp_interval(self, lo: Tensor, hi: Tensor) -> Tensor:
        """
        Compute log probability of interval P(lo ≤ X ≤ hi).
        
        Args:
            lo: Lower bound of interval
            hi: Upper bound of interval
            
        Returns:
            log P(lo ≤ X ≤ hi)
            
        Implementation Notes:
            - Uses arithmetic masking instead of torch.where for torch.compile compatibility
            - Handles extreme z-scores (|z| > 4) with approximations to avoid numerical issues
            - Automatically selects between standard and complement formulations for stability
            
        Three computation branches:
            1. Extreme case (|z| > 4): Use midpoint approximation
            2. Standard: log(Φ(z_high) - Φ(z_low)) when z_low ≤ min(0, -z_high)
            3. Complement: log(Φ(-z_low) - Φ(-z_high)) when z_low > min(0, -z_high)
        """
        # Standardize interval bounds to z-scores
        z_low = (lo - self.center) * self.prec
        z_high = (hi - self.center) * self.prec
        
        # Detect extreme cases where both z-scores are far from 0
        # In these cases, the normal CDF is very close to 0 or 1
        is_extreme = (((z_low > 4.0) & (z_high > 4.0)) |  # Both far right
                    ((z_low < -4.0) & (z_high < -4.0))).float()  # Both far left
        
        # For non-extreme cases, choose formulation for numerical stability
        # Use complement form when z_low > max(0, -z_high) to avoid catastrophic cancellation
        use_complement = ((z_low > -z_high) & (z_low > 0)).float()
        
        # === Compute all three branches with safe fallbacks ===
        
        # === Branch 1: Extreme case approximation ===
        # When |z| >> 0, use rectangle approximation: P ≈ φ(z_mid) * width
        z_mid = 0.5 * (z_low + z_high)
        log_pdf_mid = -0.5 * z_mid**2 - _LOG_SQRT_2PI  # log φ(z_mid)
        width = hi - lo
        # log P ≈ log(φ(z_mid) * width / σ) = log φ(z_mid) + log(width) + log(prec)
        approx_logp = log_pdf_mid + torch.log(width + 1e-30) + torch.log(self.prec)
        
        # === Branch 2: Standard computation log(Φ(z_high) - Φ(z_low)) ===
        # Good when z_low ≤ 0 or when z_low and z_high have different signs
        # Replace extreme values with safe dummies to avoid NaN in unused branches
        safe_z_high_std = z_high * (1.0 - is_extreme) + is_extreme * 1.0
        safe_z_low_std = z_low * (1.0 - is_extreme) + is_extreme * 0.0
        log_cdf_high = _log_ndtr(safe_z_high_std)
        log_cdf_low = _log_ndtr(safe_z_low_std)
        standard = _logsubexp(log_cdf_high, log_cdf_low) - self._log_Z
        
        # === Branch 3: Complement computation log(Φ(-z_low) - Φ(-z_high)) ===
        # Good when both z_low and z_high are positive (right tail)
        # Uses the identity: P(z_low < Z < z_high) = P(Z > z_low) - P(Z > z_high)
        safe_z_low_comp = z_low * (1.0 - is_extreme) + is_extreme * 0.0
        safe_z_high_comp = z_high * (1.0 - is_extreme) + is_extreme * 1.0
        log_sf_low = _log_ndtr(-safe_z_low_comp)   # log P(Z > z_low)
        log_sf_high = _log_ndtr(-safe_z_high_comp) # log P(Z > z_high)
        complement = _logsubexp(log_sf_low, log_sf_high) - self._log_Z
        
        # === Arithmetic combination of all three branches ===
        # This avoids torch.where which can propagate NaN gradients
        
        # First, select between standard and complement for non-extreme cases
        non_extreme_value = complement * use_complement + standard * (1.0 - use_complement)
        
        # Then, select between extreme approximation and non-extreme computation
        result = approx_logp * is_extreme + non_extreme_value * (1.0 - is_extreme)
        
        return result

    def entropy(self) -> Tensor:
        """
        Compute differential entropy of the truncated normal distribution.
        
        Returns:
            Differential entropy H(X) = -∫ p(x) log p(x) dx
            
        Formula (from Michelot et al. 2011):
            H(X) = log(σ√(2πe)) + (α*φ(α) - β*φ(β))/Z - log(Z)
            
        where:
            - α = (0-μ)/σ, β = (1-μ)/σ are standardized truncation points
            - φ is the standard normal PDF
            - Z = Φ(β) - Φ(α) is the normalization constant
        """
        alpha = (0.0 - self.center) * self.prec
        beta  = (1.0 - self.center) * self.prec
        
        # Standard normal PDF
        phi   = lambda t: torch.exp(-0.5 * t.square()) / math.sqrt(_TWO_PI)
        
        # Entropy correction term from truncation
        num = alpha * phi(alpha) - beta * phi(beta)
        
        # Full entropy formula
        return (-torch.log(self.prec) + 0.5 * math.log(_TWO_PI * math.e) + 
                num / torch.exp(self._log_Z) - self._log_Z)
    


class DiscreteGaussianActor(nn.Module):
    """
    Neural network module for discrete action sampling using truncated Gaussians.
    
    This actor converts continuous neural network outputs into discrete actions
    by sampling from truncated Gaussian distributions and discretizing.
    
    Architecture:
        Input layer: [batch_size, n_hidden] -> hidden state from policy network
        Linear layer: n_hidden -> 2 * n_actors (outputs per actor: center, precision)
        Output processing:
            - First n_actors outputs -> sigmoid -> centers ∈ (0, 1)
            - Last n_actors outputs -> softplus -> precisions > 0
    
    Action Sampling Process:
        1. Forward pass produces (center, precision) for each actor
        2. Create truncated Gaussian distributions on (0, 1)
        3. Sample from distributions using uniform random numbers
        4. Scale samples to [min_values, max_values] and round to integers
        5. Compute log probabilities over discrete intervals
    
    Training Considerations:
        - Use learning rate warmup to prevent early collapse
        - Without warmup, the model may output extreme precisions -> deterministic actions
        - This leads to poor exploration and the model getting stuck
        - Precision is soft-capped to prevent numerical instability
    
    Attributes:
        n_hidden: Input feature dimension
        n_actors: Number of parallel action distributions
        actor: Linear layer for parameter prediction
        min_values: Minimum action values per actor
        max_values: Maximum action values per actor
        precision_ceiling: Soft cap for precision values
        rangeP1: max_values - min_values + 1 (number of discrete actions)
    """
    def __init__(self, n_hidden, n_actors, 
        min_values: torch.Tensor, max_values: torch.Tensor, eps=1e-4):
        """
        Initialize DiscreteActor module.
        
        Args:
            n_hidden: Input feature dimension from the policy network
            n_actors: Number of parallel action distributions (e.g., bid and ask)
            min_values: Minimum action values per actor, shape [n_actors]
            max_values: Maximum action values per actor, shape [n_actors]
            eps: Small constant to prevent sampling exactly 0 or 1 (default: 1e-4)
            
        Example:
            # Actor for bid/ask prices
            actor = DiscreteActor(
                n_hidden=256,
                n_actors=2,  # bid and ask
                min_values=torch.tensor([0, 0]),     # min prices
                max_values=torch.tensor([100, 100])  # max prices
            )
        """
        super().__init__()
        self.n_hidden = n_hidden
        self.n_actors = n_actors
        self.actor = nn.Linear(n_hidden, n_actors * 2)
        
        # Validate inputs
        assert min_values.shape == max_values.shape, f"min_values and max_values must have the same shape, but got {min_values.shape} and {max_values.shape}"
        assert min_values.ndim == 1, f"min_values and max_values must be 1D, but got {min_values.ndim} and {max_values.ndim}"
        assert min_values.shape[0] == n_actors, f"min_values and max_values must have length {n_actors}, but got {min_values.shape[0]} and {max_values.shape[0]}"
        
        self.eps = eps 
        
        # Register buffers (move with model to device, not saved in state_dict)
        self.register_buffer('min_values', min_values, persistent=False)
        self.register_buffer('max_values', max_values, persistent=False)
        
        # Number of discrete values per actor
        num_distinct_values = (max_values - min_values + 1)
        
        # Soft cap for precision to prevent numerical issues
        # Set to 2x the number of discrete values (≈ ±2σ covers one integer bin)
        self.register_buffer('precision_ceiling', num_distinct_values * 2)
        
        # Range plus one (used for scaling)
        self.register_buffer('rangeP1', max_values - min_values + 1, persistent=False)
        
    # @torch.compile(fullgraph=True, mode="max-autotune")
    def forward(self, x):
        """
        Forward pass to compute distribution parameters.
        
        Args:
            x: Hidden states from policy network, shape [batch_size, n_hidden]
            
        Returns:
            Tuple of (center, precision):
                - center: Mean parameters ∈ (0, 1), shape [batch_size, n_actors]
                - precision: Precision parameters > 0, shape [batch_size, n_actors]
                
        Processing pipeline:
            1. Linear layer: [B, n_hidden] -> [B, 2*n_actors]
            2. Split into center and precision raw values
            3. Center: sigmoid activation -> (0, 1)
            4. Precision: leaky_relu -> softplus -> soft cap
                - leaky_relu with bias helps avoid dead neurons
                - softplus ensures positive values
                - soft cap prevents extreme precisions
        """
        output = self.actor(x)
        mean, precision = output[:, :self.n_actors], output[:, self.n_actors:]
        
        # Map mean to (0, 1) using sigmoid
        mean = torch.sigmoid(mean)
        
        # Process precision with multiple stages:
        # 1. LeakyReLU + bias: helps gradient flow and shifts values
        # 2. Softplus: ensures positivity while maintaining smoothness  
        # 3. Add 0.5: ensures minimum precision
        precision = nn.functional.softplus(
            nn.functional.leaky_relu(precision, negative_slope=0.1) + 1.5) + 0.5 
            
        # Soft-cap precision to prevent numerical instability
        # Uses inverted softplus: -softplus(-x) → x as x → ∞, → 0 as x → -∞
        precision = -nn.functional.softplus(-precision + self.precision_ceiling) + self.precision_ceiling
        
        return mean, precision
    
    def _integer_samples_from_unit_samples(self, unit_samples):
        """
        Convert continuous samples from (0, 1) to discrete actions.
        
        Args:
            unit_samples: Continuous samples in (0, 1), shape [batch_size, n_actors]
            
        Returns:
            Integer actions in [min_values, max_values], shape [batch_size, n_actors]
            
        Discretization process:
            1. Clamp to (eps, 1-eps) to avoid boundary issues
            2. Scale: sample * (max - min + 1) maps (0, 1) to (0, max-min+1)
            3. Shift: add min_values - 0.5 to center on integers
            4. Round to nearest integer
            5. Clamp to ensure within bounds (handles floating point errors)
        """
        # Prevent exact 0 or 1 samples
        unit_samples = unit_samples.clamp(self.eps, 1.0 - self.eps)
        
        # Scale to action range and round
        # The -0.5 offset ensures uniform probability for boundary values
        integer_samples = (unit_samples * self.rangeP1 + self.min_values - 0.5).clamp(
            self.min_values, self.max_values).round().int()
        
        return integer_samples
    
    def _unit_interval_of_integer_samples(self, integer_samples):
        """
        Get the continuous interval in (0, 1) corresponding to discrete actions.
        
        Args:
            integer_samples: Discrete actions, shape [batch_size, n_actors]
            
        Returns:
            Tuple of (lower_bound, upper_bound) in (0, 1) for each action
            
        This computes the inverse of the discretization process:
        - Action k corresponds to the interval that would round to k
        - This is [k-0.5, k+0.5] in action space
        - Mapped back to (0, 1) for probability computation
        
        The +0.5 terms appear because:
        - integer ± 0.5 gives the rounding boundaries
        - Additional +0.5 aligns with the discretization offset
        """
        # Upper bound: maps k+0.5 back to unit interval
        unit_samples_ub = ((integer_samples + 0.5) + 0.5 - self.min_values) / self.rangeP1
        
        # Lower bound: maps k-0.5 back to unit interval  
        unit_samples_lb = ((integer_samples - 0.5) + 0.5 - self.min_values) / self.rangeP1
        
        return unit_samples_lb, unit_samples_ub
    
    @torch.compile(fullgraph=False, mode="max-autotune-no-cudagraphs")
    def logp_entropy_and_sample(self, x, uniform_samples):
        """
        Sample discrete actions and compute their log probabilities.
        
        Args:
            x: Hidden states from policy network, shape [batch_size, n_hidden]
            uniform_samples: Uniform random numbers for sampling, shape [batch_size, n_actors]
            
        Returns:
            Dictionary containing:
                - 'samples': Discrete actions, shape [B, n_actors], dtype int
                - 'logprobs': Log probabilities of sampled actions, shape [B, n_actors]
                - 'entropy': Entropy of distributions, shape [B]
                - 'dist_params': Dictionary with 'center' and 'precision'
                
        This is the main method used during rollout/action selection.
        
        Note:
            - Log probabilities are adjusted for discrete intervals
            - Entropy is approximated using continuous differential entropy
            - Both are scaled by log(rangeP1) to account for discretization
        """
        # Get distribution parameters from neural network
        center, prec = self(x)
        
        # Create truncated Gaussian distributions
        dist = GaussianActionDistribution(center, prec)
        
        # Sample continuous values in (0, 1)
        unit_samples = dist.sample(uniform_samples)
        
        # Convert to discrete actions
        integer_samples = self._integer_samples_from_unit_samples(unit_samples)
        
        # Get the continuous interval corresponding to each discrete action
        unit_lb, unit_ub = self._unit_interval_of_integer_samples(integer_samples)
        
        # Compute log probability over the discrete interval
        # Subtract log(rangeP1) to convert from interval probability to discrete probability
        logprobs = dist.logp_interval(unit_lb, unit_ub) - self.rangeP1.log() 

        # Approximate discrete entropy using differential entropy
        # Add log(rangeP1) because H(discrete) ≈ H(continuous) + log(Δx)
        entropy = dist.entropy() + self.rangeP1.log()
        
        return {
            'samples': integer_samples,      # [B, n_actors] int - Discrete actions
            'logprobs': logprobs,           # [B, n_actors] float - Log P(action)
            'entropy': entropy,             # [B] float - Entropy per distribution  
            'dist_params': {                # Distribution parameters for logging
                'center': center,           # [B, n_actors] float - Mean in (0,1)
                'precision': prec,          # [B, n_actors] float - 1/std
            }}
    
    @torch.compile(fullgraph=True, mode="max-autotune")
    def logp_entropy(self, x, integer_samples):
        """
        Compute log probabilities and entropy for given discrete actions.
        
        Args:
            x: Hidden states from policy network, shape [batch_size, n_hidden]
            integer_samples: Given discrete actions to evaluate, shape [batch_size, n_actors]
            
        Returns:
            Dictionary containing:
                - 'logprobs': Log probabilities of given actions, shape [B, n_actors]
                - 'entropy': Entropy of distributions, shape [B]
                - 'dist_params': Dictionary with 'center' and 'precision'
                
        This method is used during:
            - Policy gradient computation (evaluating old actions with new policy)
            - Importance sampling in off-policy algorithms
            - Computing KL divergence between policies
            
        Note:
            Unlike logp_entropy_and_sample, this doesn't sample new actions,
            it only evaluates the probability of given actions.
        """
        # Get distribution parameters from neural network
        center, prec = self(x)
        
        # Create truncated Gaussian distributions
        dist = GaussianActionDistribution(center, prec)

        # Get the continuous interval for each given discrete action
        unit_lb, unit_ub = self._unit_interval_of_integer_samples(integer_samples)
        
        # Compute log probability of the discrete action
        logprobs = dist.logp_interval(unit_lb, unit_ub) - self.rangeP1.log() 
        
        # Compute entropy (same for all actions from this distribution)
        entropy = dist.entropy() + self.rangeP1.log()
        
        return {
            'logprobs': logprobs,           # [B, n_actors] float - Log P(action)
            'entropy': entropy,             # [B] float - Entropy per distribution
            'dist_params': {                # Distribution parameters for logging
                'center': center,           # [B, n_actors] float - Mean in (0,1)
                'precision': prec,          # [B, n_actors] float - 1/std
            }}
    
class DiscreteSoftmaxActor(nn.Module):
    """
    Neural network module for discrete action sampling using softmax distributions.
    
    This actor directly parameterizes a categorical distribution over discrete actions,
    using PyTorch's Categorical distribution with fully vectorized operations.
    
    Architecture:
        Input layer: [batch_size, n_hidden] -> hidden state from policy network
        Linear layer: n_hidden -> n_actors * max_actions (logits for each action)
        Output processing:
            - Reshape to [batch_size, n_actors, max_actions]
            - Apply mask to valid actions (based on min/max values)
            - Create Categorical distributions for sampling
    
    Attributes:
        n_hidden: Input feature dimension
        n_actors: Number of parallel action distributions
        n_actions: Maximum number of actions (max of all actors)
        actor: Linear layer for logit prediction
        min_values: Minimum action values per actor
        max_values: Maximum action values per actor
        mask: Mask for invalid actions (-inf for out-of-range)
        rangeP1: max_values - min_values + 1 (number of discrete actions)
    """
    
    def __init__(self, n_hidden, n_actors, min_values: torch.Tensor, max_values: torch.Tensor):
        """
        Initialize DiscreteSoftmaxActor module.
        
        Args:
            n_hidden: Input feature dimension from the policy network
            n_actors: Number of parallel action distributions (e.g., bid and ask)
            min_values: Minimum action values per actor, shape [n_actors]
            max_values: Maximum action values per actor, shape [n_actors]
            
        Example:
            # Actor for bid/ask prices
            actor = DiscreteSoftmaxActor(
                n_hidden=256,
                n_actors=2,  # bid and ask
                min_values=torch.tensor([0, 0]),     # min prices
                max_values=torch.tensor([100, 100])  # max prices
            )
        """
        super().__init__()
        assert min_values.shape == max_values.shape, "min_values and max_values must have the same shape"
        assert min_values.ndim == 1, "min_values and max_values must be 1D"
        assert min_values.shape[0] == n_actors, f"min_values and max_values must have length {n_actors}"
        
        self.n_hidden = n_hidden
        self.n_actors = n_actors
        
        self.register_buffer('min_values', min_values, persistent=False)
        self.register_buffer('max_values', max_values, persistent=False)
        self.register_buffer('rangeP1', max_values - min_values + 1, persistent=False)
        
        self.n_actions = int(self.rangeP1.max().item())
        self.actor = layer_init(nn.Linear(n_hidden, n_actors * self.n_actions), std=0.1)
        
        # Create mask for invalid actions
        mask = torch.zeros(1, n_actors, self.n_actions)
        for j in range(n_actors):
            num_valid = int(self.rangeP1[j].item())
            if num_valid < self.n_actions:
                mask[0, j, num_valid:] = -float('inf')
        self.register_buffer('mask', mask, persistent=False)
        
        # Precompute indices for mean/variance calculation
        indices = torch.arange(self.n_actions, dtype=torch.float32).view(1, 1, -1)
        indices = indices.expand(1, n_actors, -1)
        self.register_buffer('action_indices', indices, persistent=False)
        
        # Create valid action mask (True for valid actions)
        valid_mask = ~mask.isinf()
        self.register_buffer('valid_mask', valid_mask, persistent=False)
    
    def forward(self, x):
        """
        Forward pass to compute logits for each action.
        
        Args:
            x: Hidden states from policy network, shape [batch_size, n_hidden]
            
        Returns:
            logits: Masked logits for valid actions, shape [batch_size, n_actors, n_actions]
                    Invalid actions have -inf logits
        """
        logits = self.actor(x).view(x.shape[0], self.n_actors, self.n_actions)
        return logits + self.mask
    
    def _compute_mean_precision_vectorized(self, probs):
        """
        Compute mean and precision from probability distributions using vectorized operations.
        
        Args:
            probs: Probabilities over actions, shape [batch_size, n_actors, n_actions]
            
        Returns:
            Dictionary with 'center' and 'precision':
                - center: Expected value normalized to (0, 1), shape [batch_size, n_actors]
                - precision: Inverse of std deviation, shape [batch_size, n_actors]
        """
        # Expand action indices to match batch size
        indices = self.action_indices.expand(probs.shape[0], -1, -1)
        
        # Mask probabilities for invalid actions
        masked_probs = probs * self.valid_mask
        
        # Compute mean (expected value of action indices)
        mean_actions = (masked_probs * indices).sum(dim=-1)  # [B, n_actors]
        
        # Compute variance
        indices_squared = indices ** 2
        second_moment = (masked_probs * indices_squared).sum(dim=-1)  # [B, n_actors]
        variance = second_moment - mean_actions ** 2
        std = torch.sqrt(variance.clamp(min=1e-8))
        
        # Normalize mean to (0, 1) range
        # Action 0 -> 0, Action (rangeP1-1) -> 1
        mean_normalized = mean_actions / (self.rangeP1 - 1).clamp(min=1).unsqueeze(0)
        mean_normalized = mean_normalized.clamp(0, 1)
        precision = 1.0 / std.clamp(min=1e-8)
        return {'center': mean_normalized, 'precision': precision}
    
    def logp_entropy_and_sample(self, x, uniform_samples):
        """
        Sample discrete actions and compute their log probabilities using vectorized operations.
        
        Args:
            x: Hidden states from policy network, shape [batch_size, n_hidden]
            
        Returns:
            Dictionary containing:
                - 'samples': Discrete actions, shape [B, n_actors], dtype int
                - 'logprobs': Log probabilities of sampled actions, shape [B, n_actors]
                - 'entropy': Entropy of distributions, shape [B]
                - 'dist_params': Dictionary with 'center' and 'precision'
        """
        logits = self.forward(x)  # [B, n_actors, n_actions]
        
        # Compute probabilities and log probabilities
        probs = torch.softmax(logits, dim=-1)  # [B, n_actors, n_actions]
        log_probs = torch.log_softmax(logits, dim=-1)  # [B, n_actors, n_actions]
        
        # Vectorized inverse transform sampling
        cumprobs = torch.cumsum(probs, dim=-1)  # [B, n_actors, n_actions]
        # Expand uniform samples to compare with all cumprobs
        uniform = uniform_samples.unsqueeze(-1)  # [B, n_actors, 1]
        # Find first index where cumprob > uniform
        comparison = cumprobs > uniform  # [B, n_actors, n_actions]
        samples = comparison.long().argmax(dim=-1)  # [B, n_actors]
        
        # Offset samples by min_values to get actual actions
        integer_samples = (samples + self.min_values.unsqueeze(0)).clamp(max=self.rangeP1.unsqueeze(0) - 1)
        
        # Vectorized log probability extraction using gather
        gather_indices = samples.unsqueeze(-1)  # [B, n_actors, 1]
        logprobs = torch.gather(log_probs, dim=-1, index=gather_indices).squeeze(-1)  # [B, n_actors]
        
        # Vectorized entropy calculation
        valid_probs = probs * self.valid_mask
        valid_log_probs = log_probs.masked_fill(~self.valid_mask, 0)
        entropy_per_actor = -(valid_probs * valid_log_probs).sum(dim=-1)  # [B, n_actors]
        entropy = entropy_per_actor.sum(dim=-1)  # [B]
        
        return {
            'samples': integer_samples.int(),
            'logprobs': logprobs,
            'entropy': entropy,
            'dist_params': self._compute_mean_precision_vectorized(probs)
        }
    
    def logp_entropy(self, x, integer_samples):
        """
        Compute log probabilities and entropy for given discrete actions using vectorized operations.
        
        Args:
            x: Hidden states from policy network, shape [batch_size, n_hidden]
            integer_samples: Given discrete actions to evaluate, shape [batch_size, n_actors]
            
        Returns:
            Dictionary containing:
                - 'logprobs': Log probabilities of given actions, shape [B, n_actors]
                - 'entropy': Entropy of distributions, shape [B, n_actors]
                - 'dist_params': Dictionary with 'center' and 'precision'
        """
        logits = self.forward(x)  # [B, n_actors, n_actions]
        
        # Compute probabilities and log probabilities
        probs = torch.softmax(logits, dim=-1)  # [B, n_actors, n_actions]
        log_probs = torch.log_softmax(logits, dim=-1)  # [B, n_actors, n_actions]
        
        # Convert from actual action values to zero-indexed
        zero_indexed = (integer_samples - self.min_values.unsqueeze(0)).long()  # [B, n_actors]
        
        # Create gather indices for vectorized extraction
        batch_size = x.shape[0]
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, self.n_actors)
        actor_indices = torch.arange(self.n_actors, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Extract log probabilities for given actions
        logprobs = log_probs[batch_indices, actor_indices, zero_indexed]  # [B, n_actors]
        
        # Vectorized entropy calculation
        valid_probs = probs * self.valid_mask
        valid_log_probs = log_probs.masked_fill(~self.valid_mask, 0)
        entropy = -(valid_probs * valid_log_probs).sum(dim=-1)  # [B, n_actors]
        
        return {
            'logprobs': logprobs,
            'entropy': entropy,
            'dist_params': self._compute_mean_precision_vectorized(probs)
        }
    
if __name__ == '__main__':
    # Example usage and visualization
    import matplotlib.pyplot as plt

    # Test 1: Visualize truncated Gaussian near boundary
    print("Test 1: Truncated Gaussian Distribution Visualization")
    num_samples = 1024000
    center = torch.zeros((num_samples,), device='cuda:0') + 0.99  # Near upper boundary
    prec = torch.ones((num_samples,), device='cuda:0') * 1000     # High precision (narrow)

    uniform_samples = torch.rand(num_samples, device='cuda:0')

    dist = GaussianActionDistribution(center, prec)
    samples = dist.sample(uniform_samples)
    
    # Plot histogram of samples vs theoretical PDF
    plt.figure(figsize=(10, 6))
    plt.hist(samples.cpu().numpy(), bins=100, density=True, alpha=0.7, label='Samples')
    x = torch.linspace(0, 1, num_samples, device='cuda:0')
    pdf_value = dist.log_prob(x).exp()
    plt.plot(x.cpu().numpy(), pdf_value.cpu().numpy(), 'r-', linewidth=2, label='Theoretical PDF')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Truncated Gaussian Distribution (center=0.99, precision=1000)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Test 2: Verify CDF computation accuracy
    print("\nTest 2: CDF Accuracy Verification")
    cdf = dist.log_cdf(x).exp()
    numerical_difference = (cdf - pdf_value.cumsum(dim=0) / num_samples).abs()
    
    plt.figure(figsize=(10, 6))
    plt.plot(cdf.cpu().numpy() + 0.01, label='Analytical CDF (offset +0.01)', linewidth=2)
    plt.plot(pdf_value.cumsum(dim=0).cpu().numpy() / num_samples, label='Numerical CDF', linewidth=2)
    plt.plot(x.cpu().numpy(), numerical_difference.cpu().numpy(), label='Absolute difference', linewidth=1)
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability / Difference')
    plt.title(f'CDF Accuracy Test (max difference: {numerical_difference.max().item():.6f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Test 3: Discrete sampling and entropy estimation
    print("\nTest 3: Discrete Sampling and Entropy Estimation")
    batch_size = 102400
    num_actors = 10
    
    # Create distributions with high center (0.9) and moderate precision
    center = torch.zeros((batch_size, num_actors), device='cuda:0') + 0.9
    prec = torch.ones((batch_size, num_actors), device='cuda:0') * 10
    uniform_samples = torch.rand(batch_size, num_actors, device='cuda:0')

    # Define action ranges: 0 to 100 for all actors
    min_values = torch.zeros((num_actors,), device='cuda:0')
    max_values = torch.zeros((num_actors,), device='cuda:0') + 100
    rangeP1 = max_values - min_values + 1

    # Sample discrete actions using the same process as DiscreteActor
    dist = GaussianActionDistribution(center, prec)
    unit_samples = dist.sample(uniform_samples)  # Continuous samples in [0, 1]
    
    # Discretize: scale to action range and round
    integer_samples = (unit_samples * rangeP1 + min_values - 0.5).round().int().clamp(
        min=min_values, max=max_values)
    
    # Get continuous intervals for discrete actions
    unit_samples_ub = ((integer_samples + 0.5) + 0.5 - min_values) / rangeP1
    unit_samples_lb = ((integer_samples - 0.5) + 0.5 - min_values) / rangeP1

    # Compute discrete log probabilities (scale by 1/rangeP1)
    logprobs = dist.logp_interval(unit_samples_lb, unit_samples_ub) - rangeP1.log() 
    
    # Estimate discrete entropy from continuous entropy
    entropy = dist.entropy() + rangeP1.log()
    
    # Package results
    output = {
        'samples': integer_samples,     # [B, n_actors] int - Discrete actions
        'logprobs': logprobs,          # [B, n_actors] float - Log probabilities
        'entropy': entropy,            # [B] float - Entropy estimate
        'dist_params': {               # Distribution parameters
            'center': center,          # [B, n_actors] float
            'precision': prec,         # [B, n_actors] float
        }}

    # Visualize discrete action distribution
    output_samples = output['samples']
    plt.figure(figsize=(12, 6))
    plt.hist(output_samples.cpu().numpy().flatten(), bins=101, density=True, 
             alpha=0.7, edgecolor='black', linewidth=0.5)
    plt.xlabel('Discrete Action Value')
    plt.ylabel('Probability Density')
    plt.title('Distribution of Discrete Actions (center=0.9, precision=10, range=[0,100])')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Compare estimated entropy with true discrete entropy
    print("\nEntropy Comparison:")
    discrete_probs = torch.tensor([
        (output_samples == i).float().mean().item() 
        for i in range(min_values[0].long().item(), max_values[0].long().item() + 1)])

    # Compute true discrete entropy: H = -Σ p(x) log p(x)
    true_discrete_entropy = -torch.xlogy(discrete_probs, discrete_probs).sum()
    estimated_entropy = output['entropy'][0, 0].item()
    
    print(f"True discrete entropy: {true_discrete_entropy.item():.4f}")
    print(f"Estimated entropy: {estimated_entropy:.4f}")
    print(f"Relative error: {abs(true_discrete_entropy.item() - estimated_entropy) / true_discrete_entropy.item() * 100:.2f}%")
    print(f"\nNote: The estimation is approximate and works best when the distribution ")
    print(f"is smooth relative to the discretization granularity.")
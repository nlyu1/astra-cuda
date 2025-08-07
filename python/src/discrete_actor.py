"""
Triangular distribution-based discrete action sampler for RL with continuous parameterization. 

Use case: action space is a discretization of continuous interval (e.g. bid and ask prices / size). 

Key components:
- TriangleVariableWidthDistribution: Triangular distribution with adjustable width and center
- TriangleFullSupportDistribution: Triangle covering full [0,1] range  
- TrapezoidFullSupportDistribution: Mixture of triangle and uniform for exploration
- TriangleActionDistribution: Hierarchical mixture of above distributions
- DiscreteActor: NN module that outputs distribution parameters and samples discrete actions

The approach maps continuous distributions to discrete actions via:
1. Sample from continuous distribution in [0,1]
2. Scale to action range and round to nearest integer
3. Compute log probabilities over discrete intervals
"""

import torch 
import math 
import torch.nn as nn 

def inv_sigmoid(x):
    return math.log(x) - math.log(1 - x)

class TriangleVariableWidthDistribution:
    """
    Triangular distribution with adjustable center and width.
    
    Forms an isosceles triangle with:
    - Peak at 'center' with height 1/half_width
    - Support on [center-half_width, center+half_width]
    - Linear slopes from edges to peak
    - Area normalized to 1
    
    Used as the main concentrated distribution for confident actions.
    """
    def __init__(self, center, half_width):
        """
        center: [batch_size] in (0, 1)
        half_width: [batch_size] in (0, 0.5)
        """
        # Removed assertions for torch.compile compatibility
        # Use torch.clamp instead to ensure valid ranges
        center = torch.clamp(center, min=0.0, max=1.0)
        half_width = torch.clamp(half_width, min=0.0, max=0.5)
        self.center = center 
        self.half_width = torch.clamp(half_width, max=torch.min(center, 1 - center))
        self.min_val = self.center - self.half_width
        self.max_val = self.center + self.half_width

    def log_prob(self, x):
        """
        x: [batch_size] in (0, 1)
        """
        # Removed assertion for torch.compile compatibility
        x = torch.clamp(x, min=0.0, max=1.0)
        # since half_width * max_height = 1, we have slope = max_height / half_width = 1 / half_width^2
        prob = torch.where(
            x < self.min_val, torch.zeros_like(x), torch.where(
                x > self.max_val, torch.zeros_like(x), 
                torch.where(
                    x < self.center, (x - self.min_val) * self.half_width.pow(-2),
                    -(x - self.center) * self.half_width.pow(-2) + 1 / self.half_width
                )))
        return torch.log(prob).clamp(min=-100.)
    
    def sample(self, uniform_samples):
        return torch.where(
            uniform_samples < 0.5, 
            self.min_val + self.half_width * (2 * uniform_samples).sqrt(),
            self.max_val - self.half_width * (2 * (1 - uniform_samples)).sqrt())
    
    def cdf(self, x):
        cdf_value = torch.where(
            x < self.min_val, torch.zeros_like(x), torch.where(
                x > self.max_val, torch.ones_like(x), 
                torch.where(
                    x < self.center, 
                    ((x - self.min_val) / self.half_width).pow(2) * 0.5,
                    1. - ((self.max_val - x) / self.half_width).pow(2) * 0.5,
                )))
        return cdf_value

    def entropy(self):
        return 0.5 + torch.log(self.half_width)
    
class TriangleFullSupportDistribution:
    """
    Triangular distribution covering the full [0,1] support.
    
    Properties:
    - Peak at 'center' with height 2
    - Support always [0, 1] regardless of center
    - Piecewise linear: rises from 0 to peak, falls from peak to 1
    - Enables exploration across entire action space
    """
    def __init__(self, center):
        self.center = center 
        self.entropy_value = 0.5 - torch.log(2 * torch.ones_like(center))

    def log_prob(self, x):
        # Removed assertion for torch.compile compatibility
        x = torch.clamp(x, min=0.0, max=1.0)
        prob = torch.where(
            x < self.center, 
            2. / self.center * x, 
            2. / (self.center - 1) * (x - self.center) + 2)
        return torch.log(prob).clamp(min=-100.)
    
    def cdf(self, x):
        return torch.where(
            x < self.center, 
            x.pow(2) / self.center,
            1. - (1. - x).pow(2) / (1. - self.center))
    
    def entropy(self):
        return self.entropy_value
    
    def sample(self, uniform_samples):
        return torch.where(
            uniform_samples < self.center, 
            (self.center * uniform_samples).sqrt(),
            1 - ((1 - self.center) * (1 - uniform_samples)).sqrt())
    
def binary_entropy(x):
    """Binary entropy H(p) = -p*log(p) - (1-p)*log(1-p)"""
    return -torch.xlogy(x, x) - torch.xlogy(1 - x, 1 - x)

class TrapezoidFullSupportDistribution:
    """
    Mixture of triangular and uniform distributions.
    
    With probability uniform_epsilon: sample uniformly from [0,1]
    With probability 1-uniform_epsilon: sample from TriangleFullSupport
    
    Creates a trapezoid shape that interpolates between triangle and uniform.
    """
    def __init__(self, center, uniform_epsilon):
        """
        center: Peak location for triangle component
        uniform_epsilon: Mixing weight for uniform component
        """
        self.triangle = TriangleFullSupportDistribution(center)
        self.uniform_epsilon = uniform_epsilon

    def log_prob(self, x):
        return torch.logaddexp(
            self.triangle.log_prob(x) + torch.log1p(-self.uniform_epsilon),
            torch.log(self.uniform_epsilon))
    
    def cdf(self, x):
        return self.uniform_epsilon * x + (1 - self.uniform_epsilon) * self.triangle.cdf(x)
    
    def sample(self, uniform_samples):
        return torch.where(
            uniform_samples[..., 0] < self.uniform_epsilon,
            uniform_samples[..., 1],
            self.triangle.sample(uniform_samples[..., 1]))
    
    def entropy(self):
        """
        Returns an estimate of entropy as mixture of entropy + 0.5 * binary entropy of epsilon-Bernoulli
        """
        return ((1 - self.uniform_epsilon) * self.triangle.entropy()
            + 0.5 * binary_entropy(self.uniform_epsilon))
    
class TriangleActionDistribution:
    """
    Hierarchical mixture model for action sampling.
    
    Two-level mixture:
    1. With probability (1-epsilon_fullsupport): sample from concentrated TriangleVariableWidth
    2. With probability epsilon_fullsupport: sample from TrapezoidFullSupport
    
    The TrapezoidFullSupport itself mixes triangle and uniform based on epsilon_uniform.
    This creates a flexible distribution that can range from highly concentrated to uniform.
    """
    def __init__(self, center, half_width, epsilon_fullsupport, epsilon_uniform):
        """
        center: Mode of distributions, shape [...] in (0, 1)
        half_width: Width of concentrated distribution, shape [...] in (0, 0.5)
        epsilon_fullsupport: Prob of using full-support distribution, shape [...] in (0, 1)
        epsilon_uniform: Uniform mixing weight in full-support dist, shape [...] in (0, 1)
        """
        self.main = TriangleVariableWidthDistribution(center, half_width)
        self.support = TrapezoidFullSupportDistribution(center, epsilon_uniform)
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

        Mixture of entropy <= actual entropy <= mixture of entropy + binary entropy of epsilon. 
        We approximate actual entropy ~= mixture of entropy + bce * 0.75

        We use the 0.75 weighting because when models are confident, width will be small, 
        so the full-support and main distributions will be nearly disjoint, in which case 
        entropy will be closer to the upper bound, but all of this's heuristic, anyways. 
        """
        return (
            self.epsilon_fullsupport * self.support.entropy() 
            + (1 - self.epsilon_fullsupport) * self.main.entropy()
            + 0.75 * binary_entropy(self.epsilon_fullsupport))

class DiscreteActor(nn.Module):
    """
    Neural network module for discrete action sampling using triangular distributions.
    
    Architecture:
    - Input: hidden state [B, n_hidden]
    - Linear layer outputs 4*n_actors values through squishing to [0, 1]
    - Splits into: center, half_width, epsilon_fullsupport, epsilon_uniform
    - Creates TriangleActionDistribution and samples discrete actions
    
    Handles discrete action spaces by:
    1. Sampling continuous values in [0,1]
    2. Scaling to [min_values, max_values] and rounding
    3. Computing log-probs over discrete intervals

    **Important!**
    Ideally pair with learning rate warmup and smaller LR!!, else it's very easy to observe:
    1. Models output bad actions which receive negative gradients. 
    2. Gradients quickly push epsilon_fullsupport and epsilon_uniform to 1, which makes the model output uniform actions. 
    3. Value network receives totally useless signals, and model collapses. 
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
        integer_samples = (unit_samples * (self.rangeP1 - 2 * self.eps) + self.min_values - 0.5 + self.eps).round().int()
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
        dist = TriangleActionDistribution(center, half_width, epsilon_fs, epsilon_uniform)
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
        dist = TriangleActionDistribution(center, half_width, epsilon_fs, epsilon_uniform)

        unit_lb, unit_ub = self._unit_interval_of_integer_samples(integer_samples)
        logprobs = dist.logp_interval(unit_lb, unit_ub) - self.rangeP1.log() 
        entropy = dist.entropy() + self.rangeP1.log()
        return {
            'logprobs': logprobs, # [B, n_actors] float. Logprobs of the samples
            'entropy': entropy, # [B] float. Entropy of the computed distribution
        }
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    num_samples = 1024000
    center = torch.zeros((num_samples,), device='cuda:0') + 0.5
    half_width = torch.ones((num_samples,), device='cuda:0') * 0.1
    epsilon_fullsupport = torch.ones((num_samples,), device='cuda:0') * 0.6
    epsilon_uniform = torch.ones((num_samples,), device='cuda:0') * 1.

    uniform_samples = torch.rand(num_samples, 3, device='cuda:0')

    dist = TriangleActionDistribution(center, half_width, epsilon_fullsupport, epsilon_uniform)
    # dist = TrapezoidFullSupportDistribution(center, epsilon_uniform)
    samples = dist.sample(uniform_samples)
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

    ### Cell here ### 

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

    dist = TriangleActionDistribution(center, half_width, epsilon_fullsupport, epsilon_uniform)
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
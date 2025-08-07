# %% Tried to implement the cdf of the beta distribution. 

import torch 
import torch.nn as nn 
from torch.distributions import Beta
from torch.distributions.k
import matplotlib.pyplot as plt

# %%
import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt

class BetaODE(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
    def forward(self, t, y):
        """
        Returns the integrand of the Beta integral with sigmoid(t) re-parameterization. 

        y: [batch_size, 3] denoting (z, a, b, offset) respectively. 
        t: scalar in (-inf, inf) denoting logit(sigmoid(t))
        """
        z, a, b, inv_lower, inv_range = y.unbind(-1)
        unscaled_t = inv_range * t + inv_lower
        log_dz = (
            a * torch.nn.functional.logsigmoid(unscaled_t)
            - b * torch.nn.functional.softplus(unscaled_t)
        ) + inv_range.log()
        # print('t', t.item(), 'unscaled_t', unscaled_t.item(), 'log_dz', log_dz.item())
        dy = y * 0. 
        dy[:, 0] = torch.exp(log_dz)
        return dy

# Beta function using log-gamma
def beta_func(a, b):
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

def inv_sigmoid(x):
    return (torch.log(x) - torch.log1p(-x)).clamp(-100, 100)

class BetaCDF(nn.Module):
    def __init__(self, num_samples=1000, eps=1e-6):
        super().__init__()
        self.register_buffer('t_span', torch.linspace(0, 1, num_samples))
        self.t_span = (self.t_span - self.t_span[0]) / (self.t_span[-1] - self.t_span[0]) # Rescale to [0, 1]
        self.t_span = self.t_span * (1 - 2 * eps) + eps # Rescale to [eps, 1 - eps]
        self.beta_ode = BetaODE()

    # @torch.compile(fullgraph=True, mode="default")
    def forward(self, a, b, lower, upper):
        inv_lower, inv_upper = inv_sigmoid(lower), inv_sigmoid(upper)
        inv_range = inv_upper - inv_lower 
        normalize_value = beta_func(a, b).exp()
        # print('a', a.item(), 'b', b.item(), 'lower', lower.item(), 'upper', upper.item(), 'normalize_value', normalize_value.item())
        # print('inv_lower', inv_lower.item(), 'inv_upper', inv_upper.item(), 'inv_range', inv_range.item())
        initial_value = torch.stack([
            torch.zeros_like(a), 
            a, b, inv_lower, inv_range], dim=-1)
        beta_results = odeint(
            self.beta_ode,
            initial_value,
            self.t_span, method='euler')
        beta_values = beta_results[-1, :, 0] / normalize_value
        return beta_values

# Parameters
size = 10240
beta_cdf = BetaCDF(num_samples=100).to(device)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# %%

%%timeit
m, k = torch.rand(size, device=device).clamp(1e-3, 1-1e-3), torch.randint(1, 100, (size,), device=device).float()
a, b = m * k, (1 - m) * k

lower = torch.rand(size, device=device)
upper = (lower + torch.rand(size, device=device).clamp(max=0.1)).clamp(max=1.0)

values = beta_cdf(a, b, lower, upper)

# %%
from scipy import special

def beta_cdf_scipy(a, b, lower, upper):
    """Reference implementation using scipy for comparison"""
    return special.betainc(a, b, upper) - special.betainc(a, b, lower)

scipy_results = []
for i in range(size):
    scipy_result = beta_cdf_scipy(
        a[i].cpu().numpy(), 
        b[i].cpu().numpy(), 
        lower[i].cpu().numpy(), 
        upper[i].cpu().numpy()
    )
    scipy_results.append(scipy_result)
scipy_results = torch.tensor(scipy_results, device=device)

diffs = (values - scipy_results).abs()
careful_idx = diffs.argmax()
print('Argmax and max diffs:', diffs.argmax(), diffs.max())
print('a, b, lower, upper:', a[careful_idx].item(), b[careful_idx].item(), lower[careful_idx].item(), upper[careful_idx].item())

# %%

# %%

class DiscreteActor(nn.Module):
    def __init__(self, n_hidden, n_actors, min_values: torch.Tensor, max_values: torch.Tensor):
        """
        Model outputs the location and dispersion parameters of the beta distribution. 
        Beta(alpha, beta) where alpha = k*m, beta=k*(1-m)
        k: concentration parameter unbounded in [0, inf]. Parameterized by softplus(k_hidden). Higher k means higher concentration. 
        m: location parameter bounded in [0, 1]. Parameterized by sigmoid(m_hidden). 

        The std of beta distribution is sqrt(m(1-m) / (k + 1)). 
        So, to have max-(min across m) std of ~0.5 / scale, we need 0.5 / sqrt(k+1) = 0.5 / scale, or max_kappa ~= scale ** 2
        For example, for discretization of ~30 values, we need max_kappa ~= 900. 
        """
        super().__init__()
        self.n_hidden = n_hidden
        self.n_actors = n_actors
        self.actor = nn.Linear(n_hidden, n_actors * 2)
        assert min_values.shape == max_values.shape, f"min_values and max_values must have the same shape, but got {min_values.shape} and {max_values.shape}"
        assert min_values.ndim == 1, f"min_values and max_values must be 1D, but got {min_values.ndim} and {max_values.ndim}"
        assert min_values.shape[0] == n_actors, f"min_values and max_values must have length {n_actors}, but got {min_values.shape[0]} and {max_values.shape[0]}"
        self.register_buffer('min_values', min_values)
        self.register_buffer('max_values', max_values)
    
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
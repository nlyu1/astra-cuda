import os
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
import wandb
from jaxtyping import Float, Bool, jaxtyped

class VTraceBuffer:
    def __init__(self, args, num_features, device='cuda'):
        self.args = args 

        self.obs = torch.zeros((args.num_steps, args.num_envs, num_features), device=device)
        self.actions = torch.zeros((args.num_steps, args.num_envs, 4), device=device)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
        self.dones = torch.zeros((args.num_steps, args.num_envs), device=device)

        self.last_step = -1 
        self.updated_late_stats = False 
        self.length = args.num_steps 

    def is_depleted(self):
        return (self.last_step == self.length -1 and self.updated_late_stats)

    def update(self, updates, step):
        """
        Steps are agent steps. 
        """
        # Step continuity check 
        assert (step >= 0 and step < self.length), "Step must be within [0, num_steps)"
        if (step == 0):
            # Initialization or wrap-around 
            assert self.last_step ==-1 or self.is_depleted()
        else:
            assert self.last_step == step - 1 
        self.last_step = step 
        self.updated_late_stats = False 

        self.obs[step] = updates['obs']
        self.actions[step] = updates['actions']
        self.logprobs[step] = updates['logprobs']

    def update_late_stats(self, updates, step):
        assert self.last_step == step and not self.updated_late_stats, f"Step {step} either out of sync with {self.last_step} or already updated ({self.updated_late_stats})"
        self.updated_late_stats = True 

        self.rewards[step] = updates['rewards']
        self.dones[step] = updates['dones'] # Whether episode terminated after this step 

    def get_update_dictionary(self):
        return {
            'obs': self.obs,
            'logprobs': self.logprobs,
            'actions': self.actions,
            'rewards': self.rewards,
            'dones': self.dones,
        }

def vtrace_losses(
    rewards: Float[torch.Tensor, "T B"],
    dones: Bool[torch.Tensor, "T B"],
    ref_logprobs: Float[torch.Tensor, "T B"],
    logprobs: Float[torch.Tensor, "T B"], # Gradient-attached logprobs 
    values: Float[torch.Tensor, "T_plus_1 B"],
    gamma: float = 0.99,
    bar_rho: float = 1.0,
    bar_c: float = 1.0
): 
    """Computes V-trace bootstrapped value targets from a batch of trajectories.

    This function implements the V-trace algorithm, which provides off-policy
    corrected multi-step value estimates for training an actor-critic agent.
    The calculation is performed backwards in time over the trajectory.

    Args:
        rewards (Float[torch.Tensor, "T B"]): Tensor of rewards. `T` is the unroll
            length (time dimension) and `B` is the batch size (number of envs).
        dones (Bool[torch.Tensor, "T B"]): Tensor of booleans indicating if an
            episode terminated at the end of a timestep.
        ref_logprobs (Float[torch.Tensor, "T B"]): Log probabilities of the taken
            actions according to the **gradient-detached** reference/behavior policy.
        logprobs (Float[torch.Tensor, "T B"]): Log probabilities of the taken actions
            according to the current, **gradient-attached** learner policy.
        values (Float[torch.Tensor, "T_plus_1 B"]): Value function estimates for each
            state in the trajectory, $V(s_0), V(s_1), ..., V(s_T)$. This tensor must
            contain one extra time step (`T+1`) for the final bootstrap value $V(s_T)$.
        gamma (float): The discount factor for future rewards.
        bar_rho (float): The clipping threshold (ρ-bar) for the policy importance
            weight ($\rho$), which corrects the temporal difference error.
        bar_c (float): The clipping threshold (c-bar) for the trace importance
            weight ($c$), which controls the bootstrapping lookahead.
    """

    # If done at current episode, next-step value is truncated 
    next_values = values[1:] * (1 - dones)
    cur_values = values[:-1]
    with torch.no_grad(): 
        prob_ratio = (logprobs - ref_logprobs).exp()
        policy_clip = torch.clamp(prob_ratio, max=bar_rho) # [T B]
        vtrace_clip = torch.clamp(prob_ratio, max=bar_c) # [T B]

        corrected_td = policy_clip * (rewards + gamma * next_values - cur_values)
        # For example, vtrace_H = V_H + clip_H * corrected_td[H] 
        # vtrace_{H-1} = V_{H-1} + clip_{H-1} * corrected_td[H-1] + clip_{H-1} * clip_H * gamma * corrected_td[H]
        # vtrace_{H+1} is set to 0 trivially 
        vtrace_td = torch.zeros_like(values)
        for t in reversed(range(cur_values.shape[0])): # [T-1, ..., 0]
            vtrace_td[t] = vtrace_clip[t] * (
                corrected_td[t] + gamma * vtrace_td[t + 1]) # for t = T - 1, the second term is 0 
        vtrace_values = values + vtrace_td 
        value_r2 = 1 - (vtrace_values[:-1] - cur_values).var() / vtrace_values[:-1].var()

    value_loss = torch.nn.functional.smooth_l1_loss(vtrace_values[:-1], cur_values).mean()

    # Calculate action losses: vanilla PG on corrected advantage
    with torch.no_grad(): 
        vtrace_advantage = rewards + gamma * (1 - dones) *vtrace_values[1:] - cur_values 
        vtrace_advantage = (vtrace_advantage - vtrace_advantage.mean()) / (vtrace_advantage.std() + 1e-8)
    pg_loss = - policy_clip * logprobs * vtrace_advantage 
    pg_loss = pg_loss.mean()

    result = {
        'policy_loss': pg_loss,
        'value_loss': value_loss,
        'value_r2': value_r2,
    }
    return result 

compiled_vtrace_losses = torch.compile(vtrace_losses, mode='max-autotune', fullgraph=True)
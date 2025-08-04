# Uses a ping-pong buffer for IMPALA asynchronous training
import os
import random
import numpy as np
import hashlib
import collections
from collections.abc import Mapping, Iterable

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from tqdm import trange
import wandb
from jaxtyping import Float, Bool, jaxtyped

class HighLowImpalaBuffer:
    def __init__(self, args, num_features, device='cuda'):
        self.args = args 

        self.obs = torch.zeros((args.num_steps, args.num_envs, num_features), device=device)
        self.actions = torch.zeros((args.num_steps, args.num_envs, 4), device=device)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
        self.dones = torch.zeros((args.num_steps, args.num_envs), device=device)

        self.actual_settlement = torch.zeros((args.num_envs,), device=device)
        # 0, 1, 2 stands for valueCheater, highLow, customer
        self.actual_private_roles = torch.zeros((args.num_envs, args.players), device=device).long()

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

        # self.obs[step] = updates['obs']
        self.actions[step] = updates['actions']
        self.logprobs[step] = updates['logprobs']

    def update_late_stats(self, updates, step):
        assert self.last_step == step and not self.updated_late_stats, f"Step {step} either out of sync with {self.last_step} or already updated ({self.updated_late_stats})"
        self.updated_late_stats = True 

        # self.rewards[step] = updates['rewards']
        self.dones[step] = updates['dones'] # Whether episode terminated after this step 

    def get_update_dictionary(self):
        return {
            'obs': self.obs,
            'logprobs': self.logprobs,
            'actions': self.actions,
            'rewards': self.rewards,
            'dones': self.dones,
            'pinfo_tensor': self.pinfo_tensor,
            'actual_settlement': self.actual_settlement,
            'actual_private_roles': self.actual_private_roles,
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
compiled_vtrace_losses = torch.compile(vtrace_losses, mode='default', fullgraph=True)
    

class HighLowImpalaTrainer:
    def __init__(self, args, agent, checkpoint_interval=10_000_000, device='cuda'):
        self.args = args 
        self.agent = agent 
        self.optimizer = torch.optim.Adam(
            agent.parameters(),
            lr=args.learning_rate,
            eps=1e-5,
        )
        self.last_checkpoint = 0 
        self.checkpoint_interval = checkpoint_interval
        
        # Initialize warmup scheduler
        self.current_step = 0
        self.warmup_steps = args.warmup_steps
        self.base_lr = args.learning_rate

        # Weights for private loss: [0, ..., 1] across different time steps
        # Decay by half every (tau) ratio away from horizon. 
        pinfo_loss_weights = 1 - torch.arange(0, 1, 1./args.num_steps).to(device)
        pinfo_loss_weights = (pinfo_loss_weights - pinfo_loss_weights.min()) / (pinfo_loss_weights.max() - pinfo_loss_weights.min())
        pinfo_loss_weights = (0.5 ** (pinfo_loss_weights / args.pdecay_tau)).unsqueeze(1) # Unsqueeze along batch dimension
        self.pinfo_loss_weights = pinfo_loss_weights / pinfo_loss_weights.mean() # Should mean to 1

        self.logging_size = self.args.update_epochs * self.args.num_minibatches
        self.explained_vars, self.value_losses, self.pg_losses, self.entropy_losses, self.approx_kls, \
            self.pred_settlement_losses, self.pred_private_roles_losses = (
            torch.zeros((self.logging_size,), device=device),
            torch.zeros((self.logging_size,), device=device),
            torch.zeros((self.logging_size,), device=device),
            torch.zeros((self.logging_size,), device=device),
            torch.zeros((self.logging_size,), device=device),
            torch.zeros((self.logging_size,), device=device),
            torch.zeros((self.logging_size,), device=device))

    def _update_learning_rate(self):
        """Update learning rate based on warmup schedule."""
        if self.current_step < self.warmup_steps:
            # Linear warmup from 0 to base_lr
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Use base learning rate after warmup
            lr = self.base_lr
        return lr

    def train(self, update_dictionary):
        obs, logprobs, actions, rewards, dones = (
            update_dictionary['obs'],
            update_dictionary['logprobs'],
            update_dictionary['actions'].long(),
            update_dictionary['rewards'],
            update_dictionary['dones'])
        # Unlike in PPO, we batch sample by environment instead 
        batch_env_indices = torch.arange(self.args.num_envs).to(obs.device)
        logging_counter = 0
        
        # Update learning rate based on warmup schedule
        current_lr = self._update_learning_rate()
        self.current_step += 1 

        # Obs: [T, B, s]. Logprobs: [T, B], Actions: [T, B, 4], Rewards [T, B], Done: [T, B]
        num_envs_per_minibatch = self.args.num_envs // self.args.num_minibatches
        assert num_envs_per_minibatch * self.args.num_minibatches == self.args.num_envs

        with torch.autocast(device_type=obs.device.type, dtype=torch.bfloat16):
            for _ in range(self.args.update_epochs):
                batch_env_indices = torch.randperm(self.args.num_envs).to(obs.device)
                for start in range(0, self.args.num_envs, num_envs_per_minibatch):
                    end = start + num_envs_per_minibatch
                    minibatch_env_indices = batch_env_indices[start:end]

                    step_results = self._train_step(
                        minibatch_env_indices,
                        obs, logprobs, actions, rewards, dones, 
                        update_dictionary['actual_settlement'],
                        update_dictionary['actual_private_roles'],
                        update_dictionary['pinfo_tensor'])
                    step_results['loss'].backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.explained_vars[logging_counter] = step_results['explained_vars']
                    self.value_losses[logging_counter] = step_results['value_losses']
                    self.pg_losses[logging_counter] = step_results['pg_losses']
                    self.entropy_losses[logging_counter] = step_results['entropy_losses']
                    self.approx_kls[logging_counter] = step_results['approx_kls']
                    self.pred_settlement_losses[logging_counter] = step_results['pred_settlement_loss']
                    self.pred_private_roles_losses[logging_counter] = step_results['pred_private_roles_loss']
                    logging_counter += 1

        # Batch all tensor means to reduce GPU-CPU transfers
        metrics_cpu = torch.stack([
            self.explained_vars.mean(),
            self.value_losses.mean(),
            self.pg_losses.mean(),
            self.entropy_losses.mean(),
            self.approx_kls.mean(),
            self.pred_settlement_losses.mean(),
            self.pred_private_roles_losses.mean()
        ]).detach().cpu().numpy()
        
        return {
            'metrics/explained_vars': float(metrics_cpu[0]),
            'metrics/value_losses': float(metrics_cpu[1]),
            'metrics/pg_losses': float(metrics_cpu[2]),
            'metrics/entropy': float(-metrics_cpu[3]),
            'metrics/approx_kls': float(metrics_cpu[4]),
            'metrics/pred_settlement_losses': float(metrics_cpu[5]),
            'metrics/pred_private_roles_losses': float(metrics_cpu[6]),
            'metrics/learning_rate': current_lr,
        }

    @torch.compile(mode="max-autotune-no-cudagraphs", fullgraph=True)
    def _train_step(self, 
                    minibatch_env_indices,
                    obs, logprobs, actions, rewards, dones, 
                    actual_settlement, actual_private_roles, pinfo_tensor):
        """
        T = num_steps, B = [num_envs_per_batch]
        obs: [T, B, ...]
        actions: [T, B, 4]
        pinfo_tensor: [B, Pinfo_numfeatures]
        log_probs, rewards, dones: [T, B]
        actual_settlement: [B]
        actual_private_roles: [B, num_players]
        """
        T, single_obs_shape = obs.shape[0], obs.shape[2:]
        num_envs_per_minibatch = self.args.num_envs // self.args.num_minibatches

        # Off-policy network, using surrogate loss 
        # [T, b, 4] -> [T*b, 4]
        outputs = self.agent(
            obs[:, minibatch_env_indices],
            pinfo_tensor[minibatch_env_indices], # [B, Pinfo_numfeatures]
            actions[:, minibatch_env_indices])
        # _, new_logprob, entropy, values
        new_logprob, entropy, values = outputs['logprobs'], outputs['entropy'], outputs['values']
        pred_settlement = outputs['pinfo_preds']['settle_price']
        pred_private_roles = outputs['pinfo_preds']['private_roles'] # [T*b, num_players, 3]
        
        # Private loss 
        pred_settlement_loss = torch.nn.functional.smooth_l1_loss(
            pred_settlement, 
            actual_settlement[minibatch_env_indices].unsqueeze(0).expand(T, num_envs_per_minibatch), 
            reduction='none')
        pred_settlement_loss = (pred_settlement_loss * self.pinfo_loss_weights).mean()

        flattened_pred_private_roles = pred_private_roles.reshape(-1, 3)
        flattened_actual_private_roles = actual_private_roles[minibatch_env_indices].unsqueeze(0).expand(
            T, num_envs_per_minibatch, self.args.players).reshape(-1)
        pred_private_roles_loss = torch.nn.functional.cross_entropy(
            flattened_pred_private_roles, flattened_actual_private_roles, reduction='none'
        ).reshape(T, num_envs_per_minibatch, self.args.players)
        pred_private_roles_loss = (pred_private_roles_loss * self.pinfo_loss_weights.unsqueeze(-1)).mean()

        with torch.no_grad():
            log_ratio = new_logprob - logprobs[:, minibatch_env_indices]
            ratio = log_ratio.exp() 
            approx_kl = ((ratio - 1) - log_ratio).mean()

        assert dones[-1, minibatch_env_indices].all(), "All episodes must be terminated at the end of the episode"
        augmented_values = torch.cat([ # [T+1, B]
            values, torch.zeros_like(values[-1])[None, :]])                 
        
        vtrace_results = compiled_vtrace_losses(
            rewards[:, minibatch_env_indices],
            dones[:, minibatch_env_indices],
            logprobs[:, minibatch_env_indices],
            new_logprob,
            augmented_values,
            gamma=self.args.gamma)

        entropy_loss = -entropy.mean()
        loss = (vtrace_results['policy_loss'] 
                + self.args.ent_coef * entropy_loss 
                + vtrace_results['value_loss'] * self.args.vf_coef
                + pred_settlement_loss * self.args.psettlement_coef
                + pred_private_roles_loss * self.args.proles_coef)

        return {
            'loss': loss, 
            'explained_vars': vtrace_results['value_r2'],
            'value_losses': vtrace_results['value_loss'],
            'pg_losses': vtrace_results['policy_loss'],
            'entropy_losses': -entropy.mean(),
            'approx_kls': approx_kl,
            'pred_settlement_loss': pred_settlement_loss,
            'pred_private_roles_loss': pred_private_roles_loss,
        }

    def save_checkpoint(self, step):
        if step < self.last_checkpoint + self.checkpoint_interval:
            return 
        self.last_checkpoint = step

        checkpoint_path = f"checkpoints/{self.args.exp_name}_{step}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        print('Saving checkpoint to', checkpoint_path)
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step,
            'args': self.args,
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
            },
            'warmup_state': {
                'current_step': self.current_step,
                'warmup_steps': self.warmup_steps,
                'base_lr': self.base_lr,
            }
        }, checkpoint_path)
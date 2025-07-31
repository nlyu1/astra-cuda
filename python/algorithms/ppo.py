import os
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
import wandb

class PPOBuffer:
    def __init__(self, args, num_features, device='cuda'):
        self.args = args 

        self.obs = torch.zeros((args.num_steps, args.num_envs, num_features)).to(device)
        self.actions = torch.zeros((args.num_steps, args.num_envs, 4)).to(device)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        self.last_step = -1 
        self.updated_late_stats = False 
        self.advantages = torch.zeros_like(self.rewards)
        self.bootstrapped_targets = False 
        self.length = args.num_steps 

    def expand_length(self, num_steps):
        assert num_steps > self.length 
        assert self.is_depleted(), "Buffer must be depleted before expanding"
        step_delta = num_steps - self.length 
        def construct_padding(tensor):
            padding = torch.zeros((step_delta, *tensor.shape[1:])).type_as(tensor)
            result = torch.cat([tensor, padding], dim=0)
            del tensor 
            return result 
        
        self.obs = construct_padding(self.obs)
        self.actions = construct_padding(self.actions)
        self.logprobs = construct_padding(self.logprobs)
        self.rewards = construct_padding(self.rewards)
        self.dones = construct_padding(self.dones)
        self.values = construct_padding(self.values)
        self.length = num_steps 
        self.last_step = self.length - 1 # Still mark buffer as depleted 
        torch.cuda.empty_cache()

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
        self.bootstrapped_targets = False # Target bootstrap is stale upon updating 

        self.obs[step] = updates['obs']
        self.actions[step] = updates['actions']
        self.logprobs[step] = updates['logprobs']
        self.values[step] = updates['values']

    def update_late_stats(self, updates, step):
        assert self.last_step == step and not self.updated_late_stats, f"Step {step} either out of sync with {self.last_step} or already updated ({self.updated_late_stats})"
        self.updated_late_stats = True 

        self.rewards[step] = updates['rewards']
        self.dones[step] = updates['dones'] # Whether episode terminated after this step 

    @torch.no_grad()
    def bootstrap_advantages_returns(self, next_values=0):
        """
        If not provided, will assume that truncation at args.num_steps is directly at termination

        Returns advantages [num_steps * num_envs] and GAE-bootstrapped returns [num_steps * num_envs]
        """
        assert self.last_step == self.length - 1 and self.updated_late_stats
        assert not self.bootstrapped_targets
        self.bootstrapped_targets = True 

        last_gae_lambda = 0 
        for t in reversed(range(self.length)):
            if t == self.length - 1:
                next_values = next_values 
            else:
                next_values = self.values[t+1]
            ctd = 1.0 - self.dones[t] # Whether environment continued after step t
            delta = self.rewards[t] - self.values[t] + self.args.gamma * next_values * ctd
            self.advantages[t] = last_gae_lambda = (
                delta + self.args.gamma * self.args.gae_lambda * ctd * last_gae_lambda)
        # Normalize advantages upon computation
        self.bootstrapped_target_values = self.advantages + self.values 
        # self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_update_dictionary(self):
        assert self.bootstrapped_targets

        return {
            'obs': self.obs.flatten(0, 1),
            'logprobs': self.logprobs.flatten(0, 1),
            'actions': self.actions.flatten(0, 1),
            'advantages': self.advantages.flatten(0, 1),
            'value_targets': self.bootstrapped_target_values.flatten(0, 1),
            'values': self.values.flatten(0, 1),
        }
    
class HighLowPPOTrainer:
    def __init__(self, args, agent, name='', checkpoint_interval=10_000_000, device='cuda'):
        self.args = args 
        self.agent = agent 
        self.optimizer = torch.optim.Adam(
            agent.parameters(),
            lr=args.learning_rate,
            eps=1e-5,
        )
        self.last_checkpoint = -1
        self.checkpoint_interval = checkpoint_interval
        self.name = name 

    def train(self, update_dictionary, global_step):
        obs, logprobs, actions, advantages, value_targets, values = (
            update_dictionary['obs'],
            update_dictionary['logprobs'],
            update_dictionary['actions'].long(),
            update_dictionary['advantages'],
            update_dictionary['value_targets'],
            update_dictionary['values'],
        )

        batch_indices = np.arange(self.args.batch_size)
        clip_fracs, explained_var = [], []
        # for epoch in trange(self.args.update_epochs, desc="PPO Training Epoch"):
        for epoch in range(self.args.update_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                minibatch_indices = batch_indices[start:end]

                # Off-policy network, using surrogate loss 
                reference_actions = actions[minibatch_indices]
                reference_actions = {
                    'bid_price': reference_actions[:, 0],
                    'ask_price': reference_actions[:, 1],
                    'bid_size': reference_actions[:, 2],
                    'ask_size': reference_actions[:, 3],
                }
                _, new_logprob, entropy, new_value = self.agent(
                    obs[minibatch_indices], reference_actions)
                log_ratio = new_logprob - logprobs[minibatch_indices] 
                ratio = log_ratio.exp() 

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                minibatch_advantages = advantages[minibatch_indices]
                minibatch_advantages = (
                    minibatch_advantages - minibatch_advantages.mean()
                ) / (minibatch_advantages.std() + 1e-8)

                pg_loss1 = -minibatch_advantages * ratio 
                pg_loss2 = -minibatch_advantages * torch.clamp(
                    ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = torch.nn.functional.smooth_l1_loss(
                    new_value, value_targets[minibatch_indices])
                entropy_loss = -entropy.mean()
                loss = pg_loss + self.args.ent_coef * entropy_loss + value_loss * self.args.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

                if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                    break

                y_pred, y_true = values, value_targets
                var_y = y_true.var(unbiased=False)
                explained_var.append(
                    float('nan') if var_y.item() == 0
                    else (1 - (y_true - y_pred).var(unbiased=False) / var_y).item())

        wandb.log({
            "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
            "losses/value_loss": value_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": -entropy_loss.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clip_frac": np.mean(clip_fracs),
            "losses/value_returns_r2": np.nanmean(explained_var),
        }, step=global_step)

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
            }
        }, checkpoint_path)
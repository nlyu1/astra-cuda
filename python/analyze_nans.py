# %% Import necessary libraries
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from glob import glob

sys.path.append('./src')

from high_low.agent import HighLowTransformerModel
from high_low.config import Args
from high_low.env import HighLowTrading
from high_low.impala import vtrace_losses
from discrete_actor import GaussianActionDistribution
import tyro

# %% Load debug directory
debug_dir = "nan_debug_normal_seedpool_dev_20250808_233453_1"
print(f"Analyzing debug directory: {debug_dir}")

# Check if directory exists
if not os.path.exists(debug_dir):
    raise FileNotFoundError(f"Debug directory {debug_dir} not found!")

# List contents
print("\nDirectory contents:")
for file in os.listdir(debug_dir):
    print(f"  - {file}")

# %% Load saved data
# Load the debug inputs
debug_data = torch.load(os.path.join(debug_dir, "debug_inputs.pt"), map_location='cpu', weights_only=False)
print("\nDebug data keys:")
for key in debug_data.keys():
    if isinstance(debug_data[key], torch.Tensor):
        print(f"  - {key}: shape {debug_data[key].shape}, dtype {debug_data[key].dtype}")
    else:
        print(f"  - {key}: {type(debug_data[key])}")

# Load model state
model_state = torch.load(os.path.join(debug_dir, "model_state_dict.pt"), map_location='cpu', weights_only=False)
print(f"\nModel state dict has {len(model_state)} parameters")

# Load gradient info
grad_info = torch.load(os.path.join(debug_dir, "gradient_info.pt"), map_location='cpu', weights_only=False)
print(f"\nParameters with NaN gradients:")
nan_param_count = 0
for name, info in grad_info.items():
    if info['has_nan']:
        nan_param_count += 1
        print(f"  - {name}: {info['nan_count']} NaNs")
print(f"Total: {nan_param_count} parameters with NaN gradients")

# %% Reconstruct the environment and model
args = debug_data['args']
print(f"\nRecreating environment with args:")
print(f"  - num_envs: {args.num_envs}")
print(f"  - num_steps: {args.num_steps}")
print(f"  - players: {args.players}")
print(f"  - device_id: {args.device_id}")
print(f"  - learning_rate: {args.learning_rate}")
print(f"  - entropy_coef: {args.entropy_coef}")

# Create environment
game_config = args.get_game_config()
env = HighLowTrading(game_config)

# Create model
device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
model = HighLowTransformerModel(args, env, verbose=False).to(device)

# Load the saved state
model.load_state_dict(model_state, strict=True)
model.train()  # Set to training mode to match original
print(f"\nModel loaded successfully on {device}")

# %% Move data to device

device = torch.device('cuda:0')
minibatch_env_indices = debug_data['minibatch_env_indices'].to(device)
obs = debug_data['obs'].to(device)
logprobs = debug_data['logprobs'].to(device)
actions = debug_data['actions'].long().to(device)
rewards = debug_data['rewards'].to(device)
dones = debug_data['dones'].to(device)

print(f"\nData shapes:")
print(f"  - obs: {obs.shape}")
print(f"  - actions: {actions.shape}")
print(f"  - logprobs: {logprobs.shape}")
print(f"  - rewards: {rewards.shape}")
print(f"  - dones: {dones.shape}")
print(f"  - minibatch_env_indices: {minibatch_env_indices.shape}")

# %% Check for NaNs and extreme values in inputs
print("\nChecking for NaNs in inputs:")
print(f"  - obs: {torch.isnan(obs).any().item()}")
print(f"  - actions: {torch.isnan(actions.float()).any().item()}")
print(f"  - logprobs: {torch.isnan(logprobs).any().item()}")
print(f"  - rewards: {torch.isnan(rewards).any().item()}")
print(f"  - dones: {torch.isnan(dones).any().item()}")

# Check for extreme values
print("\nChecking for extreme values:")
print(f"  - obs: min={obs.min().item():.6f}, max={obs.max().item():.6f}, mean={obs.mean().item():.6f}")
print(f"  - logprobs: min={logprobs.min().item():.6f}, max={logprobs.max().item():.6f}, mean={logprobs.mean().item():.6f}")
print(f"  - rewards: min={rewards.min().item():.6f}, max={rewards.max().item():.6f}, mean={rewards.mean().item():.6f}")

# Check for infinite values
print("\nChecking for infinite values:")
print(f"  - obs: {torch.isinf(obs).any().item()}")
print(f"  - logprobs: {torch.isinf(logprobs).any().item()}")
print(f"  - rewards: {torch.isinf(rewards).any().item()}")

# %% Create pinfo_tensor (needed for forward pass)
# This is a placeholder - in actual training this comes from the environment
num_envs_in_minibatch = len(minibatch_env_indices)
pinfo_numfeatures = 2 + 1 + args.players  # From agent.py
pinfo_tensor = torch.zeros((num_envs_in_minibatch, pinfo_numfeatures), device=device)
print(f"\nCreated pinfo_tensor with shape: {pinfo_tensor.shape}")

# %% Replicate the forward pass
print("\nReplicating forward pass...")

# Enable gradient computation
model.zero_grad()

# Run forward pass with autocast (matching training)
with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=True):
    try:
        outputs = model(obs, pinfo_tensor, actions)
        
        new_logprob = outputs['logprobs']
        entropy = outputs['entropy']
        values = outputs['values']
        pred_settlement = outputs['pinfo_preds']['settle_price']
        pred_private_roles = outputs['pinfo_preds']['private_roles']
        
        print("Forward pass successful!")
        print(f"  - new_logprob: shape {new_logprob.shape}, has_nan={torch.isnan(new_logprob).any().item()}")
        print(f"  - entropy: shape {entropy.shape}, has_nan={torch.isnan(entropy).any().item()}")
        print(f"  - values: shape {values.shape}, has_nan={torch.isnan(values).any().item()}")
        print(f"  - pred_settlement: has_nan={torch.isnan(pred_settlement).any().item()}")
        print(f"  - pred_private_roles: has_nan={torch.isnan(pred_private_roles).any().item()}")
        
    except Exception as e:
        print(f"Forward pass failed with error: {e}")
        raise

print(f"new_logprob: {new_logprob.min().item()}, {new_logprob.max().item()}")

# Get distribution parameters
T, B, F = obs.shape
if 'dist_params' in outputs:
    policy_centers = outputs['dist_params']['center'].reshape(T, B, 4)
    policy_precisions = outputs['dist_params']['precision'].reshape(T, B, 4)
    print(f"policy_centers: {policy_centers.min().item()}, {policy_centers.max().item()}")
    print(f"Policy precisions: {policy_precisions.min().item()}, {policy_precisions.max().item()}")
else:
    # Fallback: manually get them from the actor's forward pass
    print("dist_params not in outputs, manually extracting...")
    with torch.no_grad():
        encoded = model.encoder(obs.view(T*B, F)).view(T, B, model.n_embd)
        encoded = model.pos_encoding(encoded)
        features = model.transformer(encoded, mask=model.causal_mask, is_causal=True).view(T * B, model.n_embd)
        features = model.transformer_norm(features)
        dist_params = model.actors.get_dist_params(features)
        policy_centers = dist_params['center'].reshape(T, B, 4)
        policy_precisions = dist_params['precision'].reshape(T, B, 4)
        print(f"policy_centers: {policy_centers.min().item()}, {policy_centers.max().item()}")
        print(f"Policy precisions: {policy_precisions.min().item()}, {policy_precisions.max().item()}")

# Analyze where -inf occurred
print("\nAnalyzing -inf log probabilities...")
inf_mask = torch.isinf(new_logprob)
print(new_logprob.shape, policy_centers.shape)
if inf_mask.any():
    inf_indices = torch.where(inf_mask)
    print(inf_indices)
    print(f"Found {inf_mask.sum().item()} -inf values")
    print(f"Indices (timestep, batch): {list(zip(inf_indices[0].cpu().numpy()[:10], inf_indices[1].cpu().numpy()[:10]))}...")
    
    # For each -inf location, check the corresponding action, center, and precision
    print("\nExamining first few -inf cases:")
    for i in range(min(5, len(inf_indices[0]))):
        t_idx = inf_indices[0][i].item()
        b_idx = inf_indices[1][i].item()
        
        print(f"\n-inf at timestep {t_idx}, batch {b_idx}:")
        print(f"  Actions: {actions[t_idx, b_idx].cpu().numpy()}")
        print(f"  Centers: {policy_centers[t_idx, b_idx].detach().float().cpu().numpy()}")
        print(f"  Precisions: {policy_precisions[t_idx, b_idx].detach().float().cpu().numpy()}")
        print(f"  Logprobs: {new_logprob[t_idx, b_idx].detach().float().cpu().numpy()}")
        
        # Check if action is outside the valid range
        min_vals = torch.tensor([1, 1, 0, 0], device=device)
        max_vals = torch.tensor([args.max_contract_value, args.max_contract_value, args.max_contracts_per_trade, args.max_contracts_per_trade], device=device)
        
        action_vals = actions[t_idx, b_idx]
        outside_range = (action_vals < min_vals) | (action_vals > max_vals)
        if outside_range.any():
            print(f"  WARNING: Action outside valid range!")
            for j in range(4):
                if outside_range[j]:
                    print(f"    Action[{j}]={action_vals[j].item()} not in [{min_vals[j].item()}, {max_vals[j].item()}]")
        break

# %% Replicate the discrete_action pass for debugging -inf
print("\n" + "="*50)
print("DEBUGGING -INF LOG PROBABILITY")
print("="*50)

# Use the first -inf case
for i in range(min(5, len(inf_indices[0]))):
    t_idx = inf_indices[0][i].item()
    b_idx = inf_indices[1][i].item()
    
    print(f"\nAnalyzing timestep {t_idx}, batch {b_idx}:")

    # Get the specific values
    action = actions[t_idx, b_idx]
    center = policy_centers[t_idx, b_idx]
    precision = policy_precisions[t_idx, b_idx]

    print(f"Action: {action.cpu().numpy()}")
    print(f"Center: {center.detach().float().cpu().numpy()}")
    print(f"Precision: {precision.detach().cpu().numpy()}")

    # Get min/max values for each action type
    min_vals = torch.tensor([1, 1, 0, 0], device=device)
    max_vals = torch.tensor([args.max_contract_value, args.max_contract_value, args.max_contracts_per_trade, args.max_contracts_per_trade], device=device)
    rangeP1 = max_vals - min_vals + 1

    print(f"\nValid ranges:")
    for i in range(4):
        print(f"  Action {i}: [{min_vals[i].item()}, {max_vals[i].item()}], rangeP1={rangeP1[i].item()}")

    # Manually compute the log probability step by step
    print("\n" + "-"*30)
    print("MANUAL COMPUTATION:")

    with torch.no_grad():
        # Create distribution for each action dimension
        for i in range(4):
            print(f"\nAction dimension {i}:")
            print(f"  Action value: {action[i].item()}")
            print(f"  Center: {center[i].item():.6f}")
            print(f"  Precision: {precision[i].item():.6f}")
            
            # Create distribution
            dist_i = GaussianActionDistribution(center[i:i+1], precision[i:i+1])
            
            # Compute unit interval bounds for the integer action
            unit_lb = ((action[i] - 0.5) + 0.5 - min_vals[i]) / rangeP1[i]
            unit_ub = ((action[i] + 0.5) + 0.5 - min_vals[i]) / rangeP1[i]
            
            print(f"  Unit interval: [{unit_lb.item():.6f}, {unit_ub.item():.6f}]")
            
            # Check the distribution's truncation bounds
            alpha = (0.0 - center[i]) * precision[i]
            beta = (1.0 - center[i]) * precision[i]
            print(f"  Alpha (z-score at 0): {alpha.item():.6f}")
            print(f"  Beta (z-score at 1): {beta.item():.6f}")
            
            # Check log CDF values
            print(f"  _log_F_alpha: {dist_i._log_F_alpha.item():.6f}")
            print(f"  _log_F_beta: {dist_i._log_F_beta.item():.6f}")
            print(f"  _log_Z: {dist_i._log_Z.item():.6f}")
            
            # Compute log probability of the interval
            logp_interval = dist_i.logp_interval(unit_lb, unit_ub)
            logp_scaled = logp_interval - rangeP1[i].log()
            
            print(f"  logp_interval: {logp_interval.item():.6f}")
            print(f"  log(rangeP1): {rangeP1[i].log().item():.6f}")
            print(f"  Final logp: {logp_scaled.item():.6f}")
            
            if torch.isinf(logp_scaled):
                print(f"  WARNING: -inf detected!")
                
                # Debug the interval computation
                z_low = (unit_lb - center[i]) * precision[i]
                z_high = (unit_ub - center[i]) * precision[i]
                log_cdf_low = torch.special.log_ndtr(z_low)
                log_cdf_high = torch.special.log_ndtr(z_high)
                
                print(f"  Debug interval computation:")
                print(f"    z_low: {z_low.item():.6f}")
                print(f"    z_high: {z_high.item():.6f}")
                print(f"    log_ndtr(z_low): {log_cdf_low.item():.6f}")
                print(f"    log_ndtr(z_high): {log_cdf_high.item():.6f}")
                
                # Check if the interval is outside the support
                if unit_ub <= 0.0 or unit_lb >= 1.0:
                    print(f"    ERROR: Interval completely outside [0,1] support!")
                elif z_high < alpha.item():
                    print(f"    ERROR: Interval below truncation lower bound!")
                elif z_low > beta.item():
                    print(f"    ERROR: Interval above truncation upper bound!")

    # Now replicate the full computation
    print("\n" + "-"*30)
    print("FULL COMPUTATION WITH AUTOCAST:")
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=True):
        dist = GaussianActionDistribution(center, precision)
        
        # Get the log probabilities using the DiscreteActor logic
        unit_lb, unit_ub = model.actors._unit_interval_of_integer_samples(action.unsqueeze(0))
        unit_lb, unit_ub = unit_lb.squeeze(0), unit_ub.squeeze(0)
        
        logprobs = dist.logp_interval(unit_lb, unit_ub) - rangeP1.log()
        
        print(f"Logprobs per action: {logprobs.detach().float().cpu().numpy()}")
        print(f"Total logprob: {logprobs.sum().item():.6f}")

# %% Compute losses (matching _train_step)
print("\nComputing losses...")

T = obs.shape[0]
num_envs_per_minibatch = len(minibatch_env_indices)

# Create dummy actual values for private info losses
actual_settlement = torch.zeros((num_envs_per_minibatch,), device=device)
actual_private_roles = torch.zeros((num_envs_per_minibatch, args.players), device=device).long()

# Private loss weights (from trainer init)
pinfo_loss_weights = 1 - torch.arange(0, 1, 1./args.num_steps).to(device)
pinfo_loss_weights = (pinfo_loss_weights - pinfo_loss_weights.min()) / (pinfo_loss_weights.max() - pinfo_loss_weights.min())
pinfo_loss_weights = (0.5 ** (pinfo_loss_weights / args.pdecay_tau)).unsqueeze(1)
pinfo_loss_weights = pinfo_loss_weights / pinfo_loss_weights.mean()

from jaxtyping import Float, Bool, jaxtyped
def vtrace_losses(
    rewards: Float[torch.Tensor, "T B"],
    dones: Bool[torch.Tensor, "T B"],
    ref_logprobs: Float[torch.Tensor, "T B"],
    logprobs: Float[torch.Tensor, "T B"], # Gradient-attached logprobs 
    values: Float[torch.Tensor, "T_plus_1 B"],
    gamma: float = 0.99,
    gae_lambda: float = 0, # (0, 1) interpolates between (1-step TD, MC) respectively 
    log_bar_rho: float = 0.0,
    log_bar_c: float = 0.0): 
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
        log_prob_ratio = logprobs - ref_logprobs
        print(f"log_prob_ratio: {log_prob_ratio.min().item()}, {log_prob_ratio.max().item()}")
        # Clip log ratio to prevent exp() overflow - log(bar_rho) and log(bar_c)
        log_policy_clip = torch.clamp(log_prob_ratio, max=log_bar_rho)
        log_vtrace_clip = torch.clamp(log_prob_ratio, max=log_bar_c)
        
        policy_clip = log_policy_clip.exp() # [T B]
        vtrace_clip = log_vtrace_clip.exp() # [T B]

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
    T = rewards.shape[0]
    with torch.no_grad(): 
        last_gae_lambda = 0 
        vtrace_advantage = torch.zeros_like(rewards)
        for t in reversed(range(T)):
            if t == T - 1:
                next_values = vtrace_values[-1]
            else:
                next_values = vtrace_values[t+1]
            ctd = 1. - dones[t]
            delta = rewards[t] - cur_values[t] + gamma * next_values * ctd
            vtrace_advantage[t] = last_gae_lambda = (
                delta + gamma * gae_lambda * ctd * last_gae_lambda)
            
        # vtrace_advantage = rewards + gamma * (1 - dones) *vtrace_values[1:] - cur_values 
        vtrace_advantage = (vtrace_advantage - vtrace_advantage.mean()) / (vtrace_advantage.std() + 1e-8)
    pg_loss = - policy_clip * logprobs * vtrace_advantage 
    pg_loss = pg_loss.mean()
    print('Hello')

    return {
        'policy_loss': pg_loss,
        'value_loss': value_loss,
        'value_r2': value_r2}

# Compute losses
with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=True):
    # Settlement loss
    pred_settlement_loss = torch.nn.functional.smooth_l1_loss(
        pred_settlement, 
        actual_settlement.unsqueeze(0).expand(T, num_envs_per_minibatch), 
        reduction='none')
    pred_settlement_loss = (pred_settlement_loss * pinfo_loss_weights).mean()
    
    # Private roles loss
    flattened_pred_private_roles = pred_private_roles.reshape(-1, 3)
    flattened_actual_private_roles = actual_private_roles.unsqueeze(0).expand(
        T, num_envs_per_minibatch, args.players).reshape(-1)
    pred_private_roles_loss = torch.nn.functional.cross_entropy(
        flattened_pred_private_roles, flattened_actual_private_roles, reduction='none'
    ).reshape(T, num_envs_per_minibatch, args.players)
    pred_private_roles_loss = (pred_private_roles_loss * pinfo_loss_weights.unsqueeze(-1)).mean()
    
    # V-trace losses
    augmented_values = torch.cat([
        values, torch.zeros_like(values[-1])[None, :]])
    
    # Note: vtrace_losses now takes log_bar_rho and log_bar_c instead of bar_rho and bar_c
    vtrace_results = vtrace_losses(
        rewards,
        dones,
        logprobs,
        new_logprob,
        augmented_values,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        log_bar_rho=0.0,  # log(1.0) = 0.0
        log_bar_c=0.0)    # log(1.0) = 0.0
    
    entropy_value = entropy.mean()
    entropy_loss = -args.entropy_coef * entropy_value
    
    # Total loss
    loss = (vtrace_results['policy_loss'] 
            + entropy_loss
            + vtrace_results['value_loss'] * args.vf_coef
            + pred_settlement_loss * args.psettlement_coef
            + pred_private_roles_loss * args.proles_coef)
    
    print(f"\nLoss components:")
    print(f"  - policy_loss: {vtrace_results['policy_loss'].item():.6f}")
    print(f"  - value_loss: {vtrace_results['value_loss'].item():.6f}")
    print(f"  - entropy_loss: {entropy_loss.item():.6f}")
    print(f"  - pred_settlement_loss: {pred_settlement_loss.item():.6f}")
    print(f"  - pred_private_roles_loss: {pred_private_roles_loss.item():.6f}")
    print(f"  - total_loss: {loss.item():.6f}")
    print(f"  - loss has NaN: {torch.isnan(loss).item()}")

# %% Compute gradients
print("\nComputing gradients...")
loss.backward()

# Check for NaN gradients
print("\nChecking for NaN gradients:")
nan_params = []
for name, param in model.named_parameters():
    if param.grad is not None:
        has_nan = torch.isnan(param.grad).any().item()
        if has_nan:
            nan_count = torch.isnan(param.grad).sum().item()
            total_count = param.grad.numel()
            nan_params.append(name)
            print(f"  - {name}: {nan_count}/{total_count} NaNs ({100*nan_count/total_count:.2f}%)")
            
            # Show some statistics about the gradient
            valid_grads = param.grad[~torch.isnan(param.grad)]
            if len(valid_grads) > 0:
                print(f"    Valid grads: min={valid_grads.min().item():.6e}, max={valid_grads.max().item():.6e}, mean={valid_grads.mean().item():.6e}")

if not nan_params:
    print("  No NaN gradients found!")
else:
    print(f"\nTotal parameters with NaN gradients: {len(nan_params)}")

# %% Analyze specific problematic parameters
if nan_params:
    print("\nAnalyzing first few parameters with NaN gradients...")
    
    for i, param_name in enumerate(nan_params[:3]):  # Analyze first 3
        print(f"\n{i+1}. Parameter: {param_name}")
        
        for name, param in model.named_parameters():
            if name == param_name:
                print(f"  - Shape: {param.shape}")
                print(f"  - Param stats: min={param.min().item():.6e}, max={param.max().item():.6e}, mean={param.mean().item():.6e}, std={param.std().item():.6e}")
                
                if param.grad is not None:
                    grad = param.grad
                    nan_mask = torch.isnan(grad)
                    print(f"  - Gradient shape: {grad.shape}")
                    print(f"  - NaN percentage: {100 * nan_mask.sum().item() / grad.numel():.2f}%")
                    
                    # Try to understand the pattern
                    if len(grad.shape) == 2:  # Matrix
                        nan_rows = nan_mask.any(dim=1).sum().item()
                        nan_cols = nan_mask.any(dim=0).sum().item()
                        print(f"  - Rows with NaN: {nan_rows}/{grad.shape[0]} ({100*nan_rows/grad.shape[0]:.1f}%)")
                        print(f"  - Cols with NaN: {nan_cols}/{grad.shape[1]} ({100*nan_cols/grad.shape[1]:.1f}%)")
                    elif len(grad.shape) == 1:  # Vector
                        print(f"  - First NaN indices: {nan_mask.nonzero().flatten().tolist()[:10]}...")

# %% Check intermediate activations with hooks
print("\n" + "="*50)
print("CHECKING INTERMEDIATE ACTIVATIONS")
print("="*50)

# Re-run forward pass with hooks to capture activations
activations = {}
def save_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks on key layers
hooks = []
hooks.append(model.encoder.register_forward_hook(save_activation('encoder')))
hooks.append(model.transformer.register_forward_hook(save_activation('transformer')))
hooks.append(model.transformer_norm.register_forward_hook(save_activation('transformer_norm')))
hooks.append(model.critic.register_forward_hook(save_activation('critic')))
hooks.append(model.actors.register_forward_hook(save_activation('actors')))

# Also check attention weights if possible
def save_attention(name):
    def hook(module, input, output):
        if hasattr(module, 'self_attn'):
            # Try to capture attention weights
            activations[f'{name}_attn'] = output
    return hook

# Run forward pass again
with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=True):
    with torch.no_grad():
        outputs = model(obs, pinfo_tensor, actions)

# Check activations
for name, act in activations.items():
    has_nan = torch.isnan(act).any().item()
    has_inf = torch.isinf(act).any().item()
    print(f"\n{name}:")
    print(f"  - Shape: {act.shape}")
    print(f"  - Has NaN: {has_nan}")
    print(f"  - Has Inf: {has_inf}")
    if not has_nan and not has_inf:
        print(f"  - Min: {act.min().item():.6e}, Max: {act.max().item():.6e}")
        print(f"  - Mean: {act.mean().item():.6e}, Std: {act.std().item():.6e}")
    else:
        valid_act = act[~(torch.isnan(act) | torch.isinf(act))]
        if valid_act.numel() > 0:
            print(f"  - Valid values - Min: {valid_act.min().item():.6e}, Max: {valid_act.max().item():.6e}")
            print(f"  - Valid values - Mean: {valid_act.mean().item():.6e}, Std: {valid_act.std().item():.6e}")

# Clean up hooks
for hook in hooks:
    hook.remove()

# %% Analyze value and policy outputs specifically
print("\n" + "="*50)
print("ANALYZING VALUE AND POLICY OUTPUTS")
print("="*50)

with torch.no_grad():
    # Check value predictions
    print("\nValue predictions:")
    print(f"  - Shape: {values.shape}")
    print(f"  - Has NaN: {torch.isnan(values).any().item()}")
    print(f"  - Has Inf: {torch.isinf(values).any().item()}")
    if not torch.isnan(values).any():
        print(f"  - Range: [{values.min().item():.6f}, {values.max().item():.6f}]")
        print(f"  - Mean: {values.mean().item():.6f}, Std: {values.std().item():.6f}")
    
    # Check log probabilities
    print("\nLog probabilities (new):")
    print(f"  - Shape: {new_logprob.shape}")
    print(f"  - Has NaN: {torch.isnan(new_logprob).any().item()}")
    print(f"  - Has Inf: {torch.isinf(new_logprob).any().item()}")
    if not torch.isnan(new_logprob).any():
        print(f"  - Range: [{new_logprob.min().item():.6f}, {new_logprob.max().item():.6f}]")
        print(f"  - Mean: {new_logprob.mean().item():.6f}, Std: {new_logprob.std().item():.6f}")
    
    # Check entropy
    print("\nEntropy:")
    print(f"  - Shape: {entropy.shape}")
    print(f"  - Has NaN: {torch.isnan(entropy).any().item()}")
    print(f"  - Has Inf: {torch.isinf(entropy).any().item()}")
    if not torch.isnan(entropy).any():
        print(f"  - Range: [{entropy.min().item():.6f}, {entropy.max().item():.6f}]")
        print(f"  - Mean: {entropy.mean().item():.6f}, Std: {entropy.std().item():.6f}")

# %% Check V-trace intermediate values
print("\n" + "="*50)
print("V-TRACE INTERMEDIATE VALUES")
print("="*50)

with torch.no_grad():
    # Recompute V-trace to inspect intermediate values
    log_prob_ratio = new_logprob - logprobs
    print(f"new_logprob: {new_logprob.min().item()}, {new_logprob.max().item()}")
    print(f"logprobs: {logprobs.min().item()}, {logprobs.max().item()}")
    log_policy_clip = torch.clamp(log_prob_ratio, max=0.0)  # log(1.0) = 0.0
    log_vtrace_clip = torch.clamp(log_prob_ratio, max=0.0)
    
    policy_clip = log_policy_clip.exp()
    vtrace_clip = log_vtrace_clip.exp()
    
    print(f"Log prob ratio stats:")
    print(f"  - Range: [{log_prob_ratio.min().item():.6f}, {log_prob_ratio.max().item():.6f}]")
    print(f"  - Has NaN: {torch.isnan(log_prob_ratio).any().item()}")
    
    print(f"\nPolicy clip (exp of clipped log ratio):")
    print(f"  - Range: [{policy_clip.min().item():.6f}, {policy_clip.max().item():.6f}]")
    print(f"  - Has NaN: {torch.isnan(policy_clip).any().item()}")
    print(f"  - Has Inf: {torch.isinf(policy_clip).any().item()}")

# %% Summary and recommendations
print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)

if nan_params:
    print(f"\nNaN gradients detected in {len(nan_params)} parameters")
    print("\nTop affected parameters:")
    for i, name in enumerate(nan_params[:5]):
        print(f"  {i+1}. {name}")
    
    print("\nPossible causes:")
    print("1. Numerical overflow in loss computation")
    print("2. Extreme values in model outputs")
    print("3. Division by zero in normalization")
    print("4. Exploding gradients due to high learning rate")
    print("5. Attention weights becoming too sharp (exp overflow)")
    
    print("\nRecommendations:")
    print("1. Check learning rate and warmup schedule")
    print("2. Add gradient clipping before NaN detection")
    print("3. Check for extreme values in rewards/observations")
    print("4. Consider using gradient accumulation with smaller batches")
    print("5. Add value clipping to prevent extreme value predictions")
    print("6. Check if attention logits are becoming too large")
else:
    print("\nNo NaN gradients detected in this reproduction!")
    print("The issue might be intermittent or require specific conditions.")
    print("\nNote: The saved debug state had NaN gradients, but they weren't reproduced.")
    print("This could mean:")
    print("1. The issue is related to optimizer state (momentum/adam statistics)")
    print("2. There's randomness in the forward pass not captured")
    print("3. The issue requires specific accumulation of gradients over time")
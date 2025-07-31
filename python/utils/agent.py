import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from typing import Dict, Optional

# -------------------------------- utilities -------------------------------- #
def layer_init(layer: nn.Module, std: float = np.sqrt(2.0), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(features, features))
        self.ln = nn.LayerNorm(features)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.ln(self.fc1(x)))


class HighLowModel(nn.Module):
    def __init__(
        self,
        high_low_config: Dict[str, int],
        hidden_size: int = 512,
        num_residual_blocks: int = 4,
    ):
        super().__init__()
        # Device will be set when .to(device) is called
        self.input_length = (
            11
            + high_low_config["steps_per_player"] * high_low_config["players"] * 6
            + high_low_config["players"] * 2
        )

        self.backbone = nn.Sequential(
            layer_init(nn.Linear(self.input_length, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            *[ResidualBlock(hidden_size) for _ in range(num_residual_blocks)],
        )

        self.actors = nn.ModuleDict({
            "bid_price": layer_init(nn.Linear(hidden_size, high_low_config["max_contract_value"])),
            "bid_size": layer_init(nn.Linear(hidden_size, 1 + high_low_config["max_contracts_per_trade"])),
            "ask_price": layer_init(nn.Linear(hidden_size, high_low_config["max_contract_value"])),
            "ask_size": layer_init(nn.Linear(hidden_size, 1 + high_low_config["max_contracts_per_trade"])),
        })

        self.critic = layer_init(nn.Linear(hidden_size, 1), std=1.0)
        
        # We will store compiled functions here, initially None
        self._compiled_forward = None
        self._compiled_value = None

    def compile(self, mode: str = "max-autotune", fullgraph: bool = True):
        """
        ✅ Compiles the performance-critical methods of the model.
        This should be called *after* the model is moved to its target device.
        """
        self._compiled_forward = torch.compile(self._forward_impl, mode=mode, fullgraph=fullgraph)
        self._compiled_value = torch.compile(self._value_impl, mode=mode, fullgraph=fullgraph)
        return self

    def forward(self, x: torch.Tensor, actions: Optional[Dict[str, torch.Tensor]] = None):
        """
        If compiled, uses the optimized graph. Otherwise, runs in eager mode.
        """
        if self._compiled_forward:
            return self._compiled_forward(x, actions)
        return self._forward_impl(x, actions)

    def get_value(self, x: torch.Tensor):
        """
        If compiled, uses the optimized graph. Otherwise, runs in eager mode.
        """
        if self._compiled_value:
            return self._compiled_value(x)
        return self._value_impl(x)

    @torch.no_grad()
    def sample_actions(self, obs: np.ndarray):
        """
        Inference helper. This does not need to be compiled itself, as the
        heavy lifting (backbone, heads) is done by the compiled functions it calls.
        """
        # The device is inferred from the model's parameters
        device = next(self.parameters()).device
        with torch.inference_mode():
            obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
            # Note: We call the internal _forward_impl here to avoid the overhead of the
            # compiled function check, since we only need the distributions.
            latent = self.backbone(obs_t)
            dists = {k: Categorical(logits=head(latent)) for k, head in self.actors.items()}
            actions = {k: d.sample() for k, d in dists.items()}

            np_actions = {k: v.cpu().numpy() for k, v in actions.items()}
            np_actions["bid_price"] += 1
            np_actions["ask_price"] += 1
            return np_actions

    def _value_impl(self, x):
        latent = self.backbone(x)
        return self.critic(latent).squeeze(-1)

    def _forward_impl(self, x, actions=None):
        latent = self.backbone(x)
        dists = {k: Categorical(logits=head(latent)) for k, head in self.actors.items()}
        values = self.critic(latent).squeeze(-1)

        if actions is None:
            sampled = {k: d.sample() for k, d in dists.items()}
            actions_for_lp = sampled
            output_actions = {k: (v + 1) if "price" in k else v for k, v in sampled.items()}
        else:
            actions_for_lp = {
                "bid_price": actions["bid_price"] - 1,
                "ask_price": actions["ask_price"] - 1,
                "bid_size": actions["bid_size"],
                "ask_size": actions["ask_size"],
            }
            output_actions = actions

        log_prob = sum(dists[k].log_prob(actions_for_lp[k]) for k in dists)
        entropy = sum(d.entropy() for d in dists.values())

        return output_actions, log_prob, entropy, values
    

class HighLowPrivateModel(nn.Module):
    def __init__(
        self,
        high_low_config: Dict[str, int],
        hidden_size: int = 512,
        num_residual_blocks: int = 4,
    ):
        super().__init__()
        # Device will be set when .to(device) is called
        self.input_length = (
            11
            + high_low_config["steps_per_player"] * high_low_config["players"] * 6
            + high_low_config["players"] * 2
        )

        self.backbone = nn.Sequential(
            layer_init(nn.Linear(self.input_length, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            *[ResidualBlock(hidden_size) for _ in range(num_residual_blocks)],
        )

        self.actors = nn.ModuleDict({
            "bid_price": layer_init(nn.Linear(hidden_size, high_low_config["max_contract_value"])),
            "bid_size": layer_init(nn.Linear(hidden_size, 1 + high_low_config["max_contracts_per_trade"])),
            "ask_price": layer_init(nn.Linear(hidden_size, high_low_config["max_contract_value"])),
            "ask_size": layer_init(nn.Linear(hidden_size, 1 + high_low_config["max_contracts_per_trade"])),
        })

        self.private_info_model = nn.ModuleDict({
            'settle_price': nn.Linear(hidden_size, 1),
            'private_roles': nn.Linear(hidden_size, high_low_config['players'] * 3),
        })

        self.critic = layer_init(nn.Linear(hidden_size, 1), std=1.0)
        self.num_players = high_low_config['players']
        
        # We will store compiled functions here, initially None
        self._compiled_forward = None
        self._compiled_value = None

    def compile(self, mode: str = "max-autotune", fullgraph: bool = True):
        """
        ✅ Compiles the performance-critical methods of the model.
        This should be called *after* the model is moved to its target device.
        """
        self._compiled_forward = torch.compile(self._forward_impl, mode=mode, fullgraph=fullgraph)
        self._compiled_value = torch.compile(self._value_impl, mode=mode, fullgraph=fullgraph)
        return self

    def forward(self, x: torch.Tensor, actions: Optional[Dict[str, torch.Tensor]] = None):
        """
        If compiled, uses the optimized graph. Otherwise, runs in eager mode.
        """
        if self._compiled_forward:
            return self._compiled_forward(x, actions)
        return self._forward_impl(x, actions)

    def get_value(self, x: torch.Tensor):
        """
        If compiled, uses the optimized graph. Otherwise, runs in eager mode.
        """
        if self._compiled_value:
            return self._compiled_value(x)
        return self._value_impl(x)

    @torch.no_grad()
    def sample_actions(self, obs: np.ndarray):
        """
        Inference helper. This does not need to be compiled itself, as the
        heavy lifting (backbone, heads) is done by the compiled functions it calls.
        """
        # The device is inferred from the model's parameters
        device = next(self.parameters()).device
        with torch.inference_mode():
            obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
            # Note: We call the internal _forward_impl here to avoid the overhead of the
            # compiled function check, since we only need the distributions.
            latent = self.backbone(obs_t)
            dists = {k: Categorical(logits=head(latent)) for k, head in self.actors.items()}
            actions = {k: d.sample() for k, d in dists.items()}

            np_actions = {k: v.cpu().numpy() for k, v in actions.items()}
            np_actions["bid_price"] += 1
            np_actions["ask_price"] += 1
            return np_actions

    def _value_impl(self, x):
        latent = self.backbone(x)
        return self.critic(latent).squeeze(-1)

    def _forward_impl(self, x, actions=None):
        latent = self.backbone(x)
        dists = {k: Categorical(logits=head(latent)) for k, head in self.actors.items()}
        values = self.critic(latent).squeeze(-1)

        if actions is None:
            sampled = {k: d.sample() for k, d in dists.items()}
            actions_for_lp = sampled
            output_actions = {k: (v + 1) if "price" in k else v for k, v in sampled.items()}
        else:
            actions_for_lp = {
                "bid_price": actions["bid_price"] - 1,
                "ask_price": actions["ask_price"] - 1,
                "bid_size": actions["bid_size"],
                "ask_size": actions["ask_size"],
            }
            output_actions = actions

        log_prob = sum(dists[k].log_prob(actions_for_lp[k]) for k in dists)
        entropy = sum(d.entropy() for d in dists.values())

        predicted_settlement = self.private_info_model['settle_price'](latent).squeeze(-1)
        predicted_private_roles = self.private_info_model['private_roles'](
            latent).reshape(-1, self.num_players, 3)
        
        forward_results = {
            'action': output_actions,
            'log_prob': log_prob,
            'entropy': entropy,
            'values': values,
            'private_info': {
                'predicted_settlement': predicted_settlement,
                'predicted_private_roles': predicted_private_roles,
            }
        }

        return forward_results
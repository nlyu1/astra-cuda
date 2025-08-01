import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import sys 
sys.path.append('../')
from model_components import layer_init, ResidualBlock
from typing import Dict

# ~350ms forward pass; extremely slow. 
class HighLowGRUModel(nn.Module):
    def __init__(self, args, env):
        super().__init__()
        self.B, self.F = env.observation_shape()
        self.T = args.steps_per_player 
        self.P = args.players
        self.M = args.max_contract_value
        self.S = args.max_contracts_per_trade
        self.device = torch.device(f'cuda:{args.device_id}')
        self.rnn_n_layers = args.rnn_n_layers
        self.rnn_hidden_size = args.rnn_hidden_size
        self.hidden_size = args.hidden_size

        self.encoder = nn.Sequential(
            ResidualBlock(self.F, args.hidden_size),
            *[ResidualBlock(args.hidden_size, args.hidden_size) for _ in range(args.pre_rnn_blocks - 1)])
        self.residual_resize = nn.Linear(args.hidden_size, self.rnn_hidden_size)
        self.gru_core = nn.GRU(
            input_size = args.hidden_size,
            hidden_size = self.rnn_hidden_size,
            num_layers = args.rnn_n_layers,
            batch_first = True)
        # hidden = torch.zeros(n_layers, B, rnn_hidden_size).to(device)
        # output, hidden = gru_core(mock_input, hidden)
        self.decoder = nn.Sequential(
            ResidualBlock(self.rnn_hidden_size, args.hidden_size),
            *[ResidualBlock(args.hidden_size, args.hidden_size) for _ in range(args.post_rnn_blocks - 1)])

        self.actors = nn.ModuleDict({
            "bid_price": layer_init(nn.Linear(args.hidden_size, self.M)),
            "ask_price": layer_init(nn.Linear(args.hidden_size, self.M)),
            "bid_size": layer_init(nn.Linear(args.hidden_size, 1 + self.S)),
            "ask_size": layer_init(nn.Linear(args.hidden_size, 1 + self.S))})

        self.pinfo_model = nn.ModuleDict({
            'settle_price': nn.Linear(args.hidden_size, 1),
            'private_roles': nn.Linear(args.hidden_size, self.P * 3)})
        self.critic = nn.Linear(args.hidden_size, 1)
        
        self.rnn_initial_belief = self.initial_belief()

    def _batch_forward(self, x, actions):
        """
        x: [B, T, F]
        actions: [B, T, 4] type long, for (bid_price, ask_price, bid_size, ask_size)
        Ranges: 1 <= bid_price, ask_price <= max_contract_value
                0 <= bid_size, ask_size <= max_contracts_per_trade

        Returns:
            values: [B, T]
            logprobs: [B, T]. Logprobs of the reference actions given policy of the current step. 
            entropy: [B, T]

        Used during training. 
        """
        # _, new_logprob, entropy, values = self.agent(
        # batch_obs, reference_actions)
        B, T, F = x.shape
        bid_px, ask_px, bid_sz, ask_sz = actions.unbind(dim=-1)
        # Safety assertions for debugging purposes. Disabled for compilation compatibility
        # assert actions.shape == (B, T, 4)
        # assert (B == self.B and T == self.T and F == self.F), "Batch_forward shape must match"
        # assert (bid_px.min() > 0 and ask_px.min() > 0), "Bid and ask prices must be positive"
        # assert (bid_px.max() <= self.M and ask_px.max() <= self.M), "Bid and ask prices must be less than or equal to max contract value"
        # assert (bid_sz.min() >= 0 and ask_sz.min() >= 0), "Bid and ask sizes must be non-negative"
        # assert (bid_sz.max() <= self.S and ask_sz.max() <= self.S), "Bid and ask sizes must be less than or equal to max contracts per trade"

        flattened_x = x.reshape(-1, F)
        flat_features = self.encoder(flattened_x) # [B*T, rnn_hidden_size]
        rnn_features, _ = self.gru_core(flat_features.reshape(B, T, self.hidden_size), self.rnn_initial_belief) # [B, T, rnn_hidden_size]
        backbone_features = self.residual_resize(flat_features) + rnn_features.reshape(B*T, self.rnn_hidden_size) # Residual connections 
        backbone_features = self.decoder(backbone_features) # [B*T, hidden_size]

        dists = {k: Categorical(logits=self.actors[k](backbone_features)) for k in self.actors} # [B*T, M] or [B*T, 1+S]
        actions_for_logprobs = {
            'bid_price': bid_px - 1, 'ask_price': ask_px - 1, # Make zero-indexed
            'bid_size': bid_sz, 'ask_size': ask_sz}
        logprobs = sum(dists[k].log_prob(actions_for_logprobs[k].reshape(B*T)) for k in dists).reshape(B, T)
        entropy = sum(d.entropy() for d in dists.values()).reshape(B, T)

        values = self.critic(backbone_features).reshape(B, T) # [B, T]
        pinfo_preds = {k: self.pinfo_model[k](backbone_features) for k in self.pinfo_model}

        return {
            'values': values, # [B, T]
            'logprobs': logprobs, # [B, T]
            'entropy': entropy, # [B, T]
            'pinfo_preds': pinfo_preds
        }

    def initial_belief(self):
        return torch.zeros(self.rnn_n_layers, self.B, self.rnn_hidden_size).to(self.device)

    @torch.jit.export
    @torch.inference_mode()
    def sample_actions(self, x: torch.Tensor, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, F]
        hidden: [rnn_n_layers, B, rnn_hidden_size]

        Returns:
            action: [B, 4] denoting (bid_price, ask_price, bid_size, ask_size). Type int. 
            hidden: [rnn_n_layers, B, rnn_hidden_size]

        Used during rollout. 
        """
        B, F = x.shape
        # assert (B == self.B and F == self.F)
        # assert hidden.shape == (self.rnn_n_layers, B, self.rnn_hidden_size)
        flat_features = self.encoder(x) # [B, idden_size]
        rnn_features, new_hidden = self.gru_core(flat_features.reshape(B, 1, self.hidden_size), hidden) # [B, 1, rnn_hidden_size]
        backbone_in = self.residual_resize(flat_features) + rnn_features.reshape(B, self.rnn_hidden_size) # Residual connection across RNN
        backbone_features = self.decoder(backbone_in) # [B, hidden_size]
        dists = {k: Categorical(logits=self.actors[k](backbone_features)) for k in self.actors} # [B, M] or [B, 1+S]
        actions = torch.stack([
            dists['bid_price'].sample() + 1,
            dists['ask_price'].sample() + 1,
            dists['bid_size'].sample(),
            dists['ask_size'].sample(),
        ], dim=-1)
        return {'action': actions.int(), 'hidden': new_hidden}
    
    def compile(self, mode: str = "max-autotune", fullgraph: bool = True):
        """
        Compiles the performance-critical methods of the model.
        This should be called *after* the model is moved to its target device.
        """
        raise NotImplementedError("Compilation not supported for GRU model")
        # self._compiled_batch_forward = torch.compile(self._batch_forward, mode=mode, fullgraph=fullgraph)
        return self

    def forward(self, x, actions = None):
        """
        If compiled, uses the optimized graph. Otherwise, runs in eager mode.
        """
        if self._compiled_batch_forward:
            return self._compiled_batch_forward(x, actions)
        else:
            raise RuntimeError("Model must be compiled before use")
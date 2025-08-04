import torch
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)  # Memory-efficient attention
torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner

import torch.nn as nn
import torch.nn.functional as F
import sys 
sys.path.append('../')
from model_components import ResidualBlock, LearnedPositionalEncoding, BetaActor

class HighLowTransformerModel(nn.Module):
    """
    Transformer-based model for High-Low Trading that replaces GRU with parallel attention.
    
    Key parameters:
    - n_hidden: Embedding dimension (replaces hidden_size)
    - n_head: Number of attention heads
    - n_layer: Number of transformer blocks
    - dropout: Dropout rate for regularization
    """
    def __init__(self, args, env, verbose=True):
        super().__init__()
        self.B, self.F = env.observation_shape()
        self.T = args.steps_per_player 
        self.P = args.players
        self.M = args.max_contract_value
        self.S = args.max_contracts_per_trade
        self.device = torch.device(f'cuda:{args.device_id}')
        
        # Transformer hyperparameters
        self.n_hidden = args.n_hidden
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.n_layer = args.n_layer
        
        # Input encoder
        self.encoder = ResidualBlock(self.F, self.n_embd)
        self.pos_encoding = LearnedPositionalEncoding(self.n_embd, max_len=self.T)
        self.pinfo_numfeatures = 2 + 1 + self.P # see pinfo_tensor method in `env.py`
        
        # Transformer core using memory-efficient configurations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.n_embd,
            nhead=self.n_head,
            dim_feedforward=self.n_embd * 4,
            dropout=0,
            activation="gelu",
            batch_first=False,
            norm_first=True)  # Pre-norm architecture
        
        # Configure encoder for memory efficiency
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layer,
            enable_nested_tensor=False)
        
        # Action heads
        self.actors = BetaActor(self.n_embd, 4)
        
        # Private information prediction heads
        self.pinfo_model = nn.ModuleDict({
            'settle_price': nn.Linear(self.n_embd, 1),
            'private_roles': nn.Linear(self.n_embd, self.P * 3)})
        
        # Value head
        self.critic = nn.Sequential(
            ResidualBlock(self.n_embd + self.pinfo_numfeatures, self.n_hidden),
            ResidualBlock(self.n_hidden, self.n_hidden),
            ResidualBlock(self.n_hidden, self.n_hidden),
            nn.Linear(self.n_hidden, 1))
        
        # Pre-generate causal masks for different sequence lengths to avoid dynamic allocation
        self.register_buffer('causal_mask', nn.Transformer.generate_square_subsequent_mask(self.T, device=self.device))

        if verbose:
            B, F = env.observation_shape()
            T = args.steps_per_player
            print(f"Batch size: {B}, Sequence length: {T}, Feature dim: {F}")
            print(f"Transformer config: {args.n_hidden}d hidden, {args.n_head} heads, {args.n_layer} layers")

            # Count parameters
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (assuming float32)")

    @torch.compile(fullgraph=True, mode="max-autotune", dynamic=True)
    def extract_action_params(self, features):
        """
        Features: [B', D]. Generates unscaled normalizations and scales appropriately. 
        """
        # Means are normalized between 0 and 1. 
        alphas, betas = self.actors(features)
        return {
            'bid_px_alpha': alphas[:, 0],
            'bid_px_beta': betas[:, 0],
            'ask_px_alpha': alphas[:, 1],
            'ask_px_beta': betas[:, 1],
            'bid_sz_alpha': alphas[:, 2],
            'bid_sz_beta': betas[:, 2],
            'ask_sz_alpha': alphas[:, 3],
            'ask_sz_beta': betas[:, 3]}
    
    @torch.compile(fullgraph=True, mode="max-autotune")
    def _action_logprobs(self, action_params, actions):
        """
        action_params: {'bid_px_mean', 'bid_px_std', ...} -> [B']
        actions: [B', 4]

        Returns: logprobs and entropy of shape [B']
        """
        bid_px, ask_px, bid_sz, ask_sz = actions.reshape(-1, 4).unbind(dim=-1)
        bidpx_lp, bidpx_ent = BetaActor.logp_entropy(bid_px, action_params['bid_px_alpha'], action_params['bid_px_beta'], 1, self.M) # [T*B]
        askpx_lp, askpx_ent = BetaActor.logp_entropy(ask_px, action_params['ask_px_alpha'], action_params['ask_px_beta'], 1, self.M) # [T*B]
        bidsz_lp, bidsz_ent = BetaActor.logp_entropy(bid_sz, action_params['bid_sz_alpha'], action_params['bid_sz_beta'], 0, self.S) # [T*B]
        asksz_lp, asksz_ent = BetaActor.logp_entropy(ask_sz, action_params['ask_sz_alpha'], action_params['ask_sz_beta'], 0, self.S) # [T*B]
        return (bidpx_lp + askpx_lp + bidsz_lp + asksz_lp), (bidpx_ent + askpx_ent + bidsz_ent + asksz_ent)
    
    @torch.compile(fullgraph=True, mode="max-autotune")
    def _batch_forward(self, x, pinfo_tensor, actions):
        """
        Fully parallel forward pass across all timesteps.
        
        x: [T, B, F]
        pinfo_tensor: [B, Pinfo_numfeatures]. Automatically expanded to all timesteps. Used for decentralized critic only. 
        actions: [T, B, 4] type long
        
        Returns:
            values: [T, B]
            logprobs: [T, B]
            entropy: [T, B]
            pinfo_preds / private_roles: [T, B, num_players, 3]
            pinfo_preds / settle_price: [T, B]
        """
        T, B, F = x.shape
        encoded = self.encoder(x.view(T*B, F)).view(T, B, self.n_embd)
        encoded = self.pos_encoding(encoded) # [T, B, D]
        # [T, B, D]. causal_mask shape [T, T] with -inf on strict upper-diagonal
        features = self.transformer(encoded, mask=self.causal_mask, is_causal=True).view(T * B, self.n_embd) # [T * B, D]
        critic_features = torch.cat([
            features.view(T, B, self.n_embd),
            pinfo_tensor.expand(T, B, self.pinfo_numfeatures)
        ], dim=-1).reshape(T*B, self.n_embd + self.pinfo_numfeatures)
        values = self.critic(critic_features).reshape(T, B)

        action_params = self.extract_action_params(features)
        logprobs, entropy = self._action_logprobs(action_params, actions.view(T*B, 4))

        pinfo_preds = {k: self.pinfo_model[k](features) for k in self.pinfo_model}
        pinfo_preds['private_roles'] = pinfo_preds['private_roles'].reshape(T, B, self.P, 3)
        pinfo_preds['settle_price'] = pinfo_preds['settle_price'].reshape(T, B)
        
        return {
            'values': values,
            'logprobs': logprobs.reshape(T, B),
            'entropy': entropy.reshape(T, B),
            'pinfo_preds': pinfo_preds
        }

    @torch.inference_mode()
    def incremental_forward(self, x, step):
        if step == 0:
            assert self.context.numel() == 0, f"Context must be empty for first step, but got {self.context.shape}"
        else:
            assert self.context.shape[0] == step, f"Context must be of length {step}, but got {self.context.shape[0]}"
        outputs = self.incremental_forward_with_context(x, self.context)
        self.context = outputs['context']
        return {
            'action': outputs['action'], 
            'logprobs': outputs['logprobs'], 
            'pinfo_preds': outputs['pinfo_preds']}

    @torch.inference_mode()
    def incremental_forward_with_context(self, x, prev_context):
        """
        x: [B, F]
        prev_context: [T_sofar, B, D]. Post-encoder, pre-posEncoding

        Returns:
            action: [B, 4]
            logprobs: [B]
            context: [T_sofar+1, B, D]
        """
        # with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
        assert x.shape[1] == self.F, f"Expected observation feature dim {self.F}, got {x.shape[1]}"
        assert prev_context.numel() == 0 or prev_context.shape[2] == self.n_embd, f"Expected context embedding dim {self.n_embd}, got {prev_context.shape[2]}"
        
        B, F = x.shape
        encoded = self.encoder(x).view(1, B, self.n_embd)
        
        if prev_context.numel() == 0: # First timestep
            context = encoded
        else: # Concatenate with previous context
            context = torch.cat([prev_context, encoded], dim=0)
        features = self._incremental_core(context)
        action_params = self.extract_action_params(features)
        bid_px = BetaActor.sample(action_params['bid_px_alpha'], action_params['bid_px_beta'], 1, self.M)
        ask_px = BetaActor.sample(action_params['ask_px_alpha'], action_params['ask_px_beta'], 1, self.M)
        bid_sz = BetaActor.sample(action_params['bid_sz_alpha'], action_params['bid_sz_beta'], 0, self.S)
        ask_sz = BetaActor.sample(action_params['ask_sz_alpha'], action_params['ask_sz_beta'], 0, self.S)

        logprobs, _entropy = self._action_logprobs(action_params, torch.stack([bid_px, ask_px, bid_sz, ask_sz], dim=-1))
        pinfo_preds = {k: self.pinfo_model[k](features) for k in self.pinfo_model}
        pinfo_preds['private_roles'] = pinfo_preds['private_roles'].reshape(B, self.P, 3)
        pinfo_preds['settle_price'] = pinfo_preds['settle_price'].reshape(B)
        return {
            'action': torch.stack([bid_px, ask_px, bid_sz, ask_sz], dim=-1), 
            'logprobs': logprobs, 
            'pinfo_preds': pinfo_preds,
            'context': context}

    def initial_context(self):
        """Returns initial context (empty tensor for compatibility)."""
        return torch.zeros(0, device=self.device)

    def reset_context(self):
        self.context = self.initial_context()

    @torch.compile(mode="max-autotune", dynamic=True)
    def _incremental_core(self, context: torch.Tensor) -> torch.Tensor:
        """
        Sample actions for a single timestep given augmented context. 
        context: [T_sofar, B, D]
        """
        context = self.pos_encoding(context)
        T_ctx = context.size(0)
        mask = nn.Transformer.generate_square_subsequent_mask(T_ctx, device=context.device)
        features = self.transformer(context, mask=mask, is_causal=True)[-1] # [B, D]
        return features # [B, D]

    def forward(self, x, pinfo_tensor, actions=None):
        assert x.shape[0] == self.T and x.shape[2] == self.F, f"Expected observation shape[0, 2] {self.T, self.F}, got {x.shape}"
        assert actions.shape[0] == self.T and actions.shape[2] == 4, f"Expected action shape[0, 2] {self.T, 4}, got {actions.shape}"
        return self._batch_forward(x, pinfo_tensor, actions)
from dataclasses import dataclass
import os

@dataclass
class Args:
    ##### Logging #####
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 20250716
    """seed of the experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "HighLowTrading_IMPALA"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    ##### Algorithm specific arguments #####
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_steps: int = 20
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 1
    """the discount factor gamma"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 2
    """the K epochs to update the policy.
    `num_minibatches * update_epochs` trades between sampling efficiency and training stability."""
    ent_coef: float = 0.1
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""

    ##### Model specific arguments #####
    n_hidden: int = 512
    """hidden size in FC components of the model"""
    n_embd: int = 128
    """attention size"""
    n_head: int = 2
    """the number of attention heads in the model"""
    n_layer: int = 2
    """number of transformer blocks"""
    pre_encoder_blocks: int = 2
    post_decoder_blocks: int = 1
    """number of residual blocks before and after transformer"""

    #### Training-specific arguments ####
    num_iterations: int = 100000000
    """Number of vectorized times to interact with environment per meta-step. 
    Per each iteration, we sample (num_envs * num_steps) frames to form a batch, 
    then split into (num_minibatches) minibatches and update network for (update_epochs) epochs."""
    iterations_per_pool_update: int = 3000
    """Number of iterations between pool updates. Set to none to do static training."""

    #### Logging specification ####
    iterations_per_checkpoint: int = 1500
    """Number of iterations between checkpoints"""
    iterations_per_heavy_logging: int = 1000
    """Number of iterations between heavy logging"""

    #### Game specification ### 
    steps_per_player: int = 20
    """the number of trading steps per player before game ends"""
    max_contracts_per_trade: int = 5
    """the maximum number of contracts in a single trade"""
    customer_max_size: int = 5
    """the maximum position size for customers"""
    max_contract_value: int = 30
    """the maximum value a contract can have""" 
    players: int = 5
    """the number of players in the game"""

    ##### Environment execution #####
    threads_per_block: int = 64
    """the number of threads per block"""
    num_blocks: int = 128
    """the number of environments per worker"""
    num_markets: int = None
    """the number of markets in the game. Filled in at runtime"""
    device_id: int = 0
    """the device id to use"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    # minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    total_timesteps: int = 0
    """the total number of timesteps (computed in runtime)"""
    num_envs: int = 0
    """the number of environments (computed in runtime)"""

    def fill_runtime_args(self):
        assert self.num_steps == self.steps_per_player, "Training pipeline handles special case."
        assert self.batch_size % self.num_minibatches == 0, "Batch size must be divisible by number of minibatches"

        self.project_name = f"HighLowTrading_{self.exp_name}"
        self.num_envs = self.num_blocks * self.threads_per_block
        self.num_markets = self.num_envs
        # This is the number of on-policy samples we collect per learning phase ``
        self.batch_size = int(self.num_envs * self.num_steps)
        # Number of times we sample from the environment 
        self.total_timesteps = self.num_iterations * self.batch_size
        print(f'Sampling {self.batch_size} frames per iteration across {self.num_envs} environments')
        print(f'Per-gradient step batch size: {self.batch_size // self.num_minibatches}. {self.num_minibatches} gradient steps for {self.update_epochs} updates')

        if self.iterations_per_pool_update is None:
            self.iterations_per_pool_update = self.num_iterations

    def get_game_config(self):
        game_config = {
            'steps_per_player': self.steps_per_player,
            'max_contracts_per_trade': self.max_contracts_per_trade,
            'customer_max_size': self.customer_max_size,
            'max_contract_value': self.max_contract_value,
            'players': self.players,
            'num_markets': self.num_envs,
            'threads_per_block': self.threads_per_block,
            'device_id': self.device_id,
        }
        return game_config
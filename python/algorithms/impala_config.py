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
    num_steps: int = 8
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.995
    """the discount factor gamma"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 2
    """the K epochs to update the policy.
    `num_minibatches * update_epochs` trades between sampling efficiency and training stability."""
    ent_coef: float = 0.05
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""

    #### Training-specific arguments ####
    num_iterations: int = 100000000
    """Number of vectorized times to interact with environment per meta-step. 
    Per each iteration, we sample (num_envs * num_steps) frames to form a batch, 
    then split into (num_minibatches) minibatches and update network for (update_epochs) epochs."""
    iterations_per_pool_update: int = None 
    """Number of iterations between pool updates. Set to none to do static training."""

    #### Logging specification ####
    iterations_per_checkpoint: int = 2000
    """Number of iterations between checkpoints"""
    iterations_per_heavy_logging: int = 1000
    """Number of iterations between heavy logging"""

    #### Game specification ### 
    steps_per_player: int = 8
    """the number of trading steps per player before game ends"""
    max_contracts_per_trade: int = 1
    """the maximum number of contracts in a single trade"""
    customer_max_size: int = 2
    """the maximum position size for customers"""
    max_contract_value: int = 10
    """the maximum value a contract can have""" 
    players: int = 5
    """the number of players in the game"""

    ##### Environment execution #####
    env_workers: int = 16
    """the number of workers. Total envs = envs_per_worker * env_workers"""
    envs_per_worker: int = 512
    """the number of environments per worker"""

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

        self.num_envs = self.envs_per_worker * self.env_workers
        # This is the number of on-policy samples we collect per learning phase ``
        self.batch_size = int(self.num_envs * self.num_steps)
        # self.minibatch_size = int(self.batch_size // self.num_minibatches) # Per-gradient step batch size 
        # assert self.minibatch_size * self.num_minibatches == self.batch_size
        # Number of times we sample from the environment 
        self.total_timesteps = self.num_iterations * self.batch_size
        print(f'Sampling {self.batch_size} frames per iteration')
        print(f'Per-gradient step batch size: {self.batch_size / self.num_minibatches}. {self.num_minibatches} gradient steps for {self.update_epochs} updates')

        if self.iterations_per_pool_update is None:
            self.iterations_per_pool_update = self.num_iterations

    def get_game_config(self):
        game_config = {
            'steps_per_player': self.steps_per_player,           # Number of trading steps per player
            'max_contracts_per_trade': self.max_contracts_per_trade,     # Maximum contracts in a single trade
            'customer_max_size': self.customer_max_size,           # Maximum position size for customers
            'max_contract_value': self.max_contract_value,         # Maximum value a contract can have
            'players': self.players                      # Total number of players in the game
        }
        return game_config
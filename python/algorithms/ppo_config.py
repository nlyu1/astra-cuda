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
    wandb_project_name: str = "HighLowTrading_PPO"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    ##### Algorithm specific arguments #####
    total_timesteps: int = 1000000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_steps: int = 8
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    ent_coef: float = 0.05
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold. Off-policy surrogate training aborts upon exceeding."""

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
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    num_envs: int = 0
    """the number of environments (computed in runtime)"""

    def fill_runtime_args(self):
        self.num_envs = self.envs_per_worker * self.env_workers
        # This is the number of on-policy samples we collect per learning phase ``
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches) # Per-gradient step batch size 
        assert self.minibatch_size * self.num_minibatches == self.batch_size
        # Number of times we sample from the environment 
        self.num_iterations = self.total_timesteps // self.batch_size
        print(f'Sampling {self.batch_size} frames per iteration')
        print(f'Per-gradient step batch size: {self.minibatch_size}. {self.num_minibatches} gradient steps for {self.update_epochs} updates')

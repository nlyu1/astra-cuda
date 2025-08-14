# %% 
import random
import time
import gc
import os
import shutil
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.expanduser("~/.cache/torch_compile")
os.environ["TORCH_COMPILE_CACHE_DIR"] = os.path.expanduser("~/.cache/torch_compile")

import numpy as np
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
torch._dynamo.config.cache_size_limit = 256 # Enable more aggressive caching
import wandb
from pathlib import Path
from tqdm import tqdm
import sys 

sys.path.append('./src')

from high_low.agent import HighLowTransformerModel
from high_low.config import Args
from high_low.env import HighLowTrading
from high_low.logger import HighLowLogger
from high_low.rollouts import RolloutGenerator
from high_low.vtrace import HighLowVTraceTrainer, HighLowVTraceBuffer
from sampler import ThompsonSampler
from timer import Timer, SegmentTimer
import tyro

args = tyro.cli(Args)
# args = Args()

args.run_name = f"HighLowMain__{args.exp_name}__{args.seed}__{int(time.time())}"
args.fill_runtime_args()
print(args)
game_config = args.get_game_config()
env = HighLowTrading(game_config)
device = torch.device(f'cuda:{args.device_id}')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed);

# %% Initialize project directory and the pool 
python_root = Path(__file__).parent
project_dir = python_root / 'checkpoints' / args.exp_name
if project_dir.exists():
    do_erase = input(f'Project directory {project_dir} already exists. Do you want to erase it? (y/n): ')
    if do_erase == 'y':
        shutil.rmtree(project_dir)
        print(f'    Erased project directory {project_dir} and starting from scratch')
    else:
        print(f'    Initializing pool from existing checkpoints in {project_dir}')
project_dir.mkdir(parents=True, exist_ok=True)
print('Saving project files to', project_dir)

# Initialize the pool by writing to seed.pt under project directory 
seed_path = project_dir / 'seed.pt'
initial_agent = HighLowTransformerModel(args, env, verbose=False).to(device)
if args.checkpoint_name != "":
    # To initialize the pool, it suffices to write weights
    path = python_root / 'checkpoints' / (args.checkpoint_name + ".pt")
    weights = torch.load(path, map_location=device, weights_only=False)['model_state_dict']
    initial_agent.load_state_dict(weights, strict=True)
    print(f"Loaded checkpoint {args.checkpoint_name} from {path}")
torch.save({'model_state_dict': initial_agent.state_dict()}, seed_path)
print(f"Saved pool seed to {seed_path}")

### Load benchmark checkpoint and initialize rollout generator ### 
benchmark_weights = None 
if args.benchmark_checkpoint_name != "": 
    benchmark_path = python_root / 'checkpoints' / (args.benchmark_checkpoint_name + ".pt")
    benchmark_weights = torch.load(benchmark_path, map_location=device, weights_only=False)['model_state_dict']
    try:
        initial_agent.load_state_dict(benchmark_weights, strict=True)
        print(f"Loaded benchmark checkpoint {args.benchmark_checkpoint_name} from {benchmark_path}")
    except Exception as e:
        print(f"Benchmark checkpoint {args.benchmark_checkpoint_name} cannot be loaded: {e}")
        exit()
    rollout_generator = RolloutGenerator(args)

cannot_be_loaded = set()
def update_from_dir(sampler): # sampler should be ThompsonSampler
    # Match against existing weights inside bandit.py. If doesn't exist, then add to pool 
    for path in project_dir.glob('*.pt'):
        name = path.stem
        if name in cannot_be_loaded or name in sampler.player_names:
            continue 
        try: # Load weights and add to bandit 
            weights = torch.load(path, weights_only=False, map_location=device)['model_state_dict']
            initial_agent.load_state_dict(weights)
            sampler.register_player(name, weights)
            print(f'Added {name} from {path} into pool. Pool now contains {len(sampler.player_names)} elements.')
        except Exception as e:
            cannot_be_loaded.add(path)
            print(f'{path} cannot be loaded: {e}')
    return sampler
pool = ThompsonSampler(args.effective_sampler_memory_size)
pool = update_from_dir(pool)

# %% Initialize buffer and main agent 
num_features = env.num_features()
buffer = HighLowVTraceBuffer(args, num_features, device)

main_agent = HighLowTransformerModel(args, env).to(device)
if args.checkpoint_name != "":
    main_agent.load_state_dict(weights, strict=False)
trainer = HighLowVTraceTrainer(args, main_agent)
npc_agents = [ # Empty initialization; these will be filled with weights from pool 
    HighLowTransformerModel(args, env, verbose=False).to(device)
    for _ in range(game_config['players'] - 1)]

observation_buffer = env.new_observation_buffer()
reward_buffer = env.new_reward_buffer() # [N, P]
returns_buffer = env.new_reward_buffer() # [N, P]
player_reward_buffer = env.new_player_reward_buffer() # [N]
self_play = True # Flag determining whether to play against pool samples or self 

pbar = tqdm(range(args.num_iterations))
done_zeros, done_ones = torch.zeros(args.num_envs, device=device).float(), torch.ones(args.num_envs, device=device).float()

# Pre-allocate GPU buffers for distribution parameters (outside loop for reuse)
dist_params_buffer = {
    'center': torch.zeros(args.num_steps, 4, device=device),
    'precision': torch.zeros(args.num_steps, 4, device=device)}

# %%

global_step = 0
timer = Timer()
segment_timer = SegmentTimer()
logger = HighLowLogger(args)
gc.disable() # Disable garbage collection for performance

segment_timer.tick('init_agents')
for iteration in pbar:
    if iteration > 0 and iteration % 200 == 0: # Manual GC every 100 iterations
        gc.collect() 
    self_play = random.random() < args.self_play_prob
    # Pick npc agents and player offset
    player_offset = np.random.randint(0, game_config['players'])
    npc_selection = pool.sample_batch(game_config['players'] - 1)
    for j, agent in enumerate(npc_agents):
        agent.load_state_dict(
            main_agent.state_dict() if self_play 
            else npc_selection['object'][j])
        agent.reset_context()
    main_agent.reset_context()

    ### Rollout ### 
    segment_timer.tick('rollout')
    settlement_preds, private_role_preds = [], []
    buffer.pinfo_tensor = env.pinfo_tensor()
    for step in range(args.num_steps):
        global_step += args.num_envs 

        for npc_id in range(player_offset):
            assert (env.current_player() == npc_id), f"Environment must be ready for NPC {npc_id}, but {env.current_player()} is acting."
            env.fill_observation_tensor(observation_buffer)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=True):
                with torch.inference_mode():
                    torch.compiler.cudagraph_mark_step_begin()
                    npc_actions = npc_agents[npc_id].incremental_forward(observation_buffer, step)['action']
            env.step(npc_actions)
        
        #### Agent action ####
        # Prior to stepping & observing new rewards, flush the cumulated rewards to buffer
        # Environment cannot terminate before player makes action (due to the design of our environment)
        assert env.current_player() == player_offset, f"Environment must be ready for player {player_offset}, but {env.current_player()} is acting."
        if step > 0: # Only update if step > 0, since step 0 is the initial state 
            env.fill_rewards_since_last_action(buffer.rewards[step - 1])
            buffer.update_late_stats(
                {'dones': done_zeros}, step - 1)

        # observation, action, log_probs, value can be calculated immediately and stored in buffer
        env.fill_observation_tensor(buffer.obs[step])
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=True):
            with torch.inference_mode():
                torch.compiler.cudagraph_mark_step_begin()
                forward_results = main_agent.incremental_forward(buffer.obs[step], step)
        action, log_probs = forward_results['action'], forward_results['logprobs']
        settlement_preds.append(forward_results['pinfo_preds']['settle_price'].clone())
        private_role_preds.append(forward_results['pinfo_preds']['private_roles'].argmax(dim=-1))

        # Store distribution parameters
        for k, v in forward_results['action_params'].items():
            dist_params_buffer[k][step] = v.mean(0)
        buffer.update({
            # Observations are implicitly updated above. 
            'actions': action,
            'logprobs': log_probs,
        }, step)
        assert (env.current_player() == player_offset), "Environment must be ready for player"
        env.step(action)

        #### Remaining player actions ####
        for player_id in range(player_offset + 1, game_config['players']):
            npc_id = player_id - 1
            assert (env.current_player() == player_id), f"Environment must be ready for NPC {npc_id}, but {env.current_player()} is acting."
            env.fill_observation_tensor(observation_buffer)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=True):
                with torch.inference_mode():
                    npc_actions = npc_agents[npc_id].incremental_forward(observation_buffer, step)['action']
            env.step(npc_actions)

    assert env.terminal(), "Environment just terminate after num_steps"
    if env.terminal():
        segment_timer.tick('update_buffer')
        env.fill_returns(returns_buffer) # This is the cumulative rewards, for logging. 
        env.fill_rewards_since_last_action(buffer.rewards[step], player_offset)
        buffer.update_late_stats({
            'rewards': player_reward_buffer,
            'dones': done_ones}, step)

        # Log and update pool sampler before resetting the environment 
        env_info = env.expose_info()
        env_pinfo_targets = env.get_pinfo_targets() 
        if not self_play: # Only log and update pool when not self-playing
            #### Update pool by computing whether main agent obtained higher returns than opponents ####
            segment_timer.tick('update_pool')
            main_agent_returns = returns_buffer[:, player_offset].mean(0)
            opponent_returns = torch.cat([returns_buffer[:, :player_offset], returns_buffer[:, player_offset + 1:]], dim=1).mean(0)
            opponent_wins = (opponent_returns > main_agent_returns).float().cpu().numpy()
            pool.update_parameters(npc_selection['index'], opponent_wins)

            #### Logging ####
            segment_timer.tick('logging')
            settlement_preds_stacked = torch.stack(settlement_preds, dim=0)
            logging_inputs = {
                'returns': returns_buffer,
                'offset': player_offset,
                'settlement_preds': settlement_preds_stacked,
                'private_role_preds': torch.stack(private_role_preds, dim=0),
                'infos': env_info | env_pinfo_targets,
                'dist_params': dist_params_buffer,
                'segment_timer': {f'performance/{k}': v for k, v in segment_timer.elapsed_times.items()}}
            
            # Only incur heavy logging when we're in seat 0 and after a certain interval 
            heavy_logging_update = (
                logger.counter - logger.last_heavy_counter > args.iterations_per_heavy_logging 
                and player_offset == 0)
            if heavy_logging_update:
                # Benchmark against benchmark checkpoint 
                if benchmark_weights is not None:
                    benchmark_payoffs = {}
                    for benchmark_offset in range(game_config['players']):
                        benchmark_state_dicts = [
                            benchmark_weights if j != benchmark_offset else main_agent.state_dict()
                            for j in range(game_config['players'])]
                        model_payoffs = rollout_generator.generate_rollout(benchmark_state_dicts)[benchmark_offset, :, 0]
                        for j, role_name in enumerate(['goodValue', 'badValue', 'highLow', 'customer', 'avg']):
                            benchmark_payoffs[f'benchmark {role_name}/{benchmark_offset}'] = model_payoffs[j]
                    for role_name in ['goodValue', 'badValue', 'highLow', 'customer', 'avg']:
                        benchmark_payoffs[f'benchmark {role_name}/avg'] = np.mean([
                            benchmark_payoffs[f'benchmark {role_name}/{benchmark_offset}']
                            for benchmark_offset in range(game_config['players'])])
                    logging_inputs['benchmark_payoffs'] = benchmark_payoffs
            logger.update_stats(logging_inputs, global_step, heavy_updates=heavy_logging_update)

        # Populate buffer's actual private info. See `env.py` env_pinfo_target method. Used for private info loss calculation. 
        buffer.actual_private_roles.copy_(env_pinfo_targets['pinfo_targets'], non_blocking=True)
        buffer.actual_settlement.copy_(env_pinfo_targets['settlement_values'], non_blocking=True)
        
        ### Reset environment ###
        env.reset()
    assert (env.current_player() == 0), "Next environment must be ready for player"

    #### Fill buffer end. Update trainer ####
    segment_timer.tick('trainer step')
    running_sps = timer.tick(args.num_envs * args.num_steps)
    wandb.log({"performance/SPS": running_sps}, step=global_step)

    update_dictionary = buffer.get_update_dictionary()
    trainer_results = trainer.train(update_dictionary)
    wandb.log(trainer_results, step=global_step)
    segment_timer.tick('checkpointing')
    saved_new_checkpoint = trainer.save_checkpoint(iteration, name='main')
    if saved_new_checkpoint:
        pool = update_from_dir(pool) # Update pool with new checkpoint 
        pool.save(project_dir / 'pool.pkl')

# Re-enable garbage collection
gc.enable()
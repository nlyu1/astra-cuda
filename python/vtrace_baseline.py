# %% Single-agent play against fixed opponents
import random
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
import wandb
from tqdm import trange, tqdm
import sys 

sys.path.append('./src')

from high_low.agent import HighLowTransformerModel
from high_low.config import Args
from high_low.env import HighLowTrading
from high_low.logger import HighLowLogger
from high_low.impala import HighLowImpalaTrainer, HighLowImpalaBuffer
from timer import Timer, OneTickTimer

from arena import Arena 

args = Args()
args.meta_steps = 10
args.exp_name = 'vtrace_single'
args.num_iterations = 10000000000000
args.checkpoint_interval = 2000 

# impala configs
args.psettlement_coef = 0.1
args.proles_coef = 0.05
args.pdecay_tau = 0.4

args.run_name = f"HighLowTradingVTrace__{args.exp_name}__{args.seed}__{int(time.time())}"
args.fill_runtime_args()
game_config = args.get_game_config()
env = HighLowTrading(game_config)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed);

# %% Argument post-processing and logging 

device = torch.device(f'cuda:{args.device_id}')
loads = []
initial_agents = {}
for path, name in loads:
    initial_agents[name] = HighLowTransformerModel(
        args, env).to(device)
    initial_agents[name].compile()
    initial_agents[name].load_state_dict(
        torch.load(path, map_location=device, weights_only=False)['model_state_dict'])
for j in range(args.players - 1 - len(loads)):
    initial_agents[f'Random{j}'] = HighLowTransformerModel(
        args, env).to(device)
    initial_agents[f'Random{j}'].compile()

num_features = env.num_features()
pool = Arena(env, initial_agents, device)
print('Debug printout')
pool.debug_printout()

buffers = [
    HighLowImpalaBuffer(args, num_features, device)
    for _ in range(2)]

local_agent = HighLowTransformerModel(args, env).to(device)
local_agent.compile()
trainer = HighLowImpalaTrainer(
    args, local_agent, 
    checkpoint_interval=args.checkpoint_interval,
    device=device)

# %%

global_step = 0
timer = Timer()

round_new_agent_name = 'challenger'
# device = 'cuda' 
# args = actor_shared_args['args']

run = wandb.init(
    project=args.wandb_project_name,
    entity=args.wandb_entity,
    config=vars(args),
    name=args.run_name,
    save_code=True,
    dir="/tmp/high_low_ppo_wandb")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# buffers = actor_shared_args['buffers']
# ready_q = actor_shared_args['ready_q']
# free_q = actor_shared_args['free_q']

game_config = args.get_game_config()

### Initialize environment, arena, and logger ###

# inference_agent = actor_shared_args['inference_agent']
# local_agent = deepcopy(inference_agent)
# local_agent.compile()

global_step, total_iterations = 0, 0
timer, ticker = Timer(), OneTickTimer()
agent_name = str(total_iterations) # This is only updated per pool update 

observation_buffer = env.new_observation_buffer()
reward_buffer = env.new_reward_buffer() # [N, P]
returns_buffer = env.new_reward_buffer() # [N, P]
player_reward_buffer = env.new_player_reward_buffer() # [N]

pbar = tqdm(range(args.num_iterations))
for iteration in pbar: 
    # actor_shared_args['shared_info']['total_iterations'] = total_iterations
    # buf_id = free_q.get()
    buf_id = 0 
    training_wait_time = ticker.tick()

    # Pick npc agents and player offset
    player_offset = np.random.randint(0, game_config['players'])
    npc_agent_names = pool.select_topk(game_config['players'] - 1)
    npc_agents = [pool.agents[name] for name in npc_agent_names]
    for agent in npc_agents:
        agent.reset_context()
    local_agent.reset_context()
    round_agent_names = ( # Used for registering pool result 
        [npc_agent_names[j] for j in range(player_offset)] 
        + [agent_name]
        + [npc_agent_names[j - 1] for j in range(player_offset + 1, game_config['players'])])

    ### Rollout ### 

    settlement_preds, private_role_preds = [], []
    for step in range(args.num_steps):
        global_step += args.num_envs 

        for npc_id in range(player_offset):
            assert (env.current_player() == npc_id), f"Environment must be ready for NPC {npc_id}, but {env.current_player()} is acting."
            env.fill_observation_tensor(observation_buffer)
            print(f'Player {npc_id} acted upon', observation_buffer[0])

            npc_actions = npc_agents[npc_id].incremental_forward(observation_buffer, step)['action']
            print(f'Player {npc_id} action:', npc_actions[0])
            env.step(npc_actions)
        
        #### Agent action ####
        # Prior to stepping & observing new rewards, flush the cumulated rewards to buffer
        # Environment cannot terminate before player makes action (due to the design of our environment)
        assert env.current_player() == player_offset, f"Environment must be ready for player {player_offset}, but {env.current_player()} is acting."

        if step > 0: # Only update if step > 0, since step 0 is the initial state 
            env.fill_rewards_since_last_action(buffers[buf_id].rewards[step - 1])
            print(f'Player {player_offset} reward since last action: {buffers[buf_id].rewards[step - 1]}')
            buffers[buf_id].update_late_stats(
                {'dones': torch.zeros(args.num_envs).to(device).float()}, step - 1)

        # observation, action, log_probs, value can be calculated immediately 
        env.fill_observation_tensor(buffers[buf_id].obs[step])
        print(f'Player! {player_offset} observation: {buffers[buf_id].obs[step][0]}')
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            with torch.inference_mode():
                forward_results = local_agent.incremental_forward(buffers[buf_id].obs[step], step)
        action, log_probs = forward_results['action'], forward_results['logprobs']
        settlement_preds.append(forward_results['pinfo_preds']['settle_price'].clone())
        private_role_preds.append(forward_results['pinfo_preds']['private_roles'].argmax(dim=-1))

        buffers[buf_id].update({
            # Observations are implicitly updated above. 
            'actions': action,
            'logprobs': log_probs,
        }, step)
        # Step the environment 
        assert (env.current_player() == player_offset), "Environment must be ready for player"
        env.step(action)

        # Step remaining players
        for player_id in range(player_offset + 1, game_config['players']):
            npc_id = player_id - 1
            assert (env.current_player() == player_id), f"Environment must be ready for NPC {npc_id}, but {env.current_player()} is acting."
            env.fill_observation_tensor(observation_buffer)
            print(f'Player {player_id} acted upon', observation_buffer[0])

            npc_actions = npc_agents[npc_id].incremental_forward(observation_buffer, step)['action']
            print(f'Player {player_id} action:', npc_actions[0])
            env.step(npc_actions)

        if env.terminal():
            # This is the cumulative rewards, for logging. 
            env.fill_returns(returns_buffer)
            env.fill_rewards_since_last_action(buffers[buf_id].rewards[step], player_offset)

            # Reset environment 
            buffers[buf_id].update_late_stats({
                'rewards': player_reward_buffer,
                'dones': torch.ones(args.num_envs).to(device).float()}, step)

            # Unfortunately, logging must happen before environment is reset
            env_info = env.states.expose_info()
            # logging_inputs = {
            #     'returns': env_returns,
            #     'offset': player_offset,
            #     'settlement_preds': torch.stack(settlement_preds, dim=0).cpu(),
            #     'private_role_preds': torch.stack(private_role_preds, dim=0).cpu(),
            #     'infos': env_info}
            # # Only incur heavy logging when we're in seat 0 and after a certain interval 
            # heavy_logging_update = (
            #     logger.counter - logger.last_heavy_counter > args.iterations_per_heavy_logging 
            #     and player_offset == 0)
            # logger.update_stats(logging_inputs, global_step, heavy_updates=heavy_logging_update)
            pool.register_playout_scores(returns_buffer.mean(0), round_agent_names)
            pbar.set_postfix({'mean_score': f'{returns_buffer.mean(0)[player_offset].item():.2f}'})
            # if heavy_logging_update:
            #     pool.log_stats()

            # Populate buffer's actual private info
            # See `high_low_trading.h` for the definition of the expose_info() function. 
            settlement_tensor = torch.from_numpy(env_info['contract'][..., -1]).pin_memory().to(device, non_blocking=True) # Shape [num_envs]
            # Use a copy here to avoid modifying the original result (used for logging)
            private_roles_tensor = torch.tensor(env_info['info_roles']).pin_memory().to(device, non_blocking=True) # Shape [num_envs, num_players]
            private_roles_tensor[private_roles_tensor == 0] = 1 # Aggregate good value (0) and bad value (1) into value cheater. 
            # (valueCheater 1, highLow 2, customer 3) -> (valueCheater 0, highLow 1, customer 2)
            
            # buffers[buf_id].actual_private_roles.copy_(private_roles_tensor - 1, non_blocking=True)
            # buffers[buf_id].actual_settlement.copy_(settlement_tensor, non_blocking=True)
            
            ### Reset environment ###
            env.reset()
        assert (env.current_player() == 0), "Next environment must be ready for player"

    #### Fill buffer end. Update trainer ####
    # assert env.is_initial_move_state(), "Environment must have just been reset"
    # running_sps = timer.tick(args.num_envs * args.num_steps)
    # wandb.log({"performance/SPS": running_sps}, step=global_step)
    # rollout_time = ticker.tick()
    # ready_q.put(buf_id)

    # if iteration > 10: # Skip first 10 iterations to avoid noise 
    #     learner_logs = deepcopy(actor_shared_args['shared_info']['learner_logs'])
    #     learner_logs['performance/RolloutTime'] = rollout_time
    #     learner_logs['performance/TrainingWaitTime'] = training_wait_time
    #     wandb.log(learner_logs)
    # local_agent.load_state_dict(actor_shared_args['inference_agent'].state_dict())

    # total_iterations += 1

    # if total_iterations % args.iterations_per_pool_update == 0:
    #     pool.register_agent(local_agent, agent_name)
    #     pool.random_maintainance()
    #     pool.debug_printout()
    #     agent_name = str(total_iterations)
run.finish()

def learner_process(learner_shared_args):
    """
    Learner keeps track of training. 
    """
    device = 'cuda' 
    args = learner_shared_args['args']
    inference_agent = learner_shared_args['inference_agent']

    trainer = HighLowImpalaTrainer(
        args, 
        deepcopy(inference_agent).to(device),
        name='impala', 
        checkpoint_interval=args.iterations_per_checkpoint,
        device=device)
    trainer.agent.compile()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    buffers = learner_shared_args['buffers']
    ready_q = learner_shared_args['ready_q']
    free_q = learner_shared_args['free_q']

    ticker = OneTickTimer()
    while True:
        buf_id = ready_q.get()
        learner_shared_args['shared_info']['learner_logs']['performance/RolloutWaitTime'] = ticker.tick()

        # This is an out of place operation; grabs from shared memory to local 
        update_dictionary = buffers[buf_id].get_update_dictionary()

        results = trainer.train(update_dictionary)
        free_q.put(buf_id)
        learner_shared_args['shared_info']['learner_logs']['performance/TrainingTime'] = ticker.tick()

        for k, v in results.items():
            learner_shared_args['shared_info']['learner_logs'][k] = v
        # Push the newly learned weights to actors 
        inference_agent.load_state_dict(trainer.agent.state_dict())
        trainer.save_checkpoint(learner_shared_args['shared_info']['total_iterations'])
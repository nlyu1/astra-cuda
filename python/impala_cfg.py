import torch 
import torch.nn as nn
import torch.multiprocessing as mp
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

import sys 
sys.path.append('./utils')
sys.path.append('./algorithms')

import random 
import numpy as np 
import wandb 
from copy import deepcopy

import time 
from impala import ImpalaBuffer, HighLowImpalaTrainer, get_uid
from agent import HighLowModel
from high_low_wrapper import HighLowWrapper
from arena import Arena
from timer import Timer, OneTickTimer
from logger import HighLowLogger
from impala_config import Args
from tqdm import trange

def actor_process(actor_shared_args):
    """
    Actor keeps track of Arena management. 
    """
    device = 'cuda' 
    args = actor_shared_args['args']

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

    buffers = actor_shared_args['buffers']
    ready_q = actor_shared_args['ready_q']
    free_q = actor_shared_args['free_q']

    game_config = args.get_game_config()

    ### Initialize environment, arena, and logger ###
    env = HighLowWrapper(args, game_config)
    loads = [
        ('./checkpoints/impala_18.pt', 'starter_18'),
        ('./checkpoints/impala_21.pt', 'starter_21'),
        ('./checkpoints/impala_24.pt', 'starter_24'),
        ('./checkpoints/impala_25.pt', 'starter_25'),
    ]
    initial_agents = {}
    for path, name in loads:
        initial_agents[name] = HighLowModel(game_config, 
            hidden_size=512, num_residual_blocks=5).to(device)
        initial_agents[name].load_state_dict(
            torch.load(path, map_location='cuda', weights_only=False)['model_state_dict'])
    pool = Arena(env, initial_agents)
    logger = HighLowLogger(args, wandb_initialized=True)
    print('Actor: initialized environment, arena, and logger')

    inference_agent = actor_shared_args['inference_agent']
    local_agent = deepcopy(inference_agent)
    local_agent.compile()
    # local_uid = get_uid(local_agent.state_dict())
    
    global_step, total_iterations = 0, 0
    timer, ticker = Timer(), OneTickTimer()
    agent_name = str(total_iterations) # This is only updated per pool update 

    for iteration in trange(args.num_iterations):
        actor_shared_args['shared_info']['total_iterations'] = total_iterations
        buf_id = free_q.get()
        training_wait_time = ticker.tick()

        # Pick npc agents and player offset
        player_offset = np.random.randint(0, game_config['players'])
        npc_agent_names = pool.select_topk(game_config['players'] - 1)
        npc_agents = [pool.agents[name].cuda() for name in npc_agent_names]
        round_agent_names = ( # Used for registering pool result 
            [npc_agent_names[j] for j in range(player_offset)] 
            + [agent_name]
            + [npc_agent_names[j - 1] for j in range(player_offset + 1, game_config['players'])])

        ### Rollout ### 
        # Keeps track of the player's accumulated reward from since its last action to right now. 
        # Used for non-terminal rewards 
        player_reward_cumulation = None

        for step in range(args.num_steps):
            global_step += args.num_envs 

            for npc_id in range(player_offset):
                assert (env.current_player() == npc_id), f"Environment must be ready for NPC {npc_id}"
                npc_obs = env.player_observations(npc_id)
                npc_actions = npc_agents[npc_id].sample_actions(npc_obs)
                npc_actions = np.stack( # num_envs, 4
                    [npc_actions['bid_price'], npc_actions['ask_price'], 
                        npc_actions['bid_size'], npc_actions['ask_size']],
                    axis=-1)
                env.step(npc_actions)
                
                # Only update after player has made an effective move 
                if player_reward_cumulation is not None: 
                    player_reward_cumulation += env.rewards()[:, player_offset]
            
            #### Fill buffer ####
            with torch.no_grad():
                # Prior to stepping & observing new rewards, flush the cumulated rewards 
                # Environment cannot terminate before player makes action (due to the design of our environment)
                if player_reward_cumulation is not None:
                    buffers[buf_id].update_late_stats({
                        'rewards': torch.tensor(player_reward_cumulation).to(device).view(-1),
                        'dones': torch.zeros(args.num_envs).to(device).float()}, step - 1)
                    player_reward_cumulation.fill(0)
                else:
                    player_reward_cumulation = np.zeros(args.num_envs)

                # observation, action, log_probs, value can be calculated immediately 
                obs = torch.from_numpy(env.observations()).pin_memory().to(device, non_blocking=True) 
                forward_results = local_agent(obs)
                action, log_probs = forward_results['action'], forward_results['log_probs']
                action = torch.stack(
                    [action['bid_price'], action['ask_price'], 
                    action['bid_size'], action['ask_size']], dim=-1)
                buffers[buf_id].update({
                    'obs': obs,
                    'actions': action,
                    'logprobs': log_probs,
                }, step)

            # Step the environment 
            assert (env.current_player() == player_offset), "Environment must be ready for player"
            env.step(action.cpu().numpy())
            player_reward_cumulation += env.rewards()[:, player_offset]

            # Step remaining players
            for player_id in range(player_offset + 1, game_config['players']):
                npc_id = player_id - 1
                assert (env.current_player() == player_id), f"Environment must be ready for NPC {npc_id}"
                npc_obs = env.player_observations(player_id)
                npc_actions = npc_agents[npc_id].sample_actions(npc_obs)
                npc_actions = np.stack( # num_envs, 4
                    [npc_actions['bid_price'], npc_actions['ask_price'], npc_actions['bid_size'], npc_actions['ask_size']],
                    axis=-1)
                env.step(npc_actions)
                player_reward_cumulation += env.rewards()[:, player_offset]

            if env.is_terminal():
                # This is the cumulative rewards, for logging. 
                env_returns = env.returns()
                tensor_reward_cumulation = torch.from_numpy(player_reward_cumulation).pin_memory().to(device, non_blocking=True)
                # Reset environment 
                buffers[buf_id].update_late_stats({
                    'rewards': tensor_reward_cumulation.view(-1),
                    'dones': torch.ones(args.num_envs).to(device).float()}, step)
                player_reward_cumulation = None

                # Unfortunately, logging must happen before environment is reset
                logging_inputs = {
                    'returns': env_returns,
                    'offset': player_offset,
                    'infos': env.states.expose_info()}
                # Only incur heavy logging when we're in seat 0 and after a certain interval 
                heavy_logging_update = (
                    logger.counter - logger.last_heavy_counter > args.iterations_per_heavy_logging 
                    and player_offset == 0)
                logger.update_stats(logging_inputs, global_step, heavy_updates=heavy_logging_update)
                pool.register_playout_scores(env_returns.mean(0), round_agent_names)
                if heavy_logging_update:
                    pool.log_stats()
                ### Reset environment ###
                env.reset()
            assert (env.current_player() == 0), "Next environment must be ready for player"

        #### Fill buffer end. Update trainer ####
        assert env.is_initial_move_state(), "Environment must have just been reset"
        running_sps = timer.tick(args.num_envs * args.num_steps)
        wandb.log({"performance/SPS": running_sps}, step=global_step)
        rollout_time = ticker.tick()
        ready_q.put(buf_id)

        if iteration > 10: # Skip first 10 iterations to avoid noise 
            learner_logs = deepcopy(actor_shared_args['shared_info']['learner_logs'])
            learner_logs['performance/RolloutTime'] = rollout_time
            learner_logs['performance/TrainingWaitTime'] = training_wait_time
            wandb.log(learner_logs)
        local_agent.load_state_dict(actor_shared_args['inference_agent'].state_dict())

        total_iterations += 1

        if total_iterations % args.iterations_per_pool_update == 0:
            pool.register_agent(local_agent, agent_name)
            pool.random_maintainance()
            pool.debug_printout()
            agent_name = str(total_iterations)
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


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    ready_q = mp.Queue(maxsize=2)   # filled by actor, consumed by learner
    free_q = mp.Queue(maxsize=2)   # filled by learner, consumed by actor
    free_q.put(0)
    free_q.put(1)

    manager = mp.Manager()
    shared_info = manager.dict()
    shared_info['learner_logs'] = manager.dict()

    # 1. Initialize arguments 
    args = Args()
    args.fill_runtime_args()
    args.exp_name = 'impala_baseline'
    args.run_name = f"HighLowTradingImpala__{args.exp_name}__{args.seed}__{int(time.time())}"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    game_config = args.get_game_config()

    device = 'cuda'
    env = HighLowWrapper(args, game_config)
    num_features = env.observations().shape[-1]
    shared_buffers = [ImpalaBuffer(args, num_features, device) for _ in range(2)]
    
    inference_agent = HighLowModel(game_config, hidden_size=512, num_residual_blocks=5).to(device)
    inference_agent.share_memory()
    
    actor_shared_args = {
        'args': args,
        'shared_info': shared_info, # Contains 'learner_logs'
        'buffers': shared_buffers,
        'ready_q': ready_q,
        'free_q': free_q,
        'inference_agent': inference_agent}
    learner_shared_args = {
        'args': args,
        'shared_info': shared_info, 
        'inference_agent': inference_agent,
        'buffers': shared_buffers,
        'ready_q': ready_q,
        'free_q': free_q}

    # 2. Create and start the processes
    p_learner = mp.Process(
        target=learner_process,
        args=(learner_shared_args,))
    p_actor = mp.Process(
        target=actor_process,
        args=(actor_shared_args,))

    p_learner.start()
    p_actor.start()

    p_learner.join()
    p_actor.join()
# %%
import random
from collections import defaultdict

import torch
import numpy as np
from tqdm import trange
from copy import deepcopy
import matplotlib.pyplot as plt
import wandb 
from high_low_wrapper import *

# %% 

def high_low_playout_scores(agents, env, num_eval_episodes=1):
    """
    Returns average playoff returns of shape [num_eval_episodes, 5]. 
    Randomizes the ordering of agents in the game. 
    """
    eval_agents = [agent.cuda() for agent in agents]

    returns = []
    for j in range(num_eval_episodes):
        agent_ordering = np.arange(5).astype(np.int32)
        np.random.shuffle(agent_ordering)
        inverse_ordering = np.argsort(agent_ordering)

        with torch.no_grad():
            while not env.is_terminal():
                for j in range(5):
                    actions = eval_agents[agent_ordering[j]].sample_actions(env.observations())
                    actions = np.stack( # num_envs, 4
                        [actions['bid_price'], actions['ask_price'], actions['bid_size'], actions['ask_size']],
                        axis=-1)
                    env.step(actions)
        rewards = env.returns()[:, inverse_ordering].mean(0)
        returns.append(rewards)
        env.reset()
    returns = np.concatenate(returns, axis=0).astype(np.float32)
    return returns

# Feel free to pass in agent. Deepcopy is done under the hood. 
class Arena:
    """
    Playout scores can be registered freely, while agents can only be sampled after they're registered. 
    """
    def __init__(self, env, initial_agents):
        self.agents = {k: deepcopy(v) for k, v in initial_agents.items()}
        for k in self.agents:
            self.agents[k].eval()
            self.agents[k].cuda()
        self.playout_scores = defaultdict(list)
        self.eval_env = env 

    def register_agent(self, agent, agent_name):
        """
        From now on, agent will be able to be used 
        """
        self.agents[agent_name] = deepcopy(agent)
        self.agents[agent_name].eval()
        self.agents[agent_name].cuda()
        
    def register_playout_scores(self, scores, agent_names):
        for agent_name, score in zip(agent_names, scores):
            if agent_name in self.agents:
                self.playout_scores[agent_name].append(score)

    def random_maintainance(self, num_iters=10):
        chosen_lists = [
            self.select_topk(5, random_prob=.8)
            for _ in range(num_iters)]
        chosen_names = set()
        for names in chosen_lists:
            for name in names:
                chosen_names.add(name)
        chosen_names = list(chosen_names)

        for chosen_agent_names in chosen_lists:
            episode_scores = high_low_playout_scores(
                [self.agents[n] for n in chosen_agent_names],
                self.eval_env,
                num_eval_episodes=1)
            self.register_playout_scores(episode_scores, chosen_agent_names)

    def get_mean_score(self, agent):
        if len(self.playout_scores[agent]) == 0:
            return 1e10 
        return np.mean(self.playout_scores[agent])
    
    def get_std_score(self, agent):
        if len(self.playout_scores[agent]) <= 3:
            return 1e10 
        return np.std(self.playout_scores[agent])

    def debug_printout(self):
        sorted_info = self.get_sorted_info()
        for name, score, std, prob in zip(
            sorted_info['names'],
            sorted_info['means'],
            sorted_info['stds'],
            sorted_info['choose_prob']):
            playable_string = "playable=" + ("true" if name in self.agents else "false")
            print(f"    {name} ({playable_string}): {score:.2f} ± {std:.2f} (prob={prob:.2f})")

    def select_topk(self, k, random_prob=.2):
        sorted_info = self.get_sorted_info()
        chosen_names = np.random.choice(
            sorted_info['names'], k, replace=False, p=sorted_info['choose_prob'])
        return chosen_names 
    
    def get_sorted_info(self):
        # ---- gather stats ---------------------------------------------------
        names  = list(self.agents.keys())
        means  = np.array([self.get_mean_score(n) for n in names], dtype=float)
        stds   = np.array([self.get_std_score(n)  for n in names], dtype=float)

        # Compute actual ranks (0 = best, 1 = second best, etc.)
        ranks = np.empty_like(means, dtype=int)
        ranks[np.argsort(-means)] = np.arange(len(means))
        
        rank_prob    = 1 / (ranks + 1 + 5)
        rank_prob    = rank_prob / rank_prob.sum()

        # ---- sort by mean so charts look tidy ------------------------------
        order           = np.argsort(means)
        names_sorted    = [names[i] for i in order]
        means_sorted    = means[order]
        stds_sorted     = stds[order]
        choose_sorted   = rank_prob[order]
        return {'names': names_sorted, 'means': means_sorted, 
                'stds': stds_sorted, 'choose_prob': choose_sorted}
    
    # ------------------------- wandb‑native logging ------------------------ #
    def log_stats(self):
        """
        Push two bar-charts to wandb:
        1. Average return for every agent in the pool
        2. Sampling probability under `random_prob`.
        """

        sorted_info = self.get_sorted_info()

        # ---- 1) returns table & bar chart ----------------------------------
        tbl = wandb.Table(columns=["agent", "mean", "std", "sampling_prob"])
        for j in range(len(sorted_info['names'])):
            tbl.add_data(
                sorted_info['names'][j],
                float(sorted_info['means'][j]),
                float(sorted_info['stds'][j]),
                float(sorted_info['choose_prob'][j])
            )

        bar_ret = wandb.plot.bar(
            tbl, "agent", "mean",
            title="Average returns",
        )

        bar_prob = wandb.plot.bar(
            tbl, "agent", "sampling_prob",
            title=f"Sampling probability",
        )
        wandb.log({
            "arena/average_returns": bar_ret,
            "arena/sampling_probability": bar_prob
        })
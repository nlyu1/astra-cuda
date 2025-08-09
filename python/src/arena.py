# %%
import random
from collections import defaultdict, deque

import torch
import numpy as np
from tqdm import trange
from copy import deepcopy
import matplotlib.pyplot as plt
import wandb 


class PlayoutFileLogger:
    def __init__(self, args, playout_file_path): 
        # Create if does not exist, else print-warn and overwrite

    def log_playout_results()

class Arena:
    """
    Playout scores can be registered freely, while agents can only be sampled after they're registered. 
    """
    def __init__(self, env, initial_agents, device, deque_buffer_size=100):
        self.device = device 
        self.agents = {k: deepcopy(v) for k, v in initial_agents.items()}
        for k in self.agents:
            self.agents[k].eval()
            self.agents[k].to(self.device)
        # Use deque to store recent scores with fixed buffer size
        self.deque_buffer_size = deque_buffer_size
        self.score_history = defaultdict(lambda: deque(maxlen=deque_buffer_size))
        # Cache mean and std for efficiency
        self.score_mean = defaultdict(lambda: 1e4)
        self.score_std = defaultdict(lambda: 1e4)
        self.eval_env = env 

    def register_agent(self, agent, agent_name):
        """
        From now on, agent will be able to be used 
        """
        self.agents[agent_name] = deepcopy(agent)
        self.agents[agent_name].eval()
        self.agents[agent_name].to(self.device)
        
    def register_playout_scores(self, scores, agent_names):
        scores = scores.cpu()
        for agent_name, score in zip(agent_names, scores):
            if agent_name in self.agents:
                # Append score to deque (automatically removes oldest if at capacity)
                self.score_history[agent_name].append(score.item())
                # Update cached statistics
                if len(self.score_history[agent_name]) == 0:
                    self.score_mean[agent_name] = 1e4
                    self.score_std[agent_name] = 1e4
                else:
                    self.score_mean[agent_name] = np.mean(self.score_history[agent_name])
                    if len(self.score_history[agent_name]) <= 3:
                        self.score_std[agent_name] = 1e4
                    else:
                        self.score_std[agent_name] = np.std(self.score_history[agent_name])

    def get_mean_score(self, agent):
        return self.score_mean[agent]
    
    def get_std_score(self, agent):
        return self.score_std[agent]

    def debug_printout(self):
        sorted_info = self.get_sorted_info()
        for name, score, std, prob in zip(
            sorted_info['names'],
            sorted_info['means'],
            sorted_info['stds'],
            sorted_info['choose_prob']):
            playable_string = "playable=" + ("true" if name in self.agents else "false")
            print(f"    {name} ({playable_string}): {score:.2f} ± {std:.2f} (prob={prob:.2f})")

    def select_topk(self, k): 
        sorted_info = self.get_sorted_info()
        chosen_names = np.random.choice(
            sorted_info['names'], k, replace=True, p=sorted_info['choose_prob'])
        return chosen_names 
    
    def get_sorted_info(self):
        # ---- gather stats ---------------------------------------------------
        names  = list(self.agents.keys())
        means  = np.array([self.get_mean_score(n) for n in names], dtype=float)
        stds   = np.array([self.get_std_score(n)  for n in names], dtype=float)

        # Compute actual ranks (0 = best, 1 = second best, etc.)
        ranks = np.empty_like(means, dtype=int)
        ranks[np.argsort(-means)] = np.arange(len(means))
        
        normalize_offset = 2
        rank_prob    = 1 / (ranks + 1 + normalize_offset)
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
    def log_stats(self, global_step=None):
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
                float(sorted_info['choose_prob'][j]))

        bar_ret = wandb.plot.bar(
            tbl, "agent", "mean",
            title="Average returns")

        bar_prob = wandb.plot.bar(
            tbl, "agent", "sampling_prob",
            title=f"Sampling probability")
        log_data = {
            "arena/average_returns": bar_ret,
            "arena/sampling_probability": bar_prob}
        
        if global_step is not None:
            wandb.log(log_data, step=global_step)
        else:
            wandb.log(log_data)
# %%
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import torch
from collections import defaultdict
python_path = Path(__file__).parent

sys.path.append(str(python_path / 'src'))
from sampler import *

# %%
# Main analysis
pool_path = python_path / 'checkpoints' / 'normal3'
print(f"Loading checkpoint from {pool_path}...")
sampler = load_from_file(pool_path / 'pool.pkl')

# %%
print("\nCreating plots...")
fig_mean = plot_stat(sampler, data_type="mean", log_scale=False)
fig_mean.show()
# %%
fig_std = plot_stat(sampler, data_type="std", log_scale=True)
fig_std.show()
# %%
fig_alpha = plot_stat(sampler, data_type="alpha", log_scale=True)
fig_alpha.show()
# %%
fig_beta = plot_stat(sampler, data_type="beta", log_scale=True)
fig_beta.show()

# %%
print("\nAnalyzing sampling distribution...")
df_sampling = approximate_sampling_probs(sampler, samples_each_round=1e4, num_rounds=10)
print("\nTop 10 players by sampling probability:")
print(df_sampling.head(10).to_string(index=False, float_format='%.4f'))

# %%
fig_sampling = px.bar(
    df_sampling.head(20), x='name', y='sample_prob', error_y='sample_std',
    title='Sampling Probabilities (Top 20 Players)',
    labels={'sample_prob': 'Sampling Probability', 'name': 'Player'}
)
fig_sampling.update_layout(xaxis_tickangle=-45, width=1200, height=600)
fig_sampling.show()
# %%
from high_low.rollouts import *

names = [
    # 'normal3/normal2_204000',
    'poolrun_selfplayonly/normal2_207000',
    'poolrun_selfplayonly/normal2_207000',
    'poolrun_selfplayonly/normal2_207000',
    'poolrun_selfplayonly/normal2_207000',
    'normal3/main_42000',
]
state_dicts = [
    torch.load(python_path / 'checkpoints' / (name+'.pt'), weights_only=False)['model_state_dict']
    for name in names]
base_args = torch.load(python_path / 'checkpoints' / (names[0]+'.pt'), weights_only=False)['args']

rgen = RolloutGenerator(base_args)
payoff_matrix = rgen.generate_rollout(state_dicts)
payoff_vector = payoff_matrix[:, 4, 0]
payoff_vector
# %%
benchmark_weights = state_dicts[0]
main_weights = state_dicts[4]
benchmark_payoffs = {}
benchmark_payoff_list = []
for benchmark_offset in range(base_args.get_game_config()['players']):
    benchmark_state_dicts = [
        benchmark_weights if j != benchmark_offset else main_weights
        for j in range(base_args.get_game_config()['players'])]
    payoff_matrix = rgen.generate_rollout(benchmark_state_dicts)
    model_payoffs = payoff_matrix[benchmark_offset, :, 0]
    print(payoff_matrix[:, 4, 0])
    benchmark_payoff_list.append(payoff_matrix.clone())
    for j, role_name in enumerate(['goodValue', 'badValue', 'highLow', 'customer', 'avg']):
        benchmark_payoffs[f'benchmark {role_name}/{benchmark_offset}'] = model_payoffs[j]
for role_name in ['goodValue', 'badValue', 'highLow', 'customer', 'avg']:
    benchmark_payoffs[f'benchmark {role_name}/avg'] = np.mean([
        benchmark_payoffs[f'benchmark {role_name}/{benchmark_offset}']
        for benchmark_offset in range(base_args.get_game_config()['players'])])
benchmark_payoffs
# %%

[b[:, 4, 0] for b in benchmark_payoff_list]
# %%

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
fig_alpha = plot_stat(sampler, data_type="alpha", log_scale=False)
fig_alpha.show()
# %%
fig_beta = plot_stat(sampler, data_type="beta", log_scale=False)
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
df_sampling
# %%
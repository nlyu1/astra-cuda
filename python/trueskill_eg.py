# %%

import torch 
from pathlib import Path
import sys
python_root = Path(__file__).parent
sys.path.append(str(python_root / 'src'))

from high_low.config import Args
from high_low.env import HighLowTrading
from high_low.agent import HighLowTransformerModel

checkpoint = torch.load(
    python_root / 'checkpoints' / 'small_seedpool_4000.pt', 
    weights_only=False)
args = checkpoint['args']
# %%
env = HighLowTrading(args.get_game_config())
env.reset()
# %%

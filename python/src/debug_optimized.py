import torch
import sys
sys.path.append('./src')
from high_low.config import Args
from high_low.env import HighLowTrading
from high_low.optimized_transformer_agent import OptimizedTransformerModel
import astra_cuda as astra
astra.register_games()

# Setup
args = Args()
args.n_hidden = 256
args.n_embd = 128
args.n_head = 2
args.n_layer = 1
args.device_id = 0

args.fill_runtime_args()
device = torch.device(f'cuda:{args.device_id}')

game_config = args.get_game_config()
env = HighLowTrading(game_config)

# Create model without compilation first
agent = OptimizedTransformerModel(args, env).to(device)

print(f"\nModel attributes:")
print(f"  self.M = {agent.M}")
print(f"  self.S = {agent.S}")
print(f"  Price logits output: {agent.price_logits.out_features}")
print(f"  Size logits output: {agent.size_logits.out_features}")

B, F = env.observation_shape()
T = args.steps_per_player

print(f"Environment info:")
print(f"  Batch size: {B}")
print(f"  Sequence length: {T}")
print(f"  Feature dim: {F}")
print(f"  Max contract value: {args.max_contract_value}")
print(f"  Max contracts per trade: {args.max_contracts_per_trade}")
print(f"  Players: {args.players}")

# Test without compilation first
mock_input = torch.randn(B, T, F).to(device)
actions = torch.randint(1, 4, (B, T, 4)).to(device)

print("\nTesting without compilation...")
try:
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        outputs = agent._batch_forward(mock_input, actions)
    print("Success! Output keys:", outputs.keys())
    print("Values shape:", outputs['values'].shape)
    print("Logprobs shape:", outputs['logprobs'].shape)
    print("Entropy shape:", outputs['entropy'].shape)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
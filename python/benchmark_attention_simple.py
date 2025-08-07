import torch
import torch.nn as nn
import time
from contextlib import contextmanager
import sys
sys.path.append('./src')
from high_low.config import Args
from high_low.agent import HighLowTransformerModel
from high_low.env import HighLowTrading

@contextmanager
def cuda_timer(name):
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    end = time.perf_counter()
    elapsed = (end - start) * 1000
    print(f"{name}: {elapsed:.2f}ms")
    return elapsed

def profile_model_components():
    # Setup
    args = Args()
    args.fill_runtime_args()
    game_config = args.get_game_config()
    env = HighLowTrading(game_config)
    device = torch.device(f'cuda:{args.device_id}')
    
    B = args.num_envs  # 4096
    T = args.steps_per_player  # 16
    
    print(f"Profiling with batch_size={B}, sequence_length={T}")
    print("-" * 60)
    
    # Create model
    model = HighLowTransformerModel(args, env, verbose=True).to(device)
    model.eval()
    
    # Generate observation
    obs_buffer = env.new_observation_buffer()
    env.fill_observation_tensor(obs_buffer)
    
    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = model.incremental_forward(obs_buffer, 0)
        model.reset_context()
    
    print("\n1. Profile incremental forward (simulating full rollout):")
    step_times = []
    model.reset_context()
    
    total_start = time.perf_counter()
    for step in range(T):
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        
        with torch.no_grad():
            _ = model.incremental_forward(obs_buffer, step)
            
        torch.cuda.synchronize()
        step_end = time.perf_counter()
        step_time = (step_end - step_start) * 1000
        step_times.append(step_time)
        print(f"  Step {step}: {step_time:.2f}ms")
    
    total_time = (time.perf_counter() - total_start) * 1000
    print(f"\nTotal time for {T} steps: {total_time:.2f}ms")
    print(f"Average per step: {sum(step_times)/len(step_times):.2f}ms")
    
    # Profile individual components
    print("\n2. Profile transformer core only:")
    model.reset_context()
    
    # Build up context
    context_times = []
    for t in range(1, T+1):
        # Create dummy context of size [t, B, D]
        context = torch.randn(t, B, args.n_embd, device=device)
        
        with cuda_timer(f"  Context length {t}"):
            _ = model._incremental_core(context)
    
    print("\n3. Theoretical speedup with KV-caching:")
    print(f"  Current approach processes: {sum(range(1, T+1))} = {sum(range(1, T+1))} total positions")
    print(f"  KV-cache would process: {T} positions only")
    print(f"  Theoretical attention speedup: {sum(range(1, T+1))/T:.1f}x")
    
    # Estimate attention fraction
    print("\n4. Component timing breakdown (rough estimate):")
    # These are rough estimates based on single forward
    x = torch.randn(B, env.observation_shape()[1], device=device)
    
    with torch.no_grad():
        with cuda_timer("  Encoder"):
            encoded = model.encoder(x)
        
        with cuda_timer("  Single transformer layer"):
            test_context = torch.randn(1, B, args.n_embd, device=device)
            mask = nn.Transformer.generate_square_subsequent_mask(1, device=device)
            encoder_layer = model.transformer.layers[0]
            _ = encoder_layer(test_context, src_mask=mask, is_causal=True)
        
        features = torch.randn(B, args.n_embd, device=device)
        with cuda_timer("  Actor forward"):
            uniform_samples = torch.rand(B, 4, 3, device=device)
            _ = model.actors.logp_entropy_and_sample(features, uniform_samples)
        
        with cuda_timer("  Critic forward"):
            critic_input = torch.randn(B, args.n_embd + model.pinfo_numfeatures, device=device)
            _ = model.critic(critic_input)
    
    print("\n5. Memory analysis:")
    kv_per_token = 2 * 2 * args.n_layer * args.n_head * (args.n_embd // args.n_head)
    print(f"  KV-cache per token: {kv_per_token} bytes = {kv_per_token/1024:.2f} KB")
    print(f"  KV-cache for full sequence: {kv_per_token * T * B / 1024 / 1024:.2f} MB")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    profile_model_components()
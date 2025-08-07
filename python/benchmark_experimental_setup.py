import torch
import time
from contextlib import contextmanager
import sys
sys.path.append('./src')
from high_low.config import Args
from high_low.agent import HighLowTransformerModel
from high_low.env import HighLowTrading
import torch.nn as nn

@contextmanager
def cuda_timer(name):
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    end = time.perf_counter()
    elapsed = (end - start) * 1000
    print(f"{name}: {elapsed:.2f}ms")

def benchmark_experimental_setup():
    # Use actual experimental parameters
    args = Args()
    args.fill_runtime_args()
    game_config = args.get_game_config()
    env = HighLowTrading(game_config)
    device = torch.device(f'cuda:{args.device_id}')
    
    print(f"Benchmarking with ACTUAL experimental parameters:")
    print(f"  Batch size (num_envs): {args.num_envs}")
    print(f"  Sequence length: {args.num_steps}")
    print(f"  Model config: {args.n_embd}d, {args.n_head} heads, {args.n_layer} layers")
    print(f"  Actor parameters: max_contract_value={args.max_contract_value}, max_contracts_per_trade={args.max_contracts_per_trade}")
    print("-" * 80)
    
    # Create model as in experiment
    model = HighLowTransformerModel(args, env, verbose=True).to(device)
    model.eval()
    
    # Generate observation as in experiment
    obs_buffer = env.new_observation_buffer()
    env.fill_observation_tensor(obs_buffer)
    
    print("\n1. Profile incremental_forward (as used in experiment):")
    
    # Reset and profile full rollout
    model.reset_context()
    step_times = []
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        model.reset_context()
        with torch.no_grad():
            for step in range(args.num_steps):
                _ = model.incremental_forward(obs_buffer, step)
    
    # Actual profiling
    model.reset_context()
    total_start = time.perf_counter()
    
    with torch.no_grad():
        for step in range(args.num_steps):
            torch.cuda.synchronize()
            step_start = time.perf_counter()
            
            output = model.incremental_forward(obs_buffer, step)
            
            torch.cuda.synchronize()
            step_end = time.perf_counter()
            step_time = (step_end - step_start) * 1000
            step_times.append(step_time)
            
            if step < 5 or step >= args.num_steps - 2:
                print(f"  Step {step}: {step_time:.2f}ms")
            elif step == 5:
                print("  ...")
    
    total_time = (time.perf_counter() - total_start) * 1000
    print(f"\nTotal time for {args.num_steps} steps: {total_time:.2f}ms")
    print(f"Average per step: {sum(step_times)/len(step_times):.2f}ms")
    
    print("\n2. Break down a single incremental_forward call:")
    
    # Profile step 8 (middle of sequence)
    model.reset_context()
    with torch.no_grad():
        for step in range(8):
            _ = model.incremental_forward(obs_buffer, step)
    
    # Now profile step 8 in detail
    with torch.no_grad():
        uniform_samples = model._populate_uniform_buffer((obs_buffer.shape[0], 4, 3))
        with cuda_timer("  Total incremental_forward (step 8)"):
            output = model.incremental_forward(obs_buffer, 8)
        
        # Profile the compiled part
        prev_context = model.context
        with cuda_timer("  incremental_forward_with_context (compiled)"):
            _ = model.incremental_forward_with_context(obs_buffer, prev_context, uniform_samples)
    
    print("\n3. Profile the actor operations (NOT compiled in incremental path):")
    
    # Get features from transformer
    with torch.no_grad():
        encoded = model.encoder(obs_buffer).view(1, args.num_envs, model.n_embd)
        context = torch.cat([model.context, encoded], dim=0)
        features = model._incremental_core(context)
        
        with cuda_timer("  actors.logp_entropy_and_sample (DiscreteActor)"):
            action_outputs = model.actors.logp_entropy_and_sample(features, uniform_samples)
    
    print("\n4. Check if actor is the bottleneck:")
    
    # Create a dummy compiled function that includes actor
    @torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
    def dummy_forward_with_actor(x, actors, uniform_samples):
        # Simulate some computation
        features = x + 1.0
        action_outputs = actors.logp_entropy_and_sample(features, uniform_samples)
        return action_outputs
    
    # Warmup
    dummy_x = torch.randn(args.num_envs, model.n_embd, device=device)
    for _ in range(10):
        _ = dummy_forward_with_actor(dummy_x, model.actors, uniform_samples)
    
    with torch.no_grad():
        with cuda_timer("  Compiled function with DiscreteActor"):
            _ = dummy_forward_with_actor(dummy_x, model.actors, uniform_samples)
    
    print("\n5. Compare with training forward (which IS fully compiled):")
    
    # Test the training forward path
    T = args.num_steps
    x = torch.randn(T, args.num_envs, env.observation_shape()[1], device=device)
    pinfo_tensor = env.pinfo_tensor()
    actions = torch.randint(0, 30, (T, args.num_envs, 4), device=device)
    
    # Warmup
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for _ in range(5):
            _ = model(x, pinfo_tensor, actions)
    
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            with cuda_timer("  Training forward (batch, compiled)"):
                _ = model(x, pinfo_tensor, actions)
    
    print("\n6. Summary:")
    print(f"  The DiscreteActor is NOT compiled in the incremental forward path!")
    print(f"  This explains why it's slower than expected during rollouts")
    print(f"  The training path uses _batch_forward which calls logp_entropy, not logp_entropy_and_sample")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    benchmark_experimental_setup()
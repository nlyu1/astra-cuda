import torch
import time
import sys
sys.path.append('./src')
from discrete_actor import DiscreteActor
from discrete_actor_parabolic import DiscreteActor as DiscreteActorParabolic

def test_compilation():
    device = torch.device('cuda:0')
    n_hidden = 256
    n_actors = 4
    batch_size = 4096
    
    # Create both actors
    min_values = torch.tensor([1, 1, 0, 0], device=device)
    max_values = torch.tensor([30, 30, 5, 5], device=device)
    
    actor_triangular = DiscreteActor(n_hidden, n_actors, min_values, max_values).to(device)
    actor_parabolic = DiscreteActorParabolic(n_hidden, n_actors, min_values, max_values).to(device)
    
    # Test inputs
    x = torch.randn(batch_size, n_hidden, device=device)
    uniform_samples = torch.rand(batch_size, n_actors, 3, device=device)
    
    print("Testing compilation behavior...")
    print("-" * 80)
    
    # Test 1: Can we compile the method directly?
    print("\n1. Testing direct compilation of logp_entropy_and_sample:")
    
    for name, actor in [("Triangular", actor_triangular), ("Parabolic", actor_parabolic)]:
        print(f"\n  {name} DiscreteActor:")
        
        # Try fullgraph compilation
        try:
            compiled_method = torch.compile(
                actor.logp_entropy_and_sample,
                fullgraph=True,
                mode="max-autotune-no-cudagraphs"
            )
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = compiled_method(x, uniform_samples)
            
            # Time it
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = compiled_method(x, uniform_samples)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            
            print(f"    ✓ Fullgraph compilation successful! Time: {elapsed:.2f}ms")
            
        except torch._dynamo.exc.Unsupported as e:
            print(f"    ✗ Fullgraph compilation failed: {str(e)[:100]}...")
            
            # Try with graph breaks allowed
            try:
                compiled_partial = torch.compile(
                    actor.logp_entropy_and_sample,
                    fullgraph=False
                )
                
                with torch.no_grad():
                    for _ in range(5):
                        _ = compiled_partial(x, uniform_samples)
                
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    _ = compiled_partial(x, uniform_samples)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                
                print(f"    ~ Partial compilation (with graph breaks) successful! Time: {elapsed:.2f}ms")
                
            except Exception as e2:
                print(f"    ✗ Even partial compilation failed: {type(e2).__name__}")
    
    # Test 2: What about when it's inside another compiled function?
    print("\n2. Testing compilation when DiscreteActor is called inside a compiled function:")
    
    def model_forward(actor, x, uniform_samples):
        # Simulate what happens in incremental_forward_with_context
        features = x + 1.0  # Dummy transformation
        return actor.logp_entropy_and_sample(features, uniform_samples)
    
    for name, actor in [("Triangular", actor_triangular), ("Parabolic", actor_parabolic)]:
        print(f"\n  {name} DiscreteActor:")
        
        try:
            compiled_model = torch.compile(
                model_forward,
                fullgraph=True,
                mode="max-autotune-no-cudagraphs"
            )
            
            # Test
            with torch.no_grad():
                _ = compiled_model(actor, x, uniform_samples)
            print(f"    ✓ Can be compiled when called inside another function!")
            
        except torch._dynamo.exc.Unsupported as e:
            print(f"    ✗ Cannot be fully compiled: {str(e)[:100]}...")
    
    # Test 3: Compare performance
    print("\n3. Performance comparison (5 runs after warmup):")
    
    for name, actor in [("Triangular", actor_triangular), ("Parabolic", actor_parabolic)]:
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = actor.logp_entropy_and_sample(x, uniform_samples)
        
        times = []
        with torch.no_grad():
            for _ in range(5):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = actor.logp_entropy_and_sample(x, uniform_samples)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
        
        print(f"  {name}: {sum(times)/len(times):.2f}ms average (min: {min(times):.2f}ms, max: {max(times):.2f}ms)")

if __name__ == "__main__":
    test_compilation()
import torch
import torch._dynamo
import sys
sys.path.append('./src')
from discrete_actor import DiscreteActor
from high_low.config import Args

# Enable detailed compilation logging
torch._dynamo.config.verbose = True
torch._dynamo.config.log_level = "INFO"

def test_discrete_actor_compilation():
    device = torch.device('cuda:0')
    n_hidden = 256
    n_actors = 4
    batch_size = 4096
    
    # Create actor
    min_values = torch.tensor([1, 1, 0, 0], device=device)
    max_values = torch.tensor([30, 30, 5, 5], device=device)
    
    actor = DiscreteActor(n_hidden, n_actors, min_values, max_values).to(device)
    
    # Test inputs
    x = torch.randn(batch_size, n_hidden, device=device)
    uniform_samples = torch.rand(batch_size, n_actors, 3, device=device)
    
    print("Testing if DiscreteActor can be compiled...")
    print("-" * 80)
    
    # Try to compile just the actor
    print("\n1. Compiling actor.logp_entropy_and_sample directly:")
    try:
        compiled_actor_method = torch.compile(
            actor.logp_entropy_and_sample, 
            fullgraph=True,
            mode="max-autotune-no-cudagraphs"
        )
        
        # Test it
        with torch.no_grad():
            result = compiled_actor_method(x, uniform_samples)
        print("✓ Successfully compiled and executed!")
        
    except Exception as e:
        print(f"✗ Failed to compile: {type(e).__name__}: {e}")
    
    # Try with fullgraph=False to see where it breaks
    print("\n2. Compiling with fullgraph=False to identify graph breaks:")
    try:
        compiled_actor_partial = torch.compile(
            actor.logp_entropy_and_sample,
            fullgraph=False,  # Allow graph breaks
            mode="default"
        )
        
        with torch.no_grad():
            result = compiled_actor_partial(x, uniform_samples)
        print("✓ Compiled with graph breaks allowed")
        
    except Exception as e:
        print(f"✗ Failed even with graph breaks: {type(e).__name__}: {e}")
    
    # Test a simplified version
    print("\n3. Testing if the forward method can be compiled:")
    try:
        compiled_forward = torch.compile(
            actor.forward,
            fullgraph=True,
            mode="max-autotune-no-cudagraphs"
        )
        
        with torch.no_grad():
            result = compiled_forward(x)
        print("✓ Forward method compiled successfully!")
        
    except Exception as e:
        print(f"✗ Failed to compile forward: {type(e).__name__}: {e}")
    
    # Create a wrapper function to test
    print("\n4. Testing compilation of a wrapper function:")
    
    @torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
    def test_wrapper(actor, x, uniform_samples):
        return actor.logp_entropy_and_sample(x, uniform_samples)
    
    try:
        with torch.no_grad():
            result = test_wrapper(actor, x, uniform_samples)
        print("✓ Wrapper function compiled successfully!")
    except Exception as e:
        print(f"✗ Failed to compile wrapper: {type(e).__name__}: {e}")
    
    # Check what's happening with graph breaks
    print("\n5. Checking compilation statistics:")
    stats = torch._dynamo.utils.CompileCounter()
    print(f"Graph breaks: {stats.graph_break}")
    print(f"Ops with no kernel: {stats.ops_with_no_kernel}")

if __name__ == "__main__":
    test_discrete_actor_compilation()
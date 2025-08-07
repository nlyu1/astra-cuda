import torch
import torch._dynamo
from torch._dynamo.utils import CompileProfiler
import sys
sys.path.append('./src')
from discrete_actor_parabolic import (
    DiscreteActor, ParabolicActionDistribution,
    ParabolicVariableWidthDistribution, ParabolicFullSupportDistribution,
    ParabolicFullSupportUniformMixture
)

# Enable debugging
torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = False

def debug_parabolic_compilation():
    device = torch.device('cuda:0')
    batch_size = 32  # Smaller for debugging
    n_actors = 4
    
    print("Debugging Parabolic Actor Compilation Issues")
    print("=" * 80)
    
    # Test 1: Individual distribution components
    print("\n1. Testing individual distribution components:")
    
    # Test ParabolicVariableWidthDistribution
    print("\n  a) ParabolicVariableWidthDistribution:")
    center = torch.rand(batch_size, device=device)
    half_width = torch.rand(batch_size, device=device) * 0.5
    
    def test_parabolic_var_width(center, half_width, x):
        dist = ParabolicVariableWidthDistribution(center, half_width)
        return dist.log_prob(x), dist.cdf(x), dist.entropy()
    
    try:
        compiled_pvw = torch.compile(test_parabolic_var_width, fullgraph=True)
        x = torch.rand(batch_size, device=device)
        with torch.no_grad():
            result = compiled_pvw(center, half_width, x)
        print("     ✓ ParabolicVariableWidthDistribution compiles!")
    except Exception as e:
        print(f"     ✗ Failed: {type(e).__name__}: {str(e)[:100]}")
    
    # Test ParabolicFullSupportDistribution
    print("\n  b) ParabolicFullSupportDistribution:")
    
    def test_parabolic_full(center, x):
        dist = ParabolicFullSupportDistribution(center)
        return dist.log_prob(x), dist.cdf(x), dist.entropy()
    
    try:
        compiled_pf = torch.compile(test_parabolic_full, fullgraph=True)
        with torch.no_grad():
            result = compiled_pf(center, x)
        print("     ✓ ParabolicFullSupportDistribution compiles!")
    except Exception as e:
        print(f"     ✗ Failed: {type(e).__name__}: {str(e)[:100]}")
    
    # Test the sampling methods specifically
    print("\n2. Testing sampling methods:")
    
    # Test ParabolicFullSupportDistribution.sample
    print("\n  a) ParabolicFullSupportDistribution.sample:")
    
    def test_pf_sample(center, uniform_u):
        dist = ParabolicFullSupportDistribution(center)
        return dist.sample(uniform_u)
    
    uniform_u = torch.rand(batch_size, device=device)
    try:
        compiled_pf_sample = torch.compile(test_pf_sample, fullgraph=True)
        with torch.no_grad():
            result = compiled_pf_sample(center, uniform_u)
        print("     ✓ Sampling compiles!")
    except Exception as e:
        print(f"     ✗ Failed: {type(e).__name__}: {str(e)[:100]}")
        
        # Try to identify the problematic line
        print("\n     Checking individual operations in sample method:")
        
        # Test the complex sampling logic
        def test_sampling_logic(c, u):
            u = torch.clamp(u, 1e-12, 1.0 - 1e-12)
            mask_left = u <= c
            
            # Left branch
            k_left = torch.where(mask_left, u / c, torch.zeros_like(u))
            m_left = 1.0 - k_left
            theta_l = torch.arccos(m_left) / 3.0 - 2.0 * torch.pi / 3.0
            v_left = 2.0 * torch.cos(theta_l)
            z_left = 1.0 + v_left
            x_left = c * z_left
            
            # Right branch
            k_right = torch.where(mask_left, torch.zeros_like(u), (u - c) / (1.0 - c))
            m_right = 1.0 - k_right
            theta_r = torch.arccos(m_right) / 3.0 - 2.0 * torch.pi / 3.0
            v_right = 2.0 * torch.cos(theta_r)
            z_right = 1.0 + v_right
            x_right = c + (1.0 - c) * z_right
            x_right = -(x_right - (1 + c) / 2.) + (1 + c) / 2.
            
            return torch.where(mask_left, x_left, x_right)
        
        try:
            compiled_logic = torch.compile(test_sampling_logic, fullgraph=True)
            with torch.no_grad():
                result = compiled_logic(center, uniform_u)
            print("       ✓ Sampling logic itself compiles!")
        except Exception as e2:
            print(f"       ✗ Sampling logic fails: {type(e2).__name__}")
    
    # Test the mixture distribution
    print("\n  c) ParabolicFullSupportUniformMixture:")
    epsilon = torch.rand(batch_size, device=device)
    
    def test_mixture(center, epsilon, uniform_samples):
        dist = ParabolicFullSupportUniformMixture(center, epsilon)
        return dist.sample(uniform_samples)
    
    uniform_samples_2d = torch.rand(batch_size, 2, device=device)
    try:
        compiled_mixture = torch.compile(test_mixture, fullgraph=True)
        with torch.no_grad():
            result = compiled_mixture(center, epsilon, uniform_samples_2d)
        print("     ✓ Mixture distribution compiles!")
    except Exception as e:
        print(f"     ✗ Failed: {type(e).__name__}: {str(e)[:100]}")
    
    # Test the full DiscreteActor
    print("\n3. Testing DiscreteActor methods:")
    
    min_values = torch.tensor([1, 1, 0, 0], device=device)
    max_values = torch.tensor([30, 30, 5, 5], device=device)
    actor = DiscreteActor(256, n_actors, min_values, max_values).to(device)
    
    x = torch.randn(batch_size, 256, device=device)
    uniform_samples = torch.rand(batch_size, n_actors, 3, device=device)
    
    # Test just the forward method
    print("\n  a) DiscreteActor.forward:")
    try:
        compiled_forward = torch.compile(actor.forward, fullgraph=True)
        with torch.no_grad():
            result = compiled_forward(x)
        print("     ✓ Forward method compiles!")
    except Exception as e:
        print(f"     ✗ Failed: {type(e).__name__}: {str(e)[:100]}")
    
    # Test logp_entropy_and_sample
    print("\n  b) DiscreteActor.logp_entropy_and_sample:")
    try:
        compiled_full = torch.compile(actor.logp_entropy_and_sample, fullgraph=True)
        with torch.no_grad():
            result = compiled_full(x, uniform_samples)
        print("     ✓ Full method compiles!")
    except Exception as e:
        print(f"     ✗ Failed: {type(e).__name__}: {str(e)[:100]}")
    
    # Test with profiler to see where it breaks
    print("\n4. Detailed profiling of compilation failure:")
    
    # Create a simple test function that should reveal the issue
    def test_actor_step_by_step(x, uniform_samples):
        # Step 1: Linear layer and parameter generation
        output = actor.actor(x)
        center = torch.sigmoid(output[:, :n_actors])
        half_width = torch.sigmoid(output[:, n_actors:2*n_actors]) * 0.5
        epsilon_fs = torch.sigmoid(output[:, 2*n_actors:3*n_actors] / 25.0 - 6.0)
        epsilon_uniform = torch.sigmoid(output[:, 3*n_actors:] / 25.0 - 6.0)
        
        # Step 2: Create distribution (THIS IS LIKELY THE ISSUE)
        # Instead of creating the object, let's inline the operations
        # This simulates what ParabolicActionDistribution does
        
        # Sample from main distribution
        main_samples = uniform_samples[..., 1]  # Would normally call main.sample()
        
        # Sample from support distribution
        support_samples = uniform_samples[..., 1]  # Would normally call support.sample()
        
        # Hierarchical sampling
        samples = torch.where(
            uniform_samples[..., 0] < epsilon_fs,
            support_samples,
            main_samples
        )
        
        return samples
    
    try:
        compiled_step = torch.compile(test_actor_step_by_step, fullgraph=True)
        with torch.no_grad():
            result = compiled_step(x, uniform_samples)
        print("  ✓ Inlined operations compile successfully!")
        print("  → The issue is likely with object creation in ParabolicActionDistribution")
    except Exception as e:
        print(f"  ✗ Even inlined operations fail: {type(e).__name__}")
    
    # Try to compile with graph breaks allowed to see what works
    print("\n5. Compilation with graph breaks allowed:")
    
    try:
        # This will show us where the graph breaks occur
        with torch._dynamo.utils.CompileProfiler() as prof:
            compiled_partial = torch.compile(
                actor.logp_entropy_and_sample,
                fullgraph=False  # Allow graph breaks
            )
            with torch.no_grad():
                result = compiled_partial(x, uniform_samples)
        
        print("  Compilation with graph breaks succeeded!")
        print(f"  Number of graph breaks: {prof.get_summary()}")
        
    except Exception as e:
        print(f"  Even partial compilation failed: {type(e).__name__}")

if __name__ == "__main__":
    debug_parabolic_compilation()
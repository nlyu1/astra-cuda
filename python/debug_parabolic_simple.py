import torch
import sys
sys.path.append('./src')
from discrete_actor_parabolic import (
    DiscreteActor, ParabolicActionDistribution,
    ParabolicVariableWidthDistribution, ParabolicFullSupportDistribution,
    ParabolicFullSupportUniformMixture
)

def debug_compilation():
    device = torch.device('cuda:0')
    batch_size = 32
    
    print("Debugging Parabolic Actor Compilation")
    print("=" * 60)
    
    # Test 1: Can we compile object creation?
    print("\n1. Testing object creation patterns:")
    
    center = torch.rand(batch_size, device=device)
    half_width = torch.rand(batch_size, device=device) * 0.5
    
    # This pattern is what's used in ParabolicActionDistribution.__init__
    def create_distributions(center, half_width, eps_fs, eps_uniform):
        # This mimics what happens in ParabolicActionDistribution
        main = ParabolicVariableWidthDistribution(center, half_width)
        support = ParabolicFullSupportUniformMixture(center, eps_uniform)
        return main, support
    
    eps_fs = torch.rand(batch_size, device=device)
    eps_uniform = torch.rand(batch_size, device=device)
    
    try:
        compiled_create = torch.compile(create_distributions, fullgraph=True)
        with torch.no_grad():
            main, support = compiled_create(center, half_width, eps_fs, eps_uniform)
        print("  ✓ Object creation CAN be compiled!")
    except Exception as e:
        print(f"  ✗ Object creation fails: {type(e).__name__}")
        print(f"     Error: {str(e)}")
    
    # Test 2: Test method calls on objects
    print("\n2. Testing method calls after object creation:")
    
    def use_distributions(center, half_width, eps_fs, eps_uniform, x):
        # Create distributions
        main = ParabolicVariableWidthDistribution(center, half_width)
        support = ParabolicFullSupportUniformMixture(center, eps_uniform)
        
        # Call methods
        main_lp = main.log_prob(x)
        support_lp = support.log_prob(x)
        
        # Mimic the logaddexp pattern from ParabolicActionDistribution
        result = torch.logaddexp(
            main_lp + torch.log1p(-eps_fs),
            support_lp + torch.log(eps_fs)
        )
        return result
    
    x = torch.rand(batch_size, device=device)
    
    try:
        compiled_use = torch.compile(use_distributions, fullgraph=True)
        with torch.no_grad():
            result = compiled_use(center, half_width, eps_fs, eps_uniform, x)
        print("  ✓ Method calls on created objects CAN be compiled!")
    except Exception as e:
        print(f"  ✗ Method calls fail: {type(e).__name__}")
        print(f"     Error: {str(e)}")
    
    # Test 3: Test the actual ParabolicActionDistribution
    print("\n3. Testing ParabolicActionDistribution directly:")
    
    def create_and_use_parabolic_action(center, half_width, eps_fs, eps_uniform, uniform_samples):
        dist = ParabolicActionDistribution(center, half_width, eps_fs, eps_uniform)
        return dist.sample(uniform_samples)
    
    uniform_samples = torch.rand(batch_size, 3, device=device)
    
    try:
        compiled_pa = torch.compile(create_and_use_parabolic_action, fullgraph=True)
        with torch.no_grad():
            result = compiled_pa(center, half_width, eps_fs, eps_uniform, uniform_samples)
        print("  ✓ ParabolicActionDistribution CAN be compiled!")
    except Exception as e:
        print(f"  ✗ ParabolicActionDistribution fails: {type(e).__name__}")
        print(f"     Error: {str(e)}")
    
    # Test 4: Test DiscreteActor's logp_entropy_and_sample
    print("\n4. Testing DiscreteActor.logp_entropy_and_sample:")
    
    n_actors = 4
    min_values = torch.tensor([1, 1, 0, 0], device=device)
    max_values = torch.tensor([30, 30, 5, 5], device=device)
    
    actor = DiscreteActor(256, n_actors, min_values, max_values).to(device)
    
    # First check if the issue is in the __init__ method
    print("\n  a) Testing if issue is in DiscreteActor.__init__:")
    
    def create_actor(n_hidden, n_actors, min_vals, max_vals):
        return DiscreteActor(n_hidden, n_actors, min_vals, max_vals)
    
    try:
        # Note: This won't actually work as expected because the actor has nn.Module
        # but let's see what error we get
        compiled_create_actor = torch.compile(create_actor, fullgraph=True)
        print("  ✓ Actor creation compiles (surprising!)")
    except Exception as e:
        print(f"  ~ Actor creation compilation: {type(e).__name__}")
    
    # Now test the actual method
    print("\n  b) Testing the actual logp_entropy_and_sample method:")
    
    x = torch.randn(batch_size, 256, device=device)
    uniform_samples_4d = torch.rand(batch_size, n_actors, 3, device=device)
    
    # Let's trace through what happens in the method
    def trace_actor_method(actor, x, uniform_samples):
        # This mimics logp_entropy_and_sample
        center, half_width, epsilon_fs, epsilon_uniform = actor(x)
        
        # The problematic line might be here:
        dist = ParabolicActionDistribution(center, half_width, epsilon_fs, epsilon_uniform)
        
        unit_samples = dist.sample(uniform_samples)
        integer_samples = actor._integer_samples_from_unit_samples(unit_samples)
        unit_lb, unit_ub = actor._unit_interval_of_integer_samples(integer_samples)
        logprobs = dist.logp_interval(unit_lb, unit_ub) - actor.rangeP1.log()
        entropy = dist.entropy() + actor.rangeP1.log()
        
        return {
            'samples': integer_samples,
            'logprobs': logprobs,
            'entropy': entropy
        }
    
    try:
        compiled_trace = torch.compile(trace_actor_method, fullgraph=True)
        with torch.no_grad():
            result = compiled_trace(actor, x, uniform_samples_4d)
        print("  ✓ Traced method compiles!")
    except Exception as e:
        print(f"  ✗ Traced method fails: {type(e).__name__}")
        print(f"     Error: {str(e)}")
        
        # The error message might give us a clue
        if "sourceless builder builtins.method" in str(e):
            print("\n  → The issue is likely with bound methods!")
            print("     When ParabolicActionDistribution stores self.main and self.support,")
            print("     their methods become 'bound methods' which torch.compile doesn't like")

if __name__ == "__main__":
    debug_compilation()
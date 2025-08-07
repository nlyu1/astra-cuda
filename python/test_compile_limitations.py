import torch
import torch.nn as nn
import time

def test_compile_limitations():
    print("Testing torch.compile limitations")
    print("-" * 80)
    
    device = torch.device('cuda:0')
    
    # Test 1: Simple nested module calls
    print("\n1. Simple nested modules (should compile fine):")
    
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(100, 100)
            
        def forward(self, x):
            return torch.relu(self.linear(x))
    
    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()
            
        @torch.compile(fullgraph=True)
        def forward(self, x):
            return self.inner(x)
    
    model1 = Outer().to(device)
    x = torch.randn(1000, 100, device=device)
    
    try:
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = model1(x)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model1(x)
            torch.cuda.synchronize()
            print(f"  Success! Time: {(time.perf_counter() - start) * 1000:.2f}ms")
    except Exception as e:
        print(f"  Failed: {type(e).__name__}: {str(e)[:100]}")
    
    # Test 2: Complex class with methods
    print("\n2. Complex class methods (like DiscreteActor):")
    
    class ComplexDistribution:
        def __init__(self, center):
            self.center = center
            
        def sample(self, uniform):
            # Similar to DiscreteActor distribution methods
            return torch.where(uniform < 0.5, self.center, 1 - self.center)
    
    class ActorLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(100, 100)
            
        def forward(self, x, uniform):
            features = self.linear(x)
            center = torch.sigmoid(features)
            dist = ComplexDistribution(center)
            return dist.sample(uniform)
    
    @torch.compile(fullgraph=True)
    def compiled_with_complex_class(x, uniform, actor):
        return actor(x, uniform)
    
    model2 = ActorLike().to(device)
    uniform = torch.rand(1000, 100, device=device)
    
    try:
        with torch.no_grad():
            _ = compiled_with_complex_class(x, uniform, model2)
        print("  Success!")
    except Exception as e:
        print(f"  Failed: {type(e).__name__}: {str(e)[:100]}")
    
    # Test 3: Using torch.where in methods
    print("\n3. Methods with torch.where (common in DiscreteActor):")
    
    class MethodsWithWhere(nn.Module):
        def __init__(self):
            super().__init__()
            
        def complex_method(self, x, y):
            # Multiple nested torch.where like in DiscreteActor
            temp = torch.where(x < 0.5, x * 2, x / 2)
            result = torch.where(y < 0.3, temp, 1 - temp)
            return result
        
        @torch.compile(fullgraph=True)
        def forward(self, x, y):
            return self.complex_method(x, y)
    
    model3 = MethodsWithWhere().to(device)
    x3 = torch.rand(1000, 100, device=device)
    y3 = torch.rand(1000, 100, device=device)
    
    try:
        with torch.no_grad():
            _ = model3(x3, y3)
        print("  Success!")
    except Exception as e:
        print(f"  Failed: {type(e).__name__}: {str(e)[:100]}")
    
    # Test 4: What actually works
    print("\n4. Inlined operations (what would work):")
    
    @torch.compile(fullgraph=True)
    def fully_inlined(x, uniform):
        # All operations inlined, no class instances
        features = torch.relu(x)
        center = torch.sigmoid(features)
        samples = torch.where(uniform < 0.5, center, 1 - center)
        return samples
    
    try:
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = fully_inlined(x, uniform)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = fully_inlined(x, uniform)
            torch.cuda.synchronize()
            print(f"  Success! Time: {(time.perf_counter() - start) * 1000:.2f}ms")
    except Exception as e:
        print(f"  Failed: {type(e).__name__}: {str(e)[:100]}")
    
    print("\n5. Summary:")
    print("  - torch.compile works well with nn.Module forward methods")
    print("  - It struggles with non-Module classes and their methods")
    print("  - Complex control flow and class instantiation can break compilation")
    print("  - DiscreteActor's distribution classes likely prevent compilation")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    test_compile_limitations()
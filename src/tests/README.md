# ASTRA Tests

This directory contains comprehensive testing infrastructure for the ASTRA GPU-accelerated trading environment system.

## Test Structure

- `unit_tests/` - C++ unit tests for core functionality
  - `market_tests.cu` - Tests for VecMarket CUDA implementation
  - `astra_utils_tests.cc` - Tests for utility functions

- `integration/` - Python integration tests  
  - `test_vec_market.py` - Tests for Python bindings and end-to-end functionality

- `benchmarks/` - Performance benchmarking tools
  - `benchmark_market.cc` - C++ benchmark for market throughput
  - `benchmark_market.py` - Python benchmark wrapper
  - `benchmark_market_optimized.py` - Optimized Python benchmark with batched operations
  - `benchmark_env.cc` - Benchmark for vectorized CUDA environment with High Low Trading game
  - `high_low_benchmark.py` - Python benchmark for High Low Trading game performance

## Running Tests

### Quick Start
```bash
# From this directory
./run_tests.sh

# Or specify a custom build directory
./run_tests.sh /path/to/build

# Include benchmarks
./run_tests.sh /path/to/build --benchmark
```

### Running Individual Test Types

#### C++ Unit Tests
```bash
# After building the project
cd ../../build/build-release
ctest
# Or run individually:
./tests/unit_tests/market_tests
./tests/unit_tests/astra_utils_tests
```

#### Python Integration Tests
```bash
# Ensure the project is built first
cd ../../build/build-release
# Using CMake target
cmake --build . --target python_tests
# Or directly:
PYTHONPATH=. python tests/integration/test_vec_market.py
```

#### Benchmarks
```bash
cd ../../build/build-release
./tests/benchmarks/benchmark_market -i 0  # GPU 0
./tests/benchmarks/benchmark_market -i 1  # GPU 1
# Or run Python benchmark:
python tests/benchmarks/benchmark_market.py

# Run the vectorized environment benchmark:
./tests/benchmarks/benchmark_env      # Default GPU 0
./tests/benchmarks/benchmark_env 1    # GPU 1
```

## Test Requirements

- C++ tests: No additional requirements beyond build dependencies
- Python tests: Install with `pip install -r requirements.txt`
  - pytest (optional but recommended)
  - torch (should already be in conda environment)

## Adding New Tests

### C++ Tests
1. Add new test file to `unit_tests/`
2. Update `unit_tests/CMakeLists.txt` to include the new test
3. Follow the pattern in existing test files

### Python Tests
1. Add new test file to `integration/`
2. Update `CMakeLists.txt` to copy the file if needed
3. Use pytest conventions for test discovery

## Test Coverage

The tests cover:
- Basic market operations (add orders, matching)
- Partial fills and order priority
- Market independence and parallelism
- Best bid/offer (BBO) functionality
- Zero-size order handling
- Ring buffer wraparound
- Price crossing logic
- Execution price determination
- Multi-GPU support
- Python bindings functionality
- Vectorized CUDA environment performance
- High Low Trading game integration with CUDA markets
- Core game framework (State, Game, GameType)
- Game registration and factory system
- Observation and information state tensors

## Benchmark Details

### benchmark_env
The vectorized environment benchmark tests the High Low Trading game with CUDA market integration:
- Tests multiple environment counts: 64, 128, 256
- Tests different thread block sizes: 256, 512, 1024
- Measures step latency and frames per second (FPS)
- Uses persistent tensors to minimize allocation overhead
- Collects rewards at each step (immediate, cumulative, terminal)
- Generates appropriate random actions for each game phase

### high_low_benchmark.py
Python-based benchmark specifically for the High Low Trading game:
- Tests various batch sizes (1024, 4096, 16384, 65536)
- Measures throughput in games per second
- Profiles memory usage and GPU utilization
- Compares performance across different device configurations
- Validates game logic correctness during benchmarking

Sample results on NVIDIA GeForce RTX 5090:
```bash
python high_low_benchmark.py --device 0 --episodes 100
```

| Envs | Blocks | Threads/Block | Device | Avg Step(ms) | Max Step(ms) | Total Time(s) | FPS |
|------|--------|---------------|--------|--------------|--------------|---------------|-----|
| 16,384 | 256 | 64 | 0 | 0.524 | 49.360 | 2.291 | 28,600,039 |
| 32,768 | 256 | 128 | 0 | 0.525 | 0.622 | 2.159 | 60,719,040 |
| 65,536 | 256 | 256 | 0 | 0.711 | 1.027 | 2.920 | 89,769,284 |
| 32,768 | 512 | 64 | 0 | 0.528 | 0.686 | 2.171 | 60,370,461 |
| 65,536 | 512 | 128 | 0 | 0.712 | 0.981 | 2.924 | 89,648,898 |
| 131,072 | 512 | 256 | 0 | 0.927 | 1.145 | 3.817 | 137,344,987 |
| 65,536 | 1024 | 64 | 0 | 0.709 | 0.887 | 2.910 | 90,078,547 |
| 131,072 | 1024 | 128 | 0 | 0.925 | 1.153 | 3.810 | 137,613,098 |
| **262,144** | **1024** | **256** | **0** | **1.550** | **2.124** | **6.378** | **164,402,193** |

**Best Configuration:** 164.4M FPS with 1024 blocks × 256 threads/block = 262,144 environments  
**Best Latency:** Average 1.550ms, Maximum 2.124ms

Sample results on NVIDIA GeForce RTX 4060 Ti:
```bash
python high_low_benchmark.py --device 1 --episodes 100
```

| Envs | Blocks | Threads/Block | Device | Avg Step(ms) | Max Step(ms) | Total Time(s) | FPS |
|------|--------|---------------|--------|--------------|--------------|---------------|-----|
| 16,384 | 256 | 64 | 1 | 0.687 | 54.566 | 2.965 | 22,101,141 |
| 32,768 | 256 | 128 | 1 | 0.999 | 1.181 | 4.116 | 31,842,022 |
| 65,536 | 256 | 256 | 1 | 1.724 | 1.975 | 7.106 | 36,891,941 |
| 32,768 | 512 | 64 | 1 | 1.002 | 1.477 | 4.127 | 31,761,474 |
| 65,536 | 512 | 128 | 1 | 1.738 | 2.059 | 7.161 | 36,604,762 |
| **131,072** | **512** | **256** | **1** | **3.212** | **4.930** | **13.232** | **39,621,602** |
| 65,536 | 1024 | 64 | 1 | 1.741 | 2.159 | 7.177 | 36,525,809 |
| 131,072 | 1024 | 128 | 1 | 3.223 | 3.630 | 13.278 | 39,484,769 |
| 262,144 | 1024 | 256 | 1 | 7.963 | 13.955 | 32.621 | 32,144,453 |

**Best Configuration:** 39.6M FPS with 512 blocks × 256 threads/block = 131,072 environments  
**Best Latency:** Average 3.212ms, Maximum 4.930ms
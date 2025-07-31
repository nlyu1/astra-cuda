# Astra: GPU-based multi-agent environments

A CUDA-based reinforcement learning library: 

1. <u>Infrastructure</u>: GPU-based independent double-auction market, efficiently supporting up to ~10k instances. 
2. <u>General sequential game API </u>: vectorized, Open-Spiel inspired interfaces (in implementation)
3. <u>Python compatibility </u>: full python / pybind compatibility with native pytorch GPU tensor interface (in implementation). 

## Benchmark results (TODO)

### Double-auction market (TODO)

Market settings:
1. [how many] price levels, [how many] maximum participants. 
2. [how many] maximum active orders, with [how many] maximum fills. 

On NVIDIA RTX 5090:
- peak throughput: [TODO]
- Latency: [TODO]
- Scaling table


## Project Structure

```
astra-cuda/
├── src/
│   ├── order_matching/     # CUDA-based parallel order matching
│   ├── core/               # Game engine APIs (vectorized & standard)
│   ├── games/              # Game implementations
│   ├── tests/              # Unit tests and benchmarks
│   ├── utils/              # Shared utilities and helpers
│   └── binding/            # Python bindings (future)
├── build/                  # Build artifacts
├── CLAUDE.md               # Development notes
├── README.md               # This file
└── usage.md                # Build instructions
```

## Core Components

### [Order Matching Engine](src/order_matching/)

Vectorized CUDA implementation of discrete double-auction markets
- Price-time priority matching
- Lock-step execution across all markets: single player submits two-sided quotes for all markets at a single timestep; quote details can vary across markets. 
- Ring buffer memory management
- Zero-copy GPU operations

### [Testing & Benchmarking](src/tests/)

- Unit tests for market operations
- Performance benchmarks

### [Utilities](src/utils/)

- Error handling and assertions
- String manipulation
- Type safety utilities
- Tensor-NumPy bridge

## Environment Setup

### Prerequisites

- CUDA Toolkit 12.0+
- NVIDIA GPU with compute capability 8.7+ (RTX 4060+)
- GCC 12+ or compatible
- Python 3.12+ with PyTorch
- CMake 4.0.3+, Ninja

### Environment Setup

```bash
# Create conda environment
conda create -n astra python=3.12
conda activate astra

# Install build tools and compilers
conda install cmake ninja gcc_linux-64 gxx_linux-64

# Install PyTorch with CUDA support, see `pytorch.org/get-started/locally/`

# CUDA toolkit (if not system-installed)
conda install cuda-toolkit -c nvidia
``` 

### Build Instructions

```bash
# From project root with astra environment activated
cmake -S ./src -B ./build/build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build ./build/build-release -j$(nproc)
```

For explicit compiler paths (recommended to avoid mixing system/conda libraries):
```bash
cmake -S ./src -B ./build/build-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ \
  -DCMAKE_CUDA_HOST_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
```

To install to python, run `pip install .` under the root directory. 

## Quick Start

First build the project. 

### Running Tests

```bash
# Run all tests
cd build/build-release
ctest --verbose

# Run specific test suite
./tests/market_tests
./tests/astra_utils_tests
```

### Running Benchmarks

```bash
# Benchmark on default GPU
./build/build-release/tests/benchmarks/benchmark_market

# Benchmark on specific GPU
./build/build-release/tests/benchmarks/benchmark_market -i 1
```

### Example Usage

```cpp
#include "order_matching/market.h"
using namespace astra::order_matching;

// Create market system with 10,000 parallel markets
VecMarket market(10000, 128, 1024, 1024, 16, 0);

// Create reusable buffers
FillBatch fills = market.NewFillBatch();
BBOBatch bbos = market.NewBBOBatch();

// Submit orders (all on GPU)
market.AddTwoSidedQuotes(bid_prices, bid_sizes, 
                         ask_prices, ask_sizes, 
                         customer_ids, fills);

// Get best bid/offer
market.GetBBOs(bbos);
```

## Advanced Configuration

### Memory Limits
Adjust in `market.h`:
```cpp
constexpr int32_t MAX_MARKETS = ...;
constexpr int32_t MAX_ACTIVE_ORDERS_PER_MARKET = ...;
constexpr int32_t MAX_ACTIVE_FILLS_PER_MARKET = ...;
constexpr int32_t MAX_CUSTOMERS = ...; 
```

### Documentation

- [Order Matching Details](src/order_matching/README.md)
- [Testing Guide](src/tests/README.md)
- [Utilities Reference](src/utils/README.md)
- [Build Instructions](usage.md)
- [Project Guidelines](CLAUDE.md)

## Acknowledgments

- [PyTorch](https://pytorch.org/) for tensor operations
- [CUDA](https://developer.nvidia.com/cuda-toolkit) for GPU computing
- [fmt](https://github.com/fmtlib/fmt) for string formatting
- [pybind11](https://github.com/pybind/pybind11) for Python bindings
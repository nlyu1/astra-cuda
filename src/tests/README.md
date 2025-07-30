# Astra CUDA Tests

This directory contains all tests for the Astra CUDA order matching system.

## Test Structure

- `unit_tests/` - C++ unit tests for core functionality
  - `market_tests.cu` - Tests for VecMarket CUDA implementation
  - `astra_utils_tests.cc` - Tests for utility functions

- `integration/` - Python integration tests  
  - `test_vec_market.py` - Tests for Python bindings and end-to-end functionality

- `benchmarks/` - Performance benchmarking tools
  - `benchmark_market.cc` - C++ benchmark for market throughput
  - `benchmark_market.py` - Python benchmark wrapper

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
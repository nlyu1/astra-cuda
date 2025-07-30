# Tests Module

Unit tests and benchmarks for the ASTRA codebase.

## Components

### Market Tests (`market_tests.cu`)

GPU unit tests for order matching:
- Partial fills
- Market independence
- Time priority (FIFO at same price)
- BBO calculations
- Ring buffer wraparound
- Zero-size order handling
- Portfolio updates

```bash
./build/build-release/tests/market_tests
```

### Utils Tests (`astra_utils_tests.cc`)

Tests for utility functions:
- String manipulation (StrCat, StrJoin, StrSplit)
- Number parsing (SimpleAtoi, SimpleAtod)
- Error handling macros
- Type casting

```bash
./build/build-release/tests/astra_utils_tests
```

### Market Benchmark (`benchmark_market.cc`)

Performance benchmarking:
- Market configurations: 64 to 65,536
- Thread block sizes: 64 to 1024
- Metrics: throughput (markets/sec), latency, speedup, efficiency

```bash
# Default GPU
./build/build-release/tests/benchmark_market

# Specific GPU
./build/build-release/tests/benchmark_market -i 1
```

## Running Tests

All tests:
```bash
cd build/build-release
ctest --verbose
```

Individual tests are standalone executables in `build/build-release/tests/`
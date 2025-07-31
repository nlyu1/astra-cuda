## Project Setup
- See @usage.md for how to build the project

## Project Overview
This project implements a vectorized CUDA-based order matching system for running parallel, independent double-auction markets on GPU.

## Build System
- **Build Tool**: CMake with CUDA support
- **Main CMake**: `src/CMakeLists.txt`
- **C++ Standard**: C++20
- **CUDA Standard**: CUDA 20
- **CUDA Architectures**: 87, 89, 90, 100, 101, 120
- **Dependencies**:
  - PyTorch (via conda environment)
  - Pybind11 (fetched via CMake)
  - CUDA Toolkit
  - Python 3.12+ with dev support

## Project Structure
```
gpu_astra/
├── src/
│   ├── CMakeLists.txt          # Main build configuration
│   ├── order_matching/
│   │   ├── CMakeLists.txt      # Order matching module build
│   │   ├── market.h            # New vectorized CUDA market header
│   │   ├── market.cc           # (To be implemented) CUDA market implementation
│   │   ├── market_.h           # Legacy reference implementation header
│   │   └── market_.cc          # Legacy reference implementation
│   └── utils/
│       └── astra_utils.h       # Utility functions and error checking
└── build/                      # Build output directory
```

## Key Design Decisions

### Vectorized Market Design
1. **Complete Independence**: Each market operates independently with no cross-market interactions
2. **Lock-step Execution**: All markets process orders simultaneously
3. **Two-sided Quotes**: Each customer submits both bid and ask in same call
4. **GPU-resident Data**: All data structures remain on GPU to minimize transfers

### Memory Layout
- **Price-level Indexing**: Direct array indexing by price level (O(1) insertion)
- **Linked Lists**: For time-priority ordering within each price level
- **Ring Buffers**: For order and fill pools with wraparound
- **Pre-allocated Memory**: Fixed-size pools, no dynamic allocation

### Simplifications from Legacy System
1. **No Priority Queues**: Replaced with price-level indexed arrays
2. **No Dynamic Customer Lists**: Each customer submits exactly one two-sided quote
3. **No Order Clearing**: Orders naturally expire each round
4. **Simplified Matching**: Process one two-sided quote per market per timestep

## Implementation Status

### Completed
- [x] Analyzed legacy implementation (`market_.h`, `market_.cc`)
- [x] Examined build system configuration
- [x] Designed new vectorized market structure
- [x] Created detailed header specifications with CUDA kernel documentation
- [x] Added comprehensive function documentation
- [x] Implemented CUDA kernels in `market.cu`:
  - `add_orders_kernel`: Adds orders to price-level linked lists with atomic operations
  - `match_orders_kernel`: Matches crossing orders and generates fills
  - `get_bbo_kernel`: Extracts best bid/offer information
- [x] Implemented VecMarket member functions (constructor, destructor, AddTwoSidedQuotes, MatchAllMarkets, GetBBOs)
- [x] Updated CMakeLists.txt to compile CUDA implementation

### Implementation Details

#### CUDA Kernel Design
1. **add_orders_kernel**:
   - One thread per market
   - Uses atomic operations for thread-safe linked list updates
   - Ring buffer design for order slots with wraparound
   - Handles both bid and ask orders in single kernel call

2. **match_orders_kernel**:
   - Scans price levels to find best bid/ask
   - Matches orders at crossing prices
   - **Executes trades at the quote (resting order) price - the order with lower tid**
   - Updates order sizes and removes filled orders
   - Generates fill records with complete information matching legacy format
   - **Runtime assertion to detect fill overflow** - triggers if more fills than MAX_ACTIVE_FILLS_PER_MARKET

3. **get_bbo_kernel**:
   - Scans price levels to find best bid (highest) and ask (lowest)
   - Returns price and size of best orders
   - NULL_INDEX indicates no orders at that side

#### Memory Management
- All tensors allocated on GPU using PyTorch
- Contiguous memory layout for coalesced access
- Atomic operations for thread-safe updates
- Ring buffer design prevents memory fragmentation

### Build Status
✅ **Successfully built!** The implementation compiles without errors and supports multi-GPU execution.

### Major Design Updates
1. **Execution Price Logic**: Trades now execute at the quote price (resting order), not always at the ask price. The order with the lower tid is considered the quote/resting order.
2. **Enhanced Fill Information**: FillBatch now includes all fields from legacy OrderFillEntry:
   - `fill_customer_ids`: The taker (aggressor) customer ID
   - `fill_quoter_ids`: The quoter (resting order) customer ID  
   - `fill_is_sell_quote`: Whether the quote was a sell order
   - `fill_quote_sizes`: Original size of the quote order
   - `fill_tid`: Transaction ID of the taker order
   - `fill_quote_tid`: Transaction ID of the quote order
3. **TID Tracking**: Orders now have transaction IDs (tid) that increment globally across all markets to maintain time priority

#### Key Fixes Applied:
1. Moved CUDA kernel declarations outside the class (CUDA kernels cannot be member functions)
2. Added missing bid_tails/ask_tails parameters to match_orders_kernel
3. Fixed torch::cuda API usage (use torch::kCUDA instead of current_device())
4. Removed unused base64.h dependency from astra_utils.cc
5. Updated all kernel signatures to be consistent between header and implementation
6. Fixed execution price logic to use quote price (resting order) based on TID comparison
7. Added TID tracking with global atomic counter
8. Updated FillBatch structure to include all legacy OrderFillEntry fields
9. Fixed atomicAdd for uint64_t using proper casting
10. Renamed BBO to BBOBatch to avoid naming conflict with legacy typedef
11. Added fmt library include to market_legacy.cc

#### Multi-GPU Support Added:
1. Added `device_id` parameter to constructor to specify which GPU to use
2. Added `threads_per_block` parameter for configurable kernel launch configuration
3. All tensor allocations now respect the specified device_id
4. Each method sets the correct CUDA device before operations
5. Kernel launches use configurable threads_per_block instead of hardcoded 256

#### Removed Unnecessary Atomics:
After user feedback, removed all atomic operations from add_orders_kernel since markets are independent and each thread handles exactly one market. TID is now a read-only value passed to the kernel, with the global counter maintained on the host and incremented after kernel completion.

### Next Steps
1. Create unit tests for verification
2. Test the implementation with sample data
3. Benchmark performance vs legacy implementation
4. Optimize kernel performance if needed
5. Add error handling and edge case validation
6. Create comprehensive unit tests comparing CUDA vs legacy implementation
7. Profile and optimize kernel performance

## Technical Specifications

### Constants
- `MAX_MARKETS`: 106,496 (maximum parallel markets)
- `PRICE_LEVELS`: 128 (price levels per market)
- `MAX_ACTIVE_ORDERS_PER_MARKET`: 64
- `MAX_ACTIVE_FILLS_PER_MARKET`: 1,024
- `NULL_INDEX`: 0xFFFFFFFF (null pointer equivalent)

### Key Implementation Details
- **Transaction ID (TID)**: Global counter maintained on host, incremented by 2 after each AddTwoSidedQuotes. All markets in a batch use the same base_tid (for bids) and base_tid+1 (for asks). TIDs only need to be unique within each market
- **Execution Price**: Determined by comparing TIDs - lower TID is the quote (resting order)
- **Fill Structure**: Matches legacy OrderFillEntry with all required fields
- **BBO Structure**: Renamed to BBOBatch to avoid conflict with legacy typedef

### Multi-GPU Usage Example
```cpp
// Create market instance on GPU 0 with default 256 threads per block
VecMarket market0(10000, 128, 64, 1024, 0);

// Create market instance on GPU 1 with 512 threads per block
VecMarket market1(20000, 128, 64, 1024, 1, 512);

// Tensors will automatically be placed on the correct device
torch::Tensor bid_prices = torch::randint(0, 100, {10000}, torch::kInt32);
torch::Tensor bid_sizes = torch::randint(1, 1000, {10000}, torch::kInt32);
// ... other tensors ...

// AddTwoSidedQuotes will automatically move tensors to the correct device
market0.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids);
```

### CUDA Kernel Strategy
- **Thread Mapping**: One thread per market
- **Memory Access**: Coalesced access patterns for GPU efficiency
- **Synchronization**: Minimal - markets are independent

### Data Structures
All use PyTorch tensors for GPU memory management:
- Order books: Price-level indexed linked lists
- Order pool: Ring buffer of active orders
- Fill pool: Ring buffer of generated fills
- Market state: Per-market counters and pointers

## Build Instructions
```bash
# Run within the "astra" anaconda environment and in root directory 

# First, verify you're in the correct conda environment:
echo "Current conda environment: $CONDA_DEFAULT_ENV"
python -c "import torch; print('PyTorch version:', torch.__version__)"

# Set up environment variables and paths for conda compilers:
export CONDA_PREFIX=/home/nlyu/miniconda3/envs/astra
export PATH=/home/nlyu/miniconda3/envs/astra/bin:$PATH

# Build with explicit compiler paths to avoid mixing system and conda libraries:
cmake -S ./src -B ./build/build-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/home/nlyu/miniconda3/envs/astra/bin/x86_64-conda-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=/home/nlyu/miniconda3/envs/astra/bin/x86_64-conda-linux-gnu-g++ \
  -DPython_EXECUTABLE=/home/nlyu/miniconda3/envs/astra/bin/python \
  -DCMAKE_CUDA_HOST_COMPILER=/home/nlyu/miniconda3/envs/astra/bin/x86_64-conda-linux-gnu-g++ && \
cmake --build ./build/build-release -j$(nproc)
```

### Build System Notes
- **LTO (Link Time Optimization)**: Currently disabled due to compiler version mismatch between CUDA's default host compiler (system GCC 12) and conda's GCC 14
- **CUDA Host Compiler**: Explicitly set to use conda's GCC to ensure consistency
- **Compiler Mixing**: Avoid mixing system and conda libraries by using explicit compiler paths

### Enabling LTO (if desired)
Yes, after syncing the host compiler and CUDA compiler, LTO can be re-enabled. To do this:
1. Ensure CUDA uses the conda compiler with `-DCMAKE_CUDA_HOST_COMPILER`
2. Re-enable LTO by changing line 7 in `src/CMakeLists.txt`:
   ```cmake
   set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
   ```
3. Re-enable the `-flto` flag in line 127 of `src/CMakeLists.txt`:
   ```cmake
   $<$<CONFIG:Release>:-flto>)
   ```
However, this may still cause issues if PyTorch or other dependencies were built with different compiler versions.

## Testing Approach
1. Unit tests for individual kernels
2. Integration tests comparing with legacy implementation
3. Performance benchmarks at various market scales
4. Stress tests at maximum capacity

## Performance Considerations
- Coalesced memory access for adjacent markets
- Minimize warp divergence in matching logic
- Use shared memory for frequently accessed data
- Consider warp-level primitives for reductions
# CUDA Parallel Double-Auction Markets Implementation Roadmap

## Simplified Design Assumptions

1. **Homogeneous Markets**: All markets trade identical instruments (same price range, tick size)
2. **Parallel Universe Model**: Each market is completely independent - like parallel game instances
3. **Indexed Batching**: Orders at index `i` in batch tensors always go to market `i`
4. **No Cross-Market Operations**: No arbitrage or routing between markets

## Design Analysis

### Key Simplification Benefits

With the constraint that `batch[i] → market[i]`, we can:

1. **Eliminate market_id storage**: No need to store which market each order belongs to
2. **Perfect memory coalescing**: Threads access consecutive memory locations
3. **Simplified kernels**: No dynamic routing logic needed
4. **Dense data structures**: Can use 2D arrays indexed by `[market_id][...]`

### Optimal Memory Layout

Given these constraints, **pure SoA becomes optimal**:

```cpp
struct ParallelMarkets {
    // Market state - indexed by [market_id][price_level]
    uint32_t* bid_sizes;      // [NUM_MARKETS][PRICE_LEVELS]
    uint32_t* ask_sizes;      // [NUM_MARKETS][PRICE_LEVELS]
    int32_t*  bid_heads;      // [NUM_MARKETS][PRICE_LEVELS]
    int32_t*  ask_heads;      // [NUM_MARKETS][PRICE_LEVELS]
    
    // BBO tracking - indexed by [market_id]
    float*    best_bid_prices;   // [NUM_MARKETS]
    float*    best_ask_prices;   // [NUM_MARKETS]
    uint32_t* best_bid_sizes;    // [NUM_MARKETS]
    uint32_t* best_ask_sizes;    // [NUM_MARKETS]
    
    // Order pools - indexed by [market_id][order_slot]
    float*    order_prices;      // [NUM_MARKETS][MAX_ORDERS_PER_MARKET]
    uint32_t* order_sizes;       // [NUM_MARKETS][MAX_ORDERS_PER_MARKET]
    int32_t*  order_next;        // [NUM_MARKETS][MAX_ORDERS_PER_MARKET]
    int32_t*  order_owners;      // [NUM_MARKETS][MAX_ORDERS_PER_MARKET]
    
    // Per-market order counter
    int32_t*  next_order_slot;   // [NUM_MARKETS]
    
    // Fill tracking
    int32_t*  fill_heads;        // [NUM_MARKETS]
};
```

### Why This Layout is Optimal

1. **AddOrders**: Perfect coalescing as thread `i` writes to `order_prices[i][slot]`
2. **MatchOrders**: Each block accesses contiguous memory for its market
3. **GetBBO**: Single coalesced read of BBO arrays

## Implementation Roadmap

### Phase 1: Core Infrastructure

#### 1.1 Basic Data Structures
```cpp
namespace cuda_auction {

constexpr int MAX_MARKETS = 1024;
constexpr int PRICE_LEVELS = 4096;
constexpr int MAX_ORDERS_PER_MARKET = 10000;
constexpr int MAX_FILLS_PER_MARKET = 10000;

class ParallelMarkets {
public:
    ParallelMarkets(int num_markets);
    
    // Batch operations where tensor[i] operates on market[i]
    void AddOrders(
        torch::Tensor prices,      // [num_markets] or [batch_size]
        torch::Tensor sizes,       // [num_markets] or [batch_size]
        torch::Tensor is_bids,     // [num_markets] or [batch_size]
        torch::Tensor owner_ids    // [num_markets] or [batch_size]
    );
    
    void MatchAllMarkets();
    
    // Returns tuple of [num_markets] tensors
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> GetBBO();
    
private:
    int num_markets_;
    // All member pointers from struct above
};
}
```

#### 1.2 Simplified AddOrders Kernel
```cpp
__global__ void addOrdersSimplified(
    ParallelMarkets markets,
    const float* prices,
    const uint32_t* sizes,
    const uint8_t* is_bids,
    const int32_t* owner_ids,
    int batch_size)
{
    int market_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (market_id >= batch_size) return;
    
    // Get order slot for this market
    int slot = atomicAdd(&markets.next_order_slot[market_id], 1);
    if (slot >= MAX_ORDERS_PER_MARKET) return;
    
    // Calculate base addresses for this market
    int market_offset = market_id * MAX_ORDERS_PER_MARKET;
    
    // Write order data - perfect coalescing!
    markets.order_prices[market_offset + slot] = prices[market_id];
    markets.order_sizes[market_offset + slot] = sizes[market_id];
    markets.order_owners[market_offset + slot] = owner_ids[market_id];
    
    // Link into price level
    int price_level = __float2int_rn(prices[market_id]);
    int pl_offset = market_id * PRICE_LEVELS + price_level;
    
    if (is_bids[market_id]) {
        int old_head = atomicExch(&markets.bid_heads[pl_offset], 
                                  market_offset + slot);
        markets.order_next[market_offset + slot] = old_head;
        atomicAdd(&markets.bid_sizes[pl_offset], sizes[market_id]);
        atomicMaxFloat(&markets.best_bid_prices[market_id], prices[market_id]);
    } else {
        // Similar for asks...
    }
}
```

#### 1.3 Simplified MatchOrders Kernel
```cpp
__global__ void matchMarketsSimplified(ParallelMarkets markets, int num_markets)
{
    int market_id = blockIdx.x;
    if (market_id >= num_markets) return;
    
    // Each block handles one market independently
    float best_bid = markets.best_bid_prices[market_id];
    float best_ask = markets.best_ask_prices[market_id];
    
    while (best_bid >= best_ask && best_bid > 0) {
        int bid_level = __float2int_rn(best_bid);
        int ask_level = __float2int_rn(best_ask);
        
        // Get heads with market offset
        int bid_head_idx = market_id * PRICE_LEVELS + bid_level;
        int ask_head_idx = market_id * PRICE_LEVELS + ask_level;
        
        int bid_order = markets.bid_heads[bid_head_idx];
        int ask_order = markets.ask_heads[ask_head_idx];
        
        if (bid_order < 0 || ask_order < 0) {
            // Update best prices and continue
            // ...
            continue;
        }
        
        // Execute trade
        uint32_t trade_size = min(markets.order_sizes[bid_order],
                                  markets.order_sizes[ask_order]);
        
        // Update sizes
        markets.order_sizes[bid_order] -= trade_size;
        markets.order_sizes[ask_order] -= trade_size;
        
        // Record fill (simplified - just track in market state)
        // ...
        
        // Remove exhausted orders
        if (markets.order_sizes[bid_order] == 0) {
            markets.bid_heads[bid_head_idx] = markets.order_next[bid_order];
        }
        if (markets.order_sizes[ask_order] == 0) {
            markets.ask_heads[ask_head_idx] = markets.order_next[ask_order];
        }
    }
}
```

### Phase 2: Python Integration

#### 2.1 Simplified Python Wrapper
```python
class CudaParallelMarkets:
    def __init__(self, num_markets: int):
        self.markets = cuda_auction.ParallelMarkets(num_markets)
        self.num_markets = num_markets
        
    def step(self, actions: torch.Tensor):
        """
        Single step of all markets in parallel.
        
        Args:
            actions: [num_markets, 4] tensor with columns:
                    [bid_price, ask_price, bid_size, ask_size]
        """
        # Convert to orders (bid and ask for each market)
        num_markets = actions.shape[0]
        
        # Create bid orders
        bid_mask = actions[:, 2] > 0  # Has bid size
        if bid_mask.any():
            self.markets.AddOrders(
                actions[bid_mask, 0],  # bid prices
                actions[bid_mask, 2],  # bid sizes
                torch.ones_like(bid_mask[bid_mask]),  # is_bid=1
                torch.arange(num_markets)[bid_mask]   # owner_ids
            )
        
        # Create ask orders  
        ask_mask = actions[:, 3] > 0  # Has ask size
        if ask_mask.any():
            self.markets.AddOrders(
                actions[ask_mask, 1],  # ask prices
                actions[ask_mask, 3],  # ask sizes
                torch.zeros_like(ask_mask[ask_mask]), # is_bid=0
                torch.arange(num_markets)[ask_mask]   # owner_ids
            )
        
        # Run matching
        self.markets.MatchAllMarkets()
    
    def get_observations(self) -> torch.Tensor:
        """Get market state for all markets"""
        bid_px, ask_px, bid_sz, ask_sz = self.markets.GetBBO()
        return torch.stack([bid_px, ask_px, bid_sz, ask_sz], dim=1)
```

### Phase 3: Integration with Astra

#### 3.1 Drop-in Replacement for Current Market
```python
class HighLowTradingCUDA:
    def __init__(self, num_envs: int, game_config: dict):
        self.num_envs = num_envs
        self.markets = CudaParallelMarkets(num_envs)
        # ... rest of game state
        
    def apply_trading_actions(self, actions: torch.Tensor):
        """Replace current market.AddOrder with CUDA version"""
        # actions: [num_envs, 4] from RL agents
        self.markets.step(actions)
        
    def get_market_observations(self) -> torch.Tensor:
        """Get BBO for all environments"""
        return self.markets.get_observations()
```

### Implementation Priority

1. **Week 1**: 
   - Basic data structure allocation
   - AddOrders kernel with perfect coalescing
   - Unit tests for order insertion

2. **Week 2**:
   - MatchOrders kernel 
   - BBO tracking
   - Fill recording

3. **Week 3**:
   - Python bindings
   - Integration tests
   - Performance benchmarking vs CPU

4. **Week 4**:
   - Integration with existing Astra pipeline
   - Full game loop testing
   - Documentation

### Expected Performance Gains

With simplified assumptions:
- **AddOrders**: ~100x faster due to perfect coalescing and no routing
- **MatchOrders**: ~50x faster with parallel market processing
- **Memory usage**: Predictable and dense (no fragmentation)
- **Overall**: 10-50x speedup for full game loop

### Key Design Decisions

1. **Fixed market indexing**: Massive simplification worth the constraint
2. **Dense allocation**: `MAX_ORDERS_PER_MARKET` wastes some memory but eliminates fragmentation
3. **No order cancellation**: Keeps implementation simple for v1
4. **Price levels as integers**: Avoids floating point comparison issues

This simplified design leverages the parallel universe assumption to create an extremely efficient implementation that should provide substantial speedup with minimal complexity.
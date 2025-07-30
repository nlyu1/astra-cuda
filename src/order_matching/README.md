# Order Matching Module

Vectorized CUDA implementation of discrete double-auction markets.

## Design

- Independent markets execute in parallel; each instance occupies a CUDA thread 
- Lock-step execution: single timestep processes all markets synchronously
- Two-sided quotes: each participant submits bid and ask simultaneously
- Price-time priority matching
- Integrated portfolio tracking (contracts and cash)

## Architecture

### VecMarket Class

Main interface in `market.h`, `market.cu`:

```cpp
VecMarket(num_markets, max_price_levels, max_orders, max_fills, 
          num_customers=16, device_id=0, threads_per_block=64)
```

Methods:
- `NewFillBatch()` - Create fill result buffer
- `NewBBOBatch()` - Create BBO data buffer
- `AddTwoSidedQuotes()` - Submit orders for all markets
- `GetBBOs()` - Retrieve best bid/offer
- `GetCustomerPortfolios()` - Access portfolio states

### Data Structures

#### Order Book Design
- **Price-Level Indexing**: Direct array indexing by price (O(1) insertion)
- **Linked Lists**: Time-priority ordering within each price level
- **Ring Buffer**: Order and fill pools with wraparound for memory efficiency

#### Memory Layout
```
Order Book Arrays:
- bid_heads_[market][price_level] -> first order index
- bid_tails_[market][price_level] -> last order index  
- ask_heads_[market][price_level] -> first order index
- ask_tails_[market][price_level] -> last order index

Order Pool Arrays:
- order_prices_[market][slot]
- order_sizes_[market][slot]
- order_customer_ids_[market][slot]
- order_tid_[market][slot]
- order_is_bid_[market][slot]
- order_next_[market][slot] -> next order in linked list

Portfolio Arrays:
- customer_portfolios_[market][customer][0] -> contracts
- customer_portfolios_[market][customer][1] -> cash
```

### CUDA Kernels

- `add_orders_kernel`: Insert orders into price-level linked lists (one thread per market)
- `match_orders_kernel`: Match crossing orders, execute at resting order price
- `get_bbo_kernel`: Find best bid/ask prices

## Usage

```cpp
VecMarket market(10000, 128, 1024, 1024, 16, 0);  // 10k markets
FillBatch fills = market.NewFillBatch();
BBOBatch bbos = market.NewBBOBatch();

// Submit orders (GPU tensors)
market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, 
                         customer_ids, fills);
market.GetBBOs(bbos);
```

## Implementation Notes

- No atomics required (markets are independent)
- Price-level arrays for O(1) insertion
- Ring buffers for order/fill recycling
- 32-bit TIDs for market-local time ordering
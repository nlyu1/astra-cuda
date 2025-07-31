#include "market.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <assert.h>
#include <fmt/format.h>
#include <sstream>
#include <vector>
#include <algorithm>

namespace astra {
namespace order_matching {

// ================================================================================
// CUDA KERNELS
// ================================================================================

__global__ void add_orders_kernel(
    const int32_t* bid_px, const int32_t* bid_sz,
    const int32_t* ask_px, const int32_t* ask_sz,
    const int32_t* customer_ids,
    int32_t* bid_heads, int32_t* bid_tails,
    int32_t* ask_heads, int32_t* ask_tails,
    int32_t* order_prices, int32_t* order_sizes,
    int32_t* order_customer_ids, int32_t* order_tid,
    bool* order_is_bid, int32_t* order_next,
    int32_t* order_next_slots, int32_t* customer_portfolios,
    int32_t base_tid, int32_t num_markets, 
    int32_t max_orders_per_market, int32_t max_price_levels,
    int32_t num_customers)
{
    // Each thread handles exactly one market's bid-ask quotes independently - no synchronization needed!
    // Markets are completely independent and customers submit exactly one two-sided quote per market.
    int32_t market_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (market_id >= num_markets) return;

    // Get bid and ask data for this market
    int32_t bid_price = bid_px[market_id];
    int32_t bid_size = bid_sz[market_id];
    int32_t ask_price = ask_px[market_id];
    int32_t ask_size = ask_sz[market_id];
    int32_t customer_id = customer_ids[market_id];

    // Process bid order if size > 0
    if (bid_size > 0 && bid_price < max_price_levels) {
        // Get next order slot for this market (ring buffer) - no atomic needed!
        int32_t bid_slot = order_next_slots[market_id]++ % max_orders_per_market;
        int32_t bid_idx = market_id * max_orders_per_market + bid_slot;

        // Fill order details
        order_prices[bid_idx] = bid_price;
        order_sizes[bid_idx] = bid_size;
        order_customer_ids[bid_idx] = customer_id;
        order_tid[bid_idx] = base_tid; 
        order_is_bid[bid_idx] = true;
        order_next[bid_idx] = NULL_INDEX;

        // Add to price level linked list
        int32_t price_idx = market_id * max_price_levels + bid_price;
        int32_t old_tail = bid_tails[price_idx];
        bid_tails[price_idx] = bid_slot;
        
        if (old_tail == NULL_INDEX) {
            // First order at this price level
            bid_heads[price_idx] = bid_slot;
        } else {
            // Link previous tail to this new order
            int32_t old_tail_idx = market_id * max_orders_per_market + old_tail;
            order_next[old_tail_idx] = bid_slot;
        }
    }

    // Process ask order if size > 0
    if (ask_size > 0 && ask_price < max_price_levels) {
        // Get next order slot for this market (ring buffer) - no atomic needed!
        int32_t ask_slot = order_next_slots[market_id]++ % max_orders_per_market;
        int32_t ask_idx = market_id * max_orders_per_market + ask_slot;

        // Fill order details
        order_prices[ask_idx] = ask_price;
        order_sizes[ask_idx] = ask_size;
        order_customer_ids[ask_idx] = customer_id;
        order_tid[ask_idx] = base_tid + 1; 
        order_is_bid[ask_idx] = false;
        order_next[ask_idx] = NULL_INDEX;

        // Add to price level linked list
        int32_t price_idx = market_id * max_price_levels + ask_price;
        int32_t old_tail = ask_tails[price_idx];
        ask_tails[price_idx] = ask_slot;
        
        if (old_tail == NULL_INDEX) {
            // First order at this price level
            ask_heads[price_idx] = ask_slot;
        } else {
            // Link previous tail to this new order
            int32_t old_tail_idx = market_id * max_orders_per_market + old_tail;
            order_next[old_tail_idx] = ask_slot;
        }
    }
}

__global__ void match_orders_kernel(
    int32_t* bid_heads, int32_t* ask_heads,
    int32_t* bid_tails, int32_t* ask_tails,
    int32_t* order_prices, int32_t* order_sizes,
    int32_t* order_customer_ids, int32_t* order_tid,
    bool* order_is_bid, int32_t* order_next,
    int32_t* fill_prices, int32_t* fill_sizes,
    int32_t* fill_customer_ids, int32_t* fill_quoter_ids,
    bool* fill_is_sell_quote, int32_t* fill_quote_sizes,
    int32_t* fill_tid, int32_t* fill_quote_tid,
    int32_t* fill_counts, int32_t* customer_portfolios,
    int32_t num_markets, int32_t max_orders_per_market,
    int32_t max_price_levels, int32_t max_fills_per_market,
    int32_t num_customers)
{
    int32_t market_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (market_id >= num_markets) return;

    int32_t fill_count = 0;

    // Find best bid and ask prices
    int best_bid_price = -1;
    int best_ask_price = max_price_levels;

    // Scan for best bid (highest price with orders)
    for (int p = max_price_levels - 1; p >= 0; p--) {
        int32_t head_idx = market_id * max_price_levels + p;
        if (bid_heads[head_idx] != NULL_INDEX) {
            best_bid_price = p;
            break;
        }
    }

    // Scan for best ask (lowest price with orders)
    for (int32_t p = 0; p < max_price_levels; p++) {
        int32_t head_idx = market_id * max_price_levels + p;
        if (ask_heads[head_idx] != NULL_INDEX) {
            best_ask_price = p;
            break;
        }
    }

    // Match orders while they cross
    while (best_bid_price >= 0 && best_ask_price < max_price_levels && 
           best_bid_price >= best_ask_price) {
        
        // Get head orders at best prices
        int32_t bid_head_idx = market_id * max_price_levels + best_bid_price;
        int32_t ask_head_idx = market_id * max_price_levels + best_ask_price;
        
        int32_t bid_slot = bid_heads[bid_head_idx];
        int32_t ask_slot = ask_heads[ask_head_idx];
        
        if (bid_slot == NULL_INDEX || ask_slot == NULL_INDEX) break;
        
        int32_t bid_idx = market_id * max_orders_per_market + bid_slot;
        int32_t ask_idx = market_id * max_orders_per_market + ask_slot;
        
        // Get order details
        int32_t bid_size = order_sizes[bid_idx];
        int32_t ask_size = order_sizes[ask_idx];
        int32_t bid_customer = order_customer_ids[bid_idx];
        int32_t ask_customer = order_customer_ids[ask_idx];
        int32_t bid_tid = order_tid[bid_idx];
        int32_t ask_tid = order_tid[ask_idx];
        int32_t bid_price = order_prices[bid_idx];
        int32_t ask_price = order_prices[ask_idx];
        
        // Determine which order is the resting (quote) order based on tid
        // The order with lower tid arrived first and is the quote
        bool bid_is_quote = bid_tid < ask_tid;
        int32_t trade_price = bid_is_quote ? bid_price : ask_price;
        int32_t trade_size = min(bid_size, ask_size);
        
        // Detailed debugging
        // if (market_id == 0 && blockIdx.x == 0) {
        //     printf("Market %d block %d: fill_count=%d, max_fills=%d\n",
        //             market_id, blockIdx.x, fill_count, max_fills_per_market);
        //     printf("  Best bid=%d, Best ask=%d\n", best_bid_price, best_ask_price);
        //     printf("  Bid head=%d, Ask head=%d\n", bid_heads[bid_head_idx], ask_heads[ask_head_idx]);
        //     printf("  Bid size=%d, Ask size=%d\n", bid_size, ask_size);
        //     printf("  Bid customer=%d, Ask customer=%d\n", bid_customer, ask_customer);
        //     printf("  Bid tid=%d, Ask tid=%d\n", bid_tid, ask_tid);
        //     printf("  Bid price=%d, Ask price=%d\n", bid_price, ask_price);
        // }
        if (fill_count >= max_fills_per_market) {
            printf("FATAL ERROR: Market %d exceeded max fills! fill_count=%d, max_fills=%d\n",
                    market_id, fill_count, max_fills_per_market);
            printf("  This indicates max_fills_per_market is too low for the trading volume\n");
            printf("  Increase MAX_ACTIVE_FILLS_PER_MARKET in high_low_trading.cc\n");
            // Set fill count to a sentinel value to indicate error
            fill_counts[market_id] = -1;
            return;
        }

        // Record fill
        int32_t fill_idx = market_id * max_fills_per_market + fill_count;
        fill_prices[fill_idx] = trade_price;
        fill_sizes[fill_idx] = trade_size;
        
        if (bid_is_quote) {
            // Bid is the quote (resting order), ask is the taker
            fill_customer_ids[fill_idx] = ask_customer;  // Taker
            fill_quoter_ids[fill_idx] = bid_customer;    // Quoter
            fill_is_sell_quote[fill_idx] = false;        // Buy quote
            fill_quote_sizes[fill_idx] = bid_size;       // Original quote size
            fill_tid[fill_idx] = ask_tid;                // Taker tid
            fill_quote_tid[fill_idx] = bid_tid;          // Quote tid
        } else {
            // Ask is the quote (resting order), bid is the taker
            fill_customer_ids[fill_idx] = bid_customer;  // Taker
            fill_quoter_ids[fill_idx] = ask_customer;    // Quoter
            fill_is_sell_quote[fill_idx] = true;         // Sell quote
            fill_quote_sizes[fill_idx] = ask_size;       // Original quote size
            fill_tid[fill_idx] = bid_tid;                // Taker tid
            fill_quote_tid[fill_idx] = ask_tid;          // Quote tid
        }
        
        // Update customer portfolios (same for both cases)
        if (bid_customer < num_customers && ask_customer < num_customers) {
            int32_t bid_customer_idx = (market_id * num_customers + bid_customer) * 2;
            int32_t ask_customer_idx = (market_id * num_customers + ask_customer) * 2;
            
            // Bid customer (buyer): gains contracts, loses cash
            customer_portfolios[bid_customer_idx] += trade_size;
            customer_portfolios[bid_customer_idx + 1] -= trade_price * trade_size;
            
            // Ask customer (seller): loses contracts, gains cash
            customer_portfolios[ask_customer_idx] -= trade_size;
            customer_portfolios[ask_customer_idx + 1] += trade_price * trade_size;
        }
        
        fill_count++;
        
        // Update order sizes
        order_sizes[bid_idx] -= trade_size;
        order_sizes[ask_idx] -= trade_size;
        
        // Remove filled orders
        if (order_sizes[bid_idx] == 0) {
            bid_heads[bid_head_idx] = order_next[bid_idx];
            if (order_next[bid_idx] == NULL_INDEX) {
                // This was the last order at this price
                bid_tails[bid_head_idx] = NULL_INDEX;
                // Find next best bid
                best_bid_price = -1;
                for (int p = max_price_levels - 1; p >= 0; p--) {
                    int32_t idx = market_id * max_price_levels + p;
                    if (bid_heads[idx] != NULL_INDEX) {
                        best_bid_price = p;
                        break;
                    }
                }
            }
        }
        
        if (order_sizes[ask_idx] == 0) {
            ask_heads[ask_head_idx] = order_next[ask_idx];
            if (order_next[ask_idx] == NULL_INDEX) {
                // This was the last order at this price
                ask_tails[ask_head_idx] = NULL_INDEX;
                // Find next best ask
                best_ask_price = max_price_levels;
                for (int32_t p = 0; p < max_price_levels; p++) {
                    int32_t idx = market_id * max_price_levels + p;
                    if (ask_heads[idx] != NULL_INDEX) {
                        best_ask_price = p;
                        break;
                    }
                }
            }
        }
    }
    
    // Store fill count for this market
    fill_counts[market_id] = fill_count;
}

__global__ void get_bbo_kernel(
    const int32_t* bid_heads, const int32_t* ask_heads,
    const int32_t* order_sizes, const int32_t* order_prices,
    int32_t* best_bid_prices, int32_t* best_bid_sizes,
    int32_t* best_ask_prices, int32_t* best_ask_sizes,
    int32_t num_markets, int32_t max_orders_per_market,
    int32_t max_price_levels)
{
    int32_t market_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (market_id >= num_markets) return;

    // Initialize outputs
    best_bid_prices[market_id] = NULL_INDEX;
    best_bid_sizes[market_id] = 0;
    best_ask_prices[market_id] = NULL_INDEX;
    best_ask_sizes[market_id] = 0;

    // Find best bid (highest price with orders)
    for (int p = max_price_levels - 1; p >= 0; p--) {
        int32_t head_idx = market_id * max_price_levels + p;
        int32_t head_slot = bid_heads[head_idx];
        if (head_slot != NULL_INDEX) {
            int32_t order_idx = market_id * max_orders_per_market + head_slot;
            best_bid_prices[market_id] = p;
            best_bid_sizes[market_id] = order_sizes[order_idx];
            break;
        }
    }

    // Find best ask (lowest price with orders)
    for (int32_t p = 0; p < max_price_levels; p++) {
        int32_t head_idx = market_id * max_price_levels + p;
        int32_t head_slot = ask_heads[head_idx];
        if (head_slot != NULL_INDEX) {
            int32_t order_idx = market_id * max_orders_per_market + head_slot;
            best_ask_prices[market_id] = p;
            best_ask_sizes[market_id] = order_sizes[order_idx];
            break;
        }
    }
}

// ================================================================================
// VECMARKET CLASS IMPLEMENTATION
// ================================================================================

VecMarket::VecMarket(int32_t num_markets, int32_t max_price_levels,
                     int32_t max_active_orders_per_market, int32_t max_active_fills_per_market,
                     int32_t num_customers, int device_id, int32_t threads_per_block)
    : num_markets_(num_markets),
      max_price_levels_(max_price_levels),
      max_orders_per_market_(max_active_orders_per_market),
      max_fills_per_market_(max_active_fills_per_market),
      num_customers_(num_customers),
      global_tid_counter_(0),
      device_id_(device_id),
      threads_per_block_(threads_per_block) {
    // Validate parameters
    ASTRA_CHECK_LE(num_markets, MAX_MARKETS);
    ASTRA_CHECK_LE(max_price_levels, PRICE_LEVELS);
    ASTRA_CHECK_LE(max_active_orders_per_market, MAX_ACTIVE_ORDERS_PER_MARKET);
    ASTRA_CHECK_LE(max_active_fills_per_market, MAX_ACTIVE_FILLS_PER_MARKET);
    ASTRA_CHECK_LE(num_customers, MAX_CUSTOMERS);
    ASTRA_CHECK_GT(threads_per_block, 0);
    ASTRA_CHECK_LE(threads_per_block, 1024); // Max threads per block for most GPUs

    // Validate and set CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    ASTRA_CHECK_GE(device_id, 0);
    ASTRA_CHECK_LT(device_id, device_count);

    // Configure tensor options for the specified device
    auto device = torch::Device(torch::kCUDA, device_id);
    auto options_options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(device);
    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);

    // Initialize price level linked lists
    bid_heads_ = torch::full({num_markets, max_price_levels}, NULL_INDEX, options_options_i32);
    ask_heads_ = torch::full({num_markets, max_price_levels}, NULL_INDEX, options_options_i32);
    bid_tails_ = torch::full({num_markets, max_price_levels}, NULL_INDEX, options_options_i32);
    ask_tails_ = torch::full({num_markets, max_price_levels}, NULL_INDEX, options_options_i32);

    // Initialize order pool
    order_prices_ = torch::zeros({num_markets, max_active_orders_per_market}, options_options_i32);
    order_sizes_ = torch::zeros({num_markets, max_active_orders_per_market}, options_options_i32);
    order_customer_ids_ = torch::zeros({num_markets, max_active_orders_per_market}, options_options_i32);
    order_tid_ = torch::zeros({num_markets, max_active_orders_per_market}, options_options_i32);
    order_is_bid_ = torch::zeros({num_markets, max_active_orders_per_market}, options_bool);
    order_next_ = torch::full({num_markets, max_active_orders_per_market}, NULL_INDEX, options_options_i32);

    // Initialize fill pool - not used internally, created on demand in MatchAllMarkets
    fill_counts_ = torch::zeros({num_markets}, options_options_i32);

    // Initialize market state
    order_next_slots_ = torch::zeros({num_markets}, options_options_i32);
    
    // Initialize customer portfolios: [num_markets, num_customers, 2]
    // Index 0: number of contracts, Index 1: cash amount
    customer_portfolios_ = torch::zeros({num_markets, num_customers, 2}, options_i32);
    
    // TID counter is maintained on host side
}

VecMarket::VecMarket(const VecMarket& other)
    : num_markets_(other.num_markets_),
      max_price_levels_(other.max_price_levels_),
      max_orders_per_market_(other.max_orders_per_market_),
      max_fills_per_market_(other.max_fills_per_market_),
      num_customers_(other.num_customers_),
      global_tid_counter_(other.global_tid_counter_),
      device_id_(other.device_id_),
      threads_per_block_(other.threads_per_block_) {
    
    // Clone all tensors to create deep copies
    // The clone() operation will create new tensors on the same device as the source
    bid_heads_ = other.bid_heads_.clone();
    ask_heads_ = other.ask_heads_.clone();
    bid_tails_ = other.bid_tails_.clone();
    ask_tails_ = other.ask_tails_.clone();
    
    order_prices_ = other.order_prices_.clone();
    order_sizes_ = other.order_sizes_.clone();
    order_customer_ids_ = other.order_customer_ids_.clone();
    order_tid_ = other.order_tid_.clone();
    order_is_bid_ = other.order_is_bid_.clone();
    order_next_ = other.order_next_.clone();
    
    fill_counts_ = other.fill_counts_.clone();
    order_next_slots_ = other.order_next_slots_.clone();
    customer_portfolios_ = other.customer_portfolios_.clone();
}

VecMarket::~VecMarket() {
    // PyTorch tensors automatically handle GPU memory cleanup
    // Ensure all kernels complete before deallocating memory
    cudaSetDevice(device_id_);
    cudaDeviceSynchronize();
}   

FillBatch VecMarket::NewFillBatch() const
{
    // Create fill tensors on the correct device
    auto device = torch::Device(torch::kCUDA, device_id_);
    auto options_options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(device);
    
    FillBatch batch;
    batch.fill_prices = torch::zeros({num_markets_, max_fills_per_market_}, options_options_i32);
    batch.fill_sizes = torch::zeros({num_markets_, max_fills_per_market_}, options_options_i32);
    batch.fill_customer_ids = torch::zeros({num_markets_, max_fills_per_market_}, options_options_i32);
    batch.fill_quoter_ids = torch::zeros({num_markets_, max_fills_per_market_}, options_options_i32);
    batch.fill_is_sell_quote = torch::zeros({num_markets_, max_fills_per_market_}, options_bool);
    batch.fill_quote_sizes = torch::zeros({num_markets_, max_fills_per_market_}, options_options_i32);
    batch.fill_tid = torch::zeros({num_markets_, max_fills_per_market_}, options_options_i32);
    batch.fill_quote_tid = torch::zeros({num_markets_, max_fills_per_market_}, options_options_i32);
    batch.fill_counts = torch::zeros({num_markets_}, options_options_i32);
    
    return batch;
}

BBOBatch VecMarket::NewBBOBatch() const
{
    // Create BBO tensors on the correct device
    auto device = torch::Device(torch::kCUDA, device_id_);
    auto options_options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    
    BBOBatch batch;
    batch.best_bid_prices = torch::zeros({num_markets_}, options_options_i32);
    batch.best_bid_sizes = torch::zeros({num_markets_}, options_options_i32);
    batch.best_ask_prices = torch::zeros({num_markets_}, options_options_i32);
    batch.best_ask_sizes = torch::zeros({num_markets_}, options_options_i32);
    // Initialize last_prices with NULL_INDEX to indicate no trades yet
    batch.last_prices = torch::full({num_markets_}, NULL_INDEX, options_options_i32);
    
    return batch;
}

void VecMarket::AddTwoSidedQuotes(
    torch::Tensor bid_px, torch::Tensor bid_sz,
    torch::Tensor ask_px, torch::Tensor ask_sz,
    torch::Tensor customer_ids, FillBatch& fills)
{   
    // Validate input dimensions
    ASTRA_CHECK_EQ(bid_px.size(0), num_markets_);
    ASTRA_CHECK_EQ(bid_sz.size(0), num_markets_);
    ASTRA_CHECK_EQ(ask_px.size(0), num_markets_);
    ASTRA_CHECK_EQ(ask_sz.size(0), num_markets_);
    ASTRA_CHECK_EQ(customer_ids.size(0), num_markets_);

    // Assert tensors are on the correct GPU and contiguous
    auto device = torch::Device(torch::kCUDA, device_id_);
    if (bid_px.device() != device) {
        AstraFatalError("bid_px tensor must be on device " + std::to_string(device_id_));
    }
    if (!bid_px.is_contiguous()) {
        AstraFatalError("bid_px tensor must be contiguous");
    }
    if (bid_sz.device() != device) {
        AstraFatalError("bid_sz tensor must be on device " + std::to_string(device_id_));
    }
    if (!bid_sz.is_contiguous()) {
        AstraFatalError("bid_sz tensor must be contiguous");
    }
    if (ask_px.device() != device) {
        AstraFatalError("ask_px tensor must be on device " + std::to_string(device_id_));
    }
    if (!ask_px.is_contiguous()) {
        AstraFatalError("ask_px tensor must be contiguous");
    }
    if (ask_sz.device() != device) {
        AstraFatalError("ask_sz tensor must be on device " + std::to_string(device_id_));
    }
    if (!ask_sz.is_contiguous()) {
        AstraFatalError("ask_sz tensor must be contiguous");
    }
    if (customer_ids.device() != device) {
        AstraFatalError("customer_ids tensor must be on device " + std::to_string(device_id_));
    }
    if (!customer_ids.is_contiguous()) {
        AstraFatalError("customer_ids tensor must be contiguous");
    }

    // Launch kernel
    dim3 blocks((num_markets_ + threads_per_block_ - 1) / threads_per_block_);
    dim3 threads(threads_per_block_);

    add_orders_kernel<<<blocks, threads>>>(
        bid_px.data_ptr<int32_t>(), bid_sz.data_ptr<int32_t>(),
        ask_px.data_ptr<int32_t>(), ask_sz.data_ptr<int32_t>(),
        customer_ids.data_ptr<int32_t>(),
        bid_heads_.data_ptr<int32_t>(), bid_tails_.data_ptr<int32_t>(),
        ask_heads_.data_ptr<int32_t>(), ask_tails_.data_ptr<int32_t>(),
        order_prices_.data_ptr<int32_t>(), order_sizes_.data_ptr<int32_t>(),
        order_customer_ids_.data_ptr<int32_t>(), order_tid_.data_ptr<int32_t>(),
        order_is_bid_.data_ptr<bool>(), order_next_.data_ptr<int32_t>(),
        order_next_slots_.data_ptr<int32_t>(), customer_portfolios_.data_ptr<int32_t>(),
        global_tid_counter_, num_markets_, max_orders_per_market_, max_price_levels_,
        num_customers_
    );

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        AstraFatalError(std::string("CUDA error in AddTwoSidedQuotes: ") + cudaGetErrorString(error));
    }
    
    // Increment global TID counter by 2 (one for all bids, one for all asks)
    global_tid_counter_ += 2;
    
    // Match orders after adding them
    MatchAllMarkets(fills);
}

void VecMarket::MatchAllMarkets(FillBatch& fills)
{   
    // Clear fill counts
    fills.fill_counts.zero_();

    // Launch matching kernel
    dim3 blocks((num_markets_ + threads_per_block_ - 1) / threads_per_block_);
    dim3 threads(threads_per_block_);

    match_orders_kernel<<<blocks, threads>>>(
        bid_heads_.data_ptr<int32_t>(), ask_heads_.data_ptr<int32_t>(),
        bid_tails_.data_ptr<int32_t>(), ask_tails_.data_ptr<int32_t>(),
        order_prices_.data_ptr<int32_t>(), order_sizes_.data_ptr<int32_t>(),
        order_customer_ids_.data_ptr<int32_t>(), order_tid_.data_ptr<int32_t>(),
        order_is_bid_.data_ptr<bool>(), order_next_.data_ptr<int32_t>(),
        fills.fill_prices.data_ptr<int32_t>(), fills.fill_sizes.data_ptr<int32_t>(),
        fills.fill_customer_ids.data_ptr<int32_t>(), fills.fill_quoter_ids.data_ptr<int32_t>(),
        fills.fill_is_sell_quote.data_ptr<bool>(), fills.fill_quote_sizes.data_ptr<int32_t>(),
        fills.fill_tid.data_ptr<int32_t>(), fills.fill_quote_tid.data_ptr<int32_t>(),
        fills.fill_counts.data_ptr<int32_t>(), customer_portfolios_.data_ptr<int32_t>(),
        num_markets_, max_orders_per_market_, max_price_levels_, max_fills_per_market_,
        num_customers_
    );

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        AstraFatalError(std::string("CUDA error in MatchAllMarkets: ") + cudaGetErrorString(error));
    }

    // Fill batch is populated in-place, nothing to return
}

void VecMarket::GetBBOs(BBOBatch& bbos)
{
    // Launch BBO kernel
    dim3 blocks((num_markets_ + threads_per_block_ - 1) / threads_per_block_);
    dim3 threads(threads_per_block_);

    get_bbo_kernel<<<blocks, threads>>>(
        bid_heads_.data_ptr<int32_t>(), ask_heads_.data_ptr<int32_t>(),
        order_sizes_.data_ptr<int32_t>(), order_prices_.data_ptr<int32_t>(),
        bbos.best_bid_prices.data_ptr<int32_t>(), bbos.best_bid_sizes.data_ptr<int32_t>(),
        bbos.best_ask_prices.data_ptr<int32_t>(), bbos.best_ask_sizes.data_ptr<int32_t>(),
        num_markets_, max_orders_per_market_, max_price_levels_
    );

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        AstraFatalError(std::string("CUDA error in GetBBOs: ") + cudaGetErrorString(error));
    }

    // BBOBatch is populated in-place, nothing to return
}

std::string VecMarket::ToString(int32_t market_id) const
{
    // Validate market_id
    if (market_id >= num_markets_) {
        return fmt::format("Error: Invalid market_id {} (num_markets = {})", market_id, num_markets_);
    }
    
    // Move tensors to CPU for reading
    auto bid_heads = bid_heads_.cpu();
    auto ask_heads = ask_heads_.cpu();
    auto order_prices = order_prices_.cpu();
    auto order_sizes = order_sizes_.cpu();
    auto order_customer_ids = order_customer_ids_.cpu();
    auto order_tid = order_tid_.cpu();
    auto order_is_bid = order_is_bid_.cpu();
    auto order_next = order_next_.cpu();
    
    // Get accessors (only for the ones we actually use)
    auto bid_heads_acc = bid_heads.accessor<int32_t, 2>();
    auto ask_heads_acc = ask_heads.accessor<int32_t, 2>();
    auto order_sizes_acc = order_sizes.accessor<int32_t, 2>();
    auto order_customer_ids_acc = order_customer_ids.accessor<int32_t, 2>();
    auto order_tid_acc = order_tid.accessor<int32_t, 2>();
    auto order_next_acc = order_next.accessor<int32_t, 2>();
    
    // Collect all sell orders (asks) for this market
    std::vector<std::tuple<int32_t, int32_t, int32_t, int32_t>> sell_orders; // price, size, customer_id, tid
    
    // Traverse all ask price levels from lowest to highest
    for (int32_t price = 0; price < max_price_levels_; ++price) {
        int32_t slot = ask_heads_acc[market_id][price];
        while (slot != NULL_INDEX) {
            int32_t size = order_sizes_acc[market_id][slot];
            if (size > 0) {
                sell_orders.push_back({
                    price,
                    size,
                    order_customer_ids_acc[market_id][slot],
                    order_tid_acc[market_id][slot]
                });
            }
            slot = order_next_acc[market_id][slot];
        }
    }
    
    // Sort sell orders by price (highest first), then by tid (earliest first)
    std::sort(sell_orders.begin(), sell_orders.end(), 
        [](const auto& a, const auto& b) {
            if (std::get<0>(a) != std::get<0>(b)) {
                return std::get<0>(a) > std::get<0>(b); // Higher price first for sells
            }
            return std::get<3>(a) < std::get<3>(b); // Earlier tid first
        });
    
    // Collect all buy orders (bids) for this market
    std::vector<std::tuple<int32_t, int32_t, int32_t, int32_t>> buy_orders; // price, size, customer_id, tid
    
    // Traverse all bid price levels from highest to lowest
    for (int price = max_price_levels_ - 1; price >= 0; --price) {
        int32_t slot = bid_heads_acc[market_id][price];
        while (slot != NULL_INDEX) {
            int32_t size = order_sizes_acc[market_id][slot];
            if (size > 0) {
                buy_orders.push_back({
                    static_cast<int32_t>(price),
                    size,
                    order_customer_ids_acc[market_id][slot],
                    order_tid_acc[market_id][slot]
                });
            }
            slot = order_next_acc[market_id][slot];
        }
    }
    
    // Build output string
    std::stringstream ss;
    
    // Sell orders section
    ss << fmt::format("####### {} sell orders #######\n", sell_orders.size());
    for (const auto& [price, size, customer_id, tid] : sell_orders) {
        ss << fmt::format("px {} @ sz {}   id={} @ t={}\n", price, size, customer_id, tid);
    }
    ss << "#############################\n";
    
    // Buy orders section
    ss << fmt::format("####### {} buy orders #######\n", buy_orders.size());
    for (const auto& [price, size, customer_id, tid] : buy_orders) {
        ss << fmt::format("px {} @ sz {}   id={} @ t={}\n", price, size, customer_id, tid);
    }
    ss << "#############################";
    
    return ss.str();
}

void VecMarket::Reset()
{
    // Reset all price level linked lists to NULL_INDEX
    bid_heads_.fill_(NULL_INDEX);
    ask_heads_.fill_(NULL_INDEX);
    bid_tails_.fill_(NULL_INDEX);
    ask_tails_.fill_(NULL_INDEX);
    
    // Reset order pool - no need to clear individual orders as they're invalidated by the linked lists
    // But we should reset the next pointers to NULL_INDEX for safety
    order_next_.fill_(NULL_INDEX);
    
    // Reset market state
    order_next_slots_.zero_();  // Reset slot counters to 0
    fill_counts_.zero_();        // Reset fill counts to 0
    
    // Reset customer portfolios to zero
    customer_portfolios_.zero_();
    
    // Reset the global TID counter
    global_tid_counter_ = 0;
}

void FillBatch::Reset() {
    fill_counts.zero_();
}

// ================================================================================
// BBOBatch IMPLEMENTATION
// ================================================================================

// CUDA kernel to update last prices based on fills
__global__ void update_last_prices_kernel(
    int32_t* last_prices,
    const int32_t* fill_counts,
    const int32_t* fill_prices,
    int32_t num_markets,
    int32_t max_fills_per_market)
{
    int32_t market_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (market_id >= num_markets) return;
    
    int32_t fill_count = fill_counts[market_id];
    
    // If there were fills, update last price to the last fill's price
    if (fill_count > 0) {
        int32_t last_fill_idx = fill_count - 1;
        int32_t price_idx = market_id * max_fills_per_market + last_fill_idx;
        last_prices[market_id] = fill_prices[price_idx];
    }
    // Otherwise, last_prices[market_id] remains unchanged
}

void BBOBatch::Reset() 
{
    // Reset all BBO data to zeros
    best_bid_prices.zero_();
    best_bid_sizes.zero_();
    best_ask_prices.zero_();
    best_ask_sizes.zero_();
    // Reset last_prices to NULL_INDEX to indicate no trades
    last_prices.fill_(NULL_INDEX);
}

void BBOBatch::UpdateLastPrices(FillBatch& fills)
{
    // Ensure tensors are on the same device
    ASTRA_CHECK_EQ(fills.fill_counts.device(), last_prices.device());
    ASTRA_CHECK_EQ(fills.fill_prices.device(), last_prices.device());
    
    // Get dimensions
    int32_t num_markets = fills.fill_counts.size(0);
    int32_t max_fills = fills.fill_prices.size(1);
    
    // Launch kernel with 64 threads per block as requested
    int32_t threads_per_block = 64;
    int32_t num_blocks = (num_markets + threads_per_block - 1) / threads_per_block;
    
    update_last_prices_kernel<<<num_blocks, threads_per_block>>>(
        last_prices.data_ptr<int32_t>(),
        fills.fill_counts.data_ptr<int32_t>(),
        fills.fill_prices.data_ptr<int32_t>(),
        num_markets,
        max_fills
    );
    
    // Check for kernel errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        AstraFatalError(std::string("CUDA error in UpdateLastPrices: ") + cudaGetErrorString(error));
    }
}

}  // namespace order_matching
}  // namespace astra
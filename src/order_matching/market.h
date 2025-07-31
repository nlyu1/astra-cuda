// ================================================================================
// VECTORIZED CUDA MARKET SYSTEM 
// ================================================================================
//
// OVERVIEW:
// Implements batched, parallel double-auction markets on GPU for high-performance
// order matching. All markets operate independently and in lock-step, processing
// two-sided quotes (bid and ask) from each customer simultaneously.
//
// KEY DESIGN PRINCIPLES:
// - All markets are completely independent (no cross-market interactions)
// - Each customer submits exactly one two-sided quote per timestep
// - All data structures remain on GPU to minimize transfers
// - Fixed-size pre-allocated memory pools (no dynamic allocation)
// - Price-level based indexing for O(1) order insertion
// - Linked list traversal within price levels for time priority
//
// MEMORY LAYOUT:
// - Order book uses price-level indexing: bid_heads_[market][price], ask_heads_[market][price]
// - Orders stored in pre-allocated pools indexed by [market][slot]
// - Ring buffer design for order and fill pools with wraparound
// - All indices use int32_t with NULL_INDEX (0xFFFFFFFF) as null pointer
//
// PARALLELIZATION STRATEGY:
// - One CUDA thread per market for order insertion and matching
// - Coalesced memory access patterns for optimal GPU performance
//
// MATCHING ALGORITHM:
// 1. Add bid order to appropriate price level (maintaining time priority)
// 2. Add ask order to appropriate price level (maintaining time priority)
// 3. Match orders starting from best bid/ask prices
// 4. Generate fills for matched orders
// 5. Update best bid/ask prices
//
// CONSTRAINTS:
// - Maximum markets: MAX_MARKETS
// - Maximum price levels per market: PRICE_LEVELS
// - Maximum active orders per market: MAX_ACTIVE_ORDERS_PER_MARKET 
// - Maximum fills per market: MAX_ACTIVE_FILLS_PER_MARKET

#pragma once 

#include <string>
#include <vector>
#include <queue>
#include <cstdint>
#include <iostream>
#include <optional>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "astra_utils.h"

namespace astra {
namespace order_matching {

constexpr int32_t MAX_MARKETS = 2048 * 2048;
constexpr int32_t PRICE_LEVELS = 128; 
constexpr int32_t MAX_ACTIVE_ORDERS_PER_MARKET = 1024; 
constexpr int32_t MAX_ACTIVE_FILLS_PER_MARKET = 1024; 
constexpr int32_t NULL_INDEX = 0xFFFFFFFF;
constexpr int32_t MAX_CUSTOMERS = 1024; 

// Structure for returning fill information
class FillBatch {
    public:
        torch::Tensor fill_prices;        // [num_markets, max_fills] int32_t - execution price
        torch::Tensor fill_sizes;         // [num_markets, max_fills] int32_t - fill size
        torch::Tensor fill_customer_ids;  // [num_markets, max_fills] int32_t - taker customer ID
        torch::Tensor fill_quoter_ids;    // [num_markets, max_fills] int32_t - quoter customer ID
        torch::Tensor fill_is_sell_quote; // [num_markets, max_fills] bool - true if sell quote, false if buy quote
        torch::Tensor fill_quote_sizes;   // [num_markets, max_fills] int32_t - size of the quote order
        torch::Tensor fill_tid;           // [num_markets, max_fills] int32_t - time ID of the taker order
        torch::Tensor fill_quote_tid;     // [num_markets, max_fills] int32_t - time ID of the quote order
        torch::Tensor fill_counts;        // [num_markets] int32_t - number of fills per market
        
        // Default constructor
        FillBatch() = default;
        
        // Copy constructor that clones all tensors
        FillBatch(const FillBatch& other)
            : fill_prices(other.fill_prices.clone()),
              fill_sizes(other.fill_sizes.clone()),
              fill_customer_ids(other.fill_customer_ids.clone()),
              fill_quoter_ids(other.fill_quoter_ids.clone()),
              fill_is_sell_quote(other.fill_is_sell_quote.clone()),
              fill_quote_sizes(other.fill_quote_sizes.clone()),
              fill_tid(other.fill_tid.clone()),
              fill_quote_tid(other.fill_quote_tid.clone()),
              fill_counts(other.fill_counts.clone()) {}
}; 

// Structure for returning best bid/offer information
class BBOBatch {
public:
    torch::Tensor best_bid_prices;  // [num_markets] int32_t
    torch::Tensor best_bid_sizes;   // [num_markets] int32_t
    torch::Tensor best_ask_prices;  // [num_markets] int32_t
    torch::Tensor best_ask_sizes;   // [num_markets] int32_t
    torch::Tensor last_prices;      // [num_markets] int32_t
    
    // Default constructor. 
    BBOBatch() = default;
    
    // Copy constructor that clones all tensors
    BBOBatch(const BBOBatch& other) 
        : best_bid_prices(other.best_bid_prices.clone()),
          best_bid_sizes(other.best_bid_sizes.clone()),
          best_ask_prices(other.best_ask_prices.clone()),
          best_ask_sizes(other.best_ask_sizes.clone()),
          last_prices(other.last_prices.clone()) {}

    void Reset();

    // Update last prices based on fills. 
    // For each market, if fill_counts > 0, update on last fill price. Otherwise carry over last last_price
    void UpdateLastPrices(FillBatch& fills); 
};

class VecMarket {
public:
    /**
     * @brief Constructs a vectorized market system for parallel order matching on GPU
     * 
     * Initializes all GPU tensors and validates input parameters against system constraints.
     * All tensors are allocated on the specified CUDA device.
     * 
     * @param num_markets Number of independent markets to simulate in parallel
     * @param max_price_levels Maximum number of price levels per market (must be <= PRICE_LEVELS)
     * @param max_active_orders_per_market Maximum orders per market (must be <= MAX_ACTIVE_ORDERS_PER_MARKET)
     * @param max_active_fills_per_market Maximum fills per market (must be <= MAX_ACTIVE_FILLS_PER_MARKET)
     * @param num_customers Number of customers per market (default: 16, must be <= MAX_CUSTOMERS)
     * @param device_id CUDA device ID to use for tensor allocation (default: 0)
     * @param threads_per_block Number of threads per block for CUDA kernels (default: 256)
     * @throws AstraFatalError if any parameter exceeds system constraints or device is invalid
     */
    VecMarket(int32_t num_markets, int32_t max_price_levels, 
        int32_t max_active_orders_per_market, int32_t max_active_fills_per_market,
        int32_t num_customers = 16, int device_id = 0, int32_t threads_per_block = 64); 
    
    /**
     * @brief Copy constructor that creates a deep copy of another VecMarket
     * 
     * Creates a new VecMarket instance with cloned tensors from the source market.
     * All tensors are deep copied to ensure complete independence between instances.
     * The global TID counter is also copied to maintain consistency.
     * 
     * @param other The VecMarket instance to copy from
     */
    VecMarket(const VecMarket& other);
    
    ~VecMarket(); 

    /**
     * @brief Creates a new FillBatch with properly allocated tensors
     * 
     * Allocates all tensors needed for a FillBatch on the specified device.
     * This should be called once and the FillBatch reused across multiple
     * AddTwoSidedQuotes calls for efficiency.
     * 
     * @return FillBatch with all tensors allocated on the correct device
     */
    FillBatch NewFillBatch() const;
    
    /**
     * @brief Creates a new BBOBatch with properly allocated tensors
     * 
     * Allocates all tensors needed for a BBOBatch on the specified device.
     * This should be called once and the BBOBatch reused across multiple
     * GetBBOs calls for efficiency.
     * 
     * @return BBOBatch with all tensors allocated on the correct device
     */
    BBOBatch NewBBOBatch() const;
    
    /**
     * @brief Adds two-sided quotes (bid and ask) for all markets in parallel
     * 
     * Each market receives exactly one bid and one ask order. Orders are processed
     * in parallel using CUDA kernels. Size-0 orders are ignored and not added to
     * the order book. This method:
     * 1. Validates input tensor dimensions
     * 2. Launches add_orders_kernel to insert orders into price-level linked lists
     * 3. Maintains time priority within each price level
     * 4. Matches orders and fills the provided FillBatch
     * 
     * CUDA Kernel: add_orders_kernel<<<num_markets, 1>>>
     * - One thread per market
     * - Adds bid order to bid_heads_/bid_tails_ linked list at price level
     * - Adds ask order to ask_heads_/ask_tails_ linked list at price level
     * - Updates order pool with order details
     * - Handles ring buffer wraparound for order slots
     * 
     * @param bid_px Bid prices for each market [num_markets] int32_t on GPU
     * @param bid_sz Bid sizes for each market [num_markets] int32_t on GPU
     * @param ask_px Ask prices for each market [num_markets] int32_t on GPU
     * @param ask_sz Ask sizes for each market [num_markets] int32_t on GPU
     * @param customer_ids Customer IDs for each market [num_markets] int32_t on GPU
     * @param fills FillBatch to populate with matched orders (must be created with NewFillBatch)
     */
    void AddTwoSidedQuotes(
        torch::Tensor bid_px, // [num_markets], uint32
        torch::Tensor bid_sz, // [num_markets], uint32
        torch::Tensor ask_px, // [num_markets], uint32
        torch::Tensor ask_sz, // [num_markets], uint32
        torch::Tensor customer_ids, // [num_markets], uint32
        FillBatch& fills
    );
    
    /**
     * @brief Populates the best bid and offer (BBO) for all markets
     * 
     * Retrieves the current best bid and ask prices and sizes for each market.
     * Markets with no bids or asks will have NULL_INDEX for prices and 0 for sizes.
     * 
     * CUDA Kernel: get_bbo_kernel<<<num_markets, 1>>>
     * - One thread per market  
     * - Finds highest bid price with orders
     * - Finds lowest ask price with orders
     * - Reads first order at each best price level for size
     * 
     * @param bbos BBOBatch to populate with best prices and sizes (must be created with NewBBOBatch)
     */
    void GetBBOs(BBOBatch& bbos);
    
    /**
     * @brief Returns a string representation of a specific market's order book
     * 
     * Formats the order book similar to the legacy implementation, showing all
     * sell orders from highest to lowest price, then all buy orders from highest
     * to lowest price.
     * 
     * @param market_id The market to display (must be < num_markets)
     * @return String representation of the market's order book
     */
    std::string ToString(int32_t market_id) const;
    
    /**
     * @brief Resets all market state to initial empty state
     * 
     * Clears all order books, resets customer portfolios to zero, and resets
     * the global TID counter. This method does not deallocate or reallocate
     * any tensors - it only resets their values. This is useful for reusing
     * the same market instance across multiple simulations.
     * 
     * After calling Reset():
     * - All order books are empty
     * - All customer portfolios are zero
     * - Order slot counters are reset to 0
     * - Global TID counter is reset to 0
     */
    void Reset();
    
    /**
     * @brief Returns a read-only view of customer portfolios
     * 
     * Returns the customer portfolio tensor which tracks the number of contracts
     * and cash amount for each customer in each market. The tensor has shape
     * [num_markets, num_customers, 2] where:
     * - Index 0: Number of contracts (positive for long, negative for short)
     * - Index 1: Cash amount
     * 
     * @return const reference to customer portfolios tensor
     */
    void CopyCustomerPortfoliosTo(torch::Tensor buffer) const { 
        // Check that shapes are equal 
        ASTRA_CHECK_EQ(buffer.dim(), 3); 
        ASTRA_CHECK_EQ(buffer.size(0), num_markets_); 
        ASTRA_CHECK_EQ(buffer.size(1), num_customers_); 
        ASTRA_CHECK_EQ(buffer.size(2), 2); 
        buffer.copy_(customer_portfolios_, /*non_blocking=*/false); 
    } 

    const torch::Tensor& GetCustomerPortfolios() const { 
        return customer_portfolios_; 
    } 
    
private:
    /**
     * @brief Matches orders for all markets in parallel and populates fills
     * 
     * Executes the matching algorithm for all markets simultaneously. For each market:
     * 1. Finds best bid and ask prices
     * 2. Matches crossing orders (bid_price >= ask_price)
     * 3. Generates fills at the resting order price
     * 4. Updates order sizes (partial fills) or removes filled orders
     * 5. Continues until no more crosses exist
     * 
     * CUDA Kernel: match_orders_kernel<<<num_markets, 1>>>
     * - One thread per market
     * - Traverses bid and ask linked lists from best prices
     * - Matches orders using price-time priority
     * - Writes fills to provided FillBatch
     * - Updates order linked lists and sizes
     * 
     * @param fills FillBatch to populate with matched orders
     */
    void MatchAllMarkets(FillBatch& fills);
    // Price level linked list heads and tails
    // Shape: [num_markets, price_levels]. Value is order pool index or NULL_INDEX
    torch::Tensor bid_heads_; // Head of linked list for each bid price level
    torch::Tensor ask_heads_; // Head of linked list for each ask price level
    torch::Tensor bid_tails_; // Tail of linked list for each bid price level (for O(1) append)
    torch::Tensor ask_tails_; // Tail of linked list for each ask price level (for O(1) append)

    // Order pool - stores all active orders
    // Shape: [num_markets, max_active_orders_per_market]
    torch::Tensor order_prices_;     // uint32 - price of this order
    torch::Tensor order_sizes_;      // uint32 - remaining size of this order
    torch::Tensor order_customer_ids_; // uint32 - customer who submitted this order
    torch::Tensor order_tid_;        // uint32 - time ID of this order
    torch::Tensor order_is_bid_;     // bool - true for bid, false for ask
    torch::Tensor order_next_;       // uint32 - next order index in price level linked list

    // Market state
    torch::Tensor order_next_slots_; // uint32[num_markets] - next available order slot (ring buffer)
    torch::Tensor fill_counts_;      // uint32[num_markets] - number of fills per market

    // Customer states: two indices correspond to number of contracts and amount of cash, respectively
    torch::Tensor customer_portfolios_; // int32[num_markets, num_customers, 2]
    int32_t global_tid_counter_;    // Global transaction ID counter (host-side)
    
    // System parameters
    int32_t num_markets_;
    int32_t max_price_levels_;
    int32_t max_orders_per_market_;
    int32_t max_fills_per_market_;
    int32_t num_customers_;
    
    // CUDA execution parameters
    int device_id_;              // CUDA device ID for this market instance
    int32_t threads_per_block_; // Number of threads per block for kernel launches
};

// ================================================================================
// CUDA KERNEL DECLARATIONS 
// ================================================================================

/**
 * @brief CUDA kernel to add orders to the order book
 * 
 * Each thread handles one market. Adds bid and ask orders to their respective
 * price level linked lists while maintaining time priority. Updates customer
 * portfolios to reflect pending orders.
 * 
 * @param bid_px Bid prices [num_markets]
 * @param bid_sz Bid sizes [num_markets]
 * @param ask_px Ask prices [num_markets]
 * @param ask_sz Ask sizes [num_markets]
 * @param customer_ids Customer IDs [num_markets]
 * @param bid_heads Bid price level heads [num_markets, price_levels]
 * @param bid_tails Bid price level tails [num_markets, price_levels]
 * @param ask_heads Ask price level heads [num_markets, price_levels]
 * @param ask_tails Ask price level tails [num_markets, price_levels]
 * @param order_prices Order price pool [num_markets, max_orders]
 * @param order_sizes Order size pool [num_markets, max_orders]
 * @param order_customer_ids Order customer pool [num_markets, max_orders]
 * @param order_tid Order time ID pool [num_markets, max_orders]
 * @param order_is_bid Order side pool [num_markets, max_orders]
 * @param order_next Order next pointers [num_markets, max_orders]
 * @param order_next_slots Next available slots [num_markets]
 * @param customer_portfolios Customer portfolio data [num_markets, num_customers, 2]
 * @param base_tid Base TID for this batch of orders
 * @param num_markets Number of markets
 * @param max_orders_per_market Maximum orders per market
 * @param max_price_levels Maximum price levels
 * @param num_customers Number of customers per market
 */
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
    int32_t num_customers
);

/**
 * @brief CUDA kernel to match orders and generate fills
 * 
 * Each thread handles one market. Matches crossing orders starting from
 * best bid and ask prices, generating fills until no more crosses exist.
 * Updates customer portfolios based on executed trades.
 * 
 * @param bid_heads Bid price level heads [num_markets, price_levels]
 * @param ask_heads Ask price level heads [num_markets, price_levels]
 * @param bid_tails Bid price level tails [num_markets, price_levels]
 * @param ask_tails Ask price level tails [num_markets, price_levels]
 * @param order_prices Order prices [num_markets, max_orders]
 * @param order_sizes Order sizes [num_markets, max_orders]
 * @param order_customer_ids Order customers [num_markets, max_orders]
 * @param order_tid Order time IDs [num_markets, max_orders]
 * @param order_is_bid Order sides [num_markets, max_orders]
 * @param order_next Order next pointers [num_markets, max_orders]
 * @param fill_prices Fill prices [num_markets, max_fills]
 * @param fill_sizes Fill sizes [num_markets, max_fills]
 * @param fill_customer_ids Fill taker customers [num_markets, max_fills]
 * @param fill_quoter_ids Fill quoter customers [num_markets, max_fills]
 * @param fill_is_sell_quote Fill quote side [num_markets, max_fills]
 * @param fill_quote_sizes Fill quote sizes [num_markets, max_fills]
 * @param fill_tid Fill taker TIDs [num_markets, max_fills]
 * @param fill_quote_tid Fill quote TIDs [num_markets, max_fills]
 * @param fill_counts Number of fills per market [num_markets]
 * @param customer_portfolios Customer portfolio data [num_markets, num_customers, 2]
 * @param num_markets Number of markets
 * @param max_orders_per_market Maximum orders per market
 * @param max_price_levels Maximum price levels
 * @param max_fills_per_market Maximum fills per market
 * @param num_customers Number of customers per market
 */
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
    int32_t num_customers
);

/**
 * @brief CUDA kernel to get best bid and offer for all markets
 * 
 * Each thread handles one market. Finds the best bid and ask prices
 * by scanning the price level arrays.
 * 
 * @param bid_heads Bid price level heads [num_markets, price_levels]
 * @param ask_heads Ask price level heads [num_markets, price_levels]
 * @param order_sizes Order sizes [num_markets, max_orders]
 * @param order_prices Order prices [num_markets, max_orders]
 * @param best_bid_prices Output best bid prices [num_markets]
 * @param best_bid_sizes Output best bid sizes [num_markets]
 * @param best_ask_prices Output best ask prices [num_markets]
 * @param best_ask_sizes Output best ask sizes [num_markets]
 * @param num_markets Number of markets
 * @param max_orders_per_market Maximum orders per market
 * @param max_price_levels Maximum price levels
 */
__global__ void get_bbo_kernel(
    const int32_t* bid_heads, const int32_t* ask_heads,
    const int32_t* order_sizes, const int32_t* order_prices,
    int32_t* best_bid_prices, int32_t* best_bid_sizes,
    int32_t* best_ask_prices, int32_t* best_ask_sizes,
    int32_t num_markets, int32_t max_orders_per_market,
    int32_t max_price_levels
);

}  // namespace order_matching
}  // namespace astra
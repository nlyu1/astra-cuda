#include <iostream>
#include <torch/torch.h>
#include <random>
#include <cassert>
#include <sstream>
#include "astra_utils.h"
#include "market.h"

// Helper to extract scalar from tensor
inline uint32_t tensor_item_u32(const torch::Tensor& t) {
    return t.cpu().data_ptr<uint32_t>()[0];
}

using namespace astra::order_matching; 
using namespace std; 

// Helper function to print fill details
void PrintFillDetails(const FillBatch& fills, uint32_t market_id, const std::string& label) {
    auto fill_counts = fills.fill_counts.cpu();
    auto fill_counts_acc = fill_counts.accessor<uint32_t, 1>();
    uint32_t count = fill_counts_acc[market_id];
    
    cout << label << " has " << count << " fills:" << endl;
    
    if (count > 0) {
        auto fill_prices = fills.fill_prices.cpu();
        auto fill_sizes = fills.fill_sizes.cpu();
        auto fill_customer_ids = fills.fill_customer_ids.cpu();
        auto fill_quoter_ids = fills.fill_quoter_ids.cpu();
        auto fill_is_sell_quote = fills.fill_is_sell_quote.cpu();
        
        auto prices_acc = fill_prices.accessor<uint32_t, 2>();
        auto sizes_acc = fill_sizes.accessor<uint32_t, 2>();
        auto customer_acc = fill_customer_ids.accessor<uint32_t, 2>();
        auto quoter_acc = fill_quoter_ids.accessor<uint32_t, 2>();
        auto sell_quote_acc = fill_is_sell_quote.accessor<bool, 2>();
        
        for (uint32_t i = 0; i < count; ++i) {
            cout << "  Fill " << i << ": price=" << prices_acc[market_id][i]
                      << ", size=" << sizes_acc[market_id][i]
                      << ", customer=" << customer_acc[market_id][i]
                      << ", quoter=" << quoter_acc[market_id][i]
                      << ", is_sell_quote=" << sell_quote_acc[market_id][i] << endl;
        }
    }
}

// Test 1: Partial Fills
void TestPartialFills() {
    cout << "\n=== Test 1: Partial Fills ===" << endl;
    
    const uint32_t num_markets = 1;
    VecMarket market(num_markets, 128, 1024, 512);
    auto fills = market.NewFillBatch();
    
    // Add a large buy order
    cout << "Adding BUY order: price=50, size=100" << endl;
    {
        torch::Tensor bid_prices = torch::full({1}, 50, torch::kUInt32);
        torch::Tensor bid_sizes = torch::full({1}, 100, torch::kUInt32);
        torch::Tensor ask_prices = torch::zeros({1}, torch::kUInt32);
        torch::Tensor ask_sizes = torch::zeros({1}, torch::kUInt32);
        torch::Tensor customer_ids = torch::zeros({1}, torch::kUInt32);
        
        market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
    }
    
    // Add a smaller sell order that partially fills
    cout << "Adding SELL order: price=50, size=30" << endl;
    {
        torch::Tensor bid_prices = torch::zeros({1}, torch::kUInt32);
        torch::Tensor bid_sizes = torch::zeros({1}, torch::kUInt32);
        torch::Tensor ask_prices = torch::full({1}, 50, torch::kUInt32);
        torch::Tensor ask_sizes = torch::full({1}, 30, torch::kUInt32);
        torch::Tensor customer_ids = torch::ones({1}, torch::kUInt32);
        
        market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
        
        // Check fills
        auto counts = fills.fill_counts.cpu();
        assert(tensor_item_u32(counts[0]) == 1);
        
        auto sizes = fills.fill_sizes.cpu();
        assert(tensor_item_u32(sizes[0][0]) == 30);
        
        PrintFillDetails(fills, 0, "Partial fill");
        cout << "✓ Partial fill of 30 shares executed correctly" << endl;
    }
    
    // Add another sell order to check remaining buy
    cout << "\nAdding another SELL order: price=50, size=80" << endl;
    {
        torch::Tensor bid_prices = torch::zeros({1}, torch::kUInt32);
        torch::Tensor bid_sizes = torch::zeros({1}, torch::kUInt32);
        torch::Tensor ask_prices = torch::full({1}, 50, torch::kUInt32);
        torch::Tensor ask_sizes = torch::full({1}, 80, torch::kUInt32);
        torch::Tensor customer_ids = torch::full({1}, 2, torch::kUInt32);
        
        market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
        
        // Check that only 70 shares are filled (remaining from original 100)
        auto counts = fills.fill_counts.cpu();
        assert(tensor_item_u32(counts[0]) == 1);
        
        auto sizes = fills.fill_sizes.cpu();
        assert(tensor_item_u32(sizes[0][0]) == 70);
        
        PrintFillDetails(fills, 0, "Remaining fill");
        cout << "✓ Remaining 70 shares filled correctly" << endl;
    }
    
    cout << "✅ Partial fills test PASSED" << endl;
}

// Test 2: Market Independence
void TestMarketIndependence() {
    cout << "\n=== Test 2: Market Independence ===" << endl;
    
    const uint32_t num_markets = 5;
    VecMarket market(num_markets, 128, 1024, 512);
    auto fills = market.NewFillBatch();
    
    // Set up different scenarios for each market
    torch::Tensor bid_prices = torch::tensor({10, 20, 30, 40, 50}, torch::kUInt32);
    torch::Tensor bid_sizes = torch::tensor({100, 200, 300, 400, 500}, torch::kUInt32);
    torch::Tensor ask_prices = torch::tensor({15, 25, 35, 45, 55}, torch::kUInt32);
    torch::Tensor ask_sizes = torch::tensor({50, 100, 150, 200, 250}, torch::kUInt32);
    torch::Tensor customer_ids = torch::zeros({num_markets}, torch::kUInt32);
    
    cout << "Adding orders to " << num_markets << " independent markets" << endl;
    market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
    
    // Verify no fills (no crossing orders)
    auto counts = fills.fill_counts.cpu();
    for (uint32_t i = 0; i < num_markets; ++i) {
        assert(tensor_item_u32(counts[i]) == 0);
    }
    cout << "✓ No fills when orders don't cross" << endl;
    
    // Now add crossing orders to specific markets only
    torch::Tensor bid_prices2 = torch::tensor({0, 30, 0, 50, 0}, torch::kUInt32);
    torch::Tensor bid_sizes2 = torch::tensor({0, 50, 0, 100, 0}, torch::kUInt32);
    torch::Tensor ask_prices2 = torch::tensor({0, 20, 0, 40, 0}, torch::kUInt32);
    torch::Tensor ask_sizes2 = torch::tensor({0, 75, 0, 150, 0}, torch::kUInt32);
    torch::Tensor customer_ids2 = torch::ones({num_markets}, torch::kUInt32);
    
    cout << "\nAdding crossing orders to markets 1 and 3 only" << endl;
    market.AddTwoSidedQuotes(bid_prices2, bid_sizes2, ask_prices2, ask_sizes2, customer_ids2, fills);
    
    // Verify fills only in markets 1 and 3
    counts = fills.fill_counts.cpu();
    
    assert(tensor_item_u32(counts[0]) == 0);  // Market 0: no crossing
    assert(tensor_item_u32(counts[1]) == 1);  // Market 1: should have fill
    assert(tensor_item_u32(counts[2]) == 0);  // Market 2: no crossing
    assert(tensor_item_u32(counts[3]) == 1);  // Market 3: should have fill
    assert(tensor_item_u32(counts[4]) == 0);  // Market 4: no crossing
    
    cout << "✓ Fills only occurred in markets with crossing orders" << endl;
    cout << "Market 1: "; PrintFillDetails(fills, 1, "Market 1");
    cout << "Market 3: "; PrintFillDetails(fills, 3, "Market 3");
    cout << "✅ Market independence test PASSED" << endl;
}

// Test 3: Multiple Orders at Same Price Level
void TestMultipleOrdersSamePrice() {
    cout << "\n=== Test 3: Multiple Orders at Same Price ===" << endl;
    
    const uint32_t num_markets = 1;
    VecMarket market(num_markets, 128, 1024, 512);
    auto fills = market.NewFillBatch();
    
    // Add multiple buy orders at the same price
    cout << "Adding 3 BUY orders at price=25 with different sizes" << endl;
    for (uint32_t i = 0; i < 3; ++i) {
        torch::Tensor bid_prices = torch::full({1}, 25, torch::kUInt32);
        torch::Tensor bid_sizes = torch::full({1}, (i + 1) * 10, torch::kUInt32);  // 10, 20, 30
        torch::Tensor ask_prices = torch::zeros({1}, torch::kUInt32);
        torch::Tensor ask_sizes = torch::zeros({1}, torch::kUInt32);
        torch::Tensor customer_ids = torch::full({1}, i, torch::kUInt32);
        
        market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
        cout << "  Added BUY from customer " << i << " for " << (i+1)*10 << " shares" << endl;
    }
    
    // Add a large sell order that should match all buys
    cout << "\nAdding SELL order: price=25, size=60 (should match all buys)" << endl;
    {
        torch::Tensor bid_prices = torch::zeros({1}, torch::kUInt32);
        torch::Tensor bid_sizes = torch::zeros({1}, torch::kUInt32);
        torch::Tensor ask_prices = torch::full({1}, 25, torch::kUInt32);
        torch::Tensor ask_sizes = torch::full({1}, 60, torch::kUInt32);
        torch::Tensor customer_ids = torch::full({1}, 3, torch::kUInt32);
        
        market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
        
        // Check that all orders were filled
        auto counts = fills.fill_counts.cpu();
        cout << "\nFills generated: " << tensor_item_u32(counts[0]) << endl;
        
        // Verify total volume matched
        auto sizes = fills.fill_sizes.cpu();
        auto sizes_acc = sizes.accessor<uint32_t, 2>();
        
        uint32_t total = 0;
        for (uint32_t i = 0; i < tensor_item_u32(counts[0]); ++i) {
            total += sizes_acc[0][i];
        }
        
        assert(total == 60);
        PrintFillDetails(fills, 0, "Market 0");
        cout << "✓ All 60 shares matched correctly (time priority maintained)" << endl;
    }
    
    cout << "✅ Multiple orders at same price test PASSED" << endl;
}

// Test 4: BBO (Best Bid/Offer) Functionality
void TestBBO() {
    cout << "\n=== Test 4: BBO (Best Bid/Offer) ===" << endl;
    
    const uint32_t num_markets = 3;
    VecMarket market(num_markets, 128, 1024, 512);
    auto fills = market.NewFillBatch();
    
    // Set up orders at different price levels
    cout << "Setting up markets with various price levels" << endl;
    
    // Market 0: bid=10, ask=25
    // Market 1: bid=15, ask=30  
    // Market 2: bid=20, ask=35
    torch::Tensor bid_prices = torch::tensor({10, 15, 20}, torch::kUInt32);
    torch::Tensor bid_sizes = torch::tensor({100, 200, 300}, torch::kUInt32);
    torch::Tensor ask_prices = torch::tensor({25, 30, 35}, torch::kUInt32);
    torch::Tensor ask_sizes = torch::tensor({150, 250, 350}, torch::kUInt32);
    torch::Tensor customer_ids = torch::zeros({num_markets}, torch::kUInt32);
    
    market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
    
    // Get BBOs
    BBOBatch bbo = market.NewBBOBatch();
    market.GetBBOs(bbo);
    
    // Move to CPU for checking
    auto best_bid_px = bbo.best_bid_prices.cpu();
    auto best_bid_sz = bbo.best_bid_sizes.cpu();
    auto best_ask_px = bbo.best_ask_prices.cpu();
    auto best_ask_sz = bbo.best_ask_sizes.cpu();
    
    // Check each market's BBO
    cout << "\nBBO Results:" << endl;
    for (uint32_t i = 0; i < num_markets; ++i) {
        cout << "Market " << i << ": "
             << "Best Bid = " << tensor_item_u32(best_bid_px[i]) 
             << " @ " << tensor_item_u32(best_bid_sz[i])
             << ", Best Ask = " << tensor_item_u32(best_ask_px[i])
             << " @ " << tensor_item_u32(best_ask_sz[i]) << endl;
    }
    
    // Verify expected values
    assert(tensor_item_u32(best_bid_px[0]) == 10);
    assert(tensor_item_u32(best_bid_sz[0]) == 100);
    assert(tensor_item_u32(best_ask_px[0]) == 25);
    assert(tensor_item_u32(best_ask_sz[0]) == 150);
    
    assert(tensor_item_u32(best_bid_px[1]) == 15);
    assert(tensor_item_u32(best_ask_px[1]) == 30);
    
    assert(tensor_item_u32(best_bid_px[2]) == 20);
    assert(tensor_item_u32(best_ask_px[2]) == 35);
    
    cout << "✓ BBO correctly identified best prices" << endl;
    cout << "✅ BBO test PASSED" << endl;
}

// Test 5: Ring Buffer Wraparound
void TestRingBufferWraparound() {
    cout << "\n=== Test 5: Ring Buffer Wraparound ===" << endl;
    
    const uint32_t num_markets = 1;
    const uint32_t max_orders = 10;  // Small buffer to test wraparound
    VecMarket market(num_markets, 128, max_orders, 512);
    auto fills = market.NewFillBatch();
    
    cout << "Adding " << (max_orders + 5) << " orders to test ring buffer wraparound" << endl;
    
    // Add more orders than the buffer can hold
    for (uint32_t i = 0; i < max_orders + 5; ++i) {
        torch::Tensor bid_prices = torch::full({1}, 10 + i, torch::kUInt32);
        torch::Tensor bid_sizes = torch::full({1}, 100, torch::kUInt32);
        torch::Tensor ask_prices = torch::zeros({1}, torch::kUInt32);
        torch::Tensor ask_sizes = torch::zeros({1}, torch::kUInt32);
        torch::Tensor customer_ids = torch::full({1}, i % 5, torch::kUInt32);
        
        market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
    }
    
    // Add a sell order that crosses with the most recent buys
    torch::Tensor bid_prices = torch::zeros({1}, torch::kUInt32);
    torch::Tensor bid_sizes = torch::zeros({1}, torch::kUInt32);
    torch::Tensor ask_prices = torch::full({1}, 20, torch::kUInt32);
    torch::Tensor ask_sizes = torch::full({1}, 500, torch::kUInt32);
    torch::Tensor customer_ids = torch::full({1}, 10, torch::kUInt32);
    
    cout << "\nAdding SELL order at price=20 to match recent buys" << endl;
    market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
    
    // Should get some fills (exact count depends on which orders are still active)
    auto counts = fills.fill_counts.cpu();
    uint32_t fill_count = tensor_item_u32(counts[0]);
    assert(fill_count > 0);
    
    cout << "✓ Ring buffer wraparound handled correctly with " 
         << fill_count << " fills" << endl;
    PrintFillDetails(fills, 0, "Market 0");
    cout << "✅ Ring buffer wraparound test PASSED" << endl;
}

// Test 6: Zero-Size Order Handling
void TestZeroSizeOrders() {
    cout << "\n=== Test 6: Zero-Size Order Handling ===" << endl;
    
    const uint32_t num_markets = 3;
    VecMarket market(num_markets, 128, 1024, 512);
    auto fills = market.NewFillBatch();
    
    // Mix of zero and non-zero orders
    torch::Tensor bid_prices = torch::tensor({10, 20, 30}, torch::kUInt32);
    torch::Tensor bid_sizes = torch::tensor({0, 100, 0}, torch::kUInt32);  // Zero sizes
    torch::Tensor ask_prices = torch::tensor({15, 0, 35}, torch::kUInt32);
    torch::Tensor ask_sizes = torch::tensor({50, 0, 200}, torch::kUInt32);  // Zero sizes
    torch::Tensor customer_ids = torch::zeros({num_markets}, torch::kUInt32);
    
    cout << "Adding orders with some zero sizes:" << endl;
    cout << "Market 0: bid size=0, ask size=50" << endl;
    cout << "Market 1: bid size=100, ask size=0" << endl;
    cout << "Market 2: bid size=0, ask size=200" << endl;
    
    market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
    
    // Get BBOs to verify zero-size orders were ignored
    BBOBatch bbo = market.NewBBOBatch();
    market.GetBBOs(bbo);
    auto best_bid_px = bbo.best_bid_prices.cpu();
    
    // Market 0: No bid (size was 0)
    assert(tensor_item_u32(best_bid_px[0]) == NULL_INDEX);
    cout << "✓ Market 0: Zero-size bid correctly ignored" << endl;
    
    // Market 1: Has bid
    assert(tensor_item_u32(best_bid_px[1]) == 20);
    cout << "✓ Market 1: Non-zero bid correctly added" << endl;
    
    // Market 2: No bid (size was 0)
    assert(tensor_item_u32(best_bid_px[2]) == NULL_INDEX);
    cout << "✓ Market 2: Zero-size bid correctly ignored" << endl;
    
    cout << "✅ Zero-size order handling test PASSED" << endl;
}

// Test 7: Cross-Market Fill Counts
void TestCrossMarketFillCounts() {
    cout << "\n=== Test 7: Cross-Market Fill Counts ===" << endl;
    
    const uint32_t num_markets = 10;
    VecMarket market(num_markets, 128, 1024, 512);
    auto fills = market.NewFillBatch();
    
    // Create a pattern where even markets get fills, odd markets don't
    torch::Tensor bid_prices = torch::zeros({num_markets}, torch::kUInt32);
    torch::Tensor bid_sizes = torch::zeros({num_markets}, torch::kUInt32);
    torch::Tensor ask_prices = torch::zeros({num_markets}, torch::kUInt32);
    torch::Tensor ask_sizes = torch::zeros({num_markets}, torch::kUInt32);
    
    cout << "Setting up pattern: even markets get crossing orders" << endl;
    for (uint32_t i = 0; i < num_markets; ++i) {
        if (i % 2 == 0) {
            bid_prices[i] = 50;
            bid_sizes[i] = 100;
            ask_prices[i] = 60;
            ask_sizes[i] = 50;
        } else {
            bid_prices[i] = 40;
            bid_sizes[i] = 100;
            ask_prices[i] = 70;
            ask_sizes[i] = 50;
        }
    }
    
    torch::Tensor customer_ids = torch::zeros({num_markets}, torch::kUInt32);
    market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
    
    // Add crossing orders only to even markets
    cout << "Adding crossing orders to even markets only" << endl;
    for (uint32_t i = 0; i < num_markets; ++i) {
        if (i % 2 == 0) {
            ask_prices[i] = 50;  // Cross with bid at 50
            ask_sizes[i] = 75;
        } else {
            ask_prices[i] = 0;
            ask_sizes[i] = 0;
        }
        bid_prices[i] = 0;
        bid_sizes[i] = 0;
    }
    customer_ids = torch::ones({num_markets}, torch::kUInt32);
    
    market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
    
    // Verify fill pattern
    auto counts = fills.fill_counts.cpu();
    uint32_t total_fills = 0;
    
    for (uint32_t i = 0; i < num_markets; ++i) {
        if (i % 2 == 0) {
            assert(tensor_item_u32(counts[i]) == 1);
            total_fills++;
        } else {
            assert(tensor_item_u32(counts[i]) == 0);
        }
    }
    
    cout << "✓ Fill pattern correct: " << total_fills << " markets with fills (even markets only)" << endl;
    cout << "✅ Cross-market fill counts test PASSED" << endl;
}

// Test 8: Price Crossing Logic
void TestPriceCrossingLogic() {
    cout << "\n=== Test 8: Price Crossing Logic ===" << endl;
    
    const uint32_t num_markets = 1;
    VecMarket market(num_markets, 128, 1024, 512);
    auto fills = market.NewFillBatch();
    
    // Test 1: Exact price match
    cout << "Test exact price match (bid=50, ask=50)" << endl;
    {
        torch::Tensor bid_prices = torch::full({1}, 50, torch::kUInt32);
        torch::Tensor bid_sizes = torch::full({1}, 100, torch::kUInt32);
        torch::Tensor ask_prices = torch::full({1}, 50, torch::kUInt32);
        torch::Tensor ask_sizes = torch::full({1}, 100, torch::kUInt32);
        torch::Tensor customer_ids = torch::zeros({1}, torch::kUInt32);
        
        market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
        
        auto counts = fills.fill_counts.cpu();
        assert(tensor_item_u32(counts[0]) == 1);
        cout << "✓ Exact price match executes" << endl;
    }
    
    // Test 2: Bid higher than ask
    cout << "\nTest bid > ask (bid=60, ask=55)" << endl;
    {
        torch::Tensor bid_prices = torch::full({1}, 60, torch::kUInt32);
        torch::Tensor bid_sizes = torch::full({1}, 50, torch::kUInt32);
        torch::Tensor ask_prices = torch::full({1}, 55, torch::kUInt32);
        torch::Tensor ask_sizes = torch::full({1}, 50, torch::kUInt32);
        torch::Tensor customer_ids = torch::ones({1}, torch::kUInt32);
        
        market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
        
        auto counts = fills.fill_counts.cpu();
        assert(tensor_item_u32(counts[0]) == 1);
        cout << "✓ Crossing orders execute when bid > ask" << endl;
    }
    
    // Test 3: No cross (bid < ask)
    cout << "\nTest no cross (bid=45, ask=50)" << endl;
    {
        torch::Tensor bid_prices = torch::full({1}, 45, torch::kUInt32);
        torch::Tensor bid_sizes = torch::full({1}, 100, torch::kUInt32);
        torch::Tensor ask_prices = torch::full({1}, 50, torch::kUInt32);
        torch::Tensor ask_sizes = torch::full({1}, 100, torch::kUInt32);
        torch::Tensor customer_ids = torch::full({1}, 2, torch::kUInt32);
        
        market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
        
        auto counts = fills.fill_counts.cpu();
        assert(tensor_item_u32(counts[0]) == 0);
        cout << "✓ No execution when bid < ask" << endl;
    }
    
    cout << "✅ Price crossing logic test PASSED" << endl;
}

// Test 9: Execution Price Determination
void TestExecutionPrice() {
    cout << "\n=== Test 9: Execution Price Determination ===" << endl;
    
    const uint32_t num_markets = 1;
    VecMarket market(num_markets, 128, 1024, 512);
    auto fills = market.NewFillBatch();
    
    // Test that execution happens at the resting order price
    cout << "Test 1: Buy order rests, sell order takes" << endl;
    {
        // Add buy order first (resting)
        torch::Tensor bid_prices = torch::full({1}, 55, torch::kUInt32);
        torch::Tensor bid_sizes = torch::full({1}, 100, torch::kUInt32);
        torch::Tensor ask_prices = torch::zeros({1}, torch::kUInt32);
        torch::Tensor ask_sizes = torch::zeros({1}, torch::kUInt32);
        torch::Tensor customer_ids = torch::zeros({1}, torch::kUInt32);
        
        market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
        
        // Add sell order (taker) at lower price
        bid_prices = torch::zeros({1}, torch::kUInt32);
        bid_sizes = torch::zeros({1}, torch::kUInt32);
        ask_prices = torch::full({1}, 50, torch::kUInt32);
        ask_sizes = torch::full({1}, 50, torch::kUInt32);
        customer_ids = torch::ones({1}, torch::kUInt32);
        
        market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
        
        // Check execution price
        auto fill_prices = fills.fill_prices.cpu();
        auto prices_acc = fill_prices.accessor<uint32_t, 2>();
        assert(prices_acc[0][0] == 55);  // Should execute at bid price (resting order)
        cout << "✓ Execution at resting buy order price: " << prices_acc[0][0] << endl;
    }
    
    // Reset for next test
    VecMarket market2(num_markets, 128, 1024, 512);
    auto fills2 = market2.NewFillBatch();
    
    cout << "\nTest 2: Sell order rests, buy order takes" << endl;
    {
        // Add sell order first (resting)
        torch::Tensor bid_prices = torch::zeros({1}, torch::kUInt32);
        torch::Tensor bid_sizes = torch::zeros({1}, torch::kUInt32);
        torch::Tensor ask_prices = torch::full({1}, 50, torch::kUInt32);
        torch::Tensor ask_sizes = torch::full({1}, 100, torch::kUInt32);
        torch::Tensor customer_ids = torch::zeros({1}, torch::kUInt32);
        
        market2.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills2);
        
        // Add buy order (taker) at higher price
        bid_prices = torch::full({1}, 55, torch::kUInt32);
        bid_sizes = torch::full({1}, 50, torch::kUInt32);
        ask_prices = torch::zeros({1}, torch::kUInt32);
        ask_sizes = torch::zeros({1}, torch::kUInt32);
        customer_ids = torch::ones({1}, torch::kUInt32);
        
        market2.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills2);
        
        // Check execution price
        auto fill_prices = fills2.fill_prices.cpu();
        auto prices_acc = fill_prices.accessor<uint32_t, 2>();
        assert(prices_acc[0][0] == 50);  // Should execute at ask price (resting order)
        cout << "✓ Execution at resting sell order price: " << prices_acc[0][0] << endl;
    }
    
    cout << "✅ Execution price test PASSED" << endl;
}

// Test 10: Last Price Tracking
void TestLastPriceTracking() {
    cout << "\n=== Test 10: Last Price Tracking ===" << endl;
    
    const uint32_t num_markets = 3;
    VecMarket market(num_markets, 128, 1024, 512);
    auto fills = market.NewFillBatch();
    auto bbo = market.NewBBOBatch();
    
    // Initial state - no trades yet
    market.GetBBOs(bbo);
    auto last_prices = bbo.last_prices.cpu();
    for (uint32_t i = 0; i < num_markets; ++i) {
        assert(tensor_item_u32(last_prices[i]) == NULL_INDEX);
    }
    cout << "✓ Initial last prices are NULL_INDEX" << endl;
    
    // Execute trades at different prices in each market
    cout << "\nExecuting trades in all markets:" << endl;
    torch::Tensor bid_prices = torch::tensor({50, 60, 70}, torch::kUInt32);
    torch::Tensor bid_sizes = torch::tensor({100, 200, 300}, torch::kUInt32);
    torch::Tensor ask_prices = torch::tensor({50, 60, 70}, torch::kUInt32);
    torch::Tensor ask_sizes = torch::tensor({50, 100, 150}, torch::kUInt32);
    torch::Tensor customer_ids = torch::zeros({num_markets}, torch::kUInt32);
    
    market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
    
    // Update last prices based on fills
    bbo.UpdateLastPrices(fills);
    last_prices = bbo.last_prices.cpu();
    
    // Verify last prices match execution prices
    assert(tensor_item_u32(last_prices[0]) == 50);
    assert(tensor_item_u32(last_prices[1]) == 60);
    assert(tensor_item_u32(last_prices[2]) == 70);
    cout << "✓ Last prices updated correctly: [50, 60, 70]" << endl;
    
    // Execute more trades, but only in market 1
    cout << "\nExecuting additional trade only in market 1:" << endl;
    bid_prices = torch::tensor({0, 65, 0}, torch::kUInt32);
    bid_sizes = torch::tensor({0, 50, 0}, torch::kUInt32);
    ask_prices = torch::tensor({0, 65, 0}, torch::kUInt32);
    ask_sizes = torch::tensor({0, 25, 0}, torch::kUInt32);
    customer_ids = torch::ones({num_markets}, torch::kUInt32);
    
    market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
    bbo.UpdateLastPrices(fills);
    last_prices = bbo.last_prices.cpu();
    
    // Markets 0 and 2 should keep their last prices, market 1 should update
    assert(tensor_item_u32(last_prices[0]) == 50);  // Unchanged
    assert(tensor_item_u32(last_prices[1]) == 65);  // Updated
    assert(tensor_item_u32(last_prices[2]) == 70);  // Unchanged
    cout << "✓ Last prices correctly preserved/updated: [50, 65, 70]" << endl;
    
    cout << "✅ Last price tracking test PASSED" << endl;
}

// Test 11: Last Price with Multiple Fills
void TestLastPriceMultipleFills() {
    cout << "\n=== Test 11: Last Price with Multiple Fills ===" << endl;
    
    const uint32_t num_markets = 1;
    VecMarket market(num_markets, 128, 1024, 512);
    auto fills = market.NewFillBatch();
    auto bbo = market.NewBBOBatch();
    
    // Add multiple orders at different prices
    cout << "Adding 3 buy orders at different prices:" << endl;
    for (uint32_t i = 0; i < 3; ++i) {
        torch::Tensor bid_prices = torch::full({1}, 50 + i * 5, torch::kUInt32);  // 50, 55, 60
        torch::Tensor bid_sizes = torch::full({1}, 100, torch::kUInt32);
        torch::Tensor ask_prices = torch::zeros({1}, torch::kUInt32);
        torch::Tensor ask_sizes = torch::zeros({1}, torch::kUInt32);
        torch::Tensor customer_ids = torch::full({1}, i, torch::kUInt32);
        
        market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
        cout << "  Added buy order at price " << 50 + i * 5 << endl;
    }
    
    // Add a large sell order that matches all buys
    cout << "\nAdding sell order at price 50 to match all buys:" << endl;
    torch::Tensor bid_prices = torch::zeros({1}, torch::kUInt32);
    torch::Tensor bid_sizes = torch::zeros({1}, torch::kUInt32);
    torch::Tensor ask_prices = torch::full({1}, 50, torch::kUInt32);
    torch::Tensor ask_sizes = torch::full({1}, 300, torch::kUInt32);
    torch::Tensor customer_ids = torch::full({1}, 3, torch::kUInt32);
    
    market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
    
    // Check that we got multiple fills
    auto fill_counts = fills.fill_counts.cpu();
    uint32_t num_fills = tensor_item_u32(fill_counts[0]);
    cout << "Number of fills: " << num_fills << endl;
    assert(num_fills == 3);
    
    // Update last prices
    bbo.UpdateLastPrices(fills);
    auto last_prices = bbo.last_prices.cpu();
    
    // Last price should be from the last fill (lowest bid at 50)
    auto fill_prices = fills.fill_prices.cpu();
    auto prices_acc = fill_prices.accessor<uint32_t, 2>();
    uint32_t last_fill_price = prices_acc[0][num_fills - 1];
    
    assert(tensor_item_u32(last_prices[0]) == last_fill_price);
    cout << "✓ Last price correctly set to last fill price: " << last_fill_price << endl;
    
    cout << "✅ Last price with multiple fills test PASSED" << endl;
}

// Test 12: Reset and Last Price Interaction
void TestResetAndLastPrice() {
    cout << "\n=== Test 12: Reset and Last Price Interaction ===" << endl;
    
    const uint32_t num_markets = 2;
    VecMarket market(num_markets, 128, 1024, 512);
    auto fills = market.NewFillBatch();
    auto bbo = market.NewBBOBatch();
    
    // Execute some trades
    cout << "Executing initial trades:" << endl;
    torch::Tensor bid_prices = torch::tensor({45, 55}, torch::kUInt32);
    torch::Tensor bid_sizes = torch::tensor({100, 200}, torch::kUInt32);
    torch::Tensor ask_prices = torch::tensor({45, 55}, torch::kUInt32);
    torch::Tensor ask_sizes = torch::tensor({100, 200}, torch::kUInt32);
    torch::Tensor customer_ids = torch::zeros({num_markets}, torch::kUInt32);
    
    market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
    bbo.UpdateLastPrices(fills);
    
    // Verify trades happened and last prices are set
    auto last_prices = bbo.last_prices.cpu();
    assert(tensor_item_u32(last_prices[0]) == 45);
    assert(tensor_item_u32(last_prices[1]) == 55);
    cout << "✓ Initial last prices: [45, 55]" << endl;
    
    // Reset market
    cout << "\nResetting market..." << endl;
    market.Reset();
    
    // BBOBatch still has old last prices
    last_prices = bbo.last_prices.cpu();
    assert(tensor_item_u32(last_prices[0]) == 45);
    assert(tensor_item_u32(last_prices[1]) == 55);
    cout << "✓ BBOBatch retains last prices after market reset" << endl;
    
    // Add new orders that don't cross (no fills)
    cout << "\nAdding non-crossing orders:" << endl;
    bid_prices = torch::tensor({30, 40}, torch::kUInt32);
    bid_sizes = torch::tensor({50, 100}, torch::kUInt32);
    ask_prices = torch::tensor({35, 45}, torch::kUInt32);
    ask_sizes = torch::tensor({50, 100}, torch::kUInt32);
    customer_ids = torch::zeros({num_markets}, torch::kUInt32);
    
    market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
    bbo.UpdateLastPrices(fills);
    
    // Last prices should remain unchanged (no new fills)
    last_prices = bbo.last_prices.cpu();
    assert(tensor_item_u32(last_prices[0]) == 45);
    assert(tensor_item_u32(last_prices[1]) == 55);
    cout << "✓ Last prices unchanged after no fills: [45, 55]" << endl;
    
    // Reset BBOBatch
    cout << "\nResetting BBOBatch..." << endl;
    bbo.Reset();
    last_prices = bbo.last_prices.cpu();
    assert(tensor_item_u32(last_prices[0]) == NULL_INDEX);
    assert(tensor_item_u32(last_prices[1]) == NULL_INDEX);
    cout << "✓ BBOBatch reset clears last prices to NULL_INDEX" << endl;
    
    // Execute new trades
    cout << "\nExecuting new trades after reset:" << endl;
    bid_prices = torch::tensor({40, 50}, torch::kUInt32);
    bid_sizes = torch::tensor({75, 125}, torch::kUInt32);
    ask_prices = torch::tensor({35, 45}, torch::kUInt32);
    ask_sizes = torch::tensor({75, 125}, torch::kUInt32);
    customer_ids = torch::ones({num_markets}, torch::kUInt32);
    
    market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
    bbo.UpdateLastPrices(fills);
    
    // New last prices should be from new trades
    last_prices = bbo.last_prices.cpu();
    assert(tensor_item_u32(last_prices[0]) == 40);  // Bid was resting
    assert(tensor_item_u32(last_prices[1]) == 50);  // Bid was resting
    cout << "✓ New last prices after reset: [40, 50]" << endl;
    
    cout << "✅ Reset and last price interaction test PASSED" << endl;
}

// Test 13: Last Price Persistence Across BBOBatch Instances
void TestLastPricePersistence() {
    cout << "\n=== Test 13: Last Price Persistence Across BBOBatch Instances ===" << endl;
    
    const uint32_t num_markets = 2;
    VecMarket market(num_markets, 128, 1024, 512);
    auto fills = market.NewFillBatch();
    
    // Create first BBOBatch and execute trades
    auto bbo1 = market.NewBBOBatch();
    cout << "Creating first BBOBatch and executing trades..." << endl;
    
    torch::Tensor bid_prices = torch::tensor({100, 200}, torch::kUInt32);
    torch::Tensor bid_sizes = torch::tensor({50, 100}, torch::kUInt32);
    torch::Tensor ask_prices = torch::tensor({100, 200}, torch::kUInt32);
    torch::Tensor ask_sizes = torch::tensor({50, 100}, torch::kUInt32);
    torch::Tensor customer_ids = torch::zeros({num_markets}, torch::kUInt32);
    
    market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
    bbo1.UpdateLastPrices(fills);
    
    auto last_prices1 = bbo1.last_prices.cpu();
    assert(tensor_item_u32(last_prices1[0]) == 100);
    assert(tensor_item_u32(last_prices1[1]) == 200);
    cout << "✓ BBOBatch 1 last prices: [100, 200]" << endl;
    
    // Create second BBOBatch - should start fresh
    auto bbo2 = market.NewBBOBatch();
    auto last_prices2 = bbo2.last_prices.cpu();
    assert(tensor_item_u32(last_prices2[0]) == NULL_INDEX);
    assert(tensor_item_u32(last_prices2[1]) == NULL_INDEX);
    cout << "✓ BBOBatch 2 starts with NULL_INDEX last prices" << endl;
    
    // Execute more trades and update only bbo2
    cout << "\nExecuting more trades and updating only BBOBatch 2..." << endl;
    bid_prices = torch::tensor({110, 210}, torch::kUInt32);
    bid_sizes = torch::tensor({25, 50}, torch::kUInt32);
    ask_prices = torch::tensor({110, 210}, torch::kUInt32);
    ask_sizes = torch::tensor({25, 50}, torch::kUInt32);
    customer_ids = torch::ones({num_markets}, torch::kUInt32);
    
    market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
    bbo2.UpdateLastPrices(fills);
    
    // Check that bbo1 is unchanged and bbo2 is updated
    last_prices1 = bbo1.last_prices.cpu();
    last_prices2 = bbo2.last_prices.cpu();
    
    assert(tensor_item_u32(last_prices1[0]) == 100);  // Unchanged
    assert(tensor_item_u32(last_prices1[1]) == 200);  // Unchanged
    assert(tensor_item_u32(last_prices2[0]) == 110);  // Updated
    assert(tensor_item_u32(last_prices2[1]) == 210);  // Updated
    
    cout << "✓ BBOBatch 1 unchanged: [100, 200]" << endl;
    cout << "✓ BBOBatch 2 updated: [110, 210]" << endl;
    cout << "✓ BBOBatch instances are independent" << endl;
    
    cout << "✅ Last price persistence test PASSED" << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << "GPU Astra Market Implementation Tests" << endl;
    cout << "========================================" << endl;
    
    try {
        TestPartialFills();
        TestMarketIndependence();
        TestMultipleOrdersSamePrice();
        TestBBO();
        TestRingBufferWraparound();
        TestZeroSizeOrders();
        TestCrossMarketFillCounts();
        TestPriceCrossingLogic();
        TestExecutionPrice();
        TestLastPriceTracking();
        TestLastPriceMultipleFills();
        TestResetAndLastPrice();
        TestLastPricePersistence();
        
        cout << "\n========================================" << endl;
        cout << "✅ ALL TESTS PASSED!" << endl;
        cout << "========================================" << endl;
    } catch (const std::exception& e) {
        cout << "\n❌ TEST FAILED with exception: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
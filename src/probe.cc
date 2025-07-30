#include <iostream>
#include <torch/torch.h>
#include "order_matching/market_legacy_wrapper.h"
#include "order_matching/market.h"

// Helper function to print detailed fill information
void PrintFillDetails(const astra::order_matching::FillBatch& fills, uint32_t market_id, const std::string& label) {
    auto fill_counts = fills.fill_counts.cpu();
    auto fill_counts_acc = fill_counts.accessor<uint32_t, 1>();
    uint32_t count = fill_counts_acc[market_id];
    
    std::cout << label << " fills: " << count << std::endl;
    
    if (count > 0) {
        auto fill_prices = fills.fill_prices.cpu();
        auto fill_sizes = fills.fill_sizes.cpu();
        auto fill_customer_ids = fills.fill_customer_ids.cpu();
        auto fill_quoter_ids = fills.fill_quoter_ids.cpu();
        auto fill_is_sell_quote = fills.fill_is_sell_quote.cpu();
        auto fill_tid = fills.fill_tid.cpu();
        auto fill_quote_tid = fills.fill_quote_tid.cpu();
        
        auto prices_acc = fill_prices.accessor<uint32_t, 2>();
        auto sizes_acc = fill_sizes.accessor<uint32_t, 2>();
        auto customer_acc = fill_customer_ids.accessor<uint32_t, 2>();
        auto quoter_acc = fill_quoter_ids.accessor<uint32_t, 2>();
        auto sell_quote_acc = fill_is_sell_quote.accessor<bool, 2>();
        auto tid_acc = fill_tid.accessor<int64_t, 2>();
        auto qtid_acc = fill_quote_tid.accessor<int64_t, 2>();
        
        for (uint32_t i = 0; i < count; ++i) {
            std::cout << "  Fill " << i << ": price=" << prices_acc[market_id][i]
                      << ", size=" << sizes_acc[market_id][i]
                      << ", customer=" << customer_acc[market_id][i]
                      << ", quoter=" << quoter_acc[market_id][i]
                      << ", is_sell_quote=" << sell_quote_acc[market_id][i]
                      << ", tid=" << tid_acc[market_id][i]
                      << ", qtid=" << qtid_acc[market_id][i] << std::endl;
        }
    }
}

int main() {
    std::cout << "=== Investigating Fill Count Divergence ===" << std::endl;
    
    // Test 1: The exact scenario from the main program output
    std::cout << "\nTest 1: Reproducing Market 2039 Scenario" << std::endl;
    std::cout << "=========================================" << std::endl;
    {
        const uint32_t num_markets = 1;
        const uint32_t market_id = 0;
        
        astra::order_matching::VecMarketLegacy legacy(num_markets, 128, 1024, 512, 0, 256);
        astra::order_matching::VecMarket cuda(num_markets, 128, 1024, 512, 0, 256);
        
        auto fills_legacy = legacy.NewFillBatch();
        auto fills_cuda = cuda.NewFillBatch();
        
        // Step 1: Player 0 submits BID at 30, size 2
        std::cout << "\nStep 1: Player 0 submits BID at 30, size 2" << std::endl;
        {
            torch::Tensor bid_prices = torch::full({1}, 30, torch::kUInt32);
            torch::Tensor bid_sizes = torch::full({1}, 2, torch::kUInt32);
            torch::Tensor ask_prices = torch::zeros({1}, torch::kUInt32);
            torch::Tensor ask_sizes = torch::zeros({1}, torch::kUInt32);
            torch::Tensor customer_ids = torch::full({1}, 0, torch::kUInt32);
            
            legacy.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills_legacy);
            cuda.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills_cuda);
        }
        
        // Step 2: Player 1 submits BID at 30, size 1
        std::cout << "\nStep 2: Player 1 submits BID at 30, size 1" << std::endl;
        {
            torch::Tensor bid_prices = torch::full({1}, 30, torch::kUInt32);
            torch::Tensor bid_sizes = torch::full({1}, 1, torch::kUInt32);
            torch::Tensor ask_prices = torch::zeros({1}, torch::kUInt32);
            torch::Tensor ask_sizes = torch::zeros({1}, torch::kUInt32);
            torch::Tensor customer_ids = torch::full({1}, 1, torch::kUInt32);
            
            legacy.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills_legacy);
            cuda.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills_cuda);
        }
        
        std::cout << "\nOrder Book State Before Player 2:" << std::endl;
        std::cout << "Legacy:\n" << legacy.ToString(0) << std::endl;
        std::cout << "CUDA:\n" << cuda.ToString(0) << std::endl;
        
        // Step 3: Player 2 submits BID at 28, size 2; ASK at 29, size 2
        std::cout << "\nStep 3: Player 2 submits BID at 28, size 2; ASK at 29, size 2" << std::endl;
        std::cout << "Expected: ASK at 29 should match against both BUY orders at 30" << std::endl;
        {
            torch::Tensor bid_prices = torch::full({1}, 28, torch::kUInt32);
            torch::Tensor bid_sizes = torch::full({1}, 2, torch::kUInt32);
            torch::Tensor ask_prices = torch::full({1}, 29, torch::kUInt32);
            torch::Tensor ask_sizes = torch::full({1}, 2, torch::kUInt32);
            torch::Tensor customer_ids = torch::full({1}, 2, torch::kUInt32);
            
            legacy.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills_legacy);
            cuda.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills_cuda);
            
            std::cout << "\nFills generated:" << std::endl;
            PrintFillDetails(fills_legacy, market_id, "Legacy");
            PrintFillDetails(fills_cuda, market_id, "CUDA");
        }
        
        std::cout << "\nFinal Order Book State:" << std::endl;
        std::cout << "Legacy:\n" << legacy.ToString(0) << std::endl;
        std::cout << "CUDA:\n" << cuda.ToString(0) << std::endl;
    }
    
    // Test 2: Simpler scenario to isolate the issue
    std::cout << "\n\nTest 2: Simplified Matching Scenario" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "Testing: Single sell order matching multiple buy orders" << std::endl;
    {
        const uint32_t num_markets = 1;
        const uint32_t market_id = 0;
        
        astra::order_matching::VecMarketLegacy legacy2(num_markets, 128, 1024, 512, 0, 256);
        astra::order_matching::VecMarket cuda2(num_markets, 128, 1024, 512, 0, 256);
        
        auto fills_legacy2 = legacy2.NewFillBatch();
        auto fills_cuda2 = cuda2.NewFillBatch();
        
        // Add two buy orders at price 10
        std::cout << "\nAdding BUY order: price=10, size=3 (player 0)" << std::endl;
        {
            torch::Tensor bid_prices = torch::full({1}, 10, torch::kUInt32);
            torch::Tensor bid_sizes = torch::full({1}, 3, torch::kUInt32);
            torch::Tensor ask_prices = torch::zeros({1}, torch::kUInt32);
            torch::Tensor ask_sizes = torch::zeros({1}, torch::kUInt32);
            torch::Tensor customer_ids = torch::zeros({1}, torch::kUInt32);
            
            legacy2.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills_legacy2);
            cuda2.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills_cuda2);
        }
        
        std::cout << "Adding BUY order: price=10, size=2 (player 1)" << std::endl;
        {
            torch::Tensor bid_prices = torch::full({1}, 10, torch::kUInt32);
            torch::Tensor bid_sizes = torch::full({1}, 2, torch::kUInt32);
            torch::Tensor ask_prices = torch::zeros({1}, torch::kUInt32);
            torch::Tensor ask_sizes = torch::zeros({1}, torch::kUInt32);
            torch::Tensor customer_ids = torch::ones({1}, torch::kUInt32);
            
            legacy2.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills_legacy2);
            cuda2.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills_cuda2);
        }
        
        std::cout << "\nOrder books before sell order:" << std::endl;
        std::cout << "Legacy:\n" << legacy2.ToString(0) << std::endl;
        std::cout << "CUDA:\n" << cuda2.ToString(0) << std::endl;
        
        // Add a sell order that should match both buys
        std::cout << "\nAdding SELL order: price=10, size=5 (player 2)" << std::endl;
        {
            torch::Tensor bid_prices = torch::zeros({1}, torch::kUInt32);
            torch::Tensor bid_sizes = torch::zeros({1}, torch::kUInt32);
            torch::Tensor ask_prices = torch::full({1}, 10, torch::kUInt32);
            torch::Tensor ask_sizes = torch::full({1}, 5, torch::kUInt32);
            torch::Tensor customer_ids = torch::full({1}, 2, torch::kUInt32);
            
            legacy2.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills_legacy2);
            cuda2.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills_cuda2);
            
            std::cout << "\nFills generated:" << std::endl;
            PrintFillDetails(fills_legacy2, market_id, "Legacy");
            PrintFillDetails(fills_cuda2, market_id, "CUDA");
        }
        
        std::cout << "\nFinal order books:" << std::endl;
        std::cout << "Legacy:\n" << legacy2.ToString(0) << std::endl;
        std::cout << "CUDA:\n" << cuda2.ToString(0) << std::endl;
    }
    
    // Test 3: Check if issue is related to partial fills
    std::cout << "\n\nTest 3: Partial Fill Scenario" << std::endl;
    std::cout << "==============================" << std::endl;
    {
        const uint32_t num_markets = 1;
        const uint32_t market_id = 0;
        
        astra::order_matching::VecMarketLegacy legacy3(num_markets, 128, 1024, 512, 0, 256);
        astra::order_matching::VecMarket cuda3(num_markets, 128, 1024, 512, 0, 256);
        
        auto fills_legacy3 = legacy3.NewFillBatch();
        auto fills_cuda3 = cuda3.NewFillBatch();
        
        // Add buy order
        std::cout << "\nAdding BUY order: price=20, size=10 (player 0)" << std::endl;
        {
            torch::Tensor bid_prices = torch::full({1}, 20, torch::kUInt32);
            torch::Tensor bid_sizes = torch::full({1}, 10, torch::kUInt32);
            torch::Tensor ask_prices = torch::zeros({1}, torch::kUInt32);
            torch::Tensor ask_sizes = torch::zeros({1}, torch::kUInt32);
            torch::Tensor customer_ids = torch::zeros({1}, torch::kUInt32);
            
            legacy3.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills_legacy3);
            cuda3.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills_cuda3);
        }
        
        // Add smaller sell order that partially fills
        std::cout << "Adding SELL order: price=20, size=3 (player 1)" << std::endl;
        {
            torch::Tensor bid_prices = torch::zeros({1}, torch::kUInt32);
            torch::Tensor bid_sizes = torch::zeros({1}, torch::kUInt32);
            torch::Tensor ask_prices = torch::full({1}, 20, torch::kUInt32);
            torch::Tensor ask_sizes = torch::full({1}, 3, torch::kUInt32);
            torch::Tensor customer_ids = torch::ones({1}, torch::kUInt32);
            
            legacy3.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills_legacy3);
            cuda3.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills_cuda3);
            
            std::cout << "\nFills generated:" << std::endl;
            PrintFillDetails(fills_legacy3, market_id, "Legacy");
            PrintFillDetails(fills_cuda3, market_id, "CUDA");
        }
        
        // Add another sell order to check remaining buy
        std::cout << "\nAdding another SELL order: price=20, size=5 (player 2)" << std::endl;
        {
            torch::Tensor bid_prices = torch::zeros({1}, torch::kUInt32);
            torch::Tensor bid_sizes = torch::zeros({1}, torch::kUInt32);
            torch::Tensor ask_prices = torch::full({1}, 20, torch::kUInt32);
            torch::Tensor ask_sizes = torch::full({1}, 5, torch::kUInt32);
            torch::Tensor customer_ids = torch::full({1}, 2, torch::kUInt32);
            
            legacy3.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills_legacy3);
            cuda3.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills_cuda3);
            
            std::cout << "\nFills generated:" << std::endl;
            PrintFillDetails(fills_legacy3, market_id, "Legacy");
            PrintFillDetails(fills_cuda3, market_id, "CUDA");
        }
        
        std::cout << "\nFinal order books:" << std::endl;
        std::cout << "Legacy:\n" << legacy3.ToString(0) << std::endl;
        std::cout << "CUDA:\n" << cuda3.ToString(0) << std::endl;
    }
    
    return 0;
}
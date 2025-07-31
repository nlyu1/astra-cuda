#include <iostream>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "market.h"
#include "astra_utils.h"

using namespace astra::order_matching;

void check_cuda_error(const char* label) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << label << " - CUDA error: " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

int main() {
    std::cout << "=== CUDA Market Segfault Debug Probe (Matching benchmark_env) ===" << std::endl;
    
    // Test configurations that work and fail
    struct TestConfig {
        int num_markets;
        int threads_per_block;
        const char* description;
    };
    
    std::vector<TestConfig> configs = {
        {16384, 64, "256 blocks × 64 threads (WORKS)"},
        {32768, 128, "256 blocks × 128 threads (WORKS)"},
        {65536, 256, "256 blocks × 256 threads (WORKS)"},
        {65536, 64, "1024 blocks × 64 threads (FAILS)"},
        {131072, 128, "1024 blocks × 128 threads (FAILS)"}
    };
    
    int device_id = 0;
    cudaSetDevice(device_id);
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::cout << "\nDevice: " << prop.name << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max grid dimensions: [" << prop.maxGridSize[0] << ", " 
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Total global memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
    
    // Test each configuration
    for (const auto& config : configs) {
        std::cout << "\n=== Testing " << config.description << " ===" << std::endl;
        std::cout << "Creating market with " << config.num_markets << " markets..." << std::endl;
        
        try {
            // Match exact parameters from benchmark_env
            const int num_players = 5;
            const int steps_per_player = 8;
            const int max_contracts_per_trade = 1;
            const int max_contract_value = 10;
            
            // These are the actual parameters used in high_low_trading.cc
            const int max_price_levels = max_contract_value;  // 10
            const int max_orders = num_players * steps_per_player * 2;  // 80
            const int max_fills = std::min(max_orders, max_contracts_per_trade * 2);  // min(80, 2) = 2
            const int num_customers = num_players;  // 5
            
            std::cout << "Market parameters from benchmark_env:" << std::endl;
            std::cout << "  max_price_levels: " << max_price_levels << std::endl;
            std::cout << "  max_orders_per_market: " << max_orders << std::endl;
            std::cout << "  max_fills_per_market: " << max_fills << std::endl;
            std::cout << "  num_customers: " << num_customers << std::endl;
            
            // Calculate memory requirements
            size_t mem_per_market = 0;
            mem_per_market += 4 * max_price_levels * sizeof(int32_t); // bid/ask heads/tails
            mem_per_market += 6 * max_orders * sizeof(int32_t); // order data
            mem_per_market += max_orders * sizeof(bool); // order_is_bid
            mem_per_market += 2 * sizeof(int32_t); // counters
            mem_per_market += num_customers * 2 * sizeof(int32_t); // customer portfolios
            
            size_t total_memory = config.num_markets * mem_per_market;
            std::cout << "Estimated memory usage: " << (total_memory / (1024*1024)) << " MB" << std::endl;
            
            // Debug index calculations
            int32_t max_market_id = config.num_markets - 1;
            int32_t max_customer_id = num_customers - 1;
            int32_t max_portfolio_idx = (max_market_id * num_customers + max_customer_id) * 2 + 1;
            std::cout << "Max portfolio index calculation:" << std::endl;
            std::cout << "  max_market_id: " << max_market_id << std::endl;
            std::cout << "  max_customer_id: " << max_customer_id << std::endl;
            std::cout << "  max_portfolio_idx: " << max_portfolio_idx << std::endl;
            std::cout << "  Expected tensor size: " << (config.num_markets * num_customers * 2) << std::endl;
            
            // Create market
            VecMarket market(config.num_markets, max_price_levels, max_orders, max_fills, 
                           num_customers, device_id, config.threads_per_block);
            
            std::cout << "Market created successfully!" << std::endl;
            
            // Create test tensors with realistic values from benchmark_env
            auto device = torch::Device(torch::kCUDA, device_id);
            auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
            
            // Generate similar trading patterns as benchmark_env
            auto bid_prices = torch::randint(1, max_contract_value + 1, {config.num_markets}, options_i32);
            auto ask_prices = torch::randint(1, max_contract_value + 1, {config.num_markets}, options_i32);
            auto bid_sizes = torch::randint(0, 11, {config.num_markets}, options_i32);  // 0-10 like benchmark
            auto ask_sizes = torch::randint(0, 11, {config.num_markets}, options_i32);
            
            // Customer IDs range from 0 to num_players-1
            auto customer_ids = torch::randint(0, num_customers, {config.num_markets}, options_i32);
            
            // Create fill batch
            auto fills = market.NewFillBatch();
            
            std::cout << "Running AddTwoSidedQuotes with realistic trading data..." << std::endl;
            
            // Simulate multiple rounds like in benchmark_env
            for (int round = 0; round < steps_per_player; ++round) {
                for (int player = 0; player < num_players; ++player) {
                    // Each player submits orders
                    customer_ids.fill_(player);
                    
                    // This is where the segfault happens
                    market.AddTwoSidedQuotes(bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fills);
                    
                    // Sync to ensure kernel completion
                    cudaDeviceSynchronize();
                    check_cuda_error("After AddTwoSidedQuotes");
                    
                    // Check fill counts
                    auto fill_counts_cpu = fills.fill_counts.cpu();
                    auto max_fills_generated = fill_counts_cpu.max().item<int32_t>();
                    if (max_fills_generated > max_fills) {
                        std::cerr << "ERROR: Generated " << max_fills_generated 
                                  << " fills, but max_fills_per_market is only " << max_fills << std::endl;
                    }
                }
                std::cout << "  Round " << round << " completed" << std::endl;
            }
            
            std::cout << "✓ All rounds completed successfully!" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "✗ Failed with exception: " << e.what() << std::endl;
            
            // Get more detailed CUDA error info
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                std::cout << "Last CUDA error: " << cudaGetErrorString(error) << std::endl;
            }
        }
    }
    
    return 0;
}
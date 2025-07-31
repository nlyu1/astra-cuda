// Speed benchmark for vectorized GPU market
#include <iostream>
#include <memory>
#include <string>
#include <chrono>
#include <random>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>
#include <unistd.h>  // For getopt_long
#include <getopt.h>  // For getopt_long structure

#include "market.h"

#include <torch/torch.h>

using namespace astra::order_matching;
using namespace astra;

struct BenchmarkResult {
    int32_t num_markets_per_block;  // threads per block
    int32_t num_blocks;
    int32_t total_markets;
    double latency_ms;
    double max_latency_ms;
    double operations_per_second;  // operations (batches) per second
    double markets_per_second;     // total markets processed per second
    double speedup;                // Speedup vs single-market baseline
    double efficiency;             // Parallel efficiency (speedup / total_markets)
    double total_time_seconds;
    int total_steps;
};

// Global variable to store device ID from command line
int test_device_id = 0;

BenchmarkResult test_market_throughput(int32_t num_markets_per_block, int32_t num_blocks) {
    // Calculate total markets
    int32_t total_markets = num_markets_per_block * num_blocks;
    
    // Market parameters
    const int32_t max_price_levels = 128;
    const int32_t max_active_orders_per_market = 1024;
    const int32_t max_active_fills_per_market = 1024;
    const int32_t num_customers = 16;  // Number of customers per market
    
    // Create VecMarket instance - device_id passed from main()
    auto market = std::make_unique<VecMarket>(
        total_markets,
        max_price_levels,
        max_active_orders_per_market,
        max_active_fills_per_market,
        num_customers,
        test_device_id,
        num_markets_per_block
    );
    
    // Create FillBatch and BBOBatch for reuse
    FillBatch fills = market->NewFillBatch();
    BBOBatch bbos = market->NewBBOBatch();
    
    // Prepare tensors on GPU
    auto device = torch::Device(torch::kCUDA, test_device_id);
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
    
    // Benchmark parameters
    const int warmup_steps = 100;
    const int benchmark_steps = 1000;
    
    // Create tensors directly on GPU for reuse
    torch::Tensor bid_px = torch::zeros({total_markets}, options);
    torch::Tensor bid_sz = torch::zeros({total_markets}, options);
    torch::Tensor ask_px = torch::zeros({total_markets}, options);
    torch::Tensor ask_sz = torch::zeros({total_markets}, options);
    torch::Tensor customer_ids = torch::zeros({total_markets}, options);
    
    // Initialize with random values using PyTorch's random functions
    torch::manual_seed(42);
    
    // Warmup phase
    for (int step = 0; step < warmup_steps; ++step) {
        // Generate random values directly on GPU
        // Bid prices: 40-49, Ask prices: 51-60
        bid_px = torch::randint(40, 50, {total_markets}, options);
        ask_px = torch::randint(51, 61, {total_markets}, options);
        
        // Sizes: 1-100
        bid_sz = torch::randint(1, 101, {total_markets}, options);
        ask_sz = torch::randint(1, 101, {total_markets}, options);
        
        // Customer IDs: 0-(num_customers-1)
        customer_ids = torch::randint(0, num_customers, {total_markets}, options);
        
        market->AddTwoSidedQuotes(bid_px, bid_sz, ask_px, ask_sz, customer_ids, fills);
    }
    
    // Ensure GPU operations are complete before starting benchmark
    cudaDeviceSynchronize();
    
    // Benchmark phase
    auto start_time = std::chrono::high_resolution_clock::now();
    double max_step_time_ms = 0.0;
    
    for (int step = 0; step < benchmark_steps; ++step) {
        // Generate random values directly on GPU
        // Bid prices: 40-49, Ask prices: 51-60
        bid_px = torch::randint(40, 50, {total_markets}, options);
        ask_px = torch::randint(51, 61, {total_markets}, options);
        
        // Sizes: 1-100
        bid_sz = torch::randint(1, 101, {total_markets}, options);
        ask_sz = torch::randint(1, 101, {total_markets}, options);
        
        // Customer IDs: 0-(num_customers-1)
        customer_ids = torch::randint(0, num_customers, {total_markets}, options);
        
        auto step_start = std::chrono::high_resolution_clock::now();
        
        // Add quotes and match orders
        market->AddTwoSidedQuotes(bid_px, bid_sz, ask_px, ask_sz, customer_ids, fills);
        
        // Ensure kernel completion for accurate timing
        cudaDeviceSynchronize();
        
        auto step_end = std::chrono::high_resolution_clock::now();
        
        auto step_duration = std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start);
        double step_time_ms = step_duration.count() / 1000.0;
        max_step_time_ms = std::max(max_step_time_ms, step_time_ms);
        
        // Optionally get BBOs and customer portfolios to ensure full pipeline
        if (step % 100 == 0) {
            market->GetBBOs(bbos);
            // Get customer portfolios to test the new functionality
            const auto& portfolios = market->GetCustomerPortfolios();
            // Use DoNotOptimize to prevent the optimizer from removing this access
            DoNotOptimize(portfolios);
            cudaDeviceSynchronize();
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Calculate metrics
    double total_time_seconds = duration.count() / 1000000.0;
    double latency_ms = (total_time_seconds * 1000.0) / benchmark_steps;
    double operations_per_second = benchmark_steps / total_time_seconds;
    double markets_per_second = (static_cast<double>(benchmark_steps) * static_cast<double>(total_markets)) / total_time_seconds;
    
    return BenchmarkResult{
        num_markets_per_block,
        num_blocks,
        total_markets,
        latency_ms,
        max_step_time_ms,
        operations_per_second,
        markets_per_second,
        0.0,  // speedup will be calculated later
        0.0,  // efficiency will be calculated later
        total_time_seconds,
        benchmark_steps
    };
}

void print_benchmark_results(const std::vector<BenchmarkResult>& results) {
    std::cout << std::fixed << std::setprecision(3);
    
    // Print header
    std::cout << std::setw(15) << "Markets/Block" 
              << std::setw(12) << "Blocks"
              << std::setw(15) << "Total Markets"
              << std::setw(12) << "Lat(ms)"
              << std::setw(18) << "Markets/sec"
              << std::setw(12) << "Speedup"
              << std::setw(12) << "Efficiency"
              << std::endl;
    
    std::cout << std::string(106, '-') << std::endl;
    
    // Print results
    for (const auto& result : results) {
        std::cout << std::setw(15) << result.num_markets_per_block
                  << std::setw(12) << result.num_blocks
                  << std::setw(15) << result.total_markets
                  << std::setw(12) << std::fixed << std::setprecision(3) << result.latency_ms
                  << std::setw(18) << std::fixed << std::setprecision(0) << result.markets_per_second
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.speedup << "x"
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.efficiency * 100 << "%"
                  << std::endl;
    }
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -i, --gpu_id <id>    Specify GPU device ID (default: 0)" << std::endl;
    std::cout << "  -h, --help           Show this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    static struct option long_options[] = {
        {"gpu_id", required_argument, 0, 'i'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "i:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'i':
                test_device_id = std::stoi(optarg);
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    std::cout << "=== GPU Market Order Matching Benchmark ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "- Max price levels: 128" << std::endl;
    std::cout << "- Max orders per market: 1024" << std::endl;
    std::cout << "- Max fills per market: 1024" << std::endl;
    std::cout << "- Number of customers per market: 16" << std::endl;
    std::cout << "- Random bid/ask orders with crossing prices" << std::endl;
    std::cout << std::endl;
    
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    // Validate GPU ID
    if (test_device_id < 0 || test_device_id >= device_count) {
        std::cerr << "Error: Invalid GPU ID " << test_device_id 
                  << ". Available GPUs: 0-" << (device_count - 1) << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, test_device_id);
    std::cout << "Using GPU " << test_device_id << ": " << prop.name << std::endl;
    std::cout << "- Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "- Max grid dimensions: " << prop.maxGridSize[0] << " x " 
              << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;
    std::cout << std::endl;
    
    // Test configurations
    std::vector<std::pair<int32_t, int32_t>> configs;
    
    // Varying threads per block (num_markets_per_block)
    std::vector<int32_t> threads_per_block = {64, 128, 256};
    std::vector<int32_t> num_blocks = {1, 64, 128, 256, 512, 1024};
    
    // Generate all combinations that don't exceed MAX_MARKETS
    for (auto tpb : threads_per_block) {
        for (auto nb : num_blocks) {
            configs.push_back({tpb, nb});
        }
    }
    
    std::vector<BenchmarkResult> results;
    
    std::cout << "Running benchmarks..." << std::endl;
    std::cout << std::endl;
    
    for (const auto& [markets_per_block, blocks] : configs) {
        std::cout << "Testing " << markets_per_block << " markets/block × " 
                  << blocks << " blocks (total: " << markets_per_block * blocks << " markets)..." 
                  << std::flush;
        
        try {
            auto result = test_market_throughput(markets_per_block, blocks);
            results.push_back(result);
            
            std::cout << " Done (Markets/sec: " << std::fixed << std::setprecision(0) 
                      << result.markets_per_second << ", Latency: " << std::setprecision(2) 
                      << result.latency_ms << "ms)" << std::endl;
        } catch (const std::exception& e) {
            std::cout << " Failed: " << e.what() << std::endl;
        }
    }
    
    // Calculate speedup and efficiency compared to smallest configuration
    if (!results.empty()) {
        // Use the smallest configuration as baseline (64 markets)
        double baseline_markets_per_second = 0;
        for (const auto& result : results) {
            if (result.total_markets == 64) {  // Smallest configuration
                baseline_markets_per_second = result.markets_per_second;
                break;
            }
        }
        
        // Calculate speedup and efficiency
        for (auto& result : results) {
            result.speedup = result.markets_per_second / baseline_markets_per_second;
            result.efficiency = result.speedup / (result.total_markets / 64.0);  // Normalized by market count increase
        }
    }
    
    std::cout << std::endl;
    std::cout << "=== BENCHMARK RESULTS ===" << std::endl;
    std::cout << std::endl;
    
    print_benchmark_results(results);
    
    std::cout << std::endl;
    std::cout << "=== ANALYSIS ===" << std::endl;
    
    // Find best configurations
    if (!results.empty()) {
        auto best_throughput = std::max_element(results.begin(), results.end(),
            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                return a.markets_per_second < b.markets_per_second;
            });
        
        auto best_latency = std::min_element(results.begin(), results.end(),
            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                return a.latency_ms < b.latency_ms;
            });
        
        std::cout << "Best Throughput: " << std::fixed << std::setprecision(0) 
                  << best_throughput->markets_per_second << " markets/sec (" 
                  << best_throughput->num_markets_per_block << " markets/block × " 
                  << best_throughput->num_blocks << " blocks = "
                  << best_throughput->total_markets << " total markets, "
                  << std::setprecision(1) << best_throughput->speedup << "x speedup)"
                  << std::endl;
                  
        std::cout << "Best Latency: " << std::fixed << std::setprecision(3) 
                  << best_latency->latency_ms << "ms (" 
                  << best_latency->num_markets_per_block << " markets/block × " 
                  << best_latency->num_blocks << " blocks)" << std::endl;
                  
        // Find baseline for speedup analysis
        double baseline_markets_per_second = 0;
        for (const auto& result : results) {
            if (result.total_markets == 64) {
                baseline_markets_per_second = result.markets_per_second;
                break;
            }
        }
                  
        std::cout << std::endl;
        std::cout << "Speedup Analysis:" << std::endl;
        std::cout << "- Baseline (64 markets): " << std::fixed << std::setprecision(0) 
                  << baseline_markets_per_second << " markets/sec" << std::endl;
        std::cout << "- Peak speedup: " << std::fixed << std::setprecision(1) 
                  << best_throughput->speedup << "x with " 
                  << best_throughput->total_markets << " markets" << std::endl;
        std::cout << "- Peak efficiency: " << std::fixed << std::setprecision(1) 
                  << best_throughput->efficiency * 100 << "% parallel efficiency" << std::endl;
        
        // Find best configuration for different block counts
        std::cout << std::endl;
        std::cout << "Best configurations by block count:" << std::endl;
        for (auto nb : num_blocks) {
            auto best_for_blocks = std::max_element(results.begin(), results.end(),
                [nb](const BenchmarkResult& a, const BenchmarkResult& b) {
                    if (a.num_blocks != nb) return true;
                    if (b.num_blocks != nb) return false;
                    return a.markets_per_second < b.markets_per_second;
                });
            
            if (best_for_blocks != results.end() && best_for_blocks->num_blocks == nb) {
                std::cout << "- " << nb << " blocks: " 
                          << best_for_blocks->num_markets_per_block << " markets/block "
                          << "(Markets/sec: " << std::fixed << std::setprecision(0) 
                          << best_for_blocks->markets_per_second << ")" << std::endl;
            }
        }
    }
    
    return 0;
}
#include <iostream>
#include <torch/torch.h>
#include <random>
#include <vector>
#include <iomanip>
#include <sstream>
#include <chrono>
#include "order_matching/market.h"

using namespace astra::order_matching;

torch::Tensor create_tensor(uint32_t value, torch::Device device) {
    return torch::tensor({static_cast<int>(value)}, torch::TensorOptions().dtype(torch::kUInt32).device(device));
}

int main() {
    // Replicate test_partial_fills scenario
    std::cout << "Testing partial fills scenario...\n";
    
    VecMarket market(1, 128, 1024, 512);
    FillBatch fills = market.NewFillBatch();
    
    // Add a large buy order (100 @ 50)
    market.AddTwoSidedQuotes(
        create_tensor(50, torch::kCUDA),  // bid_px
        create_tensor(100, torch::kCUDA), // bid_sz
        create_tensor(0, torch::kCUDA),   // ask_px
        create_tensor(0, torch::kCUDA),   // ask_sz
        create_tensor(0, torch::kCUDA),   // customer_ids
        fills
    );
    
    std::cout << "After first order - Market state:\n" << market.ToString(0) << "\n";
    
    // Add a smaller sell order that partially fills (30 @ 50)
    market.AddTwoSidedQuotes(
        create_tensor(0, torch::kCUDA),   // bid_px
        create_tensor(0, torch::kCUDA),   // bid_sz
        create_tensor(50, torch::kCUDA),  // ask_px
        create_tensor(30, torch::kCUDA),  // ask_sz
        create_tensor(1, torch::kCUDA),   // customer_ids
        fills
    );
    
    // Check fills after second order
    auto fill_counts = fills.fill_counts.cpu();
    auto fill_sizes = fills.fill_sizes.cpu();
    std::cout << "\nAfter second order:\n";
    std::cout << "Fill count: " << fill_counts[0].item().toInt() << "\n";
    std::cout << "Fill size: " << fill_sizes[0][0].item().toInt() << "\n";
    std::cout << "Market state:\n" << market.ToString(0) << "\n";
    
    // Add another sell order to check remaining buy (80 @ 50)
    market.AddTwoSidedQuotes(
        create_tensor(0, torch::kCUDA),   // bid_px
        create_tensor(0, torch::kCUDA),   // bid_sz
        create_tensor(50, torch::kCUDA),  // ask_px
        create_tensor(80, torch::kCUDA),  // ask_sz
        create_tensor(2, torch::kCUDA),   // customer_ids
        fills
    );
    
    // Check fills after third order
    fill_counts = fills.fill_counts.cpu();
    fill_sizes = fills.fill_sizes.cpu();
    std::cout << "\nAfter third order:\n";
    std::cout << "Fill count: " << fill_counts[0].item().toInt() << "\n";
    std::cout << "Fill size at [0][0]: " << fill_sizes[0][0].item().toInt() << " (expected: 70)\n";
    std::cout << "Market state:\n" << market.ToString(0) << "\n";
    
    // Let's also print all fill data to see what's happening
    std::cout << "\nAll fill sizes in buffer:\n";
    auto fill_sizes_acc = fill_sizes.accessor<uint32_t, 2>();
    for (int i = 0; i < 5; i++) {
        std::cout << "  [0][" << i << "] = " << fill_sizes_acc[0][i] << "\n";
    }
    
    // Let's check if the tensor is still on GPU and what happens when we access it directly
    std::cout << "\nDirect tensor access (before cpu copy):\n";
    std::cout << "fills.fill_sizes device: " << fills.fill_sizes.device() << "\n";
    std::cout << "fills.fill_sizes[0][0] = " << fills.fill_sizes[0][0].item().toInt() << "\n";
    
    // Let's also check tensor addresses to see if they're the same
    std::cout << "\nTensor memory addresses:\n";
    std::cout << "fills.fill_sizes.data_ptr() = " << fills.fill_sizes.data_ptr() << "\n";
    std::cout << "fill_sizes.data_ptr() = " << fill_sizes.data_ptr() << "\n";
    
    if (fill_sizes[0][0].item().toInt() == 70) {
        std::cout << "\n✓ TEST PASSED: Fill size is correct (70)\n";
    } else {
        std::cout << "\n✗ TEST FAILED: Fill size is " << fill_sizes[0][0].item().toInt() 
                  << " instead of 70\n";
    }
    
    return 0;
}
#include <iostream>
#include <memory>
#include <random>
#include <numeric>
#include <algorithm>
#include <torch/torch.h>
#include "core.h"
#include "registration.h"
#include "games/high_low_trading/high_low_trading.h"

using namespace astra;
using namespace astra::high_low_trading;

int main() {
    // Register all available games
    RegisterGames();
    
    // Verify registration
    std::cout << "Number of registered games: " << GameRegistrar::RegisteredNames().size() << std::endl;
    if (!GameRegistrar::RegisteredNames().empty()) {
        std::cout << "First registered game: " << GameRegistrar::RegisteredNames()[0] << std::endl;
    }
    
    // Set up game parameters
    int steps_per_player = 3;
    int max_contracts_per_trade = 3;
    int customer_max_size = 5;
    int max_contract_value = 30;
    int num_players = 5;
    int num_markets = 1;
    int threads_per_block = 128;
    int device_id = 0;
    
    // Create game parameters
    GameParameters params = {
        {"steps_per_player", GameParameter(steps_per_player)},
        {"max_contracts_per_trade", GameParameter(max_contracts_per_trade)},
        {"customer_max_size", GameParameter(customer_max_size)},
        {"max_contract_value", GameParameter(max_contract_value)},
        {"players", GameParameter(num_players)},
        {"num_markets", GameParameter(num_markets)},
        {"threads_per_block", GameParameter(threads_per_block)},
        {"device_id", GameParameter(device_id)}
    };
    
    std::cout << "\nCreating HighLowTrading game with parameters:" << std::endl;
    std::cout << "- Steps per player: " << steps_per_player << std::endl;
    std::cout << "- Max contracts per trade: " << max_contracts_per_trade << std::endl;
    std::cout << "- Customer max size: " << customer_max_size << std::endl;
    std::cout << "- Max contract value: " << max_contract_value << std::endl;
    std::cout << "- Players: " << num_players << std::endl;
    std::cout << "- Number of markets: " << num_markets << std::endl;
    std::cout << "- Threads per block: " << threads_per_block << std::endl;
    std::cout << "- Device ID: " << device_id << std::endl;
    
    // Create game using factory method
    auto game = Factory(params);
    
    // Create initial state
    auto state = game->NewInitialState();
    
    std::cout << "\nGame created successfully!" << std::endl;
    std::cout << "Initial state created." << std::endl;
    std::cout << "Current player: " << state->CurrentPlayer() << std::endl;
    std::cout << "Is terminal: " << (state->IsTerminal() ? "Yes" : "No") << std::endl;
    std::cout << "Move number: " << state->MoveNumber() << std::endl;
    
    // Cast to HighLowTradingState to access game-specific methods
    auto* trading_state = static_cast<HighLowTradingState*>(state.get());
    std::cout << "\nInitial state string (index 0):" << std::endl;
    std::cout << trading_state->ToString(0) << std::endl;
    
    // Fix random seed for reproducibility
    std::mt19937 rng(42);
    
    // Configure tensor options for the specified device
    auto device = torch::Device(torch::kCUDA, device_id);
    auto options = torch::TensorOptions().device(device);
    
    std::cout << "\n========== Applying Chance Actions ==========\n" << std::endl;
    
    // Action 1: Set two candidate contract values [1, max_contract_value]
    std::uniform_int_distribution<int> contract_dist(1, max_contract_value);
    int value1 = contract_dist(rng);
    int value2 = contract_dist(rng);
    torch::Tensor contract_values = torch::tensor({{value1, value2}}, options.dtype(torch::kInt32));
    
    std::cout << "Move 0 - Contract values: [" << value1 << ", " << value2 << "]" << std::endl;
    state->ApplyAction(contract_values);
    std::cout << "Current player after move 0: " << state->CurrentPlayer() << std::endl;
    
    // Action 2: Determine high/low settlement (0 = low, 1 = high)
    std::uniform_int_distribution<int> high_low_dist(0, 1);
    int high_low = high_low_dist(rng);
    torch::Tensor high_low_choice = torch::tensor({high_low}, options.dtype(torch::kInt32));
    
    std::cout << "\nMove 1 - High/Low choice: " << (high_low ? "High" : "Low") << std::endl;
    state->ApplyAction(high_low_choice);
    std::cout << "Current player after move 1: " << state->CurrentPlayer() << std::endl;
    
    // Print settlement value
    auto settlement_value = trading_state->GetContractValue();
    std::cout << "Settlement value: " << settlement_value.item<int>() << std::endl;
    
    // Action 3: Player role permutation
    std::vector<int> permutation(num_players);
    std::iota(permutation.begin(), permutation.end(), 0); // Fill with 0, 1, 2, ..., num_players-1
    std::shuffle(permutation.begin(), permutation.end(), rng);
    torch::Tensor perm_tensor = torch::tensor({permutation}, options.dtype(torch::kInt32));
    
    std::cout << "\nMove 2 - Player permutation: [";
    for (int i = 0; i < num_players; ++i) {
        std::cout << permutation[i];
        if (i < num_players - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Explain role assignments
    std::cout << "Role assignments:" << std::endl;
    for (int player = 0; player < num_players; ++player) {
        int role = permutation[player];
        std::cout << "  Player " << player << " -> Role " << role << " (";
        if (role == 0 || role == 1) {
            std::cout << "ValueCheater - knows value " << (role == 0 ? value1 : value2);
        } else if (role == 2) {
            std::cout << "HighLowCheater - knows settlement is " << (high_low ? "High" : "Low");
        } else {
            std::cout << "Customer";
        }
        std::cout << ")" << std::endl;
    }
    
    state->ApplyAction(perm_tensor);
    std::cout << "Current player after move 2: " << state->CurrentPlayer() << std::endl;
    
    // Action 4: Customer target positions (for roles 3, 4, ..., num_players-1)
    int num_customers = num_players - 3;
    std::vector<int> target_positions(num_customers);
    std::uniform_int_distribution<int> target_dist(1, customer_max_size);
    for (int i = 0; i < num_customers; ++i) {
        target_positions[i] = target_dist(rng);
    }
    torch::Tensor targets_tensor = torch::tensor({target_positions}, options.dtype(torch::kInt32));
    
    std::cout << "\nMove 3 - Customer target positions: [";
    for (int i = 0; i < num_customers; ++i) {
        std::cout << target_positions[i];
        if (i < num_customers - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Show which players have target positions
    std::cout << "Target position assignments:" << std::endl;
    for (int role = 3; role < num_players; ++role) {
        // Find which player has this customer role
        auto it = std::find(permutation.begin(), permutation.end(), role);
        int player = std::distance(permutation.begin(), it);
        std::cout << "  Player " << player << " (Customer role " << role << ") -> Target: " 
                  << target_positions[role - 3] << " contracts" << std::endl;
    }
    
    state->ApplyAction(targets_tensor);
    std::cout << "\nCurrent player after move 3: " << state->CurrentPlayer() << std::endl;
    std::cout << "Move number: " << state->MoveNumber() << std::endl;
    std::cout << "Max chance nodes in history: " << game->MaxChanceNodesInHistory() << std::endl;
    std::cout << "\nAll chance actions completed. Game is now in trading phase." << std::endl;
    
    return 0;
}
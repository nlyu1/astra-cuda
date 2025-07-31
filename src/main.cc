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
  
  // Game parameters for 2-round, 4-player game
  GameParameters params;
  params["steps_per_player"] = GameParameter(2);
  params["max_contracts_per_trade"] = GameParameter(5);
  params["customer_max_size"] = GameParameter(5);
  params["max_contract_value"] = GameParameter(30);
  params["players"] = GameParameter(4);
  params["num_markets"] = GameParameter(128);  // We'll use environment 32
  params["threads_per_block"] = GameParameter(128);
  params["device_id"] = GameParameter(0);
  
  // Create the game and initial state
  auto game = Factory(params);
  auto state = game->NewInitialState();
  
  const int32_t ENV_INDEX = 32;  // Focus on environment 32
  const int num_envs = 128;
  
  // Helper to create 2D tensor for multi-column actions
  auto make_2d_tensor = [&](const std::vector<std::vector<int>>& values) {
    int rows = values.size();
    int cols = values[0].size();
    torch::Tensor tensor;
    tensor = torch::zeros({rows, cols}, torch::kInt32);
    auto accessor = tensor.accessor<int32_t, 2>();
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          accessor[i][j] = values[i][j];
        }
      }
    return tensor.to(torch::Device(torch::kCUDA, 0));
  };
  
  // Helper to create 1D tensor for single-column actions
  auto make_1d_tensor = [&](const std::vector<int>& values) {
    torch::Tensor tensor = torch::zeros({static_cast<int>(values.size())}, torch::kInt32);
    auto accessor = tensor.accessor<int32_t, 1>();
    for (int i = 0; i < values.size(); ++i) {
      accessor[i] = values[i];
    }
    return tensor.to(torch::Device(torch::kCUDA, 0));
  };
  
  // Chance phase 1: Set candidate contract values
  std::cout << "\n=== CHANCE PHASE 1: Setting candidate contract values ===\n";
  std::vector<std::vector<int>> contract_values(num_envs, std::vector<int>(2));
  // Set specific values for environment 32
  contract_values[ENV_INDEX][0] = 10;  // First candidate value
  contract_values[ENV_INDEX][1] = 25;  // Second candidate value
  // Random values for other environments
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> value_dist(1, 30);
  for (int i = 0; i < num_envs; ++i) {
    if (i != ENV_INDEX) {
      contract_values[i][0] = value_dist(rng);
      contract_values[i][1] = value_dist(rng);
    }
  }
  state->ApplyAction(make_2d_tensor(contract_values));
  
  // Chance phase 2: High/Low settlement choice
  std::cout << "\n=== CHANCE PHASE 2: High/Low settlement ===\n";
  std::vector<int> high_low(num_envs);
  high_low[ENV_INDEX] = 1;  // 1 = High (will use 25)
  std::uniform_int_distribution<int> binary_dist(0, 1);
  for (int i = 0; i < num_envs; ++i) {
    if (i != ENV_INDEX) {
      high_low[i] = binary_dist(rng);
    }
  }
  state->ApplyAction(make_1d_tensor(high_low));
  
  // Chance phase 3: Player permutation
  std::cout << "\n=== CHANCE PHASE 3: Player role permutation ===\n";
  std::vector<std::vector<int>> permutation(num_envs, std::vector<int>(4));
  // For environment 32: P0=ValueCheater(0), P1=ValueCheater(1), P2=HighLowCheater(2), P3=Customer(3)
  permutation[ENV_INDEX] = {0, 1, 2, 3};
  for (int i = 0; i < num_envs; ++i) {
    if (i != ENV_INDEX) {
      permutation[i] = {0, 1, 2, 3};
      std::shuffle(permutation[i].begin(), permutation[i].end(), rng);
    }
  }
  state->ApplyAction(make_2d_tensor(permutation));
  
  // Chance phase 4: Customer target sizes (only 1 customer)
  std::cout << "\n=== CHANCE PHASE 4: Customer target sizes ===\n";
  std::vector<std::vector<int>> customer_sizes(num_envs, std::vector<int>(1));
  customer_sizes[ENV_INDEX][0] = 3;  // Customer (Player 3) wants 3 contracts
  std::uniform_int_distribution<int> size_dist(1, 5);
  for (int i = 0; i < num_envs; ++i) {
    if (i != ENV_INDEX) {
      customer_sizes[i][0] = size_dist(rng);
    }
  }
  state->ApplyAction(make_2d_tensor(customer_sizes));
  
  // Cast to access game-specific methods
  auto* trading_state = static_cast<HighLowTradingState*>(state.get());
  
  // Print initial game state
  std::cout << "\n=== INITIAL GAME STATE ===\n";
  std::cout << trading_state->ToString(ENV_INDEX);
  
  // Trading phase - 2 rounds of 4 players each
  torch::Tensor rewards_buffer = torch::zeros({num_envs, 4}, torch::kInt32);
  torch::Tensor player_rewards_buffer = torch::zeros({num_envs}, torch::kInt32);
  
  for (int round = 0; round < 2; ++round) {
    std::cout << "\n=== ROUND " << (round + 1) << " ===\n";
    
    for (int player = 0; player < 4; ++player) {
      std::cout << "\n--- Player " << player << "'s turn ---\n";
      
      // Print player's information state
      std::cout << trading_state->InformationStateString(player, ENV_INDEX);
      
      // Prompt for action
      std::cout << "\nEnter quote for Player " << player << " (bid_price ask_price bid_size ask_size): ";
      int bid_px, ask_px, bid_sz, ask_sz;
      std::cin >> bid_px >> ask_px >> bid_sz >> ask_sz;
      
      // Create action tensor for all environments
      std::vector<std::vector<int>> quotes(num_envs, std::vector<int>(4, 0));
      quotes[ENV_INDEX] = {bid_px, ask_px, bid_sz, ask_sz};
      
      // Apply the action - use uint32 for trading actions
      state->ApplyAction(make_2d_tensor(quotes));
      
      // Get immediate rewards
      trading_state->FillRewards(rewards_buffer);
      auto rewards_cpu = rewards_buffer.cpu();
      auto rewards_accessor = rewards_cpu.accessor<int32_t, 2>();
      
      // Get cumulative rewards since last action for current player
      trading_state->FillRewardsSinceLastAction(player_rewards_buffer, player);
      auto player_rewards_cpu = player_rewards_buffer.cpu();
      auto player_rewards_accessor = player_rewards_cpu.accessor<int32_t, 1>();
      
      // Print rewards
      std::cout << "\nImmediate rewards: ";
      for (int p = 0; p < 4; ++p) {
        std::cout << "P" << p << "=" << rewards_accessor[ENV_INDEX][p];
        if (p < 3) std::cout << ", ";
      }
      std::cout << "\n";
      
      std::cout << "Cumulative reward since Player " << player 
                << "'s last move: " << player_rewards_accessor[ENV_INDEX] << "\n";
      
      // Print updated game state
      std::cout << "\n" << trading_state->ToString(ENV_INDEX);
    }
  }
  
  // Game is now terminal - print final results
  std::cout << "\n=== GAME OVER ===\n";
  std::cout << "\nFinal State:\n";
  std::cout << trading_state->ToString(ENV_INDEX);
  
  // Get and print final returns
  torch::Tensor returns_buffer = torch::zeros({num_envs, 4}, torch::kInt32);
  trading_state->FillReturns(returns_buffer);
  auto returns_cpu = returns_buffer.cpu();
  auto returns_accessor = returns_cpu.accessor<int32_t, 2>();
  
  std::cout << "\nFinal Returns:\n";
  for (int p = 0; p < 4; ++p) {
    std::cout << "Player " << p << ": " << returns_accessor[ENV_INDEX][p] << "\n";
  }
  
  return 0;
}
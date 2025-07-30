// ================================================================================
// HIGH LOW TRADING GAME
// ================================================================================
//
// OVERVIEW:
// A multi-player trading game where players trade contracts that will settle at 
// either a high or low value. Players have asymmetric information and different 
// incentives based on their randomly assigned roles.
//
// GAME MECHANICS:
// 1. Two contract values are randomly drawn from [1, max_contract_value]
// 2. A "high" or "low" settlement is randomly chosen
// 3. Final contract value = max(value1, value2) if "high", min(value1, value2) if "low"
// 4. Players are randomly assigned roles with private information:
//    - ValueCheaters (2): Know one of the candidate contract values
//    - HighLowCheater (1): Knows whether settlement will be "high" or "low" 
//    - Customers (rest): Have target positions they want to achieve
//
// TRADING PHASE:
// Players take turns placing quotes (bid_price, bid_size, ask_price, ask_size)
// that are matched through a continuous double auction market. Orders execute
// immediately when they cross (bid_price >= ask_price).
//
// SCORING:
// - All players: final_cash + final_position * actual_contract_value
// - Customers: additional penalty for missing target position (max contract value per each missed position)
//
// INFORMATION TENSOR LAYOUT:
// Each player observes:
// 1. Game setup & private information (11 elements):
//    - Game parameters (5): [steps_per_player, max_contracts_per_trade, 
//                           customer_max_size, max_contract_value, num_players]
//    - One-hot player role (3): [is_value_cheater, is_high_low_cheater, is_customer]
//    - Player ID encoding (2): [sin(2π*id/players), cos(2π*id/players)]
//    - Private information (1): [contract_value | high_low_signal | target_position]
//
// 2. Public information (dynamic size):
//    - All player positions (num_players * 2): [contracts, cash] for each player
//    - All historical quotes (num_quotes * 6): [bid_px, ask_px, bid_sz, ask_sz, 
//                                              sin(2π*player_id/players), cos(...)]
//
// BUILD & RUN:
// Run the following command to play interactively:
// ./open_spiel/scripts/build_and_run_tests.sh --virtualenv=false --install=true --build_only=true && ./build/examples/high_low_trading_interactive_play
// Components: `market` provides matching & filling. `action_manager` provides unstructured (integer)-structured action conversion. 

// Sample global game state string:
// ********** Game setup **********
// Contract values: 5, 25
// Contract high settle: High
// Player permutation: Player roles: P0=ValueCheater, P1=HighLowCheater, P2=Customer, P3=ValueCheater
// Player 0 target position: No requirement
// Player 1 target position: 2
// Player 2 target position: No requirement
// Player 3 target position: No requirement
// ********************************
// ********** Quote & Fills **********
// Player 0 quote: 1 @ 30 [1 x 1]
// Player 1 quote: 2 @ 29 [1 x 1]
// Player 2 quote: 29 @ 30 [1 x 1]
// Order fill: sz 1 @ px 29 on t=13. User 2 crossed with user 1's quote sz 1 @ px 29
// ********************************
// ********** Player Positions **********
// Player 0 position: [0 contracts, 0 cash]
// Player 1 position: [-1 contracts, 29 cash]
// Player 2 position: [1 contracts, -29 cash]
// Player 3 position: [0 contracts, 0 cash]
// ********************************
// ********** Current Market **********
// ####### 2 sell orders #######
// sz 1 @ px 30   id=2 @ t=15
// sz 1 @ px 30   id=0 @ t=11
// #############################
// ####### 2 buy orders #######
// sz 1 @ px 2   id=1 @ t=12
// sz 1 @ px 1   id=0 @ t=10
// #############################

#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <span>
#include <optional>

#include "torch/torch.h"

#include "core.h"
#include "market.h"
#include "astra_utils.h"

namespace astra {
namespace high_low_trading {

inline constexpr int kDefaultStepsPerPlayer = 20; 
inline constexpr int kDefaultMaxContractsPerTrade = 5; 
inline constexpr int kDefaultCustomerMaxSize = 5; 
inline constexpr int kDefaultMaxContractValue = 30;
inline constexpr int kDefaultNumPlayers = 5; 
inline constexpr int kDefaultNumMarkets = 32768;
inline constexpr int kDefaultThreadsPerBlock = 128; 
inline constexpr int kDefaultDeviceId = 0; 

enum class GamePhase {
  kChanceValue,
  kChanceHighLow, 
  kChancePermutation, 
  kCustomerSize, 
  kPlayerTrading, 
  kTerminal, 
};

class HighLowTradingGame;

// Expose these for registration. 
extern const GameType kGameType;
std::shared_ptr<const Game> Factory(const GameParameters& params);

class HighLowTradingState : public State {
 public:
  explicit HighLowTradingState(std::shared_ptr<const Game> game);
  HighLowTradingState(const HighLowTradingState& other);

  Player CurrentPlayer() const override;
  std::string ToString(uint32_t index) const override;
  bool IsTerminal() const override;
  void FillRewards(torch::Tensor reward_buffer) const override;
  void FillRewardsSinceLastAction(torch::Tensor reward_buffer, Player player_id) const override;
  void FillReturns(torch::Tensor returns_buffer) const override;
  std::string InformationStateString(Player player, uint32_t index) const override;
  // Each player's information state tensor: 
  // 1. Game setup & private information (11):
  //    - Game setup (5): [num_steps, max_contracts_per_trade, customer_max_size, max_contract_value, players]
  //    - One-hot player role (3): [is_value, is_maxmin, is_customer]
  //    - Player id (2): *player_id = [sin(2 pi player_id / players), cos(...)]. *player_id is always denoted with sin and cos
  //    - Private information (1): [contract value, max / min, customer target size]
  // 2. Public information (num_timesteps * num_players * 6 + num_players * 2): quotes, positions
  //    - Positions (num_players, 2): [num_contracts, cash_position]
  //    - Quotes (num_timesteps, num_players, 6): [bid_px, ask_px, bid_sz, ask_sz, *player_id]
  void FillInformationStateTensor(Player player, torch::Tensor values) const override;
  std::string ObservationString(Player player, uint32_t index) const override;
  void FillObservationTensor(Player player, torch::Tensor values) const override;
  std::unique_ptr<State> Clone() const override;
  
  // Returns game state information for analysis/visualization/training.
  // The returned ExposeInfo map contains the following keys:
  //
  // "contract" - torch::Tensor (int32, shape=[3]):
  //   [0] = minimum of the two candidate contract values
  //   [1] = maximum of the two candidate contract values  
  //   [2] = actual settlement value (min if "low", max if "high")
  //
  // "players" - torch::Tensor (float32, shape=[num_players, num_timesteps, 5]):
  //   For each player and timestep: [bid_price, ask_price, bid_size, ask_size, position]
  //   Updated each time a player submits quotes
  //
  // "market" - torch::Tensor (float32, shape=[num_timesteps*5, 5]):
  //   For each timestep: [best_bid_price, best_ask_price, last_trade_price, buy size, sell size]
  //   Updated after each order matching cycle
  //
  // "environment" - torch::Tensor (int32, shape=[num_players]):
  //   Complete environment chance outcomes: [value1, value2, settlement_value, customer_targets...]
  //   [0] = first candidate contract value
  //   [1] = second candidate contract value  
  //   [2] = actual settlement value (GetContractValue())
  //   [3+] = customer target positions in role order (only after chance phase)
  //
  // "info_roles" - torch::Tensor (int32, shape=[num_players]):
  //   Player role indicators (only valid after chance phase):
  //   0 = goodValue (knows the contract value that will be selected)
  //   1 = badValue (knows the contract value that will NOT be selected)
  //   2 = highLow (knows whether "high" or "low" settlement will be chosen)
  //   3 = customer (has a target position requirement)
  //
  // Note: Role assignments are only populated after MoveNumber() >= MaxChanceNodesInHistory()
  ExposeInfo expose_info() const override;

  torch::Tensor GetContractValue() const; 
   
 protected:
  void DoApplyAction(torch::Tensor move) override;
  // Helper values to handle different kinds of actions 
  void ApplyCandidateValues(torch::Tensor move);
  void ApplyHighLowSettle(torch::Tensor move);
  void ApplyPermutation(torch::Tensor move);
  void ApplyCustomerSize(torch::Tensor move);
  const HighLowTradingGame* GetGame() const;

 private:
  int num_envs_;
  int num_players_;
  int steps_per_player_;
  int device_id_; 
  std::string PublicInformationString() const; 
  /* N = num_envs, P=num_players, T=rounds_per_player */
  torch::Tensor contract_values_; // [N, 3] denoting 2 candidate values and settlement value. Int
  torch::Tensor contract_high_settle_; // [N] bool 
  // If permutation[player] = ((0 / 1), (2), (...)) then (ValueCheater, HighLowCheater, Customer)
  torch::Tensor player_permutation_; // [N, P]
  torch::Tensor inv_permutation_; // [N, P]
  torch::Tensor target_positions_; // [N, P]

  order_matching::VecMarket market_; 

  // Purely for ExposeInfo uses
  torch::Tensor player_contract_over_time_; // [N, P, T, 5] int standing for (bid_px, ask_px, bid_sz, ask_sz, contract_position)
  torch::Tensor market_contract_over_time_; // [N*P, 2+1+1] int (best bid px, best ask px, last_price, volume)
};

class HighLowTradingGame : public Game {
  public:
    explicit HighLowTradingGame(const GameParameters& params);
    std::unique_ptr<State> NewInitialState() const override;
    std::vector<int> InformationStateTensorShape() const override;
    std::vector<int> ObservationTensorShape() const override;

    int MaxGameLength() const override {
      return MaxChanceNodesInHistory() + GetStepsPerPlayer() * GetNumPlayers(); 
    }
    int MaxChanceNodesInHistory() const override {
      // Three chance moves (high/low) [N, 2]
      // Contract settlement choice [N]
      // Permutation [N, P]
      // Num_customer assignments [N, P-2]
      return 4; 
    }
    int NumPlayers() const override { return GetNumPlayers(); }

    int GetNumPlayers() const { return num_players_; }
    int GetStepsPerPlayer() const { return steps_per_player_; }
    int GetMaxContractsPerTrade() const { return max_contracts_per_trade_; }
    int GetMaxContractValue() const { return max_contract_value_; }
    int GetCustomerMaxSize() const { return customer_max_size_; }
    int GetNumMarkets() const { return num_markets_; }
    int GetThreadsPerBlock() const { return threads_per_block_; }
    int GetDeviceId() const { return device_id_; }
  private:
    int steps_per_player_; 
    int max_contracts_per_trade_; 
    int customer_max_size_; 
    int max_contract_value_; 
    int num_players_; 
    int num_markets_;
    int threads_per_block_;
    int device_id_;
};

}  // namespace high_low_trading
}  // namespace astra
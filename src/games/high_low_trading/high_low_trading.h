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
#include "action_manager.h" // Game default parameters are all packaged inside "action_manager.h" 
#include "astra_utils.h"

namespace astra {
namespace high_low_trading {

class HighLowTradingGame;

class PlayerPosition {
  public:
    int num_contracts=0; 
    int cash_balance=0; 
    std::string ToString() const; 
}; 

// Expose these for registration. 
extern const GameType kGameType;
std::shared_ptr<const Game> Factory(const GameParameters& params);

class HighLowTradingState : public State {
 public:
  explicit HighLowTradingState(std::shared_ptr<const Game> game);
  HighLowTradingState(const HighLowTradingState& other);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<RewardType> Returns() const override;
  std::vector<RewardType> Rewards() const override;
  std::string InformationStateString(Player player) const override;
  // Each player's information state tensor: 
  // 1. Game setup & private information (11):
  //    - Game setup (5): [num_steps, max_contracts_per_trade, customer_max_size, max_contract_value, players]
  //    - One-hot player role (3): [is_value, is_maxmin, is_customer]
  //    - Player id (2): *player_id = [sin(2 pi player_id / players), cos(...)]. *player_id is always denoted with sin and cos
  //    - Private information (1): [contract value, max / min, customer target size]
  // 2. Public information (num_timesteps * num_players * 6 + num_players * 2): quotes, positions
  //    - Positions (num_players, 2): [num_contracts, cash_position]
  //    - Quotes (num_timesteps, num_players, 6): [bid_px, ask_px, bid_sz, ask_sz, *player_id]
  void InformationStateTensor(Player player,
                              std::span<ObservationScalarType> values) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         std::span<ObservationScalarType> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::vector<Action> LegalActions() const override;
  std::unique_ptr<State> ResampleFromInfostate(
      int player_id, std::function<double()> rng) const override;
  
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
   
 protected:
  void DoApplyAction(Action move) override;
  const HighLowTradingGame* GetGame() const;
  const ActionManager& GetActionManager() const;
  int GetContractValue() const; 

 private:
  // Helper function to reduce bloating 
  void HandleQuoteAction(order_matching::customerId customer_id, PlayerQuoteAction customer_quote); 

  std::string PublicInformationString() const; 
  std::array<ChanceContractValueAction, 2> contract_values_; 
  ChanceHighLowAction contract_high_settle_; 
  
  // PERMUTATION SYSTEM: Maps players to roles randomly
  // Role IDs (fixed): 0=ValueCheater1, 1=ValueCheater2, 2=HighLowCheater, 3+=Customers
  // 
  // permutation_[player_id] → role_id (what role does this player have?)
  // inv_permutation_[role_id] → player_id (which player has this role?)
  //
  // Example: permutation_=[2,0,3,1,4] means Player0→HighLow, Player1→Value1, Player2→Customer, etc.
  //          inv_permutation_=[1,3,0,2,4] means Value1→Player1, HighLow→Player0, etc.
  ChancePermutationAction player_permutation_; 
  std::vector<std::pair<int, PlayerQuoteAction>> player_quotes_; 
  std::vector<PlayerPosition> player_positions_; 
  
  // Target positions for customer players (indexed by player_id, not role_id)
  // 0 = no target (non-customer players), non-zero = customer target position
  // Assignment: customer role_id → player_id via inv_permutation_[role_id] → target stored at player_id
  std::vector<int> player_target_positions_; 
  // Records the last applied actions' resulting change in each player's position 
  std::vector<int> player_position_delta_; 
  std::vector<order_matching::OrderFillEntry> order_fills_; 
  order_matching::Market market_; 

  // Purely for ExposeInfo uses
  at::Tensor player_contract_over_time_; // [num_players, num_timesteps, 5] standing for (bid_px, ask_px, bid_sz, ask_sz, contract_position)
  at::Tensor market_contract_over_time_; // [num_timesteps*5, 2+1+1] (best bid px, best ask px, last_price, volume)
};

class HighLowTradingGame : public Game {
  public:
    explicit HighLowTradingGame(const GameParameters& params);
    int NumDistinctActions() const override; 
    int MaxChanceOutcomes() const override; 
    std::unique_ptr<State> NewInitialState() const override;
    std::vector<int> InformationStateTensorShape() const override;
    std::vector<int> ObservationTensorShape() const override;

    int MaxGameLength() const override {
      return MaxChanceNodesInHistory() + GetStepsPerPlayer() * GetNumPlayers(); 
    }
    int MaxChanceNodesInHistory() const override {
      // See action_manager.h: four chance moves (high, low, choice, permutation) with num_customer assignments. 
      return 4 + (GetNumPlayers() - 3); 
    }
    RewardType MinUtility() const override { return -MaxUtility(); }
    RewardType MaxUtility() const override {
      return (GetMaxContractValue() - 1) * GetMaxContractsPerTrade() * GetStepsPerPlayer() * GetNumPlayers(); 
    }
    int NumPlayers() const override { return GetNumPlayers(); }
    std::optional<RewardType> UtilitySum() const override { 
      AstraFatalError("UtilitySum not implemented.");
    }

    const ActionManager& GetActionManager() const { return action_manager_; }

    int GetNumPlayers() const { return action_manager_.GetNumPlayers(); }
    int GetStepsPerPlayer() const { return action_manager_.GetStepsPerPlayer(); }
    int GetMaxContractsPerTrade() const { return action_manager_.GetMaxContractsPerTrade(); }
    int GetMaxContractValue() const { return action_manager_.GetMaxContractValue(); }
    int GetCustomerMaxSize() const { return action_manager_.GetCustomerMaxSize(); }

  private:
    ActionManager action_manager_; 
};

}  // namespace high_low_trading
}  // namespace astra
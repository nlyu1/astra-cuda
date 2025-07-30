#include "high_low_trading.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <string>
#include <utility>

#include <fmt/format.h>
#include "market.h"
#include "game_parameters.h"
#include "core.h"
#include "astra_utils.h"

#include <torch/torch.h>

namespace astra {
namespace high_low_trading {

// Facts about the game
const GameType kGameType{/*short_name=*/"high_low_trading",
                         /*long_name=*/"High Low Trading",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kGeneralSum, 
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/10,
                         /*min_num_players=*/4,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/{
                          {"steps_per_player", GameParameter(kDefaultStepsPerPlayer)},
                          {"max_contracts_per_trade", GameParameter(kDefaultMaxContractsPerTrade)},
                          {"customer_max_size", GameParameter(kDefaultCustomerMaxSize)},
                          {"max_contract_value", GameParameter(kDefaultMaxContractValue)},
                          {"players", GameParameter(kDefaultNumPlayers)},
                         },
                         /*default_loadable=*/true};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new HighLowTradingGame(params));
}

HighLowTradingState::HighLowTradingState(std::shared_ptr<const Game> game)
    : State(game) {
    auto derived_game = static_cast<const HighLowTradingGame*>(game.get()); 
    if (derived_game == nullptr) {
      AstraFatalError("HighLowTradingState: game is not a HighLowTradingGame"); 
    }

    // Heuristic for how many fills there'll be
    return; 
}

// Todo: Implement deep copy of fields
// HighLowTradingState::HighLowTradingState(const HighLowTradingState& other)
//     : State(other),
//       contract_values_(other.contract_values_),
//       contract_high_settle_(other.contract_high_settle_),
//       player_permutation_(other.player_permutation_),
//       player_quotes_(other.player_quotes_),
//       player_positions_(other.player_positions_),
//       player_position_delta_(other.player_position_delta_),
//       player_target_positions_(other.player_target_positions_),
//       order_fills_(other.order_fills_),
//       market_(other.market_),
//       // Deep copy the tensors to ensure truly independent copies
//       player_contract_over_time_(other.player_contract_over_time_.clone()),
//       market_contract_over_time_(other.market_contract_over_time_.clone()) {
// }

Player HighLowTradingState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } 
  int move_number = MoveNumber();
  if (move_number < game_->MaxChanceNodesInHistory()) {
    return kChancePlayerId; 
  } else {
    return (move_number - game_->MaxChanceNodesInHistory()) % NumPlayers(); 
  }
  return kTerminalPlayerId;
}

void HighLowTradingState::DoApplyAction(torch::Tensor move) {
  AstraFatalError("Not implemented"); 
}

const HighLowTradingGame* HighLowTradingState::GetGame() const {
  return static_cast<const HighLowTradingGame*>(game_.get()); 
}

std::string HighLowTradingState::PublicInformationString() const {
  std::ostringstream result;
  // result << "********** Game Configuration **********\n";
  // result << fmt::format("Steps per player: {}\n", GetGame()->GetStepsPerPlayer());
  // result << fmt::format("Max contracts per trade: {}\n", GetGame()->GetMaxContractsPerTrade());
  // result << fmt::format("Customer max size: {}\n", GetGame()->GetCustomerMaxSize());
  // result << fmt::format("Max contract value: {}\n", GetGame()->GetMaxContractValue());
  // result << fmt::format("Number of players: {}\n", GetGame()->GetNumPlayers());
  // result << "****************************************\n\n";

  // result << "********** Player Positions **********\n";
  // for (int i = 0; i < NumPlayers(); ++i) {
  //   result << fmt::format("Player {} position: {}\n", i, player_positions_[i].ToString());
  // }
  // result << "**************************************\n\n";
  
  // result << "********** Fills & Quotes **********\n";
  // result << fmt::format("Number of fills: {}\n", order_fills_.size());
  // for (auto fill : order_fills_) {
  //   result << fmt::format("Order fill: {}\n", fill.ToString());
  // }
  // for (auto quote : player_quotes_) {
  //   result << fmt::format("Player {} quote: {}\n", quote.first, quote.second.ToString());
  // }
  // result << "***********************************\n\n";

  // result << "********** Current Market **********\n";
  // result << fmt::format("{}\n", market_.ToString());
  return result.str();
}

std::string HighLowTradingState::ToString(uint32_t index) const {
  std::ostringstream result;
  // result << "********** Game setup **********\n";
  // result << fmt::format("Contract values: {}, {}\n", contract_values_[0].contract_value_, contract_values_[1].contract_value_);
  // result << fmt::format("Contract high settle: {}\n", contract_high_settle_.is_high_ ? "High" : "Low");
  // result << fmt::format("Player permutation: {}\n", player_permutation_.ToString());
  // for (int i = 0; i < NumPlayers(); ++i) {
  //   auto target_position = player_target_positions_[i];
  //   if (target_position == 0) {
  //     result << fmt::format("Player {} target position: No requirement\n", i);
  //   } else {
  //     result << fmt::format("Player {} target position: {}\n", i, target_position);
  //   }
  // }
  // result << "********************************\n\n";
  // result << PublicInformationString();
  return result.str();
}

bool HighLowTradingState::IsTerminal() const {
  return MoveNumber() >= game_->MaxGameLength(); 
}

int HighLowTradingState::GetContractValue() const {
  ASTRA_CHECK_GE(MoveNumber(), 3); 
  return 0;
}

// We model instantaneous reward as (portfolio value at terminal timestep)
// To provide more granular rewards, each contract fill movement in the direction 
// of the target position is rewarded with "max_contract_value" reward. 
// Harmful moves are penalized
void HighLowTradingState::FillRewards(at::Tensor reward_buffer) const {
  AstraFatalError("Not implemented"); 
}

// Returns is modeled as (customer penalty) + portfolio value
void HighLowTradingState::FillReturns(at::Tensor returns_buffer) const {
  AstraFatalError("Not implemented"); 
}

void HighLowTradingState::FillRewardsSinceLastAction(at::Tensor reward_buffer, Player player_id) const {
  AstraFatalError("Not implemented"); 
}

ExposeInfo HighLowTradingState::expose_info() const {
  if (!IsTerminal()) {
    AstraFatalError("ExposeInfo called on non-terminal state");
  }
  ExposeInfo info; 
  
  // int min_value, max_value;
  // min_value = std::min(contract_values_[0].contract_value_, contract_values_[1].contract_value_); 
  // max_value = std::max(contract_values_[0].contract_value_, contract_values_[1].contract_value_); 
  // info["contract"] = torch::zeros({3}, torch::kInt32); 
  // auto contract_tensor = std::get<torch::Tensor>(info["contract"]);
  // auto contract_accessor = contract_tensor.accessor<int, 1>();
  // contract_accessor[0] = min_value; 
  // contract_accessor[1] = max_value; 
  // contract_accessor[2] = GetContractValue(); 

  // info["players"] = player_contract_over_time_; 
  // info["market"] = market_contract_over_time_; 

  // auto returns = Returns(); 
  // info["total_return"] = std::reduce(returns.begin(), returns.end());
  
  // // Environment tensor: [contract_value1, contract_value2, settlement_value, customer_sizes...]
  // // Size: 3 + number_of_customers = 3 + (NumPlayers() - 3) = NumPlayers()
  // info["environment"] = torch::zeros({NumPlayers()}, torch::kInt32); 
  // auto environment_tensor = std::get<torch::Tensor>(info["environment"]);
  // auto environment_accessor = environment_tensor.accessor<int, 1>();
  // environment_accessor[0] = contract_values_[0].contract_value_; 
  // environment_accessor[1] = contract_values_[1].contract_value_; 
  // environment_accessor[2] = GetContractValue(); 
  
  // // For each customer role (starting from role 3), get their target position
  // for (int role_id = 3; role_id < NumPlayers(); ++role_id) {
  //   int customer_player_id = player_permutation_.inv_permutation_[role_id]; 
  //   environment_accessor[role_id] = player_target_positions_[customer_player_id]; 
  // }

  // info["target_positions"] = torch::zeros({NumPlayers()}, torch::kInt32); 
  // auto target_positions_tensor = std::get<torch::Tensor>(info["target_positions"]);
  // auto target_positions_accessor = target_positions_tensor.accessor<int, 1>();
  // for (int i = 0; i < NumPlayers(); ++i) {
  //   target_positions_accessor[i] = player_target_positions_[i]; 
  // }
  
  // // 0, 1, 2, 3 = goodValue, badValue, highLow, customer
  // info["info_roles"] = torch::zeros({NumPlayers()}, torch::kInt32); 
  // auto info_roles_tensor = std::get<torch::Tensor>(info["info_roles"]);
  // auto info_roles_accessor = info_roles_tensor.accessor<int, 1>();
  
  // // Only assign roles if we're past the chance phase
  // for (int i = 0; i < NumPlayers(); ++i) {
  //   int perm_id = player_permutation_.permutation_[i]; 
  //   if (perm_id == 0 || perm_id == 1) {
  //     if (contract_values_[perm_id].contract_value_ == GetContractValue()) {
  //       info_roles_accessor[i] = 0; // goodValue
  //     } else {
  //       info_roles_accessor[i] = 1; // badValue
  //     }
  //   } else if (perm_id == 2) {
  //     info_roles_accessor[i] = 2; // highLow
  //   } else {
  //     info_roles_accessor[i] = 3; // customer
  //   }
  // }
  
  return info; 
}

std::unique_ptr<State> HighLowTradingState::Clone() const {
  return std::unique_ptr<State>(new HighLowTradingState(*this));
}

std::vector<int> HighLowTradingGame::InformationStateTensorShape() const {
  // See `high_low_trading.h` for what each entry means 
  return {11 + GetStepsPerPlayer() * GetNumPlayers() * 6 + GetNumPlayers() * 2};
}

void HighLowTradingState::FillInformationStateTensor(Player player,
  torch::Tensor values) const {
  ASTRA_CHECK_GE(player, 0);
  ASTRA_CHECK_LT(player, GetGame()->NumPlayers());
  values.fill_(0);
  // int offset = 0;
  
  // // 1. Game setup (5): [num_steps, max_contracts_per_trade, customer_max_size, max_contract_value, players]
  // values[offset++] = static_cast<ObservationScalarType>(GetGame()->GetStepsPerPlayer());
  // values[offset++] = static_cast<ObservationScalarType>(GetGame()->GetMaxContractsPerTrade());
  // values[offset++] = static_cast<ObservationScalarType>(GetGame()->GetCustomerMaxSize());
  // values[offset++] = static_cast<ObservationScalarType>(GetGame()->GetMaxContractValue());
  // values[offset++] = static_cast<ObservationScalarType>(GetGame()->GetNumPlayers());
  
  // // 2. One-hot player role (3): [is_value, is_maxmin, is_customer]
  // if (MoveNumber() >= GetGame()->MaxChanceNodesInHistory()) {
  //   int perm_id = player_permutation_.permutation_[player]; 
  //   if (perm_id == 0 || perm_id == 1) {
  //     values[offset] = 1.0; // is_value
  //   } else if (perm_id == 2) {
  //     values[offset + 1] = 1.0; // is_maxmin (high/low cheater)
  //   } else {
  //     values[offset + 2] = 1.0; // is_customer
  //   }
  // }
  // offset += 3;
  
  // // 3. Player id (2): [sin(2 pi player_id / players), cos(2 pi player_id / players)]
  // double angle = 2.0 * M_PI * player / GetGame()->GetNumPlayers();
  // values[offset++] = static_cast<ObservationScalarType>(std::sin(angle));
  // values[offset++] = static_cast<ObservationScalarType>(std::cos(angle));
  
  // // 4. Private information (1): [contract value, max / min, customer target size]
  // if (MoveNumber() >= GetGame()->MaxChanceNodesInHistory()) {
  //   int perm_id = player_permutation_.permutation_[player]; 
  //   if (perm_id == 0 || perm_id == 1) {
  //     values[offset] = static_cast<ObservationScalarType>(contract_values_[perm_id].contract_value_);
  //   } else if (perm_id == 2) {
  //     values[offset] = contract_high_settle_.is_high_ ? 1.0 : -1.0;
  //   } else {
  //     values[offset] = static_cast<ObservationScalarType>(player_target_positions_[player]);
  //   }
  // }
  // offset += 1;
  
  // // 5. Positions (num_players, 2): [num_contracts, cash_position] - fixed length first
  // int num_players = GetGame()->GetNumPlayers();
  // for (int p = 0; p < num_players; ++p) {
  //   values[offset++] = static_cast<ObservationScalarType>(player_positions_[p].num_contracts);
  //   values[offset++] = static_cast<ObservationScalarType>(player_positions_[p].cash_balance);
  // }
  
  // // 6. Quotes: [bid_px, ask_px, bid_sz, ask_sz, *player_id] - fill remaining space
  // for (int quote_idx = 0; quote_idx < static_cast<int>(player_quotes_.size()); ++quote_idx) {
  //   const auto& quote_pair = player_quotes_[quote_idx];
  //   int acting_player = quote_pair.first;
  //   const auto& quote = quote_pair.second;
    
  //   // Assert we have enough space (InformationStateTensorShape should guarantee this)
  //   ASTRA_CHECK_LE(offset + 6, static_cast<int>(values.size()));
    
  //   values[offset++] = static_cast<ObservationScalarType>(quote.bid_price_);
  //   values[offset++] = static_cast<ObservationScalarType>(quote.ask_price_);
  //   values[offset++] = static_cast<ObservationScalarType>(quote.bid_size_);
  //   values[offset++] = static_cast<ObservationScalarType>(quote.ask_size_);
    
  //   // Player id as sin/cos
  //   double p_angle = 2.0 * M_PI * acting_player / num_players;
  //   values[offset++] = static_cast<ObservationScalarType>(std::sin(p_angle));
  //   values[offset++] = static_cast<ObservationScalarType>(std::cos(p_angle));
  // }
}

std::string HighLowTradingState::InformationStateString(Player player, uint32_t index) const {
  ASTRA_CHECK_GE(player, 0);
  ASTRA_CHECK_LT(player, GetGame()->NumPlayers());
  
  std::ostringstream result;
  
  // // Add player's role information
  // result << "********** Private Information **********\n";

  
  // // Check if we're past the permutation phase
  // if (MoveNumber() >= GetGame()->MaxChanceNodesInHistory()) {
  //   int perm_id = player_permutation_.permutation_[player]; 
  //   std::string role_name;

  //   if (perm_id == 0 || perm_id == 1) {
  //     role_name = "ValueCheater";
  //   } else if (perm_id == 2) {
  //     role_name = "HighLowCheater";
  //   } else {
  //     role_name = "Customer";
  //   }
  //   result << fmt::format("My role: {}\n", role_name);
    
  //   // Add private information based on role
  //   if (perm_id == 0 || perm_id == 1) {
  //       // ValueCheaters know the contract values
  //       result << fmt::format("Candidate contract value: {}\n",
  //                            contract_values_[perm_id].contract_value_);
  //   } else if (perm_id == 2) {
  //       // HighLowCheaters know which settlement (high or low) will be chosen
  //       result << fmt::format("Settlement will be: {}\n",
  //                            contract_high_settle_.is_high_ ? "High" : "Low");
  //   } else {
  //       // Customers know their target position
  //       auto target_position = player_target_positions_[player];
  //       if (target_position != 0) {
  //         result << fmt::format("My target position: {}\n", target_position);
  //       } else {
  //         result << "Not supposed to happen. Customer target position should not be 0 \n"; 
  //         AstraFatalError("Not supposed to happen. Customer roles should be assigned"); 
  //       }
  //   }
  //   // Start with public information that all players can see
  //   result << PublicInformationString();
    
  // } else {
  //   result << "Private info pending...\n";
  // }
  
  result << "***************************\n";
  
  return result.str();
}

// Observations are exactly the info states. Preserve Markov condition. 
std::vector<int> HighLowTradingGame::ObservationTensorShape() const {
  return InformationStateTensorShape(); 
}

std::string HighLowTradingState::ObservationString(Player player, uint32_t index) const {
  return InformationStateString(player, index); 
}

void HighLowTradingState::ObservationTensor(Player player,
  std::span<ObservationScalarType> values) const {
  return InformationStateTensor(player, values); 
}

HighLowTradingGame::HighLowTradingGame(const GameParameters& params)
    : Game(kGameType, params),
      action_manager_(Config(
        ParameterValue<int>("steps_per_player"), 
        ParameterValue<int>("max_contracts_per_trade"),
        ParameterValue<int>("customer_max_size"),
        ParameterValue<int>("max_contract_value"),
        ParameterValue<int>("players")
      )) {
}

std::unique_ptr<State> HighLowTradingGame::NewInitialState() const {
  return std::unique_ptr<State>(new HighLowTradingState(shared_from_this()));
}

int HighLowTradingGame::NumDistinctActions() const {
  return GetActionManager().valid_action_range(GamePhase::kPlayerTrading).second + 1; 
}

}  // namespace high_low_trading
}  // namespace astra
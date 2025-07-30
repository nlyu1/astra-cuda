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
                          {"num_markets", GameParameter(kDefaultNumMarkets)},
                          {"threads_per_block", GameParameter(kDefaultThreadsPerBlock)},
                          {"device_id", GameParameter(kDefaultDeviceId)}
                         },
                         /*default_loadable=*/true};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new HighLowTradingGame(params));
}

HighLowTradingState::HighLowTradingState(std::shared_ptr<const Game> game)
    : State(game),
      num_envs_(static_cast<const HighLowTradingGame*>(game.get())->GetNumMarkets()),
      num_players_(static_cast<const HighLowTradingGame*>(game.get())->GetNumPlayers()),
      steps_per_player_(static_cast<const HighLowTradingGame*>(game.get())->GetStepsPerPlayer()),
      device_id_(static_cast<const HighLowTradingGame*>(game.get())->GetDeviceId()),
      market_(
        /*num_markets=*/static_cast<const HighLowTradingGame*>(game.get())->GetNumMarkets(), 
        /*max_price_levels=*/static_cast<const HighLowTradingGame*>(game.get())->GetMaxContractValue(),
        /*max_active_orders_per_market=*/static_cast<const HighLowTradingGame*>(game.get())->GetNumPlayers() * 
                                         static_cast<const HighLowTradingGame*>(game.get())->GetStepsPerPlayer() * 2,
        /*max_active_fills_per_market=*/std::min(
          static_cast<const HighLowTradingGame*>(game.get())->GetNumPlayers() * 
          static_cast<const HighLowTradingGame*>(game.get())->GetStepsPerPlayer() * 2,
          static_cast<const HighLowTradingGame*>(game.get())->GetMaxContractsPerTrade() * 2),
        /*num_customers=*/static_cast<const HighLowTradingGame*>(game.get())->GetNumPlayers(), 
        /*device_id=*/static_cast<const HighLowTradingGame*>(game.get())->GetDeviceId(), 
        /*threads_per_block=*/static_cast<const HighLowTradingGame*>(game.get())->GetThreadsPerBlock()
      ) {
    auto derived_game = static_cast<const HighLowTradingGame*>(game.get()); 
    if (derived_game == nullptr) {
      AstraFatalError("HighLowTradingState: game is not a HighLowTradingGame"); 
    }

    // Configure tensor options for the specified device
    auto device = torch::Device(torch::kCUDA, device_id_);
    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(device);

    // Initialize tensors with appropriate shapes
    contract_values_ = torch::zeros({num_envs_, 3}, options_i32);
    contract_high_settle_ = torch::zeros({num_envs_}, options_bool);
    player_permutation_ = torch::zeros({num_envs_, num_players_}, options_i32);
    inv_permutation_ = torch::zeros({num_envs_, num_players_}, options_i32);
    target_positions_ = torch::zeros({num_envs_, num_players_}, options_i32);
    
    // Initialize tracking tensors for ExposeInfo
    player_contract_over_time_ = torch::zeros({num_envs_, num_players_, steps_per_player_, 5}, options_i32);
    market_contract_over_time_ = torch::zeros({num_envs_ * num_players_, 4}, options_i32); // best bid, best ask, last_price, volume
}

HighLowTradingState::HighLowTradingState(const HighLowTradingState& other)
    : State(other),
      num_envs_(other.num_envs_),
      num_players_(other.num_players_),
      steps_per_player_(other.steps_per_player_),
      device_id_(other.device_id_),
      // Deep copy the tensors to ensure truly independent copies
      contract_values_(other.contract_values_.clone()),
      contract_high_settle_(other.contract_high_settle_.clone()),
      player_permutation_(other.player_permutation_.clone()),
      inv_permutation_(other.inv_permutation_.clone()),
      target_positions_(other.target_positions_.clone()),
      market_(other.market_),
      player_contract_over_time_(other.player_contract_over_time_.clone()),
      market_contract_over_time_(other.market_contract_over_time_.clone()) {
}

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

void HighLowTradingState::ApplyCandidateValues(torch::Tensor move) {
  // First chance move: setting two candidate contract values
  ASTRA_CHECK_EQ(move.dim(), 2);
  ASTRA_CHECK_EQ(move.size(1), 2);
  
  // Ensure values are within valid range [1, max_contract_value]
  auto min_val = torch::min(move).item<int>();
  auto max_val = torch::max(move).item<int>();
  ASTRA_CHECK_GE(min_val, 1);
  ASTRA_CHECK_LE(max_val, GetGame()->GetMaxContractValue());
  
  // Populate first two columns of contract_values
  contract_values_.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}) = move;
}

void HighLowTradingState::ApplyHighLowSettle(torch::Tensor move) {
  // Second chance move: determining high/low settlement
  ASTRA_CHECK_EQ(move.dim(), 1);
      
  // Ensure values are 0 or 1
  auto min_val = torch::min(move).item<int>();
  auto max_val = torch::max(move).item<int>();
  ASTRA_CHECK_GE(min_val, 0);
  ASTRA_CHECK_LE(max_val, 1);

  // Store high/low choice
  contract_high_settle_ = move.to(torch::kBool);

  // Compute max and min values from the two candidate values
  auto max_values = torch::max(contract_values_.index({torch::indexing::Slice(), 0}), 
                                contract_values_.index({torch::indexing::Slice(), 1}));
  auto min_values = torch::min(contract_values_.index({torch::indexing::Slice(), 0}), 
                                contract_values_.index({torch::indexing::Slice(), 1}));

  // Set settlement value: move * max + (1 - move) * min
  contract_values_.index({torch::indexing::Slice(), 2}) = 
      move * max_values + (1 - move) * min_values;
}

void HighLowTradingState::ApplyPermutation(torch::Tensor move) {
  // Third chance move: player role permutation. We won't check that this is a valid permutation
  ASTRA_CHECK_EQ(move.dim(), 2);
  ASTRA_CHECK_EQ(move.size(1), num_players_);
  
  // Ensure values are valid player indices [0, P-1]
  auto min_val = torch::min(move).item<int>();
  auto max_val = torch::max(move).item<int>();
  ASTRA_CHECK_GE(min_val, 0);
  ASTRA_CHECK_LT(max_val, num_players_);
  
  // Populate player permutation
  player_permutation_ = move;
  inv_permutation_ = move.argsort(/*dim=*/1, /*descending=*/false);
  // Need to double-check correctness
  // for (int j = 0; j < num_envs_; ++j) {
  //   for (int p = 0; p < num_players_; ++p) {
  //     ASTRA_CHECK_EQ(move[j, p], player_permutation_[j, inv_permutation_[j, p]]);
  //   }
  // }
}

void HighLowTradingState::ApplyCustomerSize(torch::Tensor move) {
  ASTRA_CHECK_EQ(move.dim(), 2);
  ASTRA_CHECK_EQ(move.size(1), num_players_ - 3);
  
  // Customer roles start at index 3 (roles 0,1 are ValueCheaters, role 2 is HighLowCheater)
  // move contains target positions for customer roles [3, 4, ..., P-1]
  // inv_permutation_[j, r] tells us which player has role r in environment j
  
  // For each customer role r in [3, P-1]:
  //   - The player with role r is inv_permutation_[j, r]  
  //   - The target position comes from move[j, r - 3]
  
  // Get player indices for each customer role across all environments
  // inv_permutation_[:, 3:] gives us which players have customer roles
  auto customer_player_indices = inv_permutation_.index({
      torch::indexing::Slice(), 
      torch::indexing::Slice(3, torch::indexing::None)
  });
  
  // Use scatter to assign target positions to the correct players
  // We need to scatter move values to the positions indicated by customer_player_indices
  target_positions_.scatter_(
      /*dim=*/1, 
      /*index=*/customer_player_indices.to(torch::kInt64),
      /*src=*/move.to(target_positions_.dtype())
  );
}

void HighLowTradingState::DoApplyAction(torch::Tensor move) {
  int move_number = MoveNumber();
  ASTRA_CHECK_EQ(move.size(0), num_envs_); 
  
  if (move_number == 0) {
    ApplyCandidateValues(move);
  } else if (move_number == 1) {
    ApplyHighLowSettle(move);
  } else if (move_number == 2) {
    ApplyPermutation(move);
  } else if (move_number == 3) {
    ApplyCustomerSize(move);
  } else {
    // We're in player trading mode 
    AstraFatalError("Move not implemented"); 
  }
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

torch::Tensor HighLowTradingState::GetContractValue() const {
  ASTRA_CHECK_GE(MoveNumber(), 2); 
  // Return the settlement values (third column) for all environments
  return contract_values_.index({torch::indexing::Slice(), 2}); 
}

// We model instantaneous reward as (portfolio value at terminal timestep)
// To provide more granular rewards, each contract fill movement in the direction 
// of the target position is rewarded with "max_contract_value" reward. 
// Harmful moves are penalized
void HighLowTradingState::FillRewards(torch::Tensor reward_buffer) const {
  AstraFatalError("Not implemented"); 
}

// Returns is modeled as (customer penalty) + portfolio value
void HighLowTradingState::FillReturns(torch::Tensor returns_buffer) const {
  AstraFatalError("Not implemented"); 
}

void HighLowTradingState::FillRewardsSinceLastAction(torch::Tensor reward_buffer, Player player_id) const {
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

void HighLowTradingState::FillObservationTensor(Player player,
  torch::Tensor values) const {
  return FillInformationStateTensor(player, values); 
}

HighLowTradingGame::HighLowTradingGame(const GameParameters& params)
    : Game(kGameType, params),
      steps_per_player_(ParameterValue<int>("steps_per_player")),
      max_contracts_per_trade_(ParameterValue<int>("max_contracts_per_trade")),
      customer_max_size_(ParameterValue<int>("customer_max_size")),
      max_contract_value_(ParameterValue<int>("max_contract_value")),
      num_players_(ParameterValue<int>("players")),
      num_markets_(ParameterValue<int>("num_markets")),
      threads_per_block_(ParameterValue<int>("threads_per_block")),
      device_id_(ParameterValue<int>("device_id")) {
}

std::unique_ptr<State> HighLowTradingGame::NewInitialState() const {
  return std::unique_ptr<State>(new HighLowTradingState(shared_from_this()));
}

}  // namespace high_low_trading
}  // namespace astra
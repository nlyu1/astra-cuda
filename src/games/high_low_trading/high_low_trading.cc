#include "high_low_trading.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <string>
#include <utility>

#include <fmt/format.h>
#include "game_parameters.h"
#include "core.h"
#include "astra_utils.h"
#include "market.h"
#include "action_manager.h" 

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
    : State(game),
      contract_values_{ChanceContractValueAction(0), ChanceContractValueAction(0)},
      contract_high_settle_(false),
      player_permutation_(NumPlayers()),
      player_target_positions_(NumPlayers(), 0),
      player_position_delta_(NumPlayers(), 0),
      player_positions_(NumPlayers()),
      market_() {
    auto derived_game = static_cast<const HighLowTradingGame*>(game.get()); 
    if (derived_game == nullptr) {
      AstraFatalError("HighLowTradingState: game is not a HighLowTradingGame"); 
    }

    // Heuristic for how many fills there'll be
    int steps_per_player = derived_game->GetStepsPerPlayer(); 
    int num_players = NumPlayers();
    order_fills_.reserve(steps_per_player * num_players / 2);
    player_quotes_.reserve(steps_per_player * num_players); 

    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32);
    player_contract_over_time_ = torch::zeros({num_players, steps_per_player, 5}, tensor_options);
    market_contract_over_time_ = torch::zeros({steps_per_player * 5, 5}, tensor_options);
}

HighLowTradingState::HighLowTradingState(const HighLowTradingState& other)
    : State(other),
      contract_values_(other.contract_values_),
      contract_high_settle_(other.contract_high_settle_),
      player_permutation_(other.player_permutation_),
      player_quotes_(other.player_quotes_),
      player_positions_(other.player_positions_),
      player_position_delta_(other.player_position_delta_),
      player_target_positions_(other.player_target_positions_),
      order_fills_(other.order_fills_),
      market_(other.market_),
      // Deep copy the tensors to ensure truly independent copies
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

void HighLowTradingState::DoApplyAction(Action move) {
  int move_number = MoveNumber(); 
  auto action_manager = GetActionManager(); 
  auto game_phase = action_manager.game_phase_of_timestep(move_number); 
  ASTRA_CHECK_LT(move_number, game_->MaxGameLength()); 
  
  auto [min_action, max_action] = action_manager.valid_action_range(game_phase); 
  if (max_action < 0) {
    AstraFatalError(StrCat(
        "Invalid action range for move_number=", move_number, 
        ", game_phase=", static_cast<int>(game_phase),
        ", range=[", min_action, ",", max_action, "]",
        ", num_players=", GetGame()->GetNumPlayers(),
        ", max_contract_value=", GetGame()->GetMaxContractValue(),
        ", customer_max_size=", GetGame()->GetCustomerMaxSize()));
  }
  
  ASTRA_CHECK_LE(move, max_action); 

  auto structured_action = action_manager.Encode(move_number, move); 

  if (move_number < 2) {
    contract_values_[move_number] = std::get<ChanceContractValueAction>(structured_action).contract_value_; 
  } else if (move_number == 2) {
    contract_high_settle_ = std::get<ChanceHighLowAction>(structured_action).is_high_; 
  } else if (move_number == 3) {
    player_permutation_ = std::get<ChancePermutationAction>(structured_action); 
  } else if (move_number < game_->MaxChanceNodesInHistory()) {
    auto target_position = std::get<ChanceCustomerSizeAction>(structured_action).customer_size_; 
    // Since permutation_[player_id] = permed id (0, 1, 2 = valueCheater1, valueCheater2, highLowCheater, etc), 
    // we need to identify the inverse permutation 
    int customer_player_id = player_permutation_.inv_permutation_[move_number - 4 + 3]; 
    player_target_positions_[customer_player_id] = target_position; 
  } else {
    order_matching::customerId customer_id = static_cast<order_matching::customerId>(CurrentPlayer()); 
    auto customer_quote = std::get<PlayerQuoteAction>(structured_action); 
    // Clear out position diff
    std::fill(player_position_delta_.begin(), player_position_delta_.end(), 0);
    HandleQuoteAction(customer_id, customer_quote); 
  }
}

void HighLowTradingState::HandleQuoteAction(order_matching::customerId customer_id, PlayerQuoteAction customer_quote) {
  player_quotes_.push_back(std::make_pair(CurrentPlayer(), customer_quote)); 

  // This is the number of trading move
  int move_number = MoveNumber() - GetGame()->MaxChanceNodesInHistory();
  auto fills = market_.AddOrder(order_matching::OrderEntry(customer_quote.bid_price_, customer_quote.bid_size_, 2 * move_number, customer_id, true)); 
  auto ask_fills = market_.AddOrder(order_matching::OrderEntry(customer_quote.ask_price_, customer_quote.ask_size_, 2 * move_number + 1, customer_id, false)); 

  // Merge fills 
  fills.insert(fills.end(), ask_fills.begin(), ask_fills.end()); 
  order_fills_.insert(order_fills_.end(), fills.begin(), fills.end()); 
  // Adjust customer positions based on fills: 
  int sell_volume = 0;
  int buy_volume = 0; 
  for (auto fill : fills) {
    if (fill.is_sell_quote) {
      // Customer hit a sell quote (customer is buying)
      player_positions_[fill.customer_id].num_contracts += fill.size; 
      player_positions_[fill.customer_id].cash_balance -= fill.price * fill.size; 
      player_positions_[fill.quoter_id].num_contracts -= fill.size; 
      player_positions_[fill.quoter_id].cash_balance += fill.price * fill.size; 
      buy_volume += fill.size; 
      player_position_delta_[fill.customer_id] += fill.size; 
      player_position_delta_[fill.quoter_id] -= fill.size; 
    } else {
      // Customer hit a buy quote (customer is selling)
      player_positions_[fill.customer_id].num_contracts -= fill.size; 
      player_positions_[fill.customer_id].cash_balance += fill.price * fill.size; 
      player_positions_[fill.quoter_id].num_contracts += fill.size; 
      player_positions_[fill.quoter_id].cash_balance -= fill.price * fill.size; 
      sell_volume += fill.size; 
      player_position_delta_[fill.customer_id] -= fill.size; 
      player_position_delta_[fill.quoter_id] += fill.size; 
    }
  }

  // Update ExposeInfo and ContractInfo
  int player_num_moves = move_number / GetGame()->NumPlayers(); 
  auto player_accessor = player_contract_over_time_.accessor<float, 3>();
  auto market_accessor = market_contract_over_time_.accessor<float, 2>();

  player_accessor[customer_id][player_num_moves][0] = customer_quote.bid_price_; 
  player_accessor[customer_id][player_num_moves][1] = customer_quote.ask_price_; 
  player_accessor[customer_id][player_num_moves][2] = customer_quote.bid_size_; 
  player_accessor[customer_id][player_num_moves][3] = customer_quote.ask_size_; 
  player_accessor[customer_id][player_num_moves][4] = player_positions_[customer_id].num_contracts; 

  const auto& [best_bid_opt, best_ask_opt] = market_.GetBBO(); 
  // Use the (bid, ask) price if it exists, otherwise use (0, max_value) 
  market_accessor[move_number][0] = best_bid_opt.has_value() ? best_bid_opt->first : 0.0f;
  market_accessor[move_number][1] = best_ask_opt.has_value() ? best_ask_opt->first : GetGame()->GetMaxContractValue();
  // Use last fill else 0 
  market_accessor[move_number][2] = order_fills_.size() > 0 ? order_fills_.back().price : 0; 
  // Use number of fills 
  market_accessor[move_number][3] = buy_volume;
  market_accessor[move_number][4] = sell_volume; 
}

std::vector<Action> HighLowTradingState::LegalActions() const {
  if (IsTerminal()) {
      return {};
  } 
  auto [min_action, max_action] = GetActionManager().valid_action_range(GetActionManager().game_phase_of_timestep(MoveNumber())); 
  std::vector<Action> actions; 
  for (int action = min_action; action <= max_action; ++action) {
    actions.push_back(action); 
  }
  return actions; 
}

std::string PlayerPosition::ToString() const {
  return StrCat("[", num_contracts, " contracts, ", cash_balance, " cash]"); 
}

std::string HighLowTradingState::ActionToString(Player player, Action move) const {
  auto structured_action = GetActionManager().Encode(MoveNumber(), move); 
  return StrCat("Player ", player, " ", ActionVariantToString(structured_action)); 
}

const HighLowTradingGame* HighLowTradingState::GetGame() const {
  return static_cast<const HighLowTradingGame*>(game_.get()); 
}

const ActionManager& HighLowTradingState::GetActionManager() const {
  return GetGame()->GetActionManager();
}

std::string HighLowTradingState::PublicInformationString() const {
  std::ostringstream result;
  
  result << "********** Game Configuration **********\n";
  result << fmt::format("Steps per player: {}\n", GetGame()->GetStepsPerPlayer());
  result << fmt::format("Max contracts per trade: {}\n", GetGame()->GetMaxContractsPerTrade());
  result << fmt::format("Customer max size: {}\n", GetGame()->GetCustomerMaxSize());
  result << fmt::format("Max contract value: {}\n", GetGame()->GetMaxContractValue());
  result << fmt::format("Number of players: {}\n", GetGame()->GetNumPlayers());
  result << "****************************************\n\n";

  result << "********** Player Positions **********\n";
  for (int i = 0; i < NumPlayers(); ++i) {
    result << fmt::format("Player {} position: {}\n", i, player_positions_[i].ToString());
  }
  result << "**************************************\n\n";
  
  result << "********** Fills & Quotes **********\n";
  result << fmt::format("Number of fills: {}\n", order_fills_.size());
  for (auto fill : order_fills_) {
    result << fmt::format("Order fill: {}\n", fill.ToString());
  }
  for (auto quote : player_quotes_) {
    result << fmt::format("Player {} quote: {}\n", quote.first, quote.second.ToString());
  }
  result << "***********************************\n\n";

  result << "********** Current Market **********\n";
  result << fmt::format("{}\n", market_.ToString());
  return result.str();
}

std::string HighLowTradingState::ToString() const {
  std::ostringstream result;
  
  result << "********** Game setup **********\n";
  result << fmt::format("Contract values: {}, {}\n", contract_values_[0].contract_value_, contract_values_[1].contract_value_);
  result << fmt::format("Contract high settle: {}\n", contract_high_settle_.is_high_ ? "High" : "Low");
  result << fmt::format("Player permutation: {}\n", player_permutation_.ToString());
  for (int i = 0; i < NumPlayers(); ++i) {
    auto target_position = player_target_positions_[i];
    if (target_position == 0) {
      result << fmt::format("Player {} target position: No requirement\n", i);
    } else {
      result << fmt::format("Player {} target position: {}\n", i, target_position);
    }
  }
  result << "********************************\n\n";
  result << PublicInformationString();
  return result.str();
}

bool HighLowTradingState::IsTerminal() const {
  return MoveNumber() >= game_->MaxGameLength(); 
}

int HighLowTradingState::GetContractValue() const {
  ASTRA_CHECK_GE(MoveNumber(), 3); 
  if (contract_high_settle_.is_high_) {
    return std::max(contract_values_[1].contract_value_, contract_values_[0].contract_value_); 
  } else {
    return std::min(contract_values_[1].contract_value_, contract_values_[0].contract_value_); 
  }
}

// We model instantaneous reward as (portfolio value at terminal timestep)
// To provide more granular rewards, each contract fill movement in the direction 
// of the target position is rewarded with "max_contract_value" reward. 
// Harmful moves are penalized
std::vector<RewardType> HighLowTradingState::Rewards() const {
  std::vector<RewardType> rewards; 
  rewards.resize(NumPlayers()); 

  int max_contract_value = GetGame()->GetMaxContractValue(); 
  RewardType contract_value = GetContractValue(); 

  if (IsTerminal()) {
    for (int j=0; j<NumPlayers(); ++j) {
      // Return execution fluency for each 
      RewardType portfolio_value = player_positions_[j].cash_balance + player_positions_[j].num_contracts * contract_value;
      rewards[j] = portfolio_value;
    }
  } else {
    for (int j=0; j<NumPlayers(); ++j) {
      if (player_target_positions_[j] != 0) {
        int previous_position = player_positions_[j].num_contracts - player_position_delta_[j]; 
        int previous_diff = std::abs(previous_position - player_target_positions_[j]); 
        int current_diff = std::abs(player_positions_[j].num_contracts - player_target_positions_[j]); 
        rewards[j] = (previous_diff - current_diff) * max_contract_value; 
      } else {
        rewards[j] = 0; 
      }
    }
  }
  return rewards; 
}

// Returns is modeled as (customer penalty) + portfolio value
std::vector<RewardType> HighLowTradingState::Returns() const {
  std::vector<RewardType> returns; 
  RewardType contract_value = GetContractValue(); 
  returns.resize(NumPlayers()); 
  int cumulative_customer_penalty = 0; 

  // First calculate for customers
  for (int j=0; j<NumPlayers(); ++j) {
    if (player_target_positions_[j] != 0) {
      RewardType portfolio_value = player_positions_[j].cash_balance + player_positions_[j].num_contracts * contract_value; 

      // Customers are additionally evaluated on whether they're able to obtain their target position. 
      int position_diff = player_target_positions_[j] - player_positions_[j].num_contracts; 
      int customer_penalty = std::abs(position_diff) * GetGame()->GetMaxContractValue(); 
      returns[j] = portfolio_value - customer_penalty; 
      cumulative_customer_penalty += customer_penalty; 
    }
  }
  // Then calculate for private-info cheaters. They equally share the customer penalty as returns, to keep the game zero-sum
  for (int j=0; j<NumPlayers(); ++j) {
    if (player_target_positions_[j] == 0) {
      RewardType portfolio_value = player_positions_[j].cash_balance + player_positions_[j].num_contracts * contract_value;
      returns[j] = portfolio_value;
    }
  }
  return returns; 
}

ExposeInfo HighLowTradingState::expose_info() const {
  if (!IsTerminal()) {
    AstraFatalError("ExposeInfo called on non-terminal state");
  }
  ExposeInfo info; 
  
  int min_value, max_value;
  min_value = std::min(contract_values_[0].contract_value_, contract_values_[1].contract_value_); 
  max_value = std::max(contract_values_[0].contract_value_, contract_values_[1].contract_value_); 
  info["contract"] = torch::zeros({3}, torch::kInt32); 
  auto contract_tensor = std::get<torch::Tensor>(info["contract"]);
  auto contract_accessor = contract_tensor.accessor<int, 1>();
  contract_accessor[0] = min_value; 
  contract_accessor[1] = max_value; 
  contract_accessor[2] = GetContractValue(); 

  info["players"] = player_contract_over_time_; 
  info["market"] = market_contract_over_time_; 

  auto returns = Returns(); 
  info["total_return"] = std::reduce(returns.begin(), returns.end());
  
  // Environment tensor: [contract_value1, contract_value2, settlement_value, customer_sizes...]
  // Size: 3 + number_of_customers = 3 + (NumPlayers() - 3) = NumPlayers()
  info["environment"] = torch::zeros({NumPlayers()}, torch::kInt32); 
  auto environment_tensor = std::get<torch::Tensor>(info["environment"]);
  auto environment_accessor = environment_tensor.accessor<int, 1>();
  environment_accessor[0] = contract_values_[0].contract_value_; 
  environment_accessor[1] = contract_values_[1].contract_value_; 
  environment_accessor[2] = GetContractValue(); 
  
  // For each customer role (starting from role 3), get their target position
  for (int role_id = 3; role_id < NumPlayers(); ++role_id) {
    int customer_player_id = player_permutation_.inv_permutation_[role_id]; 
    environment_accessor[role_id] = player_target_positions_[customer_player_id]; 
  }

  info["target_positions"] = torch::zeros({NumPlayers()}, torch::kInt32); 
  auto target_positions_tensor = std::get<torch::Tensor>(info["target_positions"]);
  auto target_positions_accessor = target_positions_tensor.accessor<int, 1>();
  for (int i = 0; i < NumPlayers(); ++i) {
    target_positions_accessor[i] = player_target_positions_[i]; 
  }
  
  // 0, 1, 2, 3 = goodValue, badValue, highLow, customer
  info["info_roles"] = torch::zeros({NumPlayers()}, torch::kInt32); 
  auto info_roles_tensor = std::get<torch::Tensor>(info["info_roles"]);
  auto info_roles_accessor = info_roles_tensor.accessor<int, 1>();
  
  // Only assign roles if we're past the chance phase
  for (int i = 0; i < NumPlayers(); ++i) {
    int perm_id = player_permutation_.permutation_[i]; 
    if (perm_id == 0 || perm_id == 1) {
      if (contract_values_[perm_id].contract_value_ == GetContractValue()) {
        info_roles_accessor[i] = 0; // goodValue
      } else {
        info_roles_accessor[i] = 1; // badValue
      }
    } else if (perm_id == 2) {
      info_roles_accessor[i] = 2; // highLow
    } else {
      info_roles_accessor[i] = 3; // customer
    }
  }
  
  return info; 
}

std::unique_ptr<State> HighLowTradingState::Clone() const {
  return std::unique_ptr<State>(new HighLowTradingState(*this));
}

void HighLowTradingState::UndoAction(Player player, Action move) {
  // Assert that the player and move match the top element of history
  ASTRA_CHECK_FALSE(history_.empty());
  ASTRA_CHECK_EQ(history_.back().player, player);
  ASTRA_CHECK_EQ(history_.back().action, move);
  
  // Save the history without the last action
  std::vector<PlayerAction> saved_history = history_;
  saved_history.pop_back();
  
  // Reset state to initial values
  contract_values_ = {ChanceContractValueAction(0), ChanceContractValueAction(0)};
  contract_high_settle_ = ChanceHighLowAction(false);
  player_permutation_ = ChancePermutationAction(NumPlayers());
  player_target_positions_.assign(NumPlayers(), 0);
  player_positions_.assign(NumPlayers(), PlayerPosition());
  market_ = order_matching::Market();
  order_fills_.clear();
  player_quotes_.clear();
  
  // Reset base state tracking
  history_.clear();
  move_number_ = 0;
  
  // Replay all actions from saved history
  for (const auto& player_action : saved_history) {
    DoApplyAction(player_action.action);
    history_.push_back(player_action);
    ++move_number_;
  }
}

std::vector<std::pair<Action, double>> HighLowTradingState::ChanceOutcomes() const {
  ASTRA_CHECK_TRUE(IsChanceNode());
  auto [min_action, max_action] = GetActionManager().valid_action_range(
      GetActionManager().game_phase_of_timestep(MoveNumber()));
  std::vector<std::pair<Action, double>> outcomes;
  int num_actions = max_action - min_action + 1;
  double prob = 1.0 / num_actions;
  for (int action = min_action; action <= max_action; ++action) {
    outcomes.push_back({action, prob});
  }
  return outcomes;
}

std::unique_ptr<State> HighLowTradingState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
  std::unique_ptr<State> state = game_->NewInitialState();
  return state;
}

std::vector<int> HighLowTradingGame::InformationStateTensorShape() const {
  // See `high_low_trading.h` for what each entry means 
  return {11 + GetStepsPerPlayer() * GetNumPlayers() * 6 + GetNumPlayers() * 2};
}

void HighLowTradingState::InformationStateTensor(Player player,
  std::span<ObservationScalarType> values) const {
  ASTRA_CHECK_GE(player, 0);
  ASTRA_CHECK_LT(player, GetGame()->NumPlayers());
  std::fill(values.begin(), values.end(), 0);
  int offset = 0;
  
  // 1. Game setup (5): [num_steps, max_contracts_per_trade, customer_max_size, max_contract_value, players]
  values[offset++] = static_cast<ObservationScalarType>(GetGame()->GetStepsPerPlayer());
  values[offset++] = static_cast<ObservationScalarType>(GetGame()->GetMaxContractsPerTrade());
  values[offset++] = static_cast<ObservationScalarType>(GetGame()->GetCustomerMaxSize());
  values[offset++] = static_cast<ObservationScalarType>(GetGame()->GetMaxContractValue());
  values[offset++] = static_cast<ObservationScalarType>(GetGame()->GetNumPlayers());
  
  // 2. One-hot player role (3): [is_value, is_maxmin, is_customer]
  if (MoveNumber() >= GetGame()->MaxChanceNodesInHistory()) {
    int perm_id = player_permutation_.permutation_[player]; 
    if (perm_id == 0 || perm_id == 1) {
      values[offset] = 1.0; // is_value
    } else if (perm_id == 2) {
      values[offset + 1] = 1.0; // is_maxmin (high/low cheater)
    } else {
      values[offset + 2] = 1.0; // is_customer
    }
  }
  offset += 3;
  
  // 3. Player id (2): [sin(2 pi player_id / players), cos(2 pi player_id / players)]
  double angle = 2.0 * M_PI * player / GetGame()->GetNumPlayers();
  values[offset++] = static_cast<ObservationScalarType>(std::sin(angle));
  values[offset++] = static_cast<ObservationScalarType>(std::cos(angle));
  
  // 4. Private information (1): [contract value, max / min, customer target size]
  if (MoveNumber() >= GetGame()->MaxChanceNodesInHistory()) {
    int perm_id = player_permutation_.permutation_[player]; 
    if (perm_id == 0 || perm_id == 1) {
      values[offset] = static_cast<ObservationScalarType>(contract_values_[perm_id].contract_value_);
    } else if (perm_id == 2) {
      values[offset] = contract_high_settle_.is_high_ ? 1.0 : -1.0;
    } else {
      values[offset] = static_cast<ObservationScalarType>(player_target_positions_[player]);
    }
  }
  offset += 1;
  
  // 5. Positions (num_players, 2): [num_contracts, cash_position] - fixed length first
  int num_players = GetGame()->GetNumPlayers();
  for (int p = 0; p < num_players; ++p) {
    values[offset++] = static_cast<ObservationScalarType>(player_positions_[p].num_contracts);
    values[offset++] = static_cast<ObservationScalarType>(player_positions_[p].cash_balance);
  }
  
  // 6. Quotes: [bid_px, ask_px, bid_sz, ask_sz, *player_id] - fill remaining space
  for (int quote_idx = 0; quote_idx < static_cast<int>(player_quotes_.size()); ++quote_idx) {
    const auto& quote_pair = player_quotes_[quote_idx];
    int acting_player = quote_pair.first;
    const auto& quote = quote_pair.second;
    
    // Assert we have enough space (InformationStateTensorShape should guarantee this)
    ASTRA_CHECK_LE(offset + 6, static_cast<int>(values.size()));
    
    values[offset++] = static_cast<ObservationScalarType>(quote.bid_price_);
    values[offset++] = static_cast<ObservationScalarType>(quote.ask_price_);
    values[offset++] = static_cast<ObservationScalarType>(quote.bid_size_);
    values[offset++] = static_cast<ObservationScalarType>(quote.ask_size_);
    
    // Player id as sin/cos
    double p_angle = 2.0 * M_PI * acting_player / num_players;
    values[offset++] = static_cast<ObservationScalarType>(std::sin(p_angle));
    values[offset++] = static_cast<ObservationScalarType>(std::cos(p_angle));
  }
}

std::string HighLowTradingState::InformationStateString(Player player) const {
  ASTRA_CHECK_GE(player, 0);
  ASTRA_CHECK_LT(player, GetGame()->NumPlayers());
  
  std::ostringstream result;
  
  // Add player's role information
  result << "********** Private Information **********\n";

  
  // Check if we're past the permutation phase
  if (MoveNumber() >= GetGame()->MaxChanceNodesInHistory()) {
    int perm_id = player_permutation_.permutation_[player]; 
    std::string role_name;

    if (perm_id == 0 || perm_id == 1) {
      role_name = "ValueCheater";
    } else if (perm_id == 2) {
      role_name = "HighLowCheater";
    } else {
      role_name = "Customer";
    }
    result << fmt::format("My role: {}\n", role_name);
    
    // Add private information based on role
    if (perm_id == 0 || perm_id == 1) {
        // ValueCheaters know the contract values
        result << fmt::format("Candidate contract value: {}\n",
                             contract_values_[perm_id].contract_value_);
    } else if (perm_id == 2) {
        // HighLowCheaters know which settlement (high or low) will be chosen
        result << fmt::format("Settlement will be: {}\n",
                             contract_high_settle_.is_high_ ? "High" : "Low");
    } else {
        // Customers know their target position
        auto target_position = player_target_positions_[player];
        if (target_position != 0) {
          result << fmt::format("My target position: {}\n", target_position);
        } else {
          result << "Not supposed to happen. Customer target position should not be 0 \n"; 
          AstraFatalError("Not supposed to happen. Customer roles should be assigned"); 
        }
    }
    // Start with public information that all players can see
    result << PublicInformationString();
    
  } else {
    result << "Private info pending...\n";
  }
  
  result << "***************************\n";
  
  return result.str();
}

// Observations are exactly the info states. Preserve Markov condition. 
std::vector<int> HighLowTradingGame::ObservationTensorShape() const {
  return InformationStateTensorShape(); 
}

std::string HighLowTradingState::ObservationString(Player player) const {
  return InformationStateString(player); 
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

int HighLowTradingGame::MaxChanceOutcomes() const {
  return std::max({
    GetActionManager().valid_action_range(GamePhase::kChanceValue).second + 1, 
    GetActionManager().valid_action_range(GamePhase::kChanceHighLow).second + 1, 
    GetActionManager().valid_action_range(GamePhase::kChancePermutation).second + 1, 
    GetActionManager().valid_action_range(GamePhase::kCustomerSize).second + 1, 
  }) + 1; 
}

int HighLowTradingGame::NumDistinctActions() const {
  return GetActionManager().valid_action_range(GamePhase::kPlayerTrading).second + 1; 
}

}  // namespace high_low_trading
}  // namespace astra
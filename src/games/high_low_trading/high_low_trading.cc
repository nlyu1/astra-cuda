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
    player_last_positions_ = torch::zeros({num_envs_, num_players_, 2}, options_i32);
    
    // Initialize tracking tensors for ExposeInfo
    player_contract_over_time_ = torch::zeros({num_envs_, num_players_, steps_per_player_, 6}, options_i32);
    market_contract_over_time_ = torch::zeros({num_envs_, num_players_ * steps_per_player_, 3}, options_i32); // best bid, best ask, last_price

    // Initialize reward tensors
    immediate_rewards_ = torch::zeros({num_envs_, num_players_}, options_i32);
    rewards_since_last_action_ = torch::zeros({num_envs_, num_players_}, options_i32);
    
    // Initialize BBO and Fill batches
    bbo_batch_ = market_.NewBBOBatch(); 
    fill_batch_ = market_.NewFillBatch(); 
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
      player_last_positions_(other.player_last_positions_.clone()),
      market_(other.market_),
      player_contract_over_time_(other.player_contract_over_time_.clone()),
      market_contract_over_time_(other.market_contract_over_time_.clone()),
      immediate_rewards_(other.immediate_rewards_.clone()),
      rewards_since_last_action_(other.rewards_since_last_action_.clone()),
      bbo_batch_(other.bbo_batch_),  
      fill_batch_(other.fill_batch_) { 
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

  // Assert that they are between 
  auto abs_move = torch::abs(move); 
  ASTRA_CHECK_LE(torch::max(abs_move).item<int>(), GetGame()->GetCustomerMaxSize()); 
  ASTRA_CHECK_GT(torch::min(abs_move).item<int>(), 0); // Customers cannot have zero size
  
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
    ApplyPlayerTrading(move); 
  }
}

void HighLowTradingState::ApplyPlayerTrading(torch::Tensor move) {
  Player player = CurrentPlayer(); 
  int trade_move_number = MoveNumber() - game_->MaxChanceNodesInHistory(); 
  int player_move_number = trade_move_number / num_players_;  // Which round/step this player is in
  ASTRA_CHECK_GE(trade_move_number, 0); 
  ASTRA_CHECK_LT(trade_move_number, steps_per_player_ * num_players_);
  ASTRA_CHECK_EQ(move.size(1), 4); // bid_px, ask_px, bid_sz, ask_sz
  
  // Record last positions
  market_.CopyCustomerPortfoliosTo(player_last_positions_); 
  
  // Extract columns from move tensor and make them contiguous
  auto bid_prices = move.index({torch::indexing::Slice(), 0}).contiguous();
  auto ask_prices = move.index({torch::indexing::Slice(), 1}).contiguous();
  auto bid_sizes = move.index({torch::indexing::Slice(), 2}).contiguous();
  auto ask_sizes = move.index({torch::indexing::Slice(), 3}).contiguous();
  auto customer_ids = torch::ones({num_envs_}, torch::kInt32).to(torch::Device(torch::kCUDA, device_id_)) * player;
  ASTRA_CHECK_GT(torch::min(bid_prices).item<int>(), 0);
  ASTRA_CHECK_GT(torch::min(ask_prices).item<int>(), 0);
  ASTRA_CHECK_LE(torch::max(bid_prices).item<int>(), GetGame()->GetMaxContractValue());
  ASTRA_CHECK_LE(torch::max(ask_prices).item<int>(), GetGame()->GetMaxContractValue());
  ASTRA_CHECK_GE(torch::min(bid_sizes).item<int>(), 0);
  ASTRA_CHECK_GE(torch::min(ask_sizes).item<int>(), 0);
  ASTRA_CHECK_LE(torch::max(bid_sizes).item<int>(), GetGame()->GetMaxContractsPerTrade());
  ASTRA_CHECK_LE(torch::max(ask_sizes).item<int>(), GetGame()->GetMaxContractsPerTrade());

  // Compute the fills 
  market_.AddTwoSidedQuotes(
    bid_prices, bid_sizes, ask_prices, ask_sizes, customer_ids, fill_batch_);
  
  // Flush current player's accumulated rewards since last action
  rewards_since_last_action_.index({torch::indexing::Slice(), player}).zero_();

  // Compute new immediate rewards
  auto current_positions = market_.GetCustomerPortfolios(); 
  market_.GetBBOs(bbo_batch_); 
  bbo_batch_.UpdateLastPrices(fill_batch_); // Update last prices based on fills
  auto cash_diff = current_positions.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}) - 
    player_last_positions_.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}); // [N, P]
  // Update logging metrics for tracking market movement
  player_contract_over_time_.index({torch::indexing::Slice(), player, player_move_number, torch::indexing::Slice(0, 4)}).copy_(move);
  player_contract_over_time_.index({torch::indexing::Slice(), torch::indexing::Slice(), player_move_number, torch::indexing::Slice(4, torch::indexing::None)}).copy_(current_positions);
  market_contract_over_time_.index({torch::indexing::Slice(), trade_move_number, 0}).copy_(bbo_batch_.best_bid_prices);
  market_contract_over_time_.index({torch::indexing::Slice(), trade_move_number, 1}).copy_(bbo_batch_.best_ask_prices);
  market_contract_over_time_.index({torch::indexing::Slice(), trade_move_number, 2}).copy_(bbo_batch_.last_prices);

  // Compute how much closer we are towards the final target position
  auto previous_positions = player_last_positions_.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}); // [N, P]
  auto current_positions_contracts = current_positions.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}); // [N, P]
  
  auto previous_diff = torch::abs(previous_positions - target_positions_); // [N, P]
  auto current_diff = torch::abs(current_positions_contracts - target_positions_); // [N, P]
  auto is_customer = target_positions_ != 0; // [N, P]
  
  // Positive reward if current_diff <= previous_diff
  immediate_rewards_ = is_customer * (previous_diff - current_diff) * GetGame()->GetMaxContractValue(); 
  immediate_rewards_.add_(cash_diff); // Add immediate cash value (in-place)

  if (IsTerminal()) {
    // At termination, add contract settlement value
    immediate_rewards_.addcmul_( // In-place multiply-add: += contracts * settlement_value
      current_positions_contracts,
      contract_values_.index({torch::indexing::Slice(), 2}).unsqueeze(-1)
    );
  }

  // Update rewards since last action for all players
  rewards_since_last_action_.add_(immediate_rewards_); // In-place add 
}

void HighLowTradingState::FillRewards(torch::Tensor reward_buffer) const {
  ASTRA_CHECK_EQ(reward_buffer.dim(), 2); 
  ASTRA_CHECK_EQ(reward_buffer.size(0), num_envs_); 
  ASTRA_CHECK_EQ(reward_buffer.size(1), num_players_); 
  reward_buffer.copy_(immediate_rewards_); 
}

void HighLowTradingState::FillRewardsSinceLastAction(
    torch::Tensor reward_buffer, Player player_id) const {
  ASTRA_CHECK_EQ(reward_buffer.dim(), 1); 
  ASTRA_CHECK_EQ(reward_buffer.size(0), num_envs_); 
  reward_buffer.copy_(rewards_since_last_action_.index({torch::indexing::Slice(), player_id})); 
}

// Calculate terminal returns as portfolio value minus customer penalties
void HighLowTradingState::FillReturns(torch::Tensor returns_buffer) const {
  ASTRA_CHECK_EQ(returns_buffer.dim(), 2); 
  ASTRA_CHECK_EQ(returns_buffer.size(0), num_envs_); 
  ASTRA_CHECK_EQ(returns_buffer.size(1), num_players_); 
  
  returns_buffer.zero_();
  
  // Get current portfolios and customer indicators
  auto is_customer = target_positions_ != 0; // [N, P]
  auto portfolios = market_.GetCustomerPortfolios(); // [N, P, 2]
  
  // Add portfolio value: cash + (contracts * settlement_value)
  returns_buffer.add_(portfolios.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}));
  returns_buffer.addcmul_( // In-place multiply-add: returns += contracts * settlement_value
    portfolios.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}), // contracts
    contract_values_.index({torch::indexing::Slice(), 2}).unsqueeze(-1) // settlement value [N, 1]
  );
  
  // Subtract penalty for missed target positions (customers only)
  auto position_diff = torch::abs(
    portfolios.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}) - target_positions_);
  // returns += -is_customer * position_diff * max_value
  // Convert bool to int to allow negation
  returns_buffer.add_(
    -is_customer.to(torch::kInt32) * position_diff * GetGame()->GetMaxContractValue());
}

void HighLowTradingState::DoReset() {
  // Debug: Check if tensors are properly initialized
  if (!contract_values_.defined()) {
    AstraFatalError("contract_values_ is not defined in DoReset!");
  }
  
  contract_values_.zero_();
  contract_high_settle_.zero_();
  player_permutation_.zero_();
  inv_permutation_.zero_();
  target_positions_.zero_();
  player_last_positions_.zero_();
  player_contract_over_time_.zero_();
  market_contract_over_time_.zero_();
  immediate_rewards_.zero_();
  rewards_since_last_action_.zero_();
  market_.Reset();
  bbo_batch_.Reset();
  fill_batch_.Reset(); 
}

const HighLowTradingGame* HighLowTradingState::GetGame() const {
  return static_cast<const HighLowTradingGame*>(game_.get()); 
}

std::string HighLowTradingState::ToString(int32_t index) const {
  std::ostringstream result;
  result << "********** Game setup **********\n";
  
  // Get contract values from tensor
  if (MoveNumber() >= 1) {
    auto contract_values_cpu = contract_values_.cpu();
    auto contract_values_accessor = contract_values_cpu.accessor<int32_t, 2>();
    result << fmt::format("Contract values: {}, {}\n", 
                         contract_values_accessor[index][0], 
                         contract_values_accessor[index][1]);
  }
  
  // Get high/low settlement choice
  if (MoveNumber() >= 2) {
    auto contract_high_settle_cpu = contract_high_settle_.cpu();
    auto contract_high_settle_accessor = contract_high_settle_cpu.accessor<bool, 1>();
    result << fmt::format("Contract high settle: {}\n", 
                         contract_high_settle_accessor[index] ? "High" : "Low");
  }
  
  // Get player permutation
  if (MoveNumber() >= 3) {
    auto player_permutation_cpu = player_permutation_.cpu();
    auto player_permutation_accessor = player_permutation_cpu.accessor<int32_t, 2>();
    result << "Player permutation: Player roles: ";
    for (int i = 0; i < num_players_; ++i) {
      int role = player_permutation_accessor[index][i];
      if (role == 0 || role == 1) {
        result << fmt::format("P{}=ValueCheater, ", i);
      } else if (role == 2) {
        result << fmt::format("P{}=HighLowCheater, ", i);
      } else {
        result << fmt::format("P{}=Customer, ", i);
      }
    }
    result.seekp(-2, std::ios_base::end); // Remove trailing ", "
    result << "\n";
  }
  
  // Get target positions
  if (MoveNumber() >= 4) {
    auto target_positions_cpu = target_positions_.cpu();
    auto target_positions_accessor = target_positions_cpu.accessor<int32_t, 2>();
    for (int i = 0; i < num_players_; ++i) {
      int target_position = target_positions_accessor[index][i];
      if (target_position == 0) {
        result << fmt::format("Player {} target position: No requirement\n", i);
      } else {
        result << fmt::format("Player {} target position: {}\n", i, target_position);
      }
    }
  }
  
  result << "********************************\n\n";
  
  // Add public information if we're past the chance moves
  if (MoveNumber() >= game_->MaxChanceNodesInHistory()) {
    result << PublicInformationString(index);
  }
  
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


ExposeInfo HighLowTradingState::expose_info() const {
  if (!IsTerminal()) {
    AstraFatalError("ExposeInfo called on non-terminal state");
  }
  ExposeInfo info; 

  // info["contract"] = contract_values_.clone();
  info["players"] = player_contract_over_time_.clone();
  info["market"] = market_contract_over_time_.clone();
  // info["permutation"] = player_permutation_.clone();
  // info["target_positions"] = target_positions_.clone();
  return info; 
}

std::unique_ptr<State> HighLowTradingState::Clone() const {
  return std::unique_ptr<State>(new HighLowTradingState(*this));
}

std::vector<int64_t> HighLowTradingGame::InformationStateTensorShape() const {
  AstraFatalError("InformationStateTensorShape not implemented");
}

void HighLowTradingState::FillInformationStateTensor(Player player,
  torch::Tensor values) const {
  ASTRA_CHECK_GE(player, 0);
  ASTRA_CHECK_LT(player, GetGame()->NumPlayers());
  AstraFatalError("FillInformationStateTensor not implemented");
}

std::string HighLowTradingState::ObservationString(Player player, int32_t index) const {
  return InformationStateString(player, index);
}

std::vector<int64_t> HighLowTradingGame::ObservationTensorShape() const {
  return {static_cast<int64_t>(num_markets_), static_cast<int64_t>(6 + 1 + 6 * GetNumPlayers() + 5)}; 
}

void HighLowTradingState::FillObservationTensor(Player player,
  torch::Tensor values) const {
    ASTRA_CHECK_GE(player, 0);
    ASTRA_CHECK_LT(player, GetGame()->NumPlayers());
    ASTRA_CHECK_GE(MoveNumber(), GetGame()->MaxChanceNodesInHistory());
    values.fill_(0);
    
    // Check tensor shape - use sizes() instead of shape()
    auto expected_shape = GetGame()->ObservationTensorShape();
    ASTRA_CHECK_EQ(values.sizes().size(), expected_shape.size());
    for (size_t i = 0; i < expected_shape.size(); ++i) {
        ASTRA_CHECK_EQ(values.size(i), expected_shape[i]);
    }
    int offset = 0; 
    
    // Fill player role, id, and private information
    // Get permutation IDs for this player across all environments
    torch::Tensor perm_ids = player_permutation_.index({torch::indexing::Slice(), player});
    // Create boolean masks
    torch::Tensor value_cheater_mask = (perm_ids == 0) | (perm_ids == 1);
    torch::Tensor high_low_cheater_mask = (perm_ids == 2);
    torch::Tensor customer_mask = (perm_ids > 2);
    // Fill role indicators
    values.index({torch::indexing::Slice(), offset}) = value_cheater_mask.to(torch::kFloat32);
    values.index({torch::indexing::Slice(), offset + 1}) = high_low_cheater_mask.to(torch::kFloat32);
    values.index({torch::indexing::Slice(), offset + 2}) = customer_mask.to(torch::kFloat32);
    // Fill player ID encoding
    double angle = 2.0 * M_PI * player / GetGame()->GetNumPlayers();
    values.index({torch::indexing::Slice(), offset + 3}).fill_(std::sin(angle));
    values.index({torch::indexing::Slice(), offset + 4}).fill_(std::cos(angle));
    // Fill private information based on role
    torch::Tensor mask_val0 = (perm_ids == 0);
    values.index_put_({mask_val0, offset + 5}, 
                      contract_values_.index({mask_val0, 0}).to(torch::kFloat32));
    torch::Tensor mask_val1 = (perm_ids == 1);
    values.index_put_({mask_val1, offset + 5}, 
                      contract_values_.index({mask_val1, 1}).to(torch::kFloat32));
    values.index_put_({high_low_cheater_mask, offset + 5}, 
                      contract_high_settle_.index({high_low_cheater_mask}).to(torch::kFloat32));
    values.index_put_({customer_mask, offset + 5}, 
                      target_positions_.index({customer_mask, player}).to(torch::kFloat32));

    int current_trade_move = std::max(0, MoveNumber() - GetGame()->MaxChanceNodesInHistory());
    int current_round = current_trade_move / num_players_;
    values.index_put_({torch::indexing::Slice(), offset + 6}, 
                      torch::ones({num_envs_}).to(torch::kFloat32) * current_round / steps_per_player_);
    offset += 7;
    
    // All player's quotes and positions
    torch::Tensor current_round_data = player_contract_over_time_.index({
        torch::indexing::Slice(), 
        torch::indexing::Slice(), 
        current_round, 
        torch::indexing::Slice()
    }).reshape({num_envs_, num_players_ * 6});
    values.index({torch::indexing::Slice(), 
                  torch::indexing::Slice(offset, offset + 6 * num_players_)}) = 
        current_round_data.to(torch::kFloat32);
    offset += 6 * num_players_;
    
    // BBOs and last trade prices
    values.index({torch::indexing::Slice(), offset}) = 
        bbo_batch_.best_bid_prices.to(torch::kFloat32);
    values.index({torch::indexing::Slice(), offset + 1}) = 
        bbo_batch_.best_ask_prices.to(torch::kFloat32);
    values.index({torch::indexing::Slice(), offset + 2}) = 
        bbo_batch_.best_bid_sizes.to(torch::kFloat32);
    values.index({torch::indexing::Slice(), offset + 3}) = 
        bbo_batch_.best_ask_sizes.to(torch::kFloat32);
    values.index({torch::indexing::Slice(), offset + 4}) = 
        bbo_batch_.last_prices.to(torch::kFloat32);
    offset += 5;
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

std::string HighLowTradingState::InformationStateString(Player player, int32_t index) const {
  ASTRA_CHECK_GE(player, 0);
  ASTRA_CHECK_LT(player, GetGame()->NumPlayers());
  
  std::ostringstream result;
  
  // Add player's role information
  result << "********** Private Information **********\n";
  
  // Check if we're past the permutation phase
  if (MoveNumber() >= GetGame()->MaxChanceNodesInHistory()) {
    // Get player's role from permutation
    auto player_permutation_cpu = player_permutation_.cpu();
    auto player_permutation_accessor = player_permutation_cpu.accessor<int32_t, 2>();
    int perm_id = player_permutation_accessor[index][player];
    
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
      // ValueCheaters know one of the contract values
      auto contract_values_cpu = contract_values_.cpu();
      auto contract_values_accessor = contract_values_cpu.accessor<int32_t, 2>();
      result << fmt::format("Candidate contract value: {}\n",
                           contract_values_accessor[index][perm_id]);
    } else if (perm_id == 2) {
      // HighLowCheaters know which settlement (high or low) will be chosen
      auto contract_high_settle_cpu = contract_high_settle_.cpu();
      auto contract_high_settle_accessor = contract_high_settle_cpu.accessor<bool, 1>();
      result << fmt::format("Settlement will be: {}\n",
                           contract_high_settle_accessor[index] ? "High" : "Low");
    } else {
      // Customers know their target position
      auto target_positions_cpu = target_positions_.cpu();
      auto target_positions_accessor = target_positions_cpu.accessor<int32_t, 2>();
      int target_position = target_positions_accessor[index][player];
      if (target_position != 0) {
        result << fmt::format("My target position: {}\n", target_position);
      } else {
        result << "Not supposed to happen. Customer target position should not be 0\n";
        AstraFatalError("Not supposed to happen. Customer roles should be assigned");
      }
    }
    result << "******************************************\n\n";
    
    // Add public information that all players can see
    result << PublicInformationString(index);
    
  } else {
    result << "Private info pending...\n";
    result << "***************************\n";
  }
  
  return result.str();
}

std::string HighLowTradingState::PublicInformationString(int32_t index) const {
  std::ostringstream result;
  result << "********** Game Configuration **********\n";
  result << fmt::format("Environment: {} / {}\n", index, num_envs_);
  result << fmt::format("Steps per player: {}\n", steps_per_player_);
  result << fmt::format("Max contracts per trade: {}\n", GetGame()->GetMaxContractsPerTrade());
  result << fmt::format("Customer max size: {}\n", GetGame()->GetCustomerMaxSize());
  result << fmt::format("Max contract value: {}\n", GetGame()->GetMaxContractValue()); 
  result << fmt::format("Number of players: {}\n", num_players_);
  result << "****************************************\n\n";

  result << "********** Player Positions **********\n";
  // Get current positions from market
  auto portfolios = market_.GetCustomerPortfolios(); // [N, P, 2]
  auto portfolios_cpu = portfolios.cpu();
  auto portfolios_accessor = portfolios_cpu.accessor<int32_t, 3>();
  
  for (int i = 0; i < num_players_; ++i) {
    int contracts = portfolios_accessor[index][i][0];
    int cash = portfolios_accessor[index][i][1];
    result << fmt::format("Player {} position: [{} contracts, {} cash]\n", i, contracts, cash);
  }
  result << "**************************************\n\n";
  
  result << "********** Quotes & Market Movement **********\n";
  // Get fills from fill_batch_
  if (fill_batch_.fill_counts.defined()) {
    auto fill_counts_cpu = fill_batch_.fill_counts.cpu();
    auto fill_counts_accessor = fill_counts_cpu.accessor<int32_t, 1>();
    int32_t num_fills_for_env = fill_counts_accessor[index];
    
    result << fmt::format("Number of fills in the last round: {}\n", num_fills_for_env);
    
    if (num_fills_for_env > 0) {
      // Access fill data - these are 2D tensors [num_markets, max_fills]
      auto fill_prices_cpu = fill_batch_.fill_prices.cpu();
      auto fill_sizes_cpu = fill_batch_.fill_sizes.cpu();
      auto fill_customer_ids_cpu = fill_batch_.fill_customer_ids.cpu();
      auto fill_quoter_ids_cpu = fill_batch_.fill_quoter_ids.cpu();
      auto fill_is_sell_quote_cpu = fill_batch_.fill_is_sell_quote.cpu();
      auto fill_quote_sizes_cpu = fill_batch_.fill_quote_sizes.cpu();
      auto fill_tid_cpu = fill_batch_.fill_tid.cpu();
      auto fill_quote_tid_cpu = fill_batch_.fill_quote_tid.cpu();
      
      auto fill_prices_accessor = fill_prices_cpu.accessor<int32_t, 2>();
      auto fill_sizes_accessor = fill_sizes_cpu.accessor<int32_t, 2>();
      auto fill_customer_ids_accessor = fill_customer_ids_cpu.accessor<int32_t, 2>();
      auto fill_quoter_ids_accessor = fill_quoter_ids_cpu.accessor<int32_t, 2>();
      auto fill_is_sell_quote_accessor = fill_is_sell_quote_cpu.accessor<bool, 2>();
      auto fill_quote_sizes_accessor = fill_quote_sizes_cpu.accessor<int32_t, 2>();
      auto fill_tid_accessor = fill_tid_cpu.accessor<int32_t, 2>();
      auto fill_quote_tid_accessor = fill_quote_tid_cpu.accessor<int32_t, 2>();
      
      // Iterate through fills for this environment
      for (int32_t i = 0; i < num_fills_for_env; ++i) {
        result << fmt::format(
          "Order fill: id={} {} {} contracts at px={} on t={}. Quote: id={} {} sz={}, submitted t={}\n",
          fill_customer_ids_accessor[index][i],
          fill_is_sell_quote_accessor[index][i] ? "bought" : "sold",
          fill_sizes_accessor[index][i],
          fill_prices_accessor[index][i],
          fill_tid_accessor[index][i],
          fill_quoter_ids_accessor[index][i],
          fill_is_sell_quote_accessor[index][i] ? "sale" : "bid",
          fill_quote_sizes_accessor[index][i],
          fill_quote_tid_accessor[index][i]
        );
      }
    }
  } else {
    result << "Number of fills in the last round: 0\n";
  }
  
  // Print historical quotes from player_contract_over_time_
  int current_trade_move = std::max(0, MoveNumber() - game_->MaxChanceNodesInHistory());
  int current_round = current_trade_move / num_players_;
  int player_in_round = current_trade_move % num_players_;
  
  auto player_contract_cpu = player_contract_over_time_.cpu();
  auto player_contract_accessor = player_contract_cpu.accessor<int32_t, 4>();
  
  for (int round = 0; round <= current_round && round < steps_per_player_; ++round) {
    int max_player = (round < current_round) ? num_players_ : player_in_round;
    for (int p = 0; p < max_player; ++p) {
      int bid_px = player_contract_accessor[index][p][round][0];
      int ask_px = player_contract_accessor[index][p][round][1];
      int bid_sz = player_contract_accessor[index][p][round][2];
      int ask_sz = player_contract_accessor[index][p][round][3];
      
      if (bid_px > 0 || ask_px > 0) {  // Only print if quote was actually placed
        result << fmt::format("Player {} quote: {} @ {} [{} x {}]\n", 
                            p, bid_px, ask_px, bid_sz, ask_sz);
      }
    }
  }
  
  // Print market movement over time
  result << "\n--- Market Movement ---\n";
  auto market_contract_cpu = market_contract_over_time_.cpu();
  auto market_contract_accessor = market_contract_cpu.accessor<int32_t, 3>();
  
  for (int t = 0; t < current_trade_move && t < (num_players_ * steps_per_player_); ++t) {
    int32_t best_bid = static_cast<int32_t>(market_contract_accessor[index][t][0]);
    int32_t best_ask = static_cast<int32_t>(market_contract_accessor[index][t][1]);
    int32_t last_price = static_cast<int32_t>(market_contract_accessor[index][t][2]);
    
    result << fmt::format("Time {}: Bid: {} @ Ask: {}{}\n", 
                         t,
                         best_bid == order_matching::NULL_INDEX ? "none" : std::to_string(best_bid),
                         best_ask == order_matching::NULL_INDEX ? "none" : std::to_string(best_ask),
                         last_price != order_matching::NULL_INDEX ? fmt::format(", Last: {}", last_price) : "");
  }
  
  result << "***********************************\n\n";

  result << "********** Current Market **********\n";
  result << market_.ToString(index);
  return result.str();
}

}  // namespace high_low_trading
}  // namespace astra
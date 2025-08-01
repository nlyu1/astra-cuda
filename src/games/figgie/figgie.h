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
namespace figgie {

inline constexpr int kDefaultStepsPerPlayer = 20; 
inline constexpr int kDefaultNumPlayers = 4;
inline constexpr int kDefaultNumMarkets = 32768;
inline constexpr int kDefaultThreadsPerBlock = 128;
inline constexpr int kDefaultDeviceId = 0; 
inline constexpr int kNumSuites = 4; 
inline constexpr int kAnte = 50; 

class FiggieGame;
// ID for different suites: (Clubs, Hearts, Spades, Diamonds) = (0, 1, 2, 3) so (black, red) are (even, odd). 

// Expose these for registration. 
extern const GameType kGameType;
std::shared_ptr<const Game> Factory(const GameParameters& params);

class FiggieState : public State {
 public:
  explicit FiggieState(std::shared_ptr<const Game> game);
  FiggieState(const FiggieState& other);

  Player CurrentPlayer() const override;
  std::string ToString(int32_t index) const override;
  bool IsTerminal() const override;
  void FillRewards(torch::Tensor reward_buffer) const override;
  void FillRewardsSinceLastAction(torch::Tensor reward_buffer, Player player_id) const override;
  void FillReturns(torch::Tensor returns_buffer) const override;
  std::string InformationStateString(Player player, int32_t index) const override;
  void FillInformationStateTensor(Player player, torch::Tensor values) const override;
  std::string ObservationString(Player player, int32_t index) const override;
  // Each player's observation tensor:
  // 1. Player id and private information (6)
  //    - Dealt hand [each suite], sin(id), cos(id). 
  // 2. Game progress ratio (1) 
  // 3. Players' quotes and positions (players * (5 * num_suites + 1)): 
  //    [[bid_px, ask_px, bid_sz, ask_sz, accumulated_position] per each suite, and [cash_position]]
  // 4. Market features (5 * num_suites): BBO (4) and last_trade_px
  void FillObservationTensor(Player player, torch::Tensor values) const override;
  std::unique_ptr<State> Clone() const override;
  
  ExposeInfo expose_info() const override;

  torch::Tensor GetContractValue() const; 
   
 protected:
  void DoApplyAction(torch::Tensor move) override;
  
  /**
   * @brief Resets the game state to initial conditions
   * 
   * Called by the public Reset() method after move_number_ is reset to 0.
   * This method resets all game-specific state including:
   * - All chance outcome tensors (contract values, high/low settlement, permutations)
   * - Player target positions
   * - Player last positions
   * - Tracking tensors for expose_info
   * - The underlying market (clears all orders and portfolios)
   * 
   * After calling DoReset(), the state is ready for a new game simulation
   * without needing to allocate new memory.
   */
  void DoReset() override;
  
  // Helper values to handle different kinds of actions 
  void ApplyPlayerTrading(torch::Tensor move);
  void ApplyInitialPortfolio(torch::Tensor move);
  const FiggieGame* GetGame() const;

 private:
  int num_envs_;
  int num_players_;
  int steps_per_player_;
  int device_id_; 
  std::string PublicInformationString(int32_t index) const; 
  /* N = num_envs, P=num_players, T=rounds_per_player, S=num_suites */
  torch::Tensor player_cash_position_; // [N, P, S+1]. Records each players' total position and cash position. 
  torch::Tensor player_last_cash_position_; // [N, P] player's cash position prior to last move (by any player)

  torch::Tensor immediate_rewards_; // [N, P]. Reward resulting from last action by any player
  torch::Tensor rewards_since_last_action_; // [N, P]. Reward since last action by self; includes immediate_rewards_ at all times

  order_matching::VecMarket market_; 

  // Purely for ExposeInfo uses
  torch::Tensor player_action_over_time_; // [N, P, T, S, 5] int for (bid_px, ask_px, bid_sz, ask_sz, ncards after quote) for each suite
  torch::Tensor market_action_over_time_; // [N, P*T, S, 3] int (best bid px, best ask px, last_price) for each suite

  // See `market.h` 
  order_matching::BBOBatch bbo_batch_; 
  order_matching::FillBatch fill_batch_; 
};

class FiggieGame : public Game {
  public:
    explicit FiggieGame(const GameParameters& params);
    std::unique_ptr<State> NewInitialState() const override;
    std::vector<int64_t> InformationStateTensorShape() const override;
    std::vector<int64_t> ObservationTensorShape() const override;

    int MaxGameLength() const override {
      return MaxChanceNodesInHistory() + GetStepsPerPlayer() * GetNumPlayers(); 
    }
    int MaxChanceNodesInHistory() const override {
      return 1; 
    }
    int NumPlayers() const override { return GetNumPlayers(); }

    int GetNumPlayers() const { return num_players_; }
    int GetStepsPerPlayer() const { return steps_per_player_; }
    int GetNumMarkets() const { return num_markets_; }
    int GetThreadsPerBlock() const { return threads_per_block_; }
    int GetDeviceId() const { return device_id_; }
  private:
    int steps_per_player_; 
    int num_players_; 
    int num_markets_;
    int threads_per_block_;
    int device_id_;
};

}  // namespace figgie
}  // namespace astra
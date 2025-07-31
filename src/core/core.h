#pragma once

// Largely copied from the OpenSpiel core 
// Added custom support for debug values. Thrown away mean-field games etc. 

#include <functional>
#include <iostream>
#include <map>
#include <unordered_map>
#include <memory>
#include <variant>
#include <span>
#include <string>
#include <cstdint>
#include <utility>
#include <vector>
#include <optional>
#include <numeric>
#include <mutex>
#include <torch/torch.h>

#include "game_parameters.h"
#include "astra_utils.h"

namespace astra {

// Static information for a game. This will determine what algorithms are
// applicable. For example, minimax search is only applicable to two-player,
// zero-sum games with perfect information. (Though can be made applicable to
// games that are constant-sum.)
//
// The number of players is not considered part of this static game type,
// because this depends on the parameterization. See Game::NumPlayers.
struct GameType {
  // A short name with no spaces that uniquely identifies the game, e.g.
  // "msoccer". This is the key used to distinguish games.
  std::string short_name;

  // A long human-readable name, e.g. "Markov Soccer".
  std::string long_name;

  // Is the game one-player-at-a-time or do players act simultaneously?
  enum class Dynamics {
    kSimultaneous,           // In some or all nodes every player acts.
    kSequential,             // Turn-based games.
  };
  Dynamics dynamics;

  // Are there any chance nodes? If so, how is chance treated?
  // Either all possible chance outcomes are explicitly returned as
  // ChanceOutcomes(), and the result of ApplyAction() is deterministic. Or
  // just one ChanceOutcome is returned, and the result of ApplyAction() is
  // stochastic. If in doubt, it is better to implement stochastic games with
  // kExplicitStochastic, as this makes more information available to any
  // learning algorithms you choose to use (i.e. the whole chance outcome
  // distribution is visible to the algorithm, rather than just the sampled
  // outcome). For more discussion of this field, see the github issue:
  // https://github.com/deepmind/open_spiel/issues/792.
  enum class ChanceMode {
    kDeterministic,       // No chance nodes
    kExplicitStochastic,  // Has at least one chance node, all with
                          // deterministic ApplyAction()
    kSampledStochastic,   // At least one chance node with non-deterministic
                          // ApplyAction()
  };
  ChanceMode chance_mode;

  // The information type of the game.
  enum class Information {
    kOneShot,               // aka Normal-form games (single simultaneous turn).
    kPerfectInformation,    // All players know the state of the game.
    kImperfectInformation,  // Some information is hidden from some players.
  };
  Information information;

  // Whether the game has any constraints on the player utilities.
  enum class Utility {
    kZeroSum,      // Utilities of all players sum to 0
    kConstantSum,  // Utilities of all players sum to a constant
    kGeneralSum,   // Total utility of all players differs in different outcomes
    kIdentical,    // Every player gets an identical value (cooperative game).
  };
  Utility utility;

  // When are rewards handed out? Note that even if the game only specifies
  // utilities at terminal states, the default implementation of State::Rewards
  // should work for RL uses (giving 0 everywhere except terminal states).
  enum class RewardModel {
    kRewards,   // RL-style func r(s, a, s') via State::Rewards() call at s'.
    kTerminal,  // Games-style, only at terminals. Call (State::Returns()).
  };
  RewardModel reward_model;

  // How many players can play the game. If the number can vary, the actual
  // instantiation of the game should specify how many players there are.
  int max_num_players;
  int min_num_players;

  // Which type of information state representations are supported?
  // The information state is a perfect-recall state-of-the-game from the
  // perspective of one player.
  bool provides_information_state_string;
  bool provides_information_state_tensor;

  // Which type of observation representations are supported?
  // The observation is some subset of the information state with the property
  // that remembering all the player's observations and actions is sufficient
  // to reconstruct the information state.
  bool provides_observation_string;
  bool provides_observation_tensor;

  GameParameters parameter_specification;
  bool ContainsRequiredParameters() const;

  // A number of optional values that have defaults, whose values can be
  // overridden in each game.

  // Can the game be loaded with no parameters? It is strongly recommended that
  // games be loadable with default arguments.
  bool default_loadable = true;

  bool provides_information_state() const {
    return provides_information_state_tensor
        || provides_information_state_string;
  }
  bool provides_observation() const {
    return provides_observation_tensor
        || provides_observation_string;
  }

  // Is this a concrete game, i.e. an actual game? Most games in OpenSpiel are
  // concrete games. Some games that are registered are not concrete games; for
  // example, game wrappers and other game transforms, or games that are
  // constructed from a file (e.g. efg_game).
  bool is_concrete = true;
};

// Information about a concrete Game instantiation.
// This information may depend on the game parameters, and hence cannot
// be part of `GameType`.
struct GameInfo {
  // The number of players in this instantiation of the game.
  // Does not include the chance-player.
  int num_players;

  // The total utility for all players, if this is a constant-sum-utility game.
  // Should be zero if the game is zero-sum.
  std::optional<RewardType> utility_sum;

  // The maximum number of player decisions in a game. Does not include chance
  // events. For a simultaneous action game, this is the maximum number of joint
  // decisions. In a turn-based game, this is the maximum number of individual
  // decisions summed over all players.
  int max_game_length;
};

// Python exposable information 
using ExposeValue = std::variant<
    int, float, std::string, torch::Tensor, 
    std::vector<int>, std::vector<float>, std::vector<std::string>, 
    std::vector<std::vector<int>>, std::vector<std::vector<float>>, std::vector<std::vector<std::string>>>;
using ExposeInfo = std::unordered_map<std::string, ExposeValue>;

std::ostream& operator<<(std::ostream& os, const StateType& type);
std::ostream& operator<<(std::ostream& stream, GameType::Dynamics value); // check 
std::ostream& operator<<(std::ostream& stream, GameType::ChanceMode value); // check 
std::ostream& operator<<(std::ostream& stream, GameType::Information value); // check 
std::ostream& operator<<(std::ostream& stream, GameType::Utility value); // check 
std::ostream& operator<<(std::ostream& stream, GameType::RewardModel value); // check

class Game;

// An abstract class that represents a state of the game.
class State {
 public:
  virtual ~State() = default;

  // Derived classes must call one of these constructors. Note that a state must
  // be passed a pointer to the game which created it. Some methods in some
  // games rely on this and so it must correspond to a valid game object.
  // The easiest way to ensure this is to use Game::NewInitialState to create
  // new states, which will pass a pointer to the parent game object. Also,
  // since this shared pointer to the parent is required, Game objects cannot
  // be used as value types and should always be created via a shared pointer.
  // See the documentation of the Game object for further details.
  State(std::shared_ptr<const Game> game);
  State(const State&) = default;

  // Returns current player. Player numbers start from 0.
  // Negative numbers are for chance (-1) or simultaneous (-2).
  // kTerminalPlayerId should be returned on a TerminalNode().
  virtual Player CurrentPlayer() const = 0;

  // Change the state of the game by applying the specified action in turn-based
  // games or in non-simultaneous nodes of simultaneous move games.
  // This function encodes the logic of the game rules.
  //
  // In the case of chance nodes, the behavior of this function depends on
  // GameType::chance_mode. If kExplicit, then the outcome should be
  // directly applied. If kSampled, then a dummy outcome is passed and the
  // sampling of and outcome should be done in this function and then applied.
  //
  // Games should implement DoApplyAction. 
  // Returns the player who just applied the action
  virtual Player ApplyAction(torch::Tensor action_id);

  // Returns a string representation of the state. 
  virtual std::string ToString(uint32_t index) const = 0;

  // Is this a terminal state? (i.e. has the game ended?)
  virtual bool IsTerminal() const = 0;

  // Returns reward from the most recent state transition (s, a, s') for all
  // players. This is provided so that RL-style games with intermediate rewards
  // (along the episode, rather than just one value at the end) can be properly
  // implemented. The default is to return 0 except at terminal states, where
  // the terminal returns are returned.
  //
  // Note: This must agree with Returns(). That is, for any state S_t,
  //       Returns(St) = Sum(Rewards(S_0), Rewards(S_1)... Rewards(S_t)).
  //       The default implementation is only correct for games that only
  //       have a final reward. Games with intermediate rewards must override
  //       both this method and Returns().
  // Players are indexed by the last dimension 
  virtual void FillRewards(torch::Tensor reward_buffer) const {
    AstraFatalError("Not implemented"); 
  }

  // Convenient function to keep track of the players' cumulative rewards since last action
  virtual void FillRewardsSinceLastAction(torch::Tensor reward_buffer, Player player_id) const {
    AstraFatalError("Not implemented"); 
  }

  // Returns sums of all rewards for each player up to the current state.
  // For games that only have a final reward, it should be 0 for all
  // non-terminal states, and the terminal utility for the final state.
  virtual void FillReturns(torch::Tensor returns_buffer ) const  {
    AstraFatalError("Not implemented"); 
  }

  // Expose python-viewable information 
  virtual ExposeInfo expose_info() const { 
    return {};
  }

  // Is this state a chance node? Chance nodes are "states" whose actions
  // represent stochastic outcomes. "Chance" or "Nature" is thought of as a
  // player with a fixed (randomized) policy.
  virtual bool IsChanceNode() const {
    return CurrentPlayer() == kChancePlayerId;
  }

  // Is this state a player node, with a single player acting?
  virtual bool IsPlayerNode() const { return CurrentPlayer() >= 0; }

  // Is this state a node that requires simultaneous action choices from more
  // than one player? If this is ever true, then the game should be marked as
  // a simultaneous game.
  bool IsSimultaneousNode() const {
    return CurrentPlayer() == kSimultaneousPlayerId;
  }

  // Is the specified player acting at this state?
  bool IsPlayerActing(Player player) const {
    ASTRA_CHECK_GE(player, 0);
    ASTRA_CHECK_LT(player, NumPlayers());
    return CurrentPlayer() == player || IsSimultaneousNode();
  }

  // We store (player, action) pairs in the history.
  struct PlayerAction {
    Player player;
    Action action;
    bool operator==(const PlayerAction&) const;
  };

  // Return how many moves have been done so far in the game.
  // When players make simultaneous moves, this counts only as a one move.
  // Chance transitions count also as one move.
  // Note that game transformations are not required to preserve the move
  // number in the transformed game.
  int MoveNumber() const { return move_number_; }

  // Is this a first state in the game, i.e. the initial state (root node)?
  bool IsInitialState() const { return move_number_ == 0; }

  // For imperfect information games. Returns an identifier for the current
  // information state for the specified player.
  // Different ground states can yield the same information state for a player
  // when the only part of the state that differs is not observable by that
  // player (e.g. opponents' cards in Poker.)
  //
  // The identifiers must be unique across all players.
  // This allows an algorithm to maintain a single table of identifiers
  // instead of maintaining a table per player to avoid name collisions.
  //
  // A simple way to do so is for example, in a card game, if both players can
  // hold the card Jack, the identifier can contain player identification as
  // well, like P1Jack and P2Jack. However prefixing by player number is not
  // a requirement. The only thing that is necessary is that it is unambiguous
  // who is the observer.
  //
  // Games that do not have imperfect information do not need to implement
  // these methods, but most algorithms intended for imperfect information
  // games will work on perfect information games provided the InformationState
  // is returned in a form they support. For example, InformationState()
  // could simply return the history for a perfect information game.
  //
  // A valid InformationStateString must be returned at terminal states, since
  // this is required in some applications (e.g. final observation in an RL
  // environment).
  //
  // The information state should be perfect-recall, i.e. if two states
  // have a different InformationState, then all successors of one must have
  // a different InformationState to all successors of the other.
  // For example, in tic-tac-toe, the current state of the board would not be
  // a perfect-recall representation, but the sequence of moves played would
  // be.
  //
  // If you implement both InformationState and Observation, the two must be
  // consistent for all the players (even the non-acting player(s)).
  // By consistency we mean that when you maintain an Action-Observation
  // history (AOH) for different ground states, the (in)equality of two AOHs
  // implies the (in)equality of two InformationStates. In other words, AOH is a
  // factored representation of InformationState.
  //
  // For details, see Section 3.1 of https://arxiv.org/abs/1908.09453
  // or Section 2.1 of https://arxiv.org/abs/1906.11110

  // There are currently no use-case for calling this function with
  // `kChancePlayerId`. Use this:
  //   ASTRA_CHECK_GE(player, 0);
  //   ASTRA_CHECK_LT(player, num_players_);
  virtual std::string InformationStateString(Player player, uint32_t index) const {
    AstraFatalError("InformationStateString is not implemented.");
  }
  std::string InformationStateString(uint32_t index) const {
    return InformationStateString(CurrentPlayer());
  }

  // Vector form, useful for neural-net function approximation approaches.
  // The size of the vector must match Game::InformationStateShape()
  // with values in lexicographic order. E.g. for 2x4x3, order would be:
  // (0,0,0), (0,0,1), (0,0,2), (0,1,0), ... , (1,3,2).
  // This function should resize the supplied vector if required.
  //
  // A valid InformationStateTensor must be returned at terminal states, since
  // this is required in some applications (e.g. final observation in an RL
  // environment).
  //
  // Implementations should start with: 
  //   ASTRA_CHECK_GE(player, 0);
  //   ASTRA_CHECK_LT(player, num_players_);
  virtual void FillInformationStateTensor(Player player, torch::Tensor values) const {
    AstraFatalError("InformationStateTensor unimplemented!");
  }
  Player FillInformationStateTensor(torch::Tensor values) const {
    Player player = CurrentPlayer(); 
    FillInformationStateTensor(player, values);
    return player;
  }
  // We have functions for observations which are parallel to those for
  // information states. An observation should have the following properties:
  //  - It has at most the same information content as the information state
  //  - The complete history of observations and our actions over the
  //    course of the game is sufficient to reconstruct the information
  //    state for any players at any point in the game.
  //
  // For example, an observation is the cards revealed and bets made in Poker,
  // or the current state of the board in Chess.
  // Note that neither of these are valid information states, since the same
  // observation may arise from two different observation histories (i.e. they
  // are not perfect recall).
  //
  // Observations should cover all observations: a combination of both public
  // and private observations. They are not factored into these individual
  // constituent parts.
  //
  // A valid observation must be returned at terminal states, since this is
  // required in some applications (e.g. final observation in an RL
  // environment).
  //
  // Implementations should start with (and it's tested in api_test.py):
  //   ASTRA_CHECK_GE(player, 0);
  //   ASTRA_CHECK_LT(player, num_players_);
  virtual std::string ObservationString(Player player, uint32_t index) const {
    AstraFatalError("ObservationString is not implemented.");
  }
  std::string ObservationString(uint32_t index) const {
    return ObservationString(CurrentPlayer(), index);
  }

  // Returns the view of the game, preferably from `player`'s perspective.
  //
  // Implementations should start with (and it's tested in api_test.py):
  //   ASTRA_CHECK_GE(player, 0);
  //   ASTRA_CHECK_LT(player, num_players_);
  virtual void FillObservationTensor(Player player,
                                 torch::Tensor values) const {
    AstraFatalError("ObservationTensor unimplemented!");
  }
  Player FillObservationTensor(torch::Tensor values) const {
    Player player = CurrentPlayer();
    FillObservationTensor(player, values);
    return player; 
  }

  // Return a copy of this state.
  virtual std::unique_ptr<State> Clone() const = 0;

  // Creates the child from State corresponding to action.
  std::unique_ptr<State> Child(torch::Tensor action) const {
    std::unique_ptr<State> child = Clone();
    child->ApplyAction(action);
    return child;
  }

  // Returns the number of players in this game.
  int NumPlayers() const { return num_players_; }

  // Get the game object that generated this state.
  std::shared_ptr<const Game> GetGame() const { return game_; }

  // Returns the type of the state. Either Chance, Terminal, MeanField or
  // Decision. See StateType definition for definitions of the different types.
  StateType GetType() const;

  // Implement by derived classes. 
  virtual void DoReset() { AstraFatalError("DoReset is not implemented."); }

  void Reset() { move_number_ = 0; DoReset(); }

 protected:
  // See ApplyAction.
  virtual void DoApplyAction(torch::Tensor action_id) {
    AstraFatalError("DoApplyAction is not implemented.");
  }

  // The game that created this state, plus some static information about it,
  // cached here for efficient access.
  const std::shared_ptr<const Game> game_;
  const int num_players_;

  // Information that changes over the course of the game.
  int move_number_;
};


class Game : public std::enable_shared_from_this<Game> {
  public:
   virtual ~Game() = default;
   Game(const Game&) = delete;
   Game& operator=(const Game&) = delete;
 
   // Returns a newly allocated initial state.
   virtual std::unique_ptr<State> NewInitialState() const = 0;
 
   // If the game is parameterizable, returns an object with the current
   // parameter values, including defaulted values. Returns empty parameters
   // otherwise.
   GameParameters GetParameters() const {
     std::lock_guard<std::mutex> lock(mutex_defaulted_parameters_);
     GameParameters params = game_parameters_;
     params.insert(defaulted_parameters_.begin(), defaulted_parameters_.end());
     return params;
   }
 
   // The number of players in this instantiation of the game.
   // Does not include the chance-player.
   virtual int NumPlayers() const = 0;
 
   // Static information on the game type. This should match the information
   // provided when registering the game.
   const GameType& GetType() const { return game_type_; }
 
   // The total utility for all players, if this is a constant-sum-utility game.
   // Should return 0 if the game is zero-sum.
   virtual std::optional<RewardType> UtilitySum() const { return std::nullopt; }

   // Describes the structure of the information state representation in a
   // tensor-like format. This is especially useful for experiments involving
   // reinforcement learning and neural networks. Note: the actual information is
   // returned in a 1-D vector by State::InformationStateTensor -
   // see the documentation of that function for details of the data layout.
   virtual std::vector<int> InformationStateTensorShape() const {
     AstraFatalError("InformationStateTensorShape unimplemented.");
   }
 
   // Describes the structure of the observation representation in a
   // tensor-like format. This is especially useful for experiments involving
   // reinforcement learning and neural networks. Note: the actual observation is
   // returned in a 1-D vector by State::ObservationTensor -
   // see the documentation of that function for details of the data layout.
   virtual std::vector<int> ObservationTensorShape() const {
     AstraFatalError("ObservationTensorShape unimplemented.");
   }
 
   // The maximum length of any one game (in terms of number of decision nodes
   // visited in the game tree). For a simultaneous action game, this is the
   // maximum number of joint decisions. In a turn-based game, this is the
   // maximum number of individual decisions summed over all players. Outcomes
   // of chance nodes are not included in this length.
   virtual int MaxGameLength() const = 0;
 
   // The maximum number of chance nodes occurring in any history of the game.
   // This is typically something like the number of times dice are rolled.
   virtual int MaxChanceNodesInHistory() const {
     if (GetType().chance_mode == GameType::ChanceMode::kDeterministic) {
       return 0;
     }
     AstraFatalError("MaxChanceNodesInHistory() is not implemented");
   }
 
   // The maximum number of moves in the game. The value State::MoveNumber()
   // must never be higher than this value.
   virtual int MaxMoveNumber() const {
     return MaxGameLength() + MaxChanceNodesInHistory();
   }
 
   // Get and set game's internal RNG state for de/serialization purposes. These
   // two methods only need to be overridden by sampled stochastic games that
   // need to hold an RNG state. Note that stateful game implementations are
   // discouraged in general.
   virtual std::string GetRNGState() const {
     AstraFatalError("GetRNGState unimplemented.");
   }
   // SetRNGState is const despite the fact that it changes game's internal
   // state. Sampled stochastic games need to be explicit about mutability of the
   // RNG, i.e. have to use the mutable keyword.
   virtual void SetRNGState(const std::string& rng_state) const {
     AstraFatalError("SetRNGState unimplemented.");
   }
 
  protected:
   Game(GameType game_type, GameParameters game_parameters)
       : game_type_(game_type), game_parameters_(game_parameters) {}
 
   // Access to game parameters. Returns the value provided by the user. If not:
   // - Defaults to the value stored as the default in
   // game_type.parameter_specification if the `default_value` is std::nullopt
   // - Returns `default_value` if provided.
   template <typename T>
   T ParameterValue(const std::string& key,
                    std::optional<T> default_value = std::nullopt) const {
     // Return the value if found.
     auto iter = game_parameters_.find(key);
     if (iter != game_parameters_.end()) {
       return iter->second.value<T>();
     }
 
     // Pick the defaulted value.
     GameParameter default_game_parameter;
     if (default_value.has_value()) {
       default_game_parameter = GameParameter(default_value.value());
     } else {
       auto default_iter = game_type_.parameter_specification.find(key);
       if (default_iter == game_type_.parameter_specification.end()) {
         AstraFatalError(StrCat("The parameter for ", key,
                                      " is missing in game."));
       }
       default_game_parameter = default_iter->second;
     }
 
     // Return the default value, storing it.
     std::lock_guard<std::mutex> lock(mutex_defaulted_parameters_);
     iter = defaulted_parameters_.find(key);
     if (iter == defaulted_parameters_.end()) {
       // We haven't previously defaulted this value, so store the default we
       // used.
       defaulted_parameters_[key] = default_game_parameter;
     } else {
       // Already defaulted, so check we are being consistent.
       // Using different default values at different times means the game isn't
       // well-defined.
       if (default_game_parameter != iter->second) {
         AstraFatalError(StrCat("Parameter ", key, " is defaulted to ",
                                      default_game_parameter.ToReprString(),
                                      " having previously been defaulted to ",
                                      iter->second.ToReprString(), " in game."));
       }
     }
     return default_game_parameter.value<T>();
   }
 
   // The game type.
   GameType game_type_;
 
   // Any parameters supplied when constructing the game.
   GameParameters game_parameters_;
 
   // Track the parameters for which a default value has been used. This
   // enables us to report the actual value used for every parameter.
   mutable GameParameters defaulted_parameters_;
   mutable std::mutex mutex_defaulted_parameters_;
};

#define CONCAT_(x, y) x##y
#define CONCAT(x, y) CONCAT_(x, y)
// When called by e.g. high_low_trading.cc as REGISTER_ASTRA_GAME(), it 
// Instantiates a GameRegistrar class, which updates the global registry objects. 
// Note that this REQUIRES whole-archive static linking, else the linker will omit 
// the macro initialization in high_low_trading.cc 
#define REGISTER_ASTRA_GAME(info, factory) \
  GameRegistrar CONCAT(game, __COUNTER__)(info, factory);

class GameRegistrar {
  public:
   using CreateFunc =
       std::function<std::shared_ptr<const Game>(const GameParameters& params)>;
 
   GameRegistrar(const GameType& game_type, CreateFunc creator);
 
   static std::shared_ptr<const Game> CreateByName(const std::string& short_name,
                                                   const GameParameters& params);
 
   static std::vector<std::string> RegisteredNames();
   static std::vector<std::string> RegisteredConcreteNames();
   static std::vector<GameType> RegisteredGames();
   static std::vector<GameType> RegisteredConcreteGames();
   static bool IsValidName(const std::string& short_name);
   static void RegisterGame(const GameType& game_type, CreateFunc creator);
 
  private:
   // Returns a "global" map of registrations (i.e. an object that lives from
   // initialization to the end of the program). Note that we do not just use
   // a static data member, as we want the map to be initialized before first
   // use.
   static std::map<std::string, std::pair<GameType, CreateFunc>>& factories() {
     static std::map<std::string, std::pair<GameType, CreateFunc>> impl;
     return impl;
   }
 
   static std::vector<std::string> GameTypesToShortNames(
       const std::vector<GameType>& game_types);
};

}  // namespace astra
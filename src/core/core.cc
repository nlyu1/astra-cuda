#include "core.h"
#include "astra_utils.h"
#include "fmt/format.h"
#include <iostream>
#include <algorithm>


namespace astra {

State::State(std::shared_ptr<const Game> game) : game_(game),
    num_distinct_actions_(game->NumDistinctActions()),
    num_players_(game->NumPlayers()),
    move_number_(0) {}

void State::ApplyAction(Action action_id) {
    // history_ needs to be modified *after* DoApplyAction which could
    // be using it.
    
    // Cannot apply an invalid action.
    ASTRA_CHECK_NE(action_id, kInvalidAction);
    Player player = CurrentPlayer();
    DoApplyAction(action_id);
    history_.push_back({player, action_id});
    ++move_number_;
}

std::vector<ObservationScalarType> State::ObservationTensor(Player player) const {
  auto tensor_size = GetGame()->ObservationTensorSize();
  std::vector<ObservationScalarType> values(tensor_size);
  ObservationTensor(player, std::span<ObservationScalarType>(values));
  return values;
}

void State::ApplyActionWithLegalityCheck(Action action_id) {
    std::vector<Action> legal_actions = LegalActions();
    if (std::find(legal_actions.begin(), legal_actions.end(), action_id) == legal_actions.end()) {
      Player cur_player = CurrentPlayer();
      AstraFatalError(
          StrCat("Current player ", cur_player, " calling ApplyAction ",
                       "with illegal action (", action_id, "): ",
                       ActionToString(cur_player, action_id)));
    }
    ApplyAction(action_id);
}

Action State::StringToAction(Player player,
  const std::string& action_str) const {
    for (const Action action : LegalActions()) {
        if (action_str == ActionToString(player, action)) {
            return action;
        }
    }
    AstraFatalError(
        StrCat("Couldn't find an action matching ", action_str));
}

std::vector<int> State::LegalActionsMask(Player player) const {
    int length = (player == kChancePlayerId) ? game_->MaxChanceOutcomes()
                                             : num_distinct_actions_;
    std::vector<int> mask(length, 0);
    for (int action : LegalActions(player)) {
        mask[action] = 1;
    }
    return mask;
}

std::vector<float> State::InformationStateTensor(Player player) const {
    // We add this player check, to prevent errors if the game implementation
    // lacks that check (in particular as this function is the one used in
    // Python). This can lead to doing this check twice.
    // TODO(author2): Do we want to prevent executing this twice for games
    // that implement it?
    ASTRA_CHECK_GE(player, 0);
    ASTRA_CHECK_LT(player, num_players_);
    std::vector<float> info_state(game_->InformationStateTensorSize());
    InformationStateTensor(player, std::span{info_state});
    return info_state;
}

void State::ApplyActions(const std::vector<Action>& actions) {
    // history_ needs to be modified *after* DoApplyActions which could
    // be using it.
    DoApplyActions(actions);
    history_.reserve(history_.size() + actions.size());
    for (int player = 0; player < actions.size(); ++player) {
      history_.push_back({player, actions[player]});
    }
    ++move_number_;
}

void State::ApplyActionsWithLegalityChecks(const std::vector<Action>& actions) {
    for (Player player = 0; player < actions.size(); ++player) {
      std::vector<Action> legal_actions = LegalActions(player);
      if (!legal_actions.empty() &&
          std::find(legal_actions.begin(), legal_actions.end(), actions[player]) == legal_actions.end()) {
        AstraFatalError(
            StrCat("Player ", player, " calling ApplyAction ",
                         "with illegal action (", actions[player], "): ",
                         ActionToString(player, actions[player])));
      }
    }
    ApplyActions(actions);
}

StateType State::GetType() const {
    if (IsChanceNode()) {
      return StateType::kChance;
    } else if (IsTerminal()) {
      return StateType::kTerminal;
    } else {
      return StateType::kDecision;
    }
}

std::string State::Serialize() const {
    // This simple serialization doesn't work for the following games:
    // - games with sampled chance nodes, since the history doesn't give us enough
    //   information to reconstruct the state.
    // - Mean field games, since this base class does not store the history of
    //   state distributions passed in UpdateDistribution() (and it would be
    //   very expensive to do so for games with many possible states and a long
    //   time horizon).
    // If you wish to serialize states in such games, you must implement custom
    // serialization and deserialization for the state.
    ASTRA_CHECK_NE(game_->GetType().chance_mode,
                   GameType::ChanceMode::kSampledStochastic);
    return StrCat(StrJoin(History(), "\n"), "\n");
}

std::unique_ptr<State> State::ResampleFromInfostate(
    int player_id,
    std::function<double()> rng) const {
  if (GetGame()->GetType().information ==
      GameType::Information::kPerfectInformation) {
    return Clone();
  }
  AstraFatalError("ResampleFromInfostate() not implemented.");
}

std::unique_ptr<State> Game::DeserializeState(const std::string& str) const {
    // This does not work for games with sampled chance nodes and for mean field
    //  games. See comments in State::Serialize() for the explanation. If you wish
    //  to serialize states in such games, you must implement custom serialization
    //  and deserialization for the state.
    ASTRA_CHECK_NE(game_type_.chance_mode,
                   GameType::ChanceMode::kSampledStochastic);
  
    std::unique_ptr<State> state = NewInitialState();
    if (str.empty()) {
      return state;
    }
    std::vector<std::string> lines = StrSplit(str, '\n');
    for (int i = 0; i < lines.size(); ++i) {
      if (lines[i].empty()) {
          continue;
      }
      if (state->IsSimultaneousNode()) {
        std::vector<Action> actions;
        for (int p = 0; p < state->NumPlayers(); ++p, ++i) {
          ASTRA_CHECK_LT(i, lines.size());
          Action action = static_cast<Action>(std::stol(lines[i]));
          actions.push_back(action);
        }
        state->ApplyActions(actions);
        // Must decrement i here, otherwise it is incremented too many times.
        --i;
      } else {
        Action action = static_cast<Action>(std::stol(lines[i]));
        state->ApplyAction(action);
      }
    }
    return state;
}

std::string Game::Serialize() const {
    std::string str = ToString();
    if (GetType().chance_mode == GameType::ChanceMode::kSampledStochastic) {
      AstraFatalError("Serialize() not implemented for SampledStochastic games.");
    }
    return str;
}

std::string Game::ToString() const {
    GameParameters params = game_parameters_;
    params["name"] = GameParameter(game_type_.short_name);
    return GameParametersToString(params);
}

GameRegistrar::GameRegistrar(const GameType& game_type, CreateFunc creator) {
    RegisterGame(game_type, creator);
}

// Returns the available parameter keys, to be used as a utility function.
std::string ListValidParameters(
    const GameParameters& param_spec) {
  std::vector<std::string> available_keys;
  available_keys.reserve(param_spec.size());
  for (const auto& item : param_spec) {
    available_keys.push_back(item.first);
  }
  std::sort(available_keys.begin(), available_keys.end());
  return StrJoin(available_keys, ", ");
}

// Check on supplied parameters for game creation.
// Issues a SpielFatalError if any are missing, of the wrong type, or
// unexpectedly present.
void ValidateParams(const GameParameters& params,
                    const GameParameters& param_spec) {
  // Check all supplied parameters are supported and of the right type.
  for (const auto& param : params) {
    const auto it = param_spec.find(param.first);
    if (it == param_spec.end()) {
      AstraFatalError(StrCat(
          "Unknown parameter '", param.first,
          "'. Available parameters are: ", ListValidParameters(param_spec)));
    }
    if (it->second.type() != param.second.type()) {
      AstraFatalError(StrCat(
          "Wrong type for parameter ", param.first,
          ". Expected type: ", GameParameterTypeToString(it->second.type()),
          ", got ", GameParameterTypeToString(param.second.type()), " with ",
          param.second.ToString()));
    }
  }
  // Check we aren't missing any mandatory parameters.
  for (const auto& param : param_spec) {
    if (param.second.is_mandatory() && !params.count(param.first)) {
      AstraFatalError(StrCat("Missing parameter ", param.first));
    }
  }
}

std::shared_ptr<const Game> GameRegistrar::CreateByName(
    const std::string& short_name, const GameParameters& params) {
  // Find the factory for this game and load it.
  auto iter = factories().find(short_name);
  if (iter == factories().end()) {
    AstraFatalError(StrCat("Unknown game '", short_name,
                                 "'. Available games are:\n",
                                 StrJoin(RegisteredNames(), "\n")));

  } else {
    ValidateParams(params, iter->second.first.parameter_specification);
    return (iter->second.second)(params);
  }
}

std::vector<std::string> GameRegistrar::RegisteredNames() {
    return GameTypesToShortNames(RegisteredGames());
}

std::vector<GameType> GameRegistrar::RegisteredConcreteGames() {
    std::vector<GameType> games;
    for (const auto& key_val : factories()) {
      if (key_val.second.first.is_concrete) {
        games.push_back(key_val.second.first);
      }
    }
    return games;
}
  
std::vector<std::string> GameRegistrar::RegisteredConcreteNames() {
  return GameTypesToShortNames(RegisteredConcreteGames());
}

std::vector<std::string> RegisteredGames() {
    return GameRegistrar::RegisteredNames();
}

std::vector<GameType> RegisteredGameTypes() {
return GameRegistrar::RegisteredGames();
}

std::vector<std::string> GameRegistrar::GameTypesToShortNames(
    const std::vector<GameType>& game_types) {
  std::vector<std::string> names;
  names.reserve(game_types.size());
  for (const auto& game_type : game_types) {
    names.push_back(game_type.short_name);
  }
  return names;
}

void GameRegistrar::RegisterGame(const GameType& game_type,
  GameRegistrar::CreateFunc creator) {
factories()[game_type.short_name] = std::make_pair(game_type, creator);
}

std::vector<GameType> GameRegistrar::RegisteredGames() {
  std::vector<GameType> games;
  for (const auto& key_val : factories()) {
    games.push_back(key_val.second.first);
  }
  return games;
}

std::ostream& operator<<(std::ostream& os, const StateType& type) {
    switch (type) {
        case StateType::kChance: {
        os << "CHANCE";
        break;
        }
        case StateType::kDecision: {
        os << "DECISION";
        break;
        }
        case StateType::kTerminal: {
        os << "TERMINAL";
        break;
        }
    }
    return os;
}

std::ostream& operator<<(std::ostream& stream, GameType::ChanceMode value) {
  switch (value) {
    case GameType::ChanceMode::kDeterministic:
      return stream << "Deterministic";
    case GameType::ChanceMode::kExplicitStochastic:
      return stream << "ExplicitStochastic";
    case GameType::ChanceMode::kSampledStochastic:
      return stream << "SampledStochastic";
    default:
      AstraFatalError("Unknown mode.");
  }
}

std::ostream& operator<<(std::ostream& stream, GameType::Information value) {
    switch (value) {
      case GameType::Information::kOneShot:
        return stream << "OneShot";
      case GameType::Information::kPerfectInformation:
        return stream << "PerfectInformation";
      case GameType::Information::kImperfectInformation:
        return stream << "ImperfectInformation";
      default:
        AstraFatalError("Unknown value.");
    }
}

std::ostream& operator<<(std::ostream& stream, GameType::Utility value) {
    switch (value) {
      case GameType::Utility::kZeroSum:
        return stream << "ZeroSum";
      case GameType::Utility::kConstantSum:
        return stream << "ConstantSum";
      case GameType::Utility::kGeneralSum:
        return stream << "GeneralSum";
      case GameType::Utility::kIdentical:
        return stream << "Identical";
      default:
        AstraFatalError("Unknown value.");
    }
}

std::ostream& operator<<(std::ostream& stream, GameType::RewardModel value) {
    switch (value) {
      case GameType::RewardModel::kRewards:
        return stream << "Rewards";
      case GameType::RewardModel::kTerminal:
        return stream << "Terminal";
      default:
        AstraFatalError("Unknown value.");
    }
}

std::ostream& operator<<(std::ostream& stream, const State& state) {
    return stream << state.ToString();
}

}  // namespace astra
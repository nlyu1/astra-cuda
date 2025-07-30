#include "core.h"
#include "astra_utils.h"
#include "fmt/format.h"
#include <iostream>
#include <algorithm>
#include <torch/torch.h>


namespace astra {

State::State(std::shared_ptr<const Game> game) : game_(game),
    num_players_(game->NumPlayers()),
    move_number_(0) {}

Player State::ApplyAction(torch::Tensor action_id) {
    Player current_player = CurrentPlayer();
    DoApplyAction(action_id);
    ++move_number_;
    return current_player; 
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

}  // namespace astra
#include <torch/extension.h>

#include "core.h"
#include "registration.h"
#include "game_parameter_bindings.h"
#include "core_bindings.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;

namespace astra {

void AddCoreBindings(py::module& m) {
    m.def("game_parameters_from_string", GameParametersFromString,
        "Parses a string as a GameParameter dictionary.");

    m.def("game_parameters_to_string", GameParametersToString,
        "Converts a GameParameter dictionary to string.");

    // Game factory functions
    m.def("register_games", &RegisterGames, 
        "Register all implemented games. Need to be called before loading any game.");

    m.def("load_game", &GameRegistrar::CreateByName, 
        "Load a game by name with parameters",
        py::arg("short_name"), py::arg("params")=GameParameters{});

    m.def("registered_names", &GameRegistrar::RegisteredNames,
        "Get list of all registered game names");

    // State class using classh for smart pointer support
    py::classh<State> state(m, "State");
    state
        .def("current_player", &State::CurrentPlayer)
        .def("apply_action", [](State& self, torch::Tensor action_id) -> Player {
            return self.ApplyAction(action_id);
        }, py::arg("action_id"))
        .def("to_string", &State::ToString, py::arg("index") = 0)
        .def("is_terminal", &State::IsTerminal)
        .def("fill_rewards", [](State& self, torch::Tensor reward_buffer) {
            self.FillRewards(reward_buffer);
        }, py::arg("reward_buffer"))
        .def("fill_rewards_since_last_action", [](State& self, torch::Tensor reward_buffer, Player player_id) {
            self.FillRewardsSinceLastAction(reward_buffer, player_id);
        }, py::arg("reward_buffer"), py::arg("player_id"))
        .def("fill_returns", [](State& self, torch::Tensor returns_buffer) {
            self.FillReturns(returns_buffer);
        }, py::arg("returns_buffer"))
        .def("expose_info", &State::expose_info)
        .def("is_chance_node", &State::IsChanceNode)
        .def("is_player_node", &State::IsPlayerNode)
        .def("is_simultaneous_node", &State::IsSimultaneousNode)
        .def("is_player_acting", &State::IsPlayerActing, py::arg("player"))
        .def("move_number", &State::MoveNumber)
        .def("is_initial_state", &State::IsInitialState)
        .def("information_state_string", 
             py::overload_cast<Player, int32_t>(&State::InformationStateString, py::const_),
             py::arg("player"), py::arg("index") = 0)
        .def("information_state_string", 
             py::overload_cast<int32_t>(&State::InformationStateString, py::const_),
             py::arg("index") = 0)
        .def("fill_information_state_tensor", [](State& self, Player player, torch::Tensor values) {
            self.FillInformationStateTensor(player, values);
        }, py::arg("player"), py::arg("values"))
        .def("fill_information_state_tensor", [](State& self, torch::Tensor values) -> Player {
            return self.FillInformationStateTensor(values);
        }, py::arg("values"))
        .def("observation_string", 
             py::overload_cast<Player, int32_t>(&State::ObservationString, py::const_),
             py::arg("player"), py::arg("index") = 0)
        .def("observation_string", 
             py::overload_cast<int32_t>(&State::ObservationString, py::const_),
             py::arg("index") = 0)
        .def("fill_observation_tensor", [](State& self, Player player, torch::Tensor values) {
            self.FillObservationTensor(player, values);
        }, py::arg("player"), py::arg("values"))
        .def("fill_observation_tensor", [](State& self, torch::Tensor values) -> Player {
            return self.FillObservationTensor(values);
        }, py::arg("values"))
        .def("clone", &State::Clone)
        .def("child", [](const State& self, torch::Tensor action) {
            return self.Child(action);
        }, py::arg("action"))
        .def("num_players", &State::NumPlayers)
        .def("get_game", &State::GetGame)
        .def("reset", &State::Reset)
        .def("__str__", [](const State& self) { return self.ToString(0); })
        .def("__repr__", [](const State& self) { return self.ToString(0); });

    // Game class using classh for smart pointer support
    py::classh<Game> game(m, "Game");
    game
        .def("new_initial_state", &Game::NewInitialState)
        .def("get_parameters", &Game::GetParameters)
        .def("num_players", &Game::NumPlayers)
        .def("utility_sum", &Game::UtilitySum)
        .def("information_state_tensor_shape", &Game::InformationStateTensorShape)
        .def("observation_tensor_shape", &Game::ObservationTensorShape)
        .def("max_game_length", &Game::MaxGameLength)
        .def("max_chance_nodes_in_history", &Game::MaxChanceNodesInHistory)
        .def("max_move_number", &Game::MaxMoveNumber)
        .def("get_rng_state", &Game::GetRNGState)
        .def("set_rng_state", &Game::SetRNGState, py::arg("rng_state"))
        .def("__str__", [](const Game& self) { 
            return "Game(" + self.GetType().short_name + ")"; 
        })
        .def("__repr__", [](const Game& self) { 
            return "<Game '" + self.GetType().short_name + "'>"; 
        })
        .def("__eq__", [](std::shared_ptr<const Game> a, std::shared_ptr<const Game> b) {
            return b && a->GetType().short_name == b->GetType().short_name;
        });

    // Expose constants with original names
    m.attr("kInvalidAction") = py::int_(kInvalidAction);
}

}  // namespace astra
#include <iostream>
#include <memory>
#include "games/high_low_trading/high_low_trading.h"
#include "core/core.h"

using namespace astra;
using namespace astra::high_low_trading;

// Function to register games (declaration)
void RegisterGames();

int main() {
    // Register all available games
    RegisterGames();
    
    // Verify registration
    std::cout << "Number of registered games: " << GameRegistrar::RegisteredNames().size() << std::endl;
    if (!GameRegistrar::RegisteredNames().empty()) {
        std::cout << "First registered game: " << GameRegistrar::RegisteredNames()[0] << std::endl;
    }
    
    // Set up game parameters
    int steps_per_player = 3;
    int max_contracts_per_trade = 3;
    int customer_max_size = 5;
    int max_contract_value = 30;
    int num_players = 5;
    int num_markets = 1;
    int threads_per_block = 128;
    int device_id = 0;
    
    // Create game parameters
    GameParameters params = {
        {"steps_per_player", GameParameter(steps_per_player)},
        {"max_contracts_per_trade", GameParameter(max_contracts_per_trade)},
        {"customer_max_size", GameParameter(customer_max_size)},
        {"max_contract_value", GameParameter(max_contract_value)},
        {"players", GameParameter(num_players)},
        {"num_markets", GameParameter(num_markets)},
        {"threads_per_block", GameParameter(threads_per_block)},
        {"device_id", GameParameter(device_id)}
    };
    
    std::cout << "\nCreating HighLowTrading game with parameters:" << std::endl;
    std::cout << "- Steps per player: " << steps_per_player << std::endl;
    std::cout << "- Max contracts per trade: " << max_contracts_per_trade << std::endl;
    std::cout << "- Customer max size: " << customer_max_size << std::endl;
    std::cout << "- Max contract value: " << max_contract_value << std::endl;
    std::cout << "- Players: " << num_players << std::endl;
    std::cout << "- Number of markets: " << num_markets << std::endl;
    std::cout << "- Threads per block: " << threads_per_block << std::endl;
    std::cout << "- Device ID: " << device_id << std::endl;
    
    // Create game using factory method
    auto game = Factory(params);
    
    // Create initial state
    auto state = game->NewInitialState();
    
    std::cout << "\nGame created successfully!" << std::endl;
    std::cout << "Initial state created." << std::endl;
    std::cout << "Current player: " << state->CurrentPlayer() << std::endl;
    std::cout << "Is terminal: " << (state->IsTerminal() ? "Yes" : "No") << std::endl;
    std::cout << "Move number: " << state->MoveNumber() << std::endl;
    
    // Cast to HighLowTradingState to access game-specific methods
    auto* trading_state = static_cast<HighLowTradingState*>(state.get());
    std::cout << "\nInitial state string (index 0):" << std::endl;
    std::cout << trading_state->ToString(0) << std::endl;
    
    return 0;
}
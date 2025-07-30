#include <cmath>
#include <iostream>
#include <string>
#include <algorithm>
#include <numeric>
#include <fmt/format.h>
#include "action_manager.h"

namespace astra {
namespace high_low_trading {

Config::Config(int steps_per_player, int max_contracts_per_trade, int customer_max_size, int max_contract_value, int num_players): 
    steps_per_player_(steps_per_player), 
    max_contracts_per_trade_(max_contracts_per_trade), 
    customer_max_size_(customer_max_size), 
    max_contract_value_(max_contract_value), 
    num_players_(num_players) {}

std::string Config::ToString() const {
    return fmt::format("Config(steps_per_player={}, max_contracts_per_trade={}, customer_max_size={}, max_contract_value={}, num_players={})", 
                      steps_per_player_, max_contracts_per_trade_, customer_max_size_, max_contract_value_, num_players_);
}

ActionManager::ActionManager(const Config& config) : config_(config) {
    // std::cout << "ActionManager constructor called with config: " << config.ToString() << std::endl; 
    ASTRA_CHECK_GE(config.num_players_, 4); 
}

// ChanceContractValueAction methods
ChanceContractValueAction ChanceContractValueAction::decode(Action raw_action) {
    return ChanceContractValueAction(raw_action + 1);
}

Action ChanceContractValueAction::encode() const {
    return contract_value_ - 1;
}

std::string ChanceContractValueAction::ToString() const {
  return "Environment settles one piece of contract value to " + std::to_string(contract_value_);
}

// ChanceHighLowAction methods
ChanceHighLowAction ChanceHighLowAction::decode(Action raw_action) {
    if (raw_action != 0 && raw_action != 1) {
        AstraFatalError(fmt::format("Invalid raw action: {}", raw_action)); 
    }
    return ChanceHighLowAction(raw_action == 1);
}

Action ChanceHighLowAction::encode() const {
    return is_high_ ? 1 : 0;
}

std::string ChanceHighLowAction::ToString() const {
    return fmt::format("Environment chooses {} contract settlement", is_high_ ? "high" : "low");
}

// PlayerQuoteAction methods
PlayerQuoteAction PlayerQuoteAction::decode(Action raw_action, int max_contracts_per_trade, int max_contract_value) {
    // max_contract_value and max_contracts_per_trade are both inclusive
    // Bidding 0 size is allowed, but bidding 0 price is not; we manually add 1 to the price 
    // Action range: [0, (max_contracts + 1) ^ 2 * (max_contract_value) ^ 2]
    int rolling = raw_action; 
    int bid_size_denom = (max_contracts_per_trade + 1) * max_contract_value * max_contract_value; 
    int bid_size = rolling / bid_size_denom; 
    rolling = rolling % bid_size_denom; 
    int ask_size_denom = max_contract_value * max_contract_value; 
    int ask_size = rolling / ask_size_denom; 
    rolling = rolling % ask_size_denom; 
    
    int bid_price_denom = max_contract_value; 
    int bid_price = rolling / bid_price_denom + 1; 
    rolling = rolling % bid_price_denom; 
    int ask_price = rolling + 1; 
    return PlayerQuoteAction(bid_size, bid_price, ask_size, ask_price);
}

Action PlayerQuoteAction::encode(int max_contracts_per_trade, int max_contract_value) const {
    int adjusted_bid_price = bid_price_ - 1; 
    int adjusted_ask_price = ask_price_ - 1; 

    return adjusted_ask_price + adjusted_bid_price * max_contract_value + 
        ask_size_ * max_contract_value * max_contract_value + 
        bid_size_ * (max_contracts_per_trade + 1) * max_contract_value * max_contract_value;
}

std::string PlayerQuoteAction::ToString() const {
    return fmt::format("{} @ {} [{} x {}]", bid_price_, ask_price_, bid_size_, ask_size_);
}

// Helper function to compute inverse permutation
std::vector<int> compute_inverse_permutation(const std::vector<int>& permutation) {
    std::vector<int> inv_permutation(permutation.size());
    for (int i = 0; i < permutation.size(); ++i) {
        inv_permutation[permutation[i]] = i;
    }
    return inv_permutation;
}

// Constructors for ChancePermutationAction
ChancePermutationAction::ChancePermutationAction(int n) {
    permutation_.resize(n);
    inv_permutation_.resize(n);
    for (int i = 0; i < n; ++i) {
        permutation_[i] = i;
        inv_permutation_[i] = i;
    }
}

ChancePermutationAction::ChancePermutationAction(const std::vector<int>& permutation) 
    : permutation_(permutation), inv_permutation_(compute_inverse_permutation(permutation)) {
}

ChancePermutationAction::ChancePermutationAction(std::vector<int>&& permutation) 
    : permutation_(std::move(permutation)), inv_permutation_(compute_inverse_permutation(permutation_)) {
}

// ChancePermutationAction methods
ChancePermutationAction ChancePermutationAction::decode(Action raw_action, int num_players) {
    if (raw_action < 0 || raw_action >= factorial(num_players)) {
        AstraFatalError(fmt::format("Invalid raw action: {}", raw_action)); 
    }
    std::vector<int> perm = nth_permutation(raw_action, num_players); 
    return ChancePermutationAction(perm);
}

Action ChancePermutationAction::encode() const {
    return permutation_rank(permutation_);
}

// Gets the player_id in permuted basis. 
// If permutation[id] = 0 or 1, then is value cheater. If [2], then HighLowCheater. 
PlayerRole ChancePermutationAction::GetPlayerRole(int player_id) const {
    if (player_id < 0 || player_id >= permutation_.size()) {
        AstraFatalError(fmt::format("Invalid player id: {}", player_id)); 
    }
    int permed_id = permutation_[player_id]; 
    if (permed_id == 0 || permed_id == 1) {
        return PlayerRole::kValueCheater; 
    } else if (permed_id == 2) {
        return PlayerRole::kHighLowCheater; 
    } else {
        return PlayerRole::kCustomer; 
    }
}

std::string ChancePermutationAction::ToString() const {
    std::string result; 
    for (size_t i = 0; i < permutation_.size(); ++i) {
        if (i > 0) {
            result += ", ";
        }
        result += "P" + std::to_string(i) + "=";
        if (permutation_[i] == 0) {
            result += "ValueCheater1"; 
        } else if (permutation_[i] == 1) {
            result += "ValueCheater2"; 
        } else if (permutation_[i] == 2) {
            result += "HighLowCheater"; 
        } else {
            result += "Customer" + std::to_string(permutation_[i] - 3); 
        }
    }
    return result;
}

// ChanceCustomerSizeAction methods
ChanceCustomerSizeAction ChanceCustomerSizeAction::decode(Action raw_action, int customer_max_size) {
    // 0 gets mapped to most negative size; customer size can't be 0 
    // Action range: [0, 2 * CustomerMaxSize]. Customer size range: [-CustomerMaxSize, CustomerMaxSize] - {0}
    if (raw_action < 0 || raw_action > 2 * customer_max_size) {
        AstraFatalError(fmt::format("Invalid raw action: {}", raw_action)); 
    }
    int customer_size = raw_action - customer_max_size; 
    if (customer_size >= 0) {
        customer_size++;
    }
    return ChanceCustomerSizeAction(customer_size);
}

Action ChanceCustomerSizeAction::encode(int customer_max_size) const {
    int adjusted_size = customer_size_ > 0 ? customer_size_ - 1 : customer_size_; 
    return adjusted_size + customer_max_size;
}

std::string ChanceCustomerSizeAction::ToString() const {
    return fmt::format("Customer target position: {}", customer_size_);
}

// ActionVariant utility functions
std::string ActionVariantToString(const ActionVariant& action) {
    return std::visit([](const auto& a) { return a.ToString(); }, action);
}

std::ostream& operator<<(std::ostream& os, const ActionVariant& action) {
    os << ActionVariantToString(action);
    return os;
}

std::ostream& operator<<(std::ostream& os, const GamePhase& phase) {
    switch (phase) {
        case GamePhase::kChanceValue: return os << "kChanceValue";
        case GamePhase::kChanceHighLow: return os << "kChanceHighLow";
        case GamePhase::kChancePermutation: return os << "kChancePermutation";
        case GamePhase::kCustomerSize: return os << "kCustomerSize";
        case GamePhase::kPlayerTrading: return os << "kPlayerTrading";
        case GamePhase::kTerminal: return os << "kTerminal";
    }
    return os << "UnknownGamePhase";
}

std::ostream& operator<<(std::ostream& os, const PlayerRole& role) {
    switch (role) {
        case PlayerRole::kValueCheater: return os << "kValueCheater";
        case PlayerRole::kHighLowCheater: return os << "kHighLowCheater";
        case PlayerRole::kCustomer: return os << "kCustomer";
    }
    return os << "UnknownPlayerRole";
}


GamePhase ActionManager::game_phase_of_timestep(int timestep) const {
    if (timestep < 0) {
        AstraFatalError(fmt::format("Invalid timestep: {}", timestep)); 
    } else if (timestep < 2) {
        return GamePhase::kChanceValue; 
    } else if (timestep == 2) {
        return GamePhase::kChanceHighLow; 
    } else if (timestep == 3) {
        return GamePhase::kChancePermutation; 
    } else if (timestep < 1 + config_.num_players_) {
        return GamePhase::kCustomerSize; 
    } else if (timestep < 1 + config_.num_players_ + config_.steps_per_player_ * config_.num_players_) {
        return GamePhase::kPlayerTrading; 
    } else {
        return GamePhase::kTerminal; 
    }
}

int factorial(int n) {
    int result = 1; 
    for (int i = 1; i <= n; ++i) {
        result *= i; 
    }
    return result; 
}

std::pair<int, int> ActionManager::valid_action_range(GamePhase phase) const {
    // std::cout << "valid_action_range called with phase: " << phase << std::endl; 
    // std::cout << "config_: " << config_.ToString() << std::endl; 
    switch (phase) {
        case GamePhase::kChanceValue: return {0, config_.max_contract_value_ - 1}; 
        case GamePhase::kChanceHighLow: return {0, 1}; 
        case GamePhase::kChancePermutation: return {0, factorial(config_.num_players_) - 1}; 
        case GamePhase::kCustomerSize: return {0, 2 * config_.customer_max_size_ - 1}; 
        case GamePhase::kPlayerTrading: return {0, (config_.max_contracts_per_trade_ + 1) * (config_.max_contracts_per_trade_ + 1) * config_.max_contract_value_ * config_.max_contract_value_ - 1}; 
        case GamePhase::kTerminal: AstraFatalError("Invalid terminal phase for action range"); 
    }
    // Default case to suppress compiler warning
    AstraFatalError("Unhandled GamePhase in valid_action_range"); 
}

ActionVariant ActionManager::Encode(GamePhase phase, Action raw_action) const {
    auto [min_range, max_range] = valid_action_range(phase); 
    if (raw_action < min_range || raw_action > max_range) {
        AstraFatalError(fmt::format("Invalid raw action: {}", raw_action)); 
    }
    switch (phase) {
        case GamePhase::kChanceValue: {
            return ChanceContractValueAction::decode(raw_action);
        }
        case GamePhase::kChanceHighLow: {
            return ChanceHighLowAction::decode(raw_action);
        }
        case GamePhase::kChancePermutation: {
            return ChancePermutationAction::decode(raw_action, config_.num_players_);
        }
        case GamePhase::kCustomerSize: {
            return ChanceCustomerSizeAction::decode(raw_action, config_.customer_max_size_);
        }
        case GamePhase::kPlayerTrading: {
            return PlayerQuoteAction::decode(raw_action, config_.max_contracts_per_trade_, config_.max_contract_value_);
        }
        case GamePhase::kTerminal: {
            AstraFatalError("Invalid terminal phase for action conversion"); 
        }
    }
    
    // This should never be reached, but needed to suppress compiler warning
    AstraFatalError("Unhandled GamePhase in Encode"); 
}

ActionVariant ActionManager::Encode(int timestep, Action raw_action) const {
    return Encode(game_phase_of_timestep(timestep), raw_action); 
}

Action ActionManager::Decode(GamePhase phase, const ActionVariant& structured_action) const {
    switch (phase) {
        case GamePhase::kChanceValue: {
            return std::get<ChanceContractValueAction>(structured_action).encode();
        }
        case GamePhase::kChanceHighLow: {
            return std::get<ChanceHighLowAction>(structured_action).encode();
        }
        case GamePhase::kChancePermutation: {
            return std::get<ChancePermutationAction>(structured_action).encode();
        } 
        case GamePhase::kCustomerSize: {
            return std::get<ChanceCustomerSizeAction>(structured_action).encode(config_.customer_max_size_);
        }
        case GamePhase::kPlayerTrading: {
            return std::get<PlayerQuoteAction>(structured_action).encode(config_.max_contracts_per_trade_, config_.max_contract_value_);
        }
        case GamePhase::kTerminal: {
            AstraFatalError("Invalid terminal phase for action conversion"); 
        }
    }
    AstraFatalError("Unhandled GamePhase in Decode"); 
}

std::vector<int> nth_permutation(int x, int n)
{
    // ---------- pre-compute factorials up to n (fits if x < n!) ----------
    std::vector<int> fact(n + 1, 1);
    for (int i = 1; i <= n; ++i) {
        fact[i] = fact[i - 1] * i;
    }

    // ---------- Lehmer code digits ----------
    std::vector<int> lehmer(n);
    for (int i = n - 1; i >= 0; --i) {
        lehmer[n - 1 - i] = x / fact[i];
        x %= fact[i];
    }

    // ---------- decode Lehmer code into the permutation ----------
    std::vector<int> pool(n);                    // remaining elements
    std::iota(pool.begin(), pool.end(), 0);      // 0 1 2 … n-1

    std::vector<int> perm;
    perm.reserve(n);

    for (int d : lehmer) {
        perm.push_back(pool[d]);
        pool.erase(pool.begin() + d);            // O(n) per erase
    }
    return perm;
}

int permutation_rank(const std::vector<int>& perm)
{
    const int n = static_cast<int>(perm.size());

    // ---------- factorial table ----------
    std::vector<int> fact(n + 1, 1);
    for (int i = 1; i <= n; ++i) {
        fact[i] = fact[i - 1] * static_cast<int>(i);
    }

    // ---------- build a mutable pool ----------
    std::vector<int> pool(n);
    std::iota(pool.begin(), pool.end(), 0);  // 0 1 2 … n-1

    // ---------- accumulate rank ----------
    int rank = 0;
    for (int i = 0; i < n; ++i) {
        auto it = std::find(pool.begin(), pool.end(), perm[i]);
        int idx  = static_cast<int>(it - pool.begin());   // elements "skipped"
        rank    += static_cast<int>(idx) * fact[n - 1 - i];
        pool.erase(it);                                   // remove used element
    }
    return rank;
}

// VecActionManager implementation
VecActionManager::VecActionManager(const Config& config, int num_envs) 
    : config_(config), num_envs_(num_envs) {
    ASTRA_CHECK_GE(config.num_players_, 4);
    ASTRA_CHECK_GT(num_envs, 0);
}

std::vector<Action> VecActionManager::EncodeTradingActions(const torch::Tensor& actions_tensor) const {
    // Check tensor device and type
    if (!actions_tensor.is_cpu()) {
        AstraFatalError("EncodeTradingActions: tensor must be on CPU device");
    }
    if (actions_tensor.scalar_type() != torch::kInt) {
        AstraFatalError("EncodeTradingActions: tensor must have int32 dtype");
    }
    
    // Check tensor shape: [num_envs, 4]
    ASTRA_CHECK_EQ(actions_tensor.dim(), 2);
    ASTRA_CHECK_EQ(actions_tensor.size(0), num_envs_);
    ASTRA_CHECK_EQ(actions_tensor.size(1), 4);
    
    auto accessor = actions_tensor.accessor<int, 2>();
    
    std::vector<Action> result;
    result.reserve(num_envs_);
    
    for (int env = 0; env < num_envs_; ++env) {
        // Extract values: tensor columns are [bid_price, ask_price, bid_size, ask_size]
        int bid_price = accessor[env][0];
        int ask_price = accessor[env][1]; 
        int bid_size = accessor[env][2];
        int ask_size = accessor[env][3];
        
        // Validate ranges
        ASTRA_CHECK_GE(bid_size, 0);
        ASTRA_CHECK_LE(bid_size, config_.max_contracts_per_trade_);
        ASTRA_CHECK_GE(ask_size, 0);
        ASTRA_CHECK_LE(ask_size, config_.max_contracts_per_trade_);
        ASTRA_CHECK_GE(bid_price, 1);
        ASTRA_CHECK_LE(bid_price, config_.max_contract_value_);
        ASTRA_CHECK_GE(ask_price, 1);
        ASTRA_CHECK_LE(ask_price, config_.max_contract_value_);
        
        // Create PlayerQuoteAction and encode
        // Note: PlayerQuoteAction constructor takes (bid_size, bid_price, ask_size, ask_price)
        PlayerQuoteAction action(bid_size, bid_price, ask_size, ask_price);
        Action encoded = action.encode(config_.max_contracts_per_trade_, config_.max_contract_value_);
        result.push_back(encoded);
    }
    
    return result;
}

std::vector<Action> VecActionManager::EncodePermutations(const torch::Tensor& permutations_tensor) const {
    // Check tensor device and type
    if (!permutations_tensor.is_cpu()) {
        AstraFatalError("EncodePermutations: tensor must be on CPU device");
    }
    if (permutations_tensor.scalar_type() != torch::kInt) {
        AstraFatalError("EncodePermutations: tensor must have int32 dtype");
    }
    
    // Check tensor shape: [num_envs, num_players]
    ASTRA_CHECK_EQ(permutations_tensor.dim(), 2);
    ASTRA_CHECK_EQ(permutations_tensor.size(0), num_envs_);
    ASTRA_CHECK_EQ(permutations_tensor.size(1), config_.num_players_);
    
    auto accessor = permutations_tensor.accessor<int, 2>();
    
    std::vector<Action> result;
    result.reserve(num_envs_);
    
    for (int env = 0; env < num_envs_; ++env) {
        // Extract permutation for this environment
        std::vector<int> permutation(config_.num_players_);
        for (int i = 0; i < config_.num_players_; ++i) {
            permutation[i] = accessor[env][i];
        }
        
        // Validate permutation: should be a valid permutation of [0, 1, ..., num_players-1]
        std::vector<int> sorted_perm = permutation;
        std::sort(sorted_perm.begin(), sorted_perm.end());
        for (int i = 0; i < config_.num_players_; ++i) {
            ASTRA_CHECK_EQ(sorted_perm[i], i);
        }
        
        // Create ChancePermutationAction and encode
        ChancePermutationAction action(permutation);
        Action encoded = action.encode();
        result.push_back(encoded);
    }
    
    return result;
}

std::vector<Action> VecActionManager::EncodeContractValues(const torch::Tensor& values_tensor) const {
    // Check tensor device and type
    if (!values_tensor.is_cpu()) {
        AstraFatalError("EncodeContractValues: tensor must be on CPU device");
    }
    if (values_tensor.scalar_type() != torch::kInt) {
        AstraFatalError("EncodeContractValues: tensor must have int32 dtype");
    }
    
    // Check tensor shape: [num_envs]
    ASTRA_CHECK_EQ(values_tensor.dim(), 1);
    ASTRA_CHECK_EQ(values_tensor.size(0), num_envs_);
    
    auto accessor = values_tensor.accessor<int, 1>();
    
    std::vector<Action> result;
    result.reserve(num_envs_);
    
    for (int env = 0; env < num_envs_; ++env) {
        int contract_value = accessor[env];
        
        // Validate range: [1, max_contract_value]
        ASTRA_CHECK_GE(contract_value, 1);
        ASTRA_CHECK_LE(contract_value, config_.max_contract_value_);
        
        // Create ChanceContractValueAction and encode
        ChanceContractValueAction action(contract_value);
        Action encoded = action.encode();
        result.push_back(encoded);
    }
    
    return result;
}

std::vector<Action> VecActionManager::EncodeHighLowActions(const torch::Tensor& high_low_tensor) const {
    // Check tensor device and type
    if (!high_low_tensor.is_cpu()) {
        AstraFatalError("EncodeHighLowActions: tensor must be on CPU device");
    }
    if (high_low_tensor.scalar_type() != torch::kInt) {
        AstraFatalError("EncodeHighLowActions: tensor must have int32 dtype");
    }
    
    // Check tensor shape: [num_envs]
    ASTRA_CHECK_EQ(high_low_tensor.dim(), 1);
    ASTRA_CHECK_EQ(high_low_tensor.size(0), num_envs_);
    
    auto accessor = high_low_tensor.accessor<int, 1>();
    
    std::vector<Action> result;
    result.reserve(num_envs_);
    
    for (int env = 0; env < num_envs_; ++env) {
        int is_high_int = accessor[env];
        
        // Validate range: 0 or 1
        ASTRA_CHECK_GE(is_high_int, 0);
        ASTRA_CHECK_LE(is_high_int, 1);
        
        bool is_high = (is_high_int == 1);
        
        // Create ChanceHighLowAction and encode
        ChanceHighLowAction action(is_high);
        Action encoded = action.encode();
        result.push_back(encoded);
    }
    
    return result;
}

std::vector<Action> VecActionManager::EncodeCustomerSizes(const torch::Tensor& sizes_tensor) const {
    // Check tensor device and type
    if (!sizes_tensor.is_cpu()) {
        AstraFatalError("EncodeCustomerSizes: tensor must be on CPU device");
    }
    if (sizes_tensor.scalar_type() != torch::kInt) {
        AstraFatalError("EncodeCustomerSizes: tensor must have int32 dtype");
    }
    
    // Check tensor shape: [num_envs]
    ASTRA_CHECK_EQ(sizes_tensor.dim(), 1);
    ASTRA_CHECK_EQ(sizes_tensor.size(0), num_envs_);
    
    auto accessor = sizes_tensor.accessor<int, 1>();
    
    std::vector<Action> result;
    result.reserve(num_envs_);
    
    for (int env = 0; env < num_envs_; ++env) {
        int customer_size = accessor[env];
        
        // Validate range: [-customer_max_size, customer_max_size] excluding 0
        ASTRA_CHECK_GE(customer_size, -config_.customer_max_size_);
        ASTRA_CHECK_LE(customer_size, config_.customer_max_size_);
        ASTRA_CHECK_NE(customer_size, 0); // Customer size cannot be 0
        
        // Create ChanceCustomerSizeAction and encode
        ChanceCustomerSizeAction action(customer_size);
        Action encoded = action.encode(config_.customer_max_size_);
        result.push_back(encoded);
    }
    
    return result;
}

}  // namespace high_low_trading
}  // namespace astra
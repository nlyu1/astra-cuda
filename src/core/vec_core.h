#pragma once

// Synchronized, vectorized wrapper around base game 

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <optional>
#include <numeric>
#include <torch/torch.h>

#include "game_parameters.h"
#include "core.h"

namespace astra {

constexpr torch::Dtype kStateTensorType = torch::kFloat32; // For both info and obs tensors
constexpr torch::Dtype kRewardTensorType = torch::kFloat32; 

ExposeInfo vectorize_expose_info(std::span<ExposeInfo> infos);

// Assumes (BUT DOES NOT CHECK) that environments proceed in lockstep. 
// CurrentPlayer and termination should both proceed in lockstep. 
// Further assumes that game is termina-rewardl, so only provides episode-end rewards. 
class VecState {
    public: 
        ~VecState() = default; 
        // Reference state is not consumed but cloned 
        VecState(std::unique_ptr<State> reference_state, uint num_copies);  // check 
            
        // Access methods
        State& at(uint id) {return *states_.at(id);}
        const State& at(uint id) const {return *states_.at(id);}
        uint GetNumCopies() const {return num_copies_;} 
        
        // Batch operations
        void ApplyActions(std::span<const Action> actions);  // check 
        
        void ObservationTensor(Player player, torch::Tensor& buffer) const; // check 
        torch::Tensor ObservationTensor(Player player) const;  // check 
        std::vector<std::string> ObservationString(Player player) const; 

        void InformationStateTensor(Player player, torch::Tensor& buffer) const; // check 
        torch::Tensor InformationStateTensor(Player player) const; // check 
        std::vector<std::string> InformationStateString(Player player) const; // check 

        void Reset(); // check
        torch::Tensor Returns() const; // [num_copies_, num_players]

        VecState(const VecState&) = delete;
        VecState& operator=(const VecState&) = delete;
        VecState(VecState&&) = default;
        VecState& operator=(VecState&&) = default;

        // General reduce operation: (state, uid) -> output. Cannot mutate state. 
        template<typename T>
        std::vector<T> map_(std::function<T(const State&, uint)> func) const {
            std::vector<T> result;
            result.reserve(num_copies_);
            for (int env_index = 0; env_index < num_copies_; ++env_index) {
                result.emplace_back(func(*states_[env_index], env_index));
            }
            return result;
        }

    private:
        uint num_copies_; 
        std::unique_ptr<State> input_state_; 
        std::vector<std::unique_ptr<State>> states_; 
};


class AsyncVecState {
    // AsyncVecState is a vector of VecState objects, but it implicitly "hides" the asynchronous dimension
    public: 
        ~AsyncVecState() = default; 
        AsyncVecState(std::unique_ptr<State> reference_state, uint num_sync_copies, uint num_async_copies); 
            
        // Access methods
        State& at(uint id) {return vec_states_[id / num_sync_copies_]->at(id % num_sync_copies_);}
        const State& at(uint id) const {return vec_states_[id / num_sync_copies_]->at(id % num_sync_copies_);}
        std::unique_ptr<State> clone_at(uint id) const {
            return vec_states_[id / num_sync_copies_]->at(id % num_sync_copies_).Clone();}
        uint GetTotalNumCopies() const {return num_copies_;} 
        uint GetNumSyncCopies() const {return num_sync_copies_;} 
        uint GetNumAsyncCopies() const {return num_async_copies_;} 
        
        // Batch operations
        void ApplyActions(std::span<const Action> actions);  // check
        
        torch::Tensor ObservationTensor(Player player) const; // check 
        std::vector<std::string> ObservationString(Player player) const; // check 

        // Information state is the same as observation for perfect-information games
        // We only expose value-return on this top level
        torch::Tensor InformationStateTensor(Player player) const; // check 
        std::vector<std::string> InformationStateString(Player player) const; // check 

        // Query operations
        bool IsTerminal() const; // returns first env 
        std::vector<Action> LegalActions() const; // returns first env 
        Player CurrentPlayer() const; // returns first env 
        torch::Tensor Returns() const;
        torch::Tensor Rewards() const;
        ExposeInfo expose_info() const; 
        void Reset(); // check 

        // General reduce operation
        template<typename T>
        std::vector<T> map_(std::function<T(const State&, uint)> func) const {
            std::vector<T> result;
            std::vector<std::vector<T>> temp_results(num_async_copies_);
            result.reserve(num_copies_);

            #pragma omp parallel for
            for (int i = 0; i < num_async_copies_; ++i) {
                // Include offset within each synchronized environment
                uint offset = i * num_sync_copies_;
                std::function<T(const State&, uint)> wrapped_func = 
                    [&](const State& state, uint local_index) {
                        uint global_index = offset + local_index;
                        return func(state, global_index);
                    };
                temp_results[i] = vec_states_[i]->map_(wrapped_func);
            }

            for (int env_index = 0; env_index < num_async_copies_; ++env_index) {
                result.insert(result.end(), temp_results[env_index].begin(), temp_results[env_index].end());    
            }
            return result;
        }

    private:
        uint num_sync_copies_; 
        uint num_async_copies_; 
        uint num_copies_; 
        std::vector<std::unique_ptr<VecState>> vec_states_; 
};

}
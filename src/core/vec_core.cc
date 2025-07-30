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
#include <unordered_set>
#include <torch/torch.h>
#include "vec_core.h"
#include "astra_utils.h"

namespace astra {
 
// Usage: state = load_game().initial_state(); VecState(*state); 
VecState::VecState(std::unique_ptr<State> reference_state, uint num_copies): 
        input_state_(reference_state->Clone()), num_copies_(num_copies) {
    states_.reserve(num_copies);  // Pre-allocate for efficiency
    for (int j = 0; j < num_copies; j++) {
        states_.emplace_back(input_state_->Clone()); 
    }
}

void VecState::ApplyActions(std::span<const Action> actions) {
    ASTRA_CHECK_EQ(actions.size(), num_copies_);
    for (uint i = 0; i < num_copies_; ++i) {
        states_[i]->ApplyAction(actions[i]);
    }
}

void VecState::Reset() {
    for (int i = 0; i < num_copies_; ++i) {
        states_[i] = input_state_->Clone();
    }
}

std::vector<std::string> VecState::InformationStateString(Player player) const {
    return map_<std::string>([player](const State& state, uint _uid) {
        return state.InformationStateString(player);
    });
}   

std::vector<std::string> VecState::ObservationString(Player player) const {
    return map_<std::string>([player](const State& state, uint _uid) {
        return state.ObservationString(player);
    });
}

void VecState::InformationStateTensor(Player player, torch::Tensor& buffer) const {
    std::vector<int> single_shape_int = states_[0]->GetGame()->InformationStateTensorShape();
    std::vector<long> expected_shape(single_shape_int.begin(), single_shape_int.end());
    expected_shape.insert(expected_shape.begin(), num_copies_);
    
    // Received buffer with incorrect shape
    ASTRA_CHECK_EQ(buffer.sizes().vec(), expected_shape);
    // Received non-contiguous buffer
    ASTRA_CHECK_TRUE(buffer.is_contiguous());
    // Received buffer with incorrect dtype
    ASTRA_CHECK_EQ(buffer.scalar_type(), kStateTensorType);

    // --- Fill Buffer ---
    for (uint env_index = 0; env_index < num_copies_; ++env_index) {
        // Get a view of the slice for the current environment
        torch::Tensor slice_view = buffer.index({(long)env_index});
        
        // Create a span to the slice's data and fill it
        std::span<ObservationScalarType> slice_span(
            slice_view.data_ptr<ObservationScalarType>(), slice_view.numel());
        states_[env_index]->InformationStateTensor(player, slice_span);
    }
}

bool AsyncVecState::IsTerminal() const {
    return vec_states_[0]->at(0).IsTerminal();
}

std::vector<Action> AsyncVecState::LegalActions() const {
    return vec_states_[0]->at(0).LegalActions();
}

Player AsyncVecState::CurrentPlayer() const {
    return vec_states_[0]->at(0).CurrentPlayer();
}


void VecState::ObservationTensor(Player player, torch::Tensor& buffer) const {
    std::vector<int> single_shape_int = states_[0]->GetGame()->ObservationTensorShape();
    std::vector<long> expected_shape(single_shape_int.begin(), single_shape_int.end());
    expected_shape.insert(expected_shape.begin(), num_copies_);
    ASTRA_CHECK_EQ(buffer.sizes().vec(), expected_shape);
    ASTRA_CHECK_TRUE(buffer.is_contiguous());
    ASTRA_CHECK_EQ(buffer.scalar_type(), kStateTensorType);

    for (uint env_index = 0; env_index < num_copies_; ++env_index) {
        torch::Tensor slice_view = buffer.index({(long)env_index});
        std::span<ObservationScalarType> slice_span(
            slice_view.data_ptr<ObservationScalarType>(), slice_view.numel());
        states_[env_index]->ObservationTensor(player, slice_span);
    }
}

torch::Tensor VecState::InformationStateTensor(Player player) const {
    auto option = torch::TensorOptions().dtype(kStateTensorType);
    std::vector<int> single_shape_int = states_[0]->GetGame()->InformationStateTensorShape();
    std::vector<long> result_shape(single_shape_int.begin(), single_shape_int.end());
    result_shape.insert(result_shape.begin(), num_copies_);

    torch::Tensor result = torch::zeros(result_shape, option);
    InformationStateTensor(player, result); 
    return result;
}

torch::Tensor VecState::ObservationTensor(Player player) const {
    auto option = torch::TensorOptions().dtype(kStateTensorType);
    std::vector<int> single_shape_int = states_[0]->GetGame()->ObservationTensorShape();
    std::vector<long> result_shape(single_shape_int.begin(), single_shape_int.end());
    result_shape.insert(result_shape.begin(), num_copies_);

    torch::Tensor result = torch::zeros(result_shape, option);
    ObservationTensor(player, result); 
    return result;
}

// AsyncVecState is a vector of VecState objects.  

AsyncVecState::AsyncVecState(std::unique_ptr<State> reference_state, uint num_sync_copies, uint num_async_copies): 
        num_sync_copies_(num_sync_copies), num_async_copies_(num_async_copies) {
    vec_states_.clear();
    num_copies_ = num_sync_copies_ * num_async_copies_;
    vec_states_.reserve(num_async_copies_);  // Pre-allocate for efficiency
    for (int j = 0; j < num_async_copies_; j++) {
        vec_states_.emplace_back(
            std::make_unique<VecState>(reference_state->Clone(), num_sync_copies_)
        ); 
    }
}

void AsyncVecState::ApplyActions(std::span<const Action> actions) {
    ASTRA_CHECK_EQ(actions.size(), num_copies_);

    #pragma omp parallel for
    for (uint i = 0; i < num_async_copies_; ++i) {
        vec_states_[i]->ApplyActions(actions.subspan(i * num_sync_copies_, num_sync_copies_));
    }
}

torch::Tensor AsyncVecState::Returns() const {
    auto get_return = [](const State& state, uint env_id) -> torch::Tensor {
        return torch::tensor(state.Returns(), torch::TensorOptions().dtype(kRewardTensorType));
    };
    std::vector<torch::Tensor> all_returns = map_<torch::Tensor>(get_return);
    return torch::stack(all_returns, 0);
}

torch::Tensor AsyncVecState::Rewards() const {
    auto get_reward = [](const State& state, uint env_id) -> torch::Tensor {
        return torch::tensor(state.Rewards(), torch::TensorOptions().dtype(kRewardTensorType));
    };
    std::vector<torch::Tensor> all_rewards = map_<torch::Tensor>(get_reward);
    return torch::stack(all_rewards, 0);
}

void AsyncVecState::Reset() {
    for (int i = 0; i < num_async_copies_; ++i) {
        vec_states_[i]->Reset();
    }
}

std::vector<std::string> AsyncVecState::InformationStateString(Player player) const {
    return map_<std::string>([player](const State& state, uint _uid) {
        return state.InformationStateString(player);
    });
}

std::vector<std::string> AsyncVecState::ObservationString(Player player) const {
    return map_<std::string>([player](const State& state, uint _uid) {
        return state.ObservationString(player);
    });
}


torch::Tensor AsyncVecState::InformationStateTensor(Player player) const {
    // 1. Determine the total shape and create one large tensor (the buffer).
    auto option = torch::TensorOptions().dtype(kStateTensorType);
    std::vector<int> single_shape_int = at(0).GetGame()->InformationStateTensorShape();
    std::vector<long> total_shape(single_shape_int.begin(), single_shape_int.end());
    total_shape.insert(total_shape.begin(), num_copies_);
    
    torch::Tensor result = torch::empty(total_shape, option);

    #pragma omp parallel for
    for (uint i = 0; i < num_async_copies_; ++i) {
        // 3. Get a slice of the main buffer for the i-th VecState.
        //    This slice has the shape [num_sync_copies_, ...], which is what
        //    VecState::InformationStateTensor expects.
        torch::Tensor buffer_slice = result.slice(
            /*dim=*/0, 
            i * num_sync_copies_, /* Start index */
            (i + 1) * num_sync_copies_ /* End index */);
        vec_states_[i]->InformationStateTensor(player, buffer_slice);
    }
    
    return result;
}

torch::Tensor AsyncVecState::ObservationTensor(Player player) const {
    // 1. Determine the total shape and create one large tensor (the buffer).
    auto option = torch::TensorOptions().dtype(kStateTensorType);
    std::vector<int> single_shape_int = at(0).GetGame()->ObservationTensorShape();
    std::vector<long> total_shape(single_shape_int.begin(), single_shape_int.end());
    total_shape.insert(total_shape.begin(), num_copies_);
    
    torch::Tensor result = torch::empty(total_shape, option);

    #pragma omp parallel for
    for (uint i = 0; i < num_async_copies_; ++i) {
        torch::Tensor buffer_slice = result.slice(
            /*dim=*/0, 
            i * num_sync_copies_, /* Start index */
            (i + 1) * num_sync_copies_ /* End index */);
        vec_states_[i]->ObservationTensor(player, buffer_slice);
    }
    
    return result;
}

ExposeInfo AsyncVecState::expose_info() const{
    auto gathered = map_<ExposeInfo>(
        [](const State& s, uint){ return s.expose_info(); });
    return vectorize_expose_info(std::span<ExposeInfo>(gathered));
}


// ──────────────────────────────────────────────────────────────────────────
//  Combine a batch of ExposeInfo objects into one "vectorised" ExposeInfo.
//  • Scalars → 1-D vectors
//  • 1-D vectors → 2-D vectors
//  • torch::Tensor → stacked tensors along leading dimension (with shape validation)
//  • If any input already contains a 2-D vector ⇒ fatal error
//  • All infos must share exactly the same key set and the same type per key
// ──────────────────────────────────────────────────────────────────────────
ExposeInfo vectorize_expose_info(std::span<ExposeInfo> infos) {
    ASTRA_CHECK_GT(infos.size(), 0);
  
    // 1. Sanity-check that every info has the same keys
    const ExposeInfo& first = infos.front();
    std::unordered_set<std::string> ref_keys;
    ref_keys.reserve(first.size());
    for (const auto& kv : first) ref_keys.insert(kv.first);
    
    // fast key-set check without rebuilding a set each time
    for (std::size_t i = 1; i < infos.size(); ++i) {
        ASTRA_CHECK_EQ(infos[i].size(), ref_keys.size());
        for (const auto& kv : infos[i])
            ASTRA_CHECK_TRUE(ref_keys.count(kv.first));
    }
  
    // 2. Build the aggregated result
    ExposeInfo out;
  
    for (const auto& [key, exemplar] : first) {
      // ---- inside vectorize_expose_info ---------------------------------
        if (std::holds_alternative<int>(exemplar)) {
            std::vector<int> buf;  buf.reserve(infos.size());
            for (const auto& info : infos) buf.push_back(std::get<int>(info.at(key)));
            out[key] = std::move(buf);
        }
        else if (std::holds_alternative<float>(exemplar)) {
            std::vector<float> buf;  buf.reserve(infos.size());
            for (const auto& info : infos) buf.push_back(std::get<float>(info.at(key)));
            out[key] = std::move(buf);
        }
        else if (std::holds_alternative<std::string>(exemplar)) {
            std::vector<std::string> buf;  buf.reserve(infos.size());
            for (const auto& info : infos) buf.push_back(std::get<std::string>(info.at(key)));
            out[key] = std::move(buf);
        }
        else if (std::holds_alternative<torch::Tensor>(exemplar)) {
            // Collect all tensors for this key and validate shapes
            std::vector<torch::Tensor> tensors;
            tensors.reserve(infos.size());
            
            const torch::Tensor& first_tensor = std::get<torch::Tensor>(exemplar);
            auto expected_shape = first_tensor.sizes();
            
            for (auto& info : infos) {
                torch::Tensor tensor = std::get<torch::Tensor>(info.at(key));
                ASTRA_CHECK_EQ(tensor.sizes(), expected_shape); 
                tensors.emplace_back(std::move(tensor));
            }
            out[key] = torch::stack(tensors, /*dim=*/0);
        }
        else if (std::holds_alternative<std::vector<int>>(exemplar)) {
            std::vector<std::vector<int>> mat;  mat.reserve(infos.size());
            for (const auto& info : infos) mat.push_back(std::get<std::vector<int>>(info.at(key)));
            out[key] = std::move(mat);
        }
        else if (std::holds_alternative<std::vector<float>>(exemplar)) {
            std::vector<std::vector<float>> mat;  mat.reserve(infos.size());
            for (const auto& info : infos) mat.push_back(std::get<std::vector<float>>(info.at(key)));
            out[key] = std::move(mat);
        }
        else if (std::holds_alternative<std::vector<std::string>>(exemplar)) {
            std::vector<std::vector<std::string>> mat;  mat.reserve(infos.size());
            for (const auto& info : infos) mat.push_back(std::get<std::vector<std::string>>(info.at(key)));
            out[key] = std::move(mat);
        }
        else if (std::holds_alternative<std::vector<std::vector<int>>>(exemplar) ||
                std::holds_alternative<std::vector<std::vector<float>>>(exemplar) ||
                std::holds_alternative<std::vector<std::vector<std::string>>>(exemplar)){
            AstraFatalError(StrCat("vectorize_expose_info: key '", key,
                                "' already contains a 2-D vector – disallowed as input"));
        }
        else {
            AstraFatalError(StrCat("vectorize_expose_info: unknown variant type for key '", key, "'"));
        }
    }
  
    return out;
  }

}
#include <iostream>
#include <chrono>
#include <memory>
#include <string>
#include <random>
#include <vector>
#include <iomanip>
#include <numeric>

#include "market.h"
#include "high_low_trading/high_low_trading.h"
#include "game_parameters.h"
#include "core.h"

#include <torch/torch.h>

using namespace astra::order_matching;
using namespace astra::high_low_trading;
using namespace astra;

struct BenchmarkConfig {
    int num_envs;
    int threads_per_block;
    int device_id;
};

struct BenchmarkResult {
    BenchmarkConfig config;
    double avg_step_time_ms;
    double max_step_time_ms;
    double total_time_seconds;
    double fps;  // Frames (env-steps) per second
    int total_steps;
    int total_frames;
};

class EnvironmentBenchmark {
public:
    EnvironmentBenchmark(const GameParameters& params)
        : game_(std::make_shared<HighLowTradingGame>(params)),
          state_(game_->NewInitialState()),
          hlt_state_(static_cast<HighLowTradingState*>(state_.get())),
          hlt_game_(static_cast<const HighLowTradingGame*>(game_.get())),
          num_players_(hlt_game_->GetNumPlayers()),
          steps_per_player_(hlt_game_->GetStepsPerPlayer()),
          max_contract_value_(hlt_game_->GetMaxContractValue()),
          customer_max_size_(hlt_game_->GetCustomerMaxSize()),
          num_envs_(hlt_game_->GetNumMarkets()),
          device_id_(hlt_game_->GetDeviceId()),
          device_(torch::kCUDA, device_id_),
          options_i32_(torch::TensorOptions().dtype(torch::kInt32).device(device_)) {
        
        // Pre-allocate persistent tensors for actions
        candidate_values_ = torch::zeros({num_envs_, 2}, options_i32_);
        high_low_settle_ = torch::zeros({num_envs_}, options_i32_);
        permutation_ = torch::zeros({num_envs_, num_players_}, options_i32_);
        customer_sizes_ = torch::zeros({num_envs_, num_players_ - 3}, options_i32_);
        trading_action_ = torch::zeros({num_envs_, 4}, options_i32_);
        
        // Pre-allocate reward buffers
        immediate_rewards_ = torch::zeros({num_envs_, num_players_}, options_i32_);
        player_rewards_ = torch::zeros({num_envs_}, options_i32_);
        terminal_rewards_ = torch::zeros({num_envs_, num_players_}, options_i32_);
        
        // Initialize RNG
        rng_ = std::mt19937(42);
    }
    
    void RandomizeActions(int move_number) {
        if (move_number == 0) {
            // First chance move: two candidate contract values [1, max_contract_value]
            std::uniform_int_distribution<int> value_dist(1, max_contract_value_);
            for (int i = 0; i < num_envs_; ++i) {
                candidate_values_[i][0] = value_dist(rng_);
                candidate_values_[i][1] = value_dist(rng_);
            }
        } else if (move_number == 1) {
            // Second chance move: high/low settlement (0 or 1)
            std::uniform_int_distribution<int> binary_dist(0, 1);
            for (int i = 0; i < num_envs_; ++i) {
                high_low_settle_[i] = binary_dist(rng_);
            }
        } else if (move_number == 2) {
            // Third chance move: permutation
            // Generate a fixed permutation and use it for all environments
            std::vector<int> perm(num_players_);
            std::iota(perm.begin(), perm.end(), 0);
            std::shuffle(perm.begin(), perm.end(), rng_);
            
            for (int i = 0; i < num_envs_; ++i) {
                for (int j = 0; j < num_players_; ++j) {
                    permutation_[i][j] = perm[j];
                }
            }
        } else if (move_number == 3) {
            // Fourth chance move: customer sizes [-customer_max_size, customer_max_size] excluding 0
            std::uniform_int_distribution<int> size_dist(1, customer_max_size_);
            std::uniform_int_distribution<int> sign_dist(0, 1);
            
            for (int i = 0; i < num_envs_; ++i) {
                for (int j = 0; j < num_players_ - 3; ++j) {
                    int size = size_dist(rng_);
                    customer_sizes_[i][j] = sign_dist(rng_) ? size : -size;
                }
            }
        } else {
            // Player trading actions: [bid_px, ask_px, bid_sz, ask_sz]
            std::uniform_int_distribution<int> price_dist(0, max_contract_value_);
            std::uniform_int_distribution<int> size_dist(0, 10);  // Reasonable size range
            
            for (int i = 0; i < num_envs_; ++i) {
                trading_action_[i][0] = price_dist(rng_);  // bid_px
                trading_action_[i][1] = price_dist(rng_);  // ask_px
                trading_action_[i][2] = size_dist(rng_);   // bid_sz
                trading_action_[i][3] = size_dist(rng_);   // ask_sz
            }
        }
    }
    
    torch::Tensor GetCurrentAction(int move_number) {
        if (move_number == 0) return candidate_values_;
        else if (move_number == 1) return high_low_settle_;
        else if (move_number == 2) return permutation_;
        else if (move_number == 3) return customer_sizes_;
        else return trading_action_;
    }
    
    BenchmarkResult RunBenchmark(int num_episodes) {
        auto start_time = std::chrono::high_resolution_clock::now();
        double max_step_time_ms = 0.0;
        double total_step_time_ms = 0.0;
        int total_steps = 0;
        
        for (int episode = 0; episode < num_episodes; ++episode) {
            state_->Reset();
            
            while (!state_->IsTerminal()) {
                int move_number = state_->MoveNumber();
                Player current_player = state_->CurrentPlayer();
                
                // Generate random actions
                RandomizeActions(move_number);
                torch::Tensor action = GetCurrentAction(move_number);
                
                auto step_start = std::chrono::high_resolution_clock::now();
                
                // Apply action
                state_->ApplyAction(action);
                
                // Fill immediate rewards after each step
                hlt_state_->FillRewards(immediate_rewards_);
                
                // If it's a player action, also get cumulative rewards since last action
                if (current_player >= 0) {
                    hlt_state_->FillRewardsSinceLastAction(player_rewards_, current_player);
                }
                
                auto step_end = std::chrono::high_resolution_clock::now();
                
                auto step_duration = std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start);
                double step_time_ms = step_duration.count() / 1000.0;
                max_step_time_ms = std::max(max_step_time_ms, step_time_ms);
                total_step_time_ms += step_time_ms;
                
                total_steps++;
            }
            
            // Get terminal rewards
            hlt_state_->FillReturns(terminal_rewards_);
            
            // Optionally verify rewards are reasonable
            auto terminal_sum = terminal_rewards_.sum().item<int>();
            if (terminal_sum == 0 && episode == 0) {
                std::cout << "Warning: Terminal rewards sum to zero!" << std::endl;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        double total_time_seconds = duration.count() / 1000000.0;
        double avg_step_time_ms = total_step_time_ms / total_steps;
        int total_frames = total_steps * num_envs_;
        double fps = total_frames / total_time_seconds;
        
        return BenchmarkResult{
            BenchmarkConfig{num_envs_, hlt_game_->GetThreadsPerBlock(), device_id_},
            avg_step_time_ms,
            max_step_time_ms,
            total_time_seconds,
            fps,
            total_steps,
            total_frames
        };
    }
    
private:
    std::shared_ptr<const Game> game_;
    std::unique_ptr<State> state_;
    HighLowTradingState* hlt_state_;
    const HighLowTradingGame* hlt_game_;
    
    int num_players_;
    int steps_per_player_;
    int max_contract_value_;
    int customer_max_size_;
    int num_envs_;
    int device_id_;
    
    torch::Device device_;
    torch::TensorOptions options_i32_;
    
    // Persistent action tensors
    torch::Tensor candidate_values_;
    torch::Tensor high_low_settle_;
    torch::Tensor permutation_;
    torch::Tensor customer_sizes_;
    torch::Tensor trading_action_;
    
    // Reward buffers
    torch::Tensor immediate_rewards_;
    torch::Tensor player_rewards_;
    torch::Tensor terminal_rewards_;
    
    std::mt19937 rng_;
};

void print_results(const std::vector<BenchmarkResult>& results) {
    std::cout << std::fixed << std::setprecision(3);
    
    // Print header
    std::cout << std::setw(12) << "Envs"
              << std::setw(15) << "Threads/Block"
              << std::setw(10) << "Device"
              << std::setw(15) << "Avg Step(ms)"
              << std::setw(15) << "Max Step(ms)"
              << std::setw(15) << "Total Time(s)"
              << std::setw(15) << "FPS"
              << std::endl;
    
    std::cout << std::string(100, '-') << std::endl;
    
    // Print results
    for (const auto& result : results) {
        std::cout << std::setw(12) << result.config.num_envs
                  << std::setw(15) << result.config.threads_per_block
                  << std::setw(10) << result.config.device_id
                  << std::setw(15) << result.avg_step_time_ms
                  << std::setw(15) << result.max_step_time_ms
                  << std::setw(15) << result.total_time_seconds
                  << std::setw(15) << std::fixed << std::setprecision(0) << result.fps
                  << std::endl;
    }
}

int main(int argc, char** argv) {
    std::cout << "=== High Low Trading CUDA Environment Benchmark ===" << std::endl;
    
    // Parse command line arguments for device selection
    int device_id = 0;
    if (argc > 1) {
        device_id = std::atoi(argv[1]);
    }
    
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_id >= device_count) {
        std::cerr << "Error: Device " << device_id << " not found. Only " 
                  << device_count << " devices available." << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::cout << "Using device " << device_id << ": " << prop.name << std::endl;
    std::cout << std::endl;
    
    // Test configurations
    std::vector<int> env_counts = {64, 128, 256};
    std::vector<int> thread_counts = {256, 512, 1024};
    
    std::vector<BenchmarkResult> results;
    
    const int num_episodes = 10;  // Number of episodes per configuration
    
    std::cout << "Running benchmarks with " << num_episodes << " episodes per configuration..." << std::endl;
    std::cout << std::endl;
    
    for (int num_envs : env_counts) {
        for (int threads_per_block : thread_counts) {
            std::cout << "Testing " << num_envs << " environments with " 
                      << threads_per_block << " threads/block..." << std::flush;
            
            // Create game parameters
            GameParameters params = {
                {"steps_per_player", GameParameter(8)},
                {"max_contracts_per_trade", GameParameter(1)},
                {"customer_max_size", GameParameter(2)},
                {"max_contract_value", GameParameter(10)},
                {"players", GameParameter(5)},
                {"num_markets", GameParameter(num_envs)},
                {"threads_per_block", GameParameter(threads_per_block)},
                {"device_id", GameParameter(device_id)}
            };
            
            try {
                EnvironmentBenchmark benchmark(params);
                auto result = benchmark.RunBenchmark(num_episodes);
                results.push_back(result);
                
                std::cout << " Done (FPS: " << std::fixed << std::setprecision(0) 
                          << result.fps << ")" << std::endl;
            } catch (const std::exception& e) {
                std::cout << " Failed: " << e.what() << std::endl;
            }
        }
    }
    
    std::cout << std::endl;
    std::cout << "=== BENCHMARK RESULTS ===" << std::endl;
    std::cout << std::endl;
    
    print_results(results);
    
    std::cout << std::endl;
    std::cout << "=== ANALYSIS ===" << std::endl;
    
    // Find best configuration
    auto best_fps = std::max_element(results.begin(), results.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.fps < b.fps;
        });
    
    if (best_fps != results.end()) {
        std::cout << "Best FPS: " << std::fixed << std::setprecision(0) 
                  << best_fps->fps << " FPS (" 
                  << best_fps->config.num_envs << " envs, " 
                  << best_fps->config.threads_per_block << " threads/block)" << std::endl;
                  
        std::cout << "Best config latency: Avg " << std::fixed << std::setprecision(3) 
                  << best_fps->avg_step_time_ms << "ms, Max " 
                  << best_fps->max_step_time_ms << "ms" << std::endl;
    }
    
    return 0;
}
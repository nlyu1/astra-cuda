```bash
# Run within the "astra" anaconda environment and in root directory 

# First, verify you're in the correct conda environment:
echo "Current conda environment: $CONDA_DEFAULT_ENV"
python -c "import torch; print('PyTorch version:', torch.__version__)"

# Then build:
cmake -S ./src -B ./build/build-release -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build ./build/build-release -j$(nproc)

# Test market throughput. All-else being equal, 5090 offers 3x throughput compared to 4060ti. 
./build/build-release/tests/benchmarks/benchmark_market -i 1
./build/build-release/tests/benchmarks/benchmark_market -i 0
```

Quick benchmarking: running into bugs right now. 
```
cmake -S ./src -B ./build/build-release -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build ./build/build-release -j$(nproc) && ./build/build-release/tests/benchmark_env

compute-sanitizer --tool memcheck ./build/build-release/tests/benchmark_env
```

To run basic experiment:

Small game experiments

1. [Seed run](wandb.ai/lyuxingjian-na/HighLowTrading_Transformer/runs/irikjsip)
2. [Long run](wandb.ai/lyuxingjian-na/HighLowTrading_Transformer/runs/o40ukdqh)
    - Enabled settlement and self-role guessing. 
3. [Bootstrapped from long run](https://wandb.ai/lyuxingjian-na/HighLowTrading_Transformer/runs/bow7tcf0): environment added self-trade penalties. Lower entropy term. Higher weight on private information modeling. Fixed display error for settlement differences
4. [Another bootstrap](https://wandb.ai/lyuxingjian-na/HighLowTrading_Transformer/runs/v52dp8jj): less emphasis on entropy; even more emphasis on private info role and settlement. Smaller lr

```
CUDA_VISIBLE_DEVICES=1 python vtrace_baseline.py --steps_per_player 8 --max_contracts_per_trade 1 --customer_max_size 2 --max_contract_value 10 --players 5 --ent_coef 0.05 --num_steps 8 --num_iterations 3010 --iterations_per_heavy_logging 500 --iterations_per_checkpoint 3000 --exp_name smallgame_seedpool

CUDA_VISIBLE_DEVICES=1 python vtrace_baseline.py --steps_per_player 8 --max_contracts_per_trade 1 --customer_max_size 2 --max_contract_value 10 --players 5 --ent_coef 0.02 --num_steps 8 --psettlement_coef 0.1 --proles_coef 0.05 --num_iterations 3000000000 --iterations_per_pool_update 2000 --iterations_per_heavy_logging 2000 --iterations_per_checkpoint 2000 --exp_name smallgame_pool0 --checkpoint_name smallgame_pool0_6000

CUDA_VISIBLE_DEVICES=1 python vtrace_baseline.py --steps_per_player 8 --max_contracts_per_trade 1 --customer_max_size 2 --max_contract_value 10 --players 5 --ent_coef 0.008 --num_steps 8 --psettlement_coef 1. --proles_coef 0.1 --num_iterations 3000000000 --iterations_per_pool_update 2000 --iterations_per_heavy_logging 2000 --iterations_per_checkpoint 2000 --exp_name smallgame_pool1 --checkpoint_name smallgame_pool0_180000

CUDA_VISIBLE_DEVICES=1 python vtrace_baseline.py --steps_per_player 8 --max_contracts_per_trade 1 --customer_max_size 2 --max_contract_value 10 --players 5 --ent_coef 0.003 --num_steps 8 --psettlement_coef 1. --proles_coef 0.5 --learning_rate 1.5e-4 --num_iterations 3000000000 --iterations_per_pool_update 2000 --iterations_per_heavy_logging 2000 --iterations_per_checkpoint 2000 --exp_name smallgame_pool2 --checkpoint_name smallgame_pool1_330000
```

Full, normal game experiments

1. [Seed run](wandb.ai/lyuxingjian-na/HighLowTrading_Transformer/runs/l5q8vf04)
2. [Long run](wandb.ai/lyuxingjian-na/HighLowTrading_Transformer/runs/hypdfixc)
    - Enabled settlement and self-role guessing. 
3. [Bootstrapped from long run](https://wandb.ai/lyuxingjian-na/HighLowTrading_Transformer/runs/uqpoe863)
    - Less entropy bonus; greater emphasis on private info modeling. Also added intermediate rewards to discourage self-trades. 
4. [Another bootstrap](https://wandb.ai/lyuxingjian-na/HighLowTrading_Transformer/runs/6ind76mu)
    - 
```
CUDA_VISIBLE_DEVICES=0 python vtrace_baseline.py --ent_coef 0.05 --num_iterations 5010 --iterations_per_heavy_logging 500 --iterations_per_checkpoint 5000 --iterations_per_pool_update 5000 --exp_name normalgame_seedpool

CUDA_VISIBLE_DEVICES=0 python vtrace_baseline.py --ent_coef 0.02 --psettlement_coef 0.1 --proles_coef 0.05 --num_iterations 3000000000 --iterations_per_pool_update 3000 --iterations_per_heavy_logging 3000 --iterations_per_checkpoint 3000 --exp_name normalgame_pool0 --checkpoint_name normalgame_seedpool_5000

CUDA_VISIBLE_DEVICES=0 python vtrace_baseline.py --ent_coef 0.01 --psettlement_coef 1. --proles_coef 0.2 --num_iterations 3000000000 --iterations_per_pool_update 2000 --iterations_per_heavy_logging 2000 --iterations_per_checkpoint 2000 --exp_name normalgame_pool1 --checkpoint_name normalgame_pool0_114000

CUDA_VISIBLE_DEVICES=0 python vtrace_baseline.py --ent_coef 0.005 --learning_rate 1.5e-4 --psettlement_coef 1. --proles_coef 0.5 --num_iterations 3000000000 --iterations_per_pool_update 2000 --iterations_per_heavy_logging 2000 --iterations_per_checkpoint 2000 --exp_name normalgame_pool2 --checkpoint_name normalgame_pool1_204000
```

Dev run
```
CUDA_VISIBLE_DEVICES=1 python vtrace_baseline.py --steps_per_player 8 --max_contracts_per_trade 1 --customer_max_size 2 --max_contract_value 10 --players 5 --ent_coef 0.01 --num_steps 8 --psettlement_coef 1. --proles_coef 0.05 --num_iterations 3000000000 --iterations_per_pool_update 2000 --iterations_per_heavy_logging 100 --iterations_per_checkpoint 2000 --exp_name dev --checkpoint_name smallgame_pool0_180000
```

Profile run
```
CUDA_VISIBLE_DEVICES=1 python profile_cpu_bottlenecks.py --steps_per_player 8 --max_contracts_per_trade 1 --customer_max_size 2 --max_contract_value 10 --players 5 --ent_coef 0.05 --num_steps 8 --num_iterations 100 --iterations_per_heavy_logging 1 --iterations_per_checkpoint 1500 --exp_name dev
```
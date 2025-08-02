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
```
# wandb.ai/lyuxingjian-na/HighLowTrading_Transformer/runs/irikjsip
CUDA_VISIBLE_DEVICES=1 python vtrace_baseline.py --steps_per_player 8 --max_contracts_per_trade 1 --customer_max_size 2 --max_contract_value 10 --players 5 --ent_coef 0.05 --num_steps 8 --num_iterations 3100 --iterations_per_heavy_logging 500 --iterations_per_checkpoint 3000 --exp_name smallgame_seedpool

CUDA_VISIBLE_DEVICES=1 python vtrace_baseline.py --steps_per_player 8 --max_contracts_per_trade 1 --customer_max_size 2 --max_contract_value 10 --players 5 --ent_coef 0.04 --num_steps 8 --num_iterations 3000000000 --iterations_per_pool_update 3000 --iterations_per_heavy_logging 500 --iterations_per_checkpoint 3000 --exp_name smallgame_pool0 --checkpoint_name smallgame_seedpool_3000


# wandb.ai/lyuxingjian-na/HighLowTrading_Transformer/runs/l5q8vf04
CUDA_VISIBLE_DEVICES=0 python vtrace_baseline.py --ent_coef 0.05 --num_iterations 5010 --iterations_per_heavy_logging 500 --iterations_per_checkpoint 5000 --iterations_per_pool_update 5000 --exp_name normalgame_seedpool



CUDA_VISIBLE_DEVICES=1 python vtrace_baseline.py --steps_per_player 8 --max_contracts_per_trade 1 --customer_max_size 2 --max_contract_value 10 --players 5 --ent_coef 0.05 --num_steps 8 --num_iterations 100 --iterations_per_heavy_logging 1 --iterations_per_checkpoint 1500 --exp_name dev
```
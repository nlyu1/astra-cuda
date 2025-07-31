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
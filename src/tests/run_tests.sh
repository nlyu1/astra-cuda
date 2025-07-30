#!/bin/bash
# Test runner script for astra-cuda

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Running Astra CUDA Tests ===${NC}"

# Ensure we're in the correct conda environment
if [ "$CONDA_DEFAULT_ENV" != "astra" ]; then
    echo -e "${RED}Error: Not in 'astra' conda environment${NC}"
    echo "Please activate the environment with: conda activate astra"
    exit 1
fi

# Build directory
BUILD_DIR="${1:-../../build/build-release}"

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: Build directory not found: $BUILD_DIR${NC}"
    echo "Please build the project first with the instructions in usage.md"
    exit 1
fi

# Run C++ unit tests
echo -e "\n${YELLOW}Running C++ Unit Tests...${NC}"
if [ -f "$BUILD_DIR/tests/unit_tests/market_tests" ]; then
    echo "Running market_tests..."
    "$BUILD_DIR/tests/unit_tests/market_tests"
    echo -e "${GREEN}✓ market_tests passed${NC}"
else
    echo -e "${RED}market_tests not found${NC}"
fi

if [ -f "$BUILD_DIR/tests/unit_tests/astra_utils_tests" ]; then
    echo -e "\nRunning astra_utils_tests..."
    "$BUILD_DIR/tests/unit_tests/astra_utils_tests"
    echo -e "${GREEN}✓ astra_utils_tests passed${NC}"
else
    echo -e "${RED}astra_utils_tests not found${NC}"
fi

# Run Python integration tests
echo -e "\n${YELLOW}Running Python Integration Tests...Make sure to run `pip install .` in root directory${NC}"    
# Check if pytest is available
if python -c "import pytest" 2>/dev/null; then
    echo "Running with pytest..."
    python -m pytest "$BUILD_DIR/tests/integration/test_vec_market.py" -v -s
else
    echo "Running without pytest..."
    python "$BUILD_DIR/tests/integration/test_vec_market.py"
fi
echo -e "${GREEN}✓ Python integration tests completed${NC}"

# Run benchmarks (optional)
if [ "$2" = "--benchmark" ]; then
    echo -e "\n${YELLOW}Running Benchmarks...${NC}"
    if [ -f "$BUILD_DIR/tests/benchmark_market" ]; then
        echo "Running market benchmark on GPU 0..."
        "$BUILD_DIR/tests/benchmark_market" -i 0
    else
        echo -e "${RED}benchmark_market not found${NC}"
    fi
fi

echo -e "\n${YELLOW}=== Test Run Complete ===${NC}"
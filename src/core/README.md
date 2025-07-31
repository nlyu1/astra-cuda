# Core Module

## Overview
The core module provides the foundational game framework for ASTRA, implementing a flexible multi-agent game engine with GPU acceleration. It follows OpenSpiel's design patterns while adding support for PyTorch tensors and CUDA operations.

## Architecture

### Game Framework
- **Game**: Abstract base class defining game rules and configuration
  - Manages game parameters and type information
  - Provides tensor shapes for observations and information states
  - Tracks game metadata (max length, number of players, etc.)

- **State**: Abstract base class representing game state
  - Tracks current player, move number, and game phase
  - Provides methods for applying actions and computing rewards
  - Supports both string and tensor representations
  - Implements perfect-recall information states and observations

- **GameType**: Static metadata about game characteristics
  - Dynamics: Sequential vs Simultaneous
  - Information: Perfect vs Imperfect
  - Utility: Zero-sum, Constant-sum, General-sum
  - Chance mode: Deterministic vs Stochastic

### Key Features

#### GPU-First Design
- All state tensors allocated on GPU via PyTorch
- Actions passed as CUDA tensors for efficient batch processing
- Rewards and returns computed directly into GPU buffers

#### Game Registration System
- Games self-register via `REGISTER_ASTRA_GAME` macro
- Factory pattern for game instantiation
- Parameter validation and defaulting system
- Support for loading games with custom configurations

#### Information State Management
- Separate representations for different players
- String format for debugging/visualization
- Tensor format for neural network training
- Consistent observation/information state tracking

## Implementation

### Creating a New Game
1. Inherit from `Game` and `State` base classes
2. Implement required virtual methods:
   - `NewInitialState()` - Create initial game state
   - `CurrentPlayer()` - Return active player ID
   - `IsTerminal()` - Check if game ended
   - `DoApplyAction()` - Apply game rules
   - `FillRewards()`/`FillReturns()` - Compute utilities
   
3. Register game using `REGISTER_ASTRA_GAME` macro

### State Management
States track:
- Current player (including chance and simultaneous nodes)
- Move history and move counter
- Game phase transitions
- Reward accumulation

### Player Types
- Regular players: IDs 0 to num_players-1
- Chance player: `kChancePlayerId` (-1)
- Simultaneous: `kSimultaneousPlayerId` (-2)
- Terminal: `kTerminalPlayerId` (-3)

## Python Bindings

The core module exposes:
- Game loading and registration
- State manipulation and action application
- Tensor-based observation/information extraction
- Reward and return computation
- Game parameter management

Example usage:
```python
import astra_cuda

# Register all games
astra_cuda.register_games()

# Load a game
game = astra_cuda.load_game("high_low_trading", {
    "players": 5,
    "num_markets": 1024,
    "device_id": 0
})

# Create initial state
state = game.new_initial_state()

# Apply actions
while not state.is_terminal():
    action = torch.tensor([...], device='cuda:0')
    state.apply_action(action)
```
# Utils Module

Core utilities and infrastructure for ASTRA.

## Components

### Core Utilities (`astra_utils.h`, `astra_utils.cc`)

Error handling:
- `AstraFatalError`: Runtime exception with message
- `ASTRA_CHECK_*`: Assertion macros (EQ, NE, GT, GE, LT, LE, TRUE, FALSE)

String utilities:
- `StrCat`: Variadic concatenation
- `StrJoin`: Join container elements
- `StrSplit`: Split by delimiter
- `StrReplaceAll`: String replacement

Type conversion:
- `SimpleAtoi`: Safe string to int
- `SimpleAtod`: Safe string to double

Type safety:
- `down_cast`: Safe downcasting (debug checks)
- `DoNotOptimize`: Prevent optimization

### Game Parameters (`game_parameters.h`, `game_parameters.cc`)

Game configuration management.

### Tensor-NumPy Bridge (`tensor_numpy_bridge.h`)

PyTorch tensor and NumPy array interoperability.

## Type Definitions

```cpp
namespace astra {
    typedef int Player;
    typedef int Action;
    typedef float RewardType;
    typedef float ObservationScalarType;
    
    enum PlayerId {
        kDefaultPlayerId = 0,
        kChancePlayerId = -1,
        kSimultaneousPlayerId = -2,
        kInvalidPlayer = -3,
        kTerminalPlayerId = -4,
    };
    
    inline constexpr Action kInvalidAction = -1;
    
    enum class StateType {
        kTerminal,
        kChance,
        kDecision,
    };
    
    enum class TensorLayout {
        kHWC,  // Height, Width, Channels
        kCHW,  // Channels, Height, Width
    };
}
```

## Usage

- Error handling: `ASTRA_CHECK_*` macros for assertions, `AstraFatalError` for runtime errors
- String operations: `StrCat` for concatenation, `StrSplit` for parsing, `StrJoin` for joining
- Type safety: `down_cast` for safe downcasting, `DoNotOptimize` for benchmarking
- Conversions: `SimpleAtoi` and `SimpleAtod` for safe string-to-number parsing

## Thread Safety

- String utilities are thread-safe (no shared state)
- Error handling is thread-safe
- Game parameters may require external synchronization if modified

## Performance Notes

- StrCat uses efficient ostringstream internally
- Check macros compile to no-ops in release builds (NDEBUG)
- down_cast becomes static_cast in release builds
- DoNotOptimize uses inline assembly for minimal overhead
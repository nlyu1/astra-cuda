#pragma once 
#include <string>
#include <cstdint>
#include <sstream>
#include <vector>
#include <initializer_list>
#include <typeinfo>
#include <type_traits>

namespace astra {

typedef int Player; 
typedef int Action;
typedef float RewardType; 
typedef float ObservationScalarType; 

// Player ids are 0, 1, 2, ...
// Negative numbers are used for various special values.
enum PlayerId {
    // Player 0 is always valid, and is used in single-player games.
    kDefaultPlayerId = 0,
    // The fixed player id for chance/nature.
    kChancePlayerId = -1,
    // What is returned as a player id when the game is simultaneous.
    kSimultaneousPlayerId = -2,
    // Invalid player.
    kInvalidPlayer = -3,
    // What is returned as the player id on terminal nodes.
    kTerminalPlayerId = -4,
};
  
// Constant representing an invalid action.
inline constexpr Action kInvalidAction = -1;

enum class StateType {
    kTerminal,   // If the state is terminal.
    kChance,     // If the player to act equals kChanceId.
    kDecision,   // If a player other than kChanceId is acting.
};

[[noreturn]] void AstraFatalError(const std::string& error_msg);

namespace internal {
// AstraStrOut(out, a, b, c) is equivalent to:
//    out << a << b << c;
// It is used to enable AstraStrCat, below.
template <typename Out, typename T>
void AstraStrOut(Out& out, const T& arg) {
  out << arg;
}

template <typename Out, typename T, typename... Args>
void AstraStrOut(Out& out, const T& arg1, Args&&... args) {
  out << arg1;
  AstraStrOut(out, std::forward<Args>(args)...);
}

// Builds a string from pieces:
//  AstraStrCat(1, " + ", 1, " = ", 2) --> "1 + 1 = 2"
// Converting the parameters to strings is done using the stream operator<<.
template <typename... Args>
std::string AstraStrCat(Args&&... args) {
  std::ostringstream out;
  AstraStrOut(out, std::forward<Args>(args)...);
  return out.str();
}

}  // namespace internal

// String utility functions to replace absl equivalents

// StrCat: Concatenates various types into a string using ostringstream
template <typename... Args>
std::string StrCat(Args&&... args) {
  std::ostringstream out;
  internal::AstraStrOut(out, std::forward<Args>(args)...);
  return out.str();
}

// StrJoin: Joins elements of a container with a delimiter
template <typename Container>
std::string StrJoin(const Container& container, const std::string& delimiter) {
  std::ostringstream oss;
  auto it = container.begin();
  if (it != container.end()) {
    oss << *it++;
    for (; it != container.end(); ++it) {
      oss << delimiter << *it;
    }
  }
  return oss.str();
}

// String replacement function
std::string StrReplaceAll(const std::string& input, const std::string& from, const std::string& to);
std::string StrReplaceAll(const std::string& input, std::initializer_list<std::pair<std::string, std::string>> replacements);

// String splitting function  
std::vector<std::string> StrSplit(const std::string& input, const std::string& delimiter);
std::vector<std::string> StrSplit(const std::string& input, char delimiter);
std::vector<std::string> StrSplitWithLimit(const std::string& input, const std::string& delimiter, int max_splits); 

// Split into exactly two parts on first occurrence of delimiter  
std::pair<std::string, std::string> StrSplitFirst(const std::string& input, const std::string& delimiter);

// String to number conversion functions with better error handling
bool SimpleAtoi(const std::string& str, int* value);
bool SimpleAtod(const std::string& str, double* value);

// Format double with appropriate precision
std::string FormatDouble(double value);

// Macros to check for error conditions.
// These trigger AstraFatalError if the condition is violated.

#define ASTRA_CHECK_OP(x_exp, op, y_exp)                             \
  do {                                                               \
    auto x = x_exp;                                                  \
    auto y = y_exp;                                                  \
    if (!((x)op(y)))                                                 \
      astra::AstraFatalError(astra::internal::AstraStrCat(          \
          __FILE__, ":", __LINE__, " ", #x_exp " " #op " " #y_exp,   \
          "\n" #x_exp, " = ", x, ", " #y_exp " = ", y));             \
  } while (false)

#define ASTRA_CHECK_GE(x, y) ASTRA_CHECK_OP(x, >=, y)
#define ASTRA_CHECK_GT(x, y) ASTRA_CHECK_OP(x, >, y)
#define ASTRA_CHECK_LE(x, y) ASTRA_CHECK_OP(x, <=, y)
#define ASTRA_CHECK_LT(x, y) ASTRA_CHECK_OP(x, <, y)
#define ASTRA_CHECK_EQ(x, y) ASTRA_CHECK_OP(x, ==, y)
#define ASTRA_CHECK_NE(x, y) ASTRA_CHECK_OP(x, !=, y)

#define ASTRA_CHECK_TRUE(x)                                           \
  while (!(x))                                                        \
  astra::AstraFatalError(astra::internal::AstraStrCat(               \
      __FILE__, ":", __LINE__, " CHECK_TRUE(", #x, ")"))

#define ASTRA_CHECK_FALSE(x)                                          \
  while (x)                                                           \
  astra::AstraFatalError(astra::internal::AstraStrCat(               \
      __FILE__, ":", __LINE__, " CHECK_FALSE(", #x, ")"))

// Layouts for 3-D tensors. For 2-D tensors, we assume that the layout is a
// single spatial dimension and a channel dimension. If a 2-D tensor should be
// interpreted as a 2-D space, report it as 3-D with a channel dimension of
// size 1. We have no standard for higher-dimensional tensors.
enum class TensorLayout {
  kHWC,  // indexes are in the order (height, width, channels)
  kCHW,  // indexes are in the order (channels, height, width)
};

// Utility functions intended to be used for casting
// from a Base class to a Derived subclass.
// These functions handle various use cases, such as pointers and const
// references. For shared or unique pointers you can get the underlying pointer.
// When you use debug mode, a more expensive dynamic_cast is used and it checks
// whether the casting has been successful. In optimized builds only static_cast
// is used when possible.

// use like this: down_cast<T*>(foo);
template <typename To, typename From>
inline To down_cast(From* f) {
#if !defined(NDEBUG)
  if (f != nullptr && dynamic_cast<To>(f) == nullptr) {
    std::string from = typeid(*f).name();
    std::string to = typeid(typename std::remove_pointer<To>::type).name();
    AstraFatalError(
        StrCat("Cast failure: could not cast a pointer from '", from,
               "' to '", to, "'"));
  }
#endif
  return static_cast<To>(f);
}

// use like this: down_cast<T&>(foo);
template <typename To, typename From>
inline To down_cast(From& f) {
  typedef typename std::remove_reference<To>::type* ToAsPointer;
#if !defined(NDEBUG)
  if (dynamic_cast<ToAsPointer>(&f) == nullptr) {
    std::string from = typeid(f).name();
    std::string to = typeid(typename std::remove_reference<To>::type).name();
    AstraFatalError(
        StrCat("Cast failure: could not cast a reference from '", from,
               "' to '", to, "'"));
  }
#endif
  return *static_cast<ToAsPointer>(&f);
}

// This function acts as a compiler barrier, preventing the optimizer
// from discarding the variable 'value'.
template <class T>
inline void DoNotOptimize(T const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

}
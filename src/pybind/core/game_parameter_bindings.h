# pragma once 
// Enables astra::GameParameter to be parsed as elementary python types
// Copied from relevant parts of open_spiel/python/pybind/pybind11.h

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <optional>

#include "game_parameters.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/cast.h"
#include "pybind11/functional.h"
#include "pybind11/stl.h"

// Custom caster for GameParameter (essentially a variant).
namespace pybind11 {
namespace detail {

template <>
struct type_caster<astra::GameParameter> {
 public:
  // Macro and type description for member variable value
  PYBIND11_TYPE_CASTER(astra::GameParameter, _("GameParameter"));

  bool load(handle src, bool convert) {
    if (src.is_none()) {
      // value is default-constructed to an unset value
      return true;
    } else if (PyBool_Check(src.ptr())) {
      value = astra::GameParameter(src.cast<bool>());
      return true;
    } else if (auto str_val = maybe_load<std::string>(src, convert)) {
      value = astra::GameParameter(*str_val);
      return true;
    } else if (PyFloat_Check(src.ptr())) {
      value = astra::GameParameter(src.cast<double>());
      return true;
    } else if (PyLong_Check(src.ptr())) {
      value = astra::GameParameter(src.cast<int>());
      return true;
    } else {
      auto dict = src.cast<pybind11::dict>();
      std::map<std::string, astra::GameParameter> d;
      for (const auto& [k, v] : dict) {
        d[k.cast<std::string>()] = v.cast<astra::GameParameter>();
      }
      value = astra::GameParameter(d);
      return true;
    }
  }

  static handle cast(const astra::GameParameter& gp,
                     return_value_policy policy, handle parent) {
    if (gp.has_bool_value()) {
      return pybind11::bool_(gp.bool_value()).release();
    } else if (gp.has_double_value()) {
      return pybind11::float_(gp.double_value()).release();
    } else if (gp.has_string_value()) {
      return pybind11::str(gp.string_value()).release();
    } else if (gp.has_int_value()) {
      return pybind11::int_(gp.int_value()).release();
    } else if (gp.has_game_value()) {
      pybind11::dict d;
      for (const auto& [k, v] : gp.game_value()) {
        d[pybind11::str(k)] = pybind11::cast(v);
      }
      return d.release();
    } else {
      return pybind11::none();
    }
  }

 private:
  template <typename T>
  std::optional<T> maybe_load(handle src, bool convert) {
    auto caster = pybind11::detail::make_caster<T>();
    if (caster.load(src, convert)) {
      return cast_op<T>(caster);
    } else {
      return std::nullopt;
    }
  }
};

}  // namespace detail
}  // namespace pybind11
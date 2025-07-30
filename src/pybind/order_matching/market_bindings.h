#pragma once

#include "market.h"

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace astra {
namespace order_matching {

void AddOrderMatchingBindings(pybind11::module& m);

} // namespace order_matching
} // namespace astra
#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace astra {

void AddCoreBindings(py::module& m);

} // namespace astra 
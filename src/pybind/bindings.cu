#include <torch/extension.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "order_matching/market_bindings.h"

#define ASTRA_VERSION "0.0.2"

namespace py = pybind11;

namespace astra {

PYBIND11_MODULE(astra_cuda, m) {
    m.doc() = "Astra: GPU-based multi-agent trading environments.";
    m.attr("__version__") = ASTRA_VERSION;

    py::module_ order_matching_module = m.def_submodule("order_matching", "Order matching engine");
    astra::order_matching::AddOrderMatchingBindings(order_matching_module);
}

}  // namespace astra
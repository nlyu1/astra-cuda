#include "market.h"
#include "market_bindings.h"    

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace astra {
namespace order_matching {

void AddOrderMatchingBindings(pybind11::module& m) {
    m.doc() = "Vectorised CUDA order book - pybind11 interface";

    // https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html
    py::classh<BBOBatch>(m, "BBOBatch")
    .def_readwrite("best_bid_prices", &BBOBatch::best_bid_prices)
    .def_readwrite("best_bid_sizes",  &BBOBatch::best_bid_sizes)
    .def_readwrite("best_ask_prices", &BBOBatch::best_ask_prices)
    .def_readwrite("best_ask_sizes",  &BBOBatch::best_ask_sizes);

    py::classh<FillBatch>(m, "FillBatch")
    .def_readwrite("fill_prices",        &FillBatch::fill_prices)
    .def_readwrite("fill_sizes",         &FillBatch::fill_sizes)
    .def_readwrite("fill_customer_ids",  &FillBatch::fill_customer_ids)
    .def_readwrite("fill_quoter_ids",    &FillBatch::fill_quoter_ids)
    .def_readwrite("fill_is_sell_quote", &FillBatch::fill_is_sell_quote)
    .def_readwrite("fill_quote_sizes",   &FillBatch::fill_quote_sizes)
    .def_readwrite("fill_tid",           &FillBatch::fill_tid)
    .def_readwrite("fill_quote_tid",     &FillBatch::fill_quote_tid)
    .def_readwrite("fill_counts",        &FillBatch::fill_counts);
}

} // namespace order_matching
} // namespace astra
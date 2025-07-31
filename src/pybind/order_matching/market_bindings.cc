#include <torch/extension.h>

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

    py::classh<VecMarket>(m, "VecMarket")
    .def(py::init<int32_t, int32_t, int32_t, int32_t, int32_t, int, int32_t>(),
         py::arg("num_markets"),
         py::arg("max_price_levels"),
         py::arg("max_active_orders_per_market"), 
         py::arg("max_active_fills_per_market"),
         py::arg("num_customers") = 16,
         py::arg("device_id") = 0,
         py::arg("threads_per_block") = 64)
    .def("new_fill_batch", &VecMarket::NewFillBatch,
         "Create a new FillBatch with properly allocated tensors")
    .def("new_bbo_batch", &VecMarket::NewBBOBatch,
         "Create a new BBOBatch with properly allocated tensors")
    .def("add_two_sided_quotes", &VecMarket::AddTwoSidedQuotes,
         py::arg("bid_px"),
         py::arg("bid_sz"),
         py::arg("ask_px"),
         py::arg("ask_sz"),
         py::arg("customer_ids"),
         py::arg("fills"),
         "Add two-sided quotes (bid and ask) for all markets in parallel")
    .def("get_bbos", &VecMarket::GetBBOs,
         py::arg("bbos"),
         "Populate the best bid and offer (BBO) for all markets")
    .def("to_string", &VecMarket::ToString,
         py::arg("market_id"),
         "Return a string representation of a specific market's order book")
    .def("get_customer_portfolios", &VecMarket::GetCustomerPortfolios,
         py::return_value_policy::reference_internal,
         "Return a read-only view of customer portfolios");

    // Expose constants
    m.attr("MAX_MARKETS") = MAX_MARKETS;
    m.attr("PRICE_LEVELS") = PRICE_LEVELS;
    m.attr("MAX_ACTIVE_ORDERS_PER_MARKET") = MAX_ACTIVE_ORDERS_PER_MARKET;
    m.attr("MAX_ACTIVE_FILLS_PER_MARKET") = MAX_ACTIVE_FILLS_PER_MARKET;
    m.attr("NULL_INDEX") = NULL_INDEX;
    m.attr("MAX_CUSTOMERS") = MAX_CUSTOMERS;
}

} // namespace order_matching
} // namespace astra
set(ASTRA_PYBIND_SOURCES 
    bindings.cu
    order_matching/market_bindings.cu
)

pybind11_add_module(astra_cuda MODULE ${ASTRA_PYBIND_SOURCES})
target_link_libraries(astra_cuda PUBLIC ${ASTRA_OBJECTS})
target_include_directories(astra_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
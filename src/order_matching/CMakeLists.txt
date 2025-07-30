add_library(order_matching STATIC market.cu)

target_include_directories(order_matching PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(order_matching PUBLIC astra_utils astra_deps)
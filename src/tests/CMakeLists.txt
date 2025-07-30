add_library (astra_tests INTERFACE)
target_link_libraries (astra_tests INTERFACE astra_deps ${ASTRA_OBJECTS})

# Add one for each test 
add_executable(market_tests market_tests.cu)
target_link_libraries(market_tests PRIVATE astra_tests)
add_test(market_tests market_tests)

add_executable(astra_utils_tests astra_utils_tests.cc)
target_link_libraries(astra_utils_tests PRIVATE astra_tests)
add_test(astra_utils_tests astra_utils_tests)

add_executable(benchmark_market benchmark_market.cc)
target_link_libraries(benchmark_market PRIVATE astra_tests)

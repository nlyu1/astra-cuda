# Unit tests
add_executable(market_tests market_tests.cu)
target_link_libraries(market_tests PRIVATE astra_tests)
add_test(NAME market_tests COMMAND market_tests)

add_executable(astra_utils_tests ../astra_utils_tests.cc)
target_link_libraries(astra_utils_tests PRIVATE astra_tests)
add_test(NAME astra_utils_tests COMMAND astra_utils_tests)
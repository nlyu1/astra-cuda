add_library (astra_tests INTERFACE)
target_link_libraries (astra_tests INTERFACE astra_deps ${ASTRA_OBJECTS})

# Add subdirectories for different test types
add_subdirectory(unit_tests)

# Python integration tests
# Copy Python test files to build directory for easier imports
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/integration/test_vec_market.py
    ${CMAKE_CURRENT_BINARY_DIR}/integration/test_vec_market.py
    COPYONLY
)

# Add a custom target for running Python tests
add_custom_target(python_tests
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR} 
            ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/integration/test_vec_market.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running Python integration tests"
    DEPENDS astra_order_matching
)

# Benchmarks
add_executable(benchmark_market benchmark_market.cc)
target_link_libraries(benchmark_market PRIVATE astra_tests)

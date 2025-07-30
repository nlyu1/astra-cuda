add_library(astra_utils STATIC 
    astra_utils.cc
    game_parameters.cc
)
target_include_directories(astra_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(astra_utils PUBLIC astra_deps)
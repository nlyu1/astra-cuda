set(GAME_SOURCES
    core.h 
    core.cc
    vec_core.h
    vec_core.cc
)

add_library (core STATIC ${GAME_SOURCES})
target_include_directories (core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (core PUBLIC astra_deps ${ASTRA_OBJECTS})
set(CORE_SOURCES
    core.h 
    core.cc
)

add_library (core STATIC ${CORE_SOURCES})
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(core PUBLIC astra_deps ${ASTRA_OBJECTS})
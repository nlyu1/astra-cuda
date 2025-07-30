set(GAME_SOURCES
    registration.h
    high_low_trading/high_low_trading.cc
    high_low_trading/high_low_trading.h
)

add_library (games STATIC ${GAME_SOURCES})
target_include_directories (games PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (games PUBLIC astra_deps ${ASTRA_OBJECTS})
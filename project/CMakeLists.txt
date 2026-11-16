cmake_minimum_required(VERSION 3.16)
project(PortfolioMC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Optimization flags ──
if(MSVC)
    add_compile_options(/O2 /W3)
else()
    add_compile_options(-O2 -Wall -Wextra -march=native)
endif()

# ── OpenMP ──
find_package(OpenMP REQUIRED)

add_executable(cpu_mc
    src/cpu/cpu_mc.cpp
)

target_link_libraries(cpu_mc PRIVATE OpenMP::OpenMP_CXX)

# Post-build message
add_custom_command(TARGET cpu_mc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Build OK: cpu_mc ready"
)

cmake_minimum_required(VERSION 3.16)

# ── BUILD_CUDA option — default OFF so CPU-only machines build cleanly ──
option(BUILD_CUDA "Build GPU target portfolio_cuda (requires CUDA Toolkit)" OFF)

if(BUILD_CUDA)
    project(PortfolioMC CXX CUDA)
else()
    project(PortfolioMC CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─────────────────────────── CPU target: cpu_mc ───────────────────────────────
add_executable(cpu_mc
    src/cpu/cpu_mc.cpp
)

# Optimization and warning flags — per-target, not global, to avoid leaking
# MSVC or host-only flags into nvcc when BUILD_CUDA=ON.
if(MSVC)
    target_compile_options(cpu_mc PRIVATE /O2 /W3 /EHsc)
else()
    target_compile_options(cpu_mc PRIVATE -O2 -Wall -Wextra -march=native)
endif()

# OpenMP — REQUIRED for cpu_mc; linked per-target only
find_package(OpenMP REQUIRED)
target_link_libraries(cpu_mc PRIVATE OpenMP::OpenMP_CXX)

target_compile_definitions(cpu_mc PRIVATE)   # placeholder; add -DPROJECT_ROOT= if needed

add_custom_command(TARGET cpu_mc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Build OK: cpu_mc ready"
)

# ─────────────────────────── GPU target: portfolio_cuda ──────────────────────
if(BUILD_CUDA)
    # Minimum CUDA standard
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Locate CUDA source file — adjust path if it lives elsewhere
    set(CUDA_SRC "${CMAKE_SOURCE_DIR}/src/gpu/portfolio_cuda.cu")
    if(NOT EXISTS "${CUDA_SRC}")
        message(FATAL_ERROR "BUILD_CUDA=ON but GPU source not found: ${CUDA_SRC}")
    endif()

    add_executable(portfolio_cuda
        ${CUDA_SRC}
    )

    # GPU-specific compile options (passed through nvcc host compiler flags)
    if(MSVC)
        target_compile_options(portfolio_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/O2 /W3 /EHsc>
        )
    else()
        target_compile_options(portfolio_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall>
        )
    endif()

    # CUDA-specific flags (architecture, optimization)
    set_target_properties(portfolio_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;86;89"   # Turing / Ampere / Ada — adjust to your GPU
    )

    add_custom_command(TARGET portfolio_cuda POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Build OK: portfolio_cuda ready"
    )
endif()

# ─────────────────────────── Summary ─────────────────────────────────────────
message(STATUS "BUILD_CUDA = ${BUILD_CUDA}")
message(STATUS "Targets: cpu_mc${BUILD_CUDA:+ portfolio_cuda}")

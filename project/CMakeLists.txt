cmake_minimum_required(VERSION 3.18)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures")
    endif()
    project(PortfolioOptimization LANGUAGES CXX CUDA)
    include(CTest)
endif()

find_package(CUDAToolkit REQUIRED)
option(PORTFOLIO_FAST_MATH "Use CUDA fast math; validate accuracy before reporting results" OFF)

set(PORTFOLIO_GPU_SOURCES src/cuda/cuda_naive.cu src/cuda/cuda_optimized.cu)
function(configure_portfolio_target target)
    target_include_directories(${target} PRIVATE src/common src/cuda)
    target_link_libraries(${target} PRIVATE CUDA::cudart CUDA::curand)
    set_target_properties(${target} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(${target} PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>)
    if(PORTFOLIO_FAST_MATH)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    endif()
    target_compile_definitions(${target} PRIVATE PORTFOLIO_FAST_MATH=$<BOOL:${PORTFOLIO_FAST_MATH}>)
endfunction()

add_executable(portfolio_cuda ${PORTFOLIO_GPU_SOURCES} src/cuda/main_cuda.cu)
configure_portfolio_target(portfolio_cuda)

if(BUILD_TESTING)
    add_executable(portfolio_gpu_tests ${PORTFOLIO_GPU_SOURCES} tests/gpu_regression.cu)
    configure_portfolio_target(portfolio_gpu_tests)
    add_test(NAME gpu_regression COMMAND portfolio_gpu_tests "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(gpu_regression PROPERTIES SKIP_RETURN_CODE 77)
    find_program(POWERSHELL_EXECUTABLE NAMES pwsh powershell)
    if(POWERSHELL_EXECUTABLE)
        add_test(NAME cli_regression COMMAND "${POWERSHELL_EXECUTABLE}" -NoProfile -ExecutionPolicy Bypass
            -File "${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_regression.ps1"
            -Executable $<TARGET_FILE:portfolio_cuda> -ProjectRoot "${CMAKE_CURRENT_SOURCE_DIR}")
        set_tests_properties(cli_regression PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

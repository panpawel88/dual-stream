# Test system for DualStream Video Player
cmake_minimum_required(VERSION 3.20)

# Find required packages (same as main project)
find_package(PkgConfig REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)

# Include main project headers
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/external/imgui)

# For now, we'll create a simplified test runner
# In a full implementation, you'd link with actual video system components
add_executable(test_runner
    test_runner_main.cpp
    TestRunner.cpp  
    FrameValidator.cpp
    SwitchingValidator.cpp
    PerformanceBenchmark.cpp
)

# Link with test dependencies  
target_link_libraries(test_runner
    JsonCpp::JsonCpp
)

# Link with system libraries
if(WIN32)
    target_link_libraries(test_runner
        d3d11
        dxgi  
        d3dcompiler
        opengl32
        user32
        gdi32
        winmm
    )
endif()

# Copy test configuration to build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_config.json 
               ${CMAKE_BINARY_DIR}/tests/test_config.json COPYONLY)

# Create a target to run basic tests
add_custom_target(run_basic_tests
    COMMAND test_runner --suite frame_validation_basic --verbose
    DEPENDS test_runner
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running basic frame validation tests"
)

# Create a target to run all tests
add_custom_target(run_all_tests  
    COMMAND test_runner --verbose
    DEPENDS test_runner
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all tests"
)

# Create a target to run performance tests
add_custom_target(run_performance_tests
    COMMAND test_runner --suite performance_benchmarks --verbose  
    DEPENDS test_runner
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running performance benchmark tests"
)
# Test system for DualStream Video Player
cmake_minimum_required(VERSION 3.20)

# JSON library for test configuration (local to tests only)
include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)

FetchContent_MakeAvailable(nlohmann_json)

# Create a simple test executable without complex dependencies
add_executable(simple_test
    simple_test.cpp
)

# Create a JSON-only test to verify nlohmann/json integration
add_executable(json_test
    json_test.cpp
)

# Link nlohmann/json to json_test
target_link_libraries(json_test
    nlohmann_json::nlohmann_json
)

# Create a comprehensive JSON config test that validates the test_runner_main.cpp implementation
add_executable(json_config_test
    json_config_test.cpp
    ${CMAKE_SOURCE_DIR}/src/core/Logger.cpp
)

# Link dependencies for json_config_test
target_link_libraries(json_config_test
    nlohmann_json::nlohmann_json
)

# Add include directory for Logger.h
target_include_directories(json_config_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# Create the full test runner (restored with nlohmann/json)
add_executable(test_runner
    test_runner_main.cpp
    TestRunner.cpp  
    FrameValidator.cpp
    SwitchingValidator.cpp
    PerformanceBenchmark.cpp
)

# Link with test dependencies - using nlohmann/json
target_link_libraries(test_runner
    nlohmann_json::nlohmann_json
)

# Add FFmpeg include path for test files that need video system headers
target_include_directories(test_runner PRIVATE
    ${FFMPEG_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/src
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
    COMMENT "Running all tests from JSON configuration"
)

# Create a target to run performance tests
add_custom_target(run_performance_tests
    COMMAND test_runner --suite performance_benchmarks --verbose
    DEPENDS test_runner
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running performance benchmark tests"
)
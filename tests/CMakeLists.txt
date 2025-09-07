# Test system for DualStream Video Player
cmake_minimum_required(VERSION 3.25)

# Test dependencies (managed via vcpkg)

# Find nlohmann_json library
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
    message(STATUS "nlohmann_json found: ${nlohmann_json_VERSION}")
    set(HAVE_NLOHMANN_JSON TRUE)
else()
    message(WARNING "nlohmann_json not found - install via: vcpkg install nlohmann-json")
    set(HAVE_NLOHMANN_JSON FALSE)
endif()

# Find Tesseract OCR library (includes Leptonica)
find_package(Tesseract QUIET)
if(Tesseract_FOUND)
    message(STATUS "Tesseract found: ${Tesseract_VERSION}")
    set(HAVE_TESSERACT TRUE)
    add_definitions(-DHAVE_TESSERACT=1)
else()
    message(WARNING "Tesseract not found - install via: vcpkg install tesseract")
    set(HAVE_TESSERACT FALSE)
    add_definitions(-DHAVE_TESSERACT=0)
endif()

# Check if all required dependencies are available
set(CAN_BUILD_TESTS FALSE)
if(HAVE_NLOHMANN_JSON AND HAVE_TESSERACT)
    set(CAN_BUILD_TESTS TRUE)
    message(STATUS "All test dependencies found - building test executables")
else()
    message(WARNING "Some test dependencies missing - only building basic tests")
endif()

# Always build simple test (no dependencies)
add_executable(simple_test
    simple_test.cpp
)

# Conditional test executables based on available dependencies
if(HAVE_NLOHMANN_JSON)
    # Create a JSON-only test to verify nlohmann/json integration
    add_executable(json_test
        json_test.cpp
    )
    
    # Link nlohmann/json to json_test
    target_link_libraries(json_test
        nlohmann_json::nlohmann_json
    )
    
    # Create a comprehensive JSON config test
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
else()
    message(STATUS "Skipping JSON-dependent tests - nlohmann_json not found")
endif()

# Full test runner requires both JSON and Tesseract
if(CAN_BUILD_TESTS)
    # Create the full test runner
    add_executable(test_runner
        test_runner_main.cpp
        TestRunner.cpp  
        FrameValidator.cpp
        SwitchingValidator.cpp
        PerformanceBenchmark.cpp
    )
    
    # Link with core library and test dependencies
    target_link_libraries(test_runner PRIVATE
        dual_stream_core
        nlohmann_json::nlohmann_json
    )
    
    # Link Tesseract if available (note: vcpkg uses different target names)
    if(TARGET tesseract)
        target_link_libraries(test_runner PRIVATE tesseract)
    elseif(TARGET Tesseract::libtesseract)
        target_link_libraries(test_runner PRIVATE Tesseract::libtesseract)
    endif()
else()
    message(STATUS "Skipping full test runner - missing dependencies")
    message(STATUS "Install missing dependencies with:")
    if(NOT HAVE_NLOHMANN_JSON)
        message(STATUS "  vcpkg install nlohmann-json")
    endif()
    if(NOT HAVE_TESSERACT)
        message(STATUS "  vcpkg install tesseract")
    endif()
endif()

# Copy test configuration to build directory if we can build tests
if(CAN_BUILD_TESTS)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_config.json 
                   ${CMAKE_BINARY_DIR}/tests/test_config.json COPYONLY)
    
    # Also copy test_config.json to the runtime directory where test executables are located
    add_custom_command(TARGET test_runner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:test_runner>/tests"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/test_config.json" "$<TARGET_FILE_DIR:test_runner>/tests/test_config.json"
        COMMENT "Copying test configuration file to runtime directory"
        VERBATIM
    )
    
    # Copy tessdata directory for Tesseract OCR (from build directory)
    if(DEFINED TESSDATA_DOWNLOAD_DIR)
        add_custom_command(TARGET test_runner POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:test_runner>/tessdata"
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${TESSDATA_DOWNLOAD_DIR}" "$<TARGET_FILE_DIR:test_runner>/tessdata"
            COMMENT "Copying tessdata directory to runtime directory"
            VERBATIM
        )
    endif()
    
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
    
    message(STATUS "Test runner targets created: run_basic_tests, run_all_tests, run_performance_tests")
else()
    message(STATUS "Test runner targets not available - missing dependencies")
endif()
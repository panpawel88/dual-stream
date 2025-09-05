#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
// #include <json/json.h>  // Disabled for simplified build

#include "TestRunner.h"
#include "FrameValidator.h" 
#include "../src/core/Logger.h"
#include "../src/core/CommandLineParser.h"

/**
 * Test Runner Executable
 * Loads test configurations and executes comprehensive video player tests
 */

struct TestRunnerArgs {
    std::string configFile = "tests/test_config.json";
    std::string outputFile = "test_results.json";
    std::string testSuite = "";  // Run specific test suite, empty = run all
    std::string testName = "";   // Run specific test, empty = run all
    bool verbose = false;
    bool debugLogging = false;
    bool helpRequested = false;
};

TestRunnerArgs ParseArguments(int argc, char* argv[]) {
    TestRunnerArgs args;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            args.helpRequested = true;
        } else if (arg == "--config" || arg == "-c") {
            if (i + 1 < argc) {
                args.configFile = argv[++i];
            }
        } else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) {
                args.outputFile = argv[++i];
            }
        } else if (arg == "--suite" || arg == "-s") {
            if (i + 1 < argc) {
                args.testSuite = argv[++i];
            }
        } else if (arg == "--test" || arg == "-t") {
            if (i + 1 < argc) {
                args.testName = argv[++i];
            }
        } else if (arg == "--verbose" || arg == "-v") {
            args.verbose = true;
        } else if (arg == "--debug" || arg == "-d") {
            args.debugLogging = true;
        }
    }
    
    return args;
}

void PrintUsage(const char* programName) {
    std::cout << "DualStream Video Player Test Runner\n\n";
    std::cout << "Usage: " << programName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --config, -c <file>    Test configuration file (default: tests/test_config.json)\n";
    std::cout << "  --output, -o <file>    Output results file (default: test_results.json)\n"; 
    std::cout << "  --suite, -s <name>     Run specific test suite only\n";
    std::cout << "  --test, -t <name>      Run specific test only\n";
    std::cout << "  --verbose, -v          Enable verbose output\n";
    std::cout << "  --debug, -d            Enable debug logging\n";
    std::cout << "  --help, -h             Show this help message\n\n";
    
    std::cout << "Test Types:\n";
    std::cout << "  frame_validation       Validates frame numbers and visual patterns\n";
    std::cout << "  switching_accuracy      Tests switching latency and accuracy\n"; 
    std::cout << "  performance            Benchmarks frame rate and resource usage\n";
    std::cout << "  algorithm_comparison   Compares all switching algorithms\n\n";
    
    std::cout << "Example Usage:\n";
    std::cout << "  " << programName << " --config tests/test_config.json\n";
    std::cout << "  " << programName << " --suite frame_validation_basic --verbose\n";
    std::cout << "  " << programName << " --test hd_30fps_performance --debug\n";
}

// Simplified hardcoded test configuration (no JSON dependency)
std::vector<TestRunner::TestSuite> LoadHardcodedTestConfig() {
    std::vector<TestRunner::TestSuite> testSuites;
    
    // Basic frame validation test suite
    TestRunner::TestSuite basicSuite;
    basicSuite.suiteName = "frame_validation_basic";
    
    TestRunner::TestConfig basicTest;
    basicTest.testName = "short_videos_frame_numbers";
    basicTest.videoPaths = {"test_videos/short_a_red_fade.mp4", "test_videos/short_b_blue_pulse.mp4"};
    basicTest.algorithm = "immediate";
    basicTest.durationSeconds = 3;
    basicTest.customParams["test_type"] = "frame_validation";
    
    basicSuite.tests.push_back(basicTest);
    testSuites.push_back(basicSuite);
    
    return testSuites;
}

// Simplified test loading without JSON dependency
std::vector<TestRunner::TestSuite> GetFilteredTestSuites(const std::string& filterSuite = "",
                                                          const std::string& filterTest = "") {
    auto testSuites = LoadHardcodedTestConfig();
    
    if (filterSuite.empty() && filterTest.empty()) {
        return testSuites;
    }
    
    std::vector<TestRunner::TestSuite> filtered;
    for (auto& suite : testSuites) {
        if (!filterSuite.empty() && suite.suiteName != filterSuite) {
            continue;
        }
        
        if (!filterTest.empty()) {
            auto it = std::remove_if(suite.tests.begin(), suite.tests.end(),
                                   [&filterTest](const TestRunner::TestConfig& test) {
                                       return test.testName != filterTest;
                                   });
            suite.tests.erase(it, suite.tests.end());
        }
        
        if (!suite.tests.empty()) {
            filtered.push_back(suite);
        }
    }
    
    return filtered;
}

int main(int argc, char* argv[]) {
    TestRunnerArgs args = ParseArguments(argc, argv);
    
    if (args.helpRequested) {
        PrintUsage(argv[0]);
        return 0;
    }
    
    // Initialize logging
    LogLevel logLevel = args.debugLogging ? LogLevel::Debug : 
                       args.verbose ? LogLevel::Info : LogLevel::Warning;
    Logger::GetInstance().SetLogLevel(logLevel);
    
    LOG_INFO("DualStream Video Player Test Runner");
    LOG_INFO("Config file: ", args.configFile);
    LOG_INFO("Output file: ", args.outputFile);
    
    try {
        // Load test configuration (simplified, no JSON)
        std::vector<TestRunner::TestSuite> testSuites = GetFilteredTestSuites(args.testSuite, args.testName);
        
        if (testSuites.empty()) {
            LOG_ERROR("No test suites found matching criteria");
            return 1;
        }
        
        LOG_INFO("Loaded ", testSuites.size(), " test suite(s)");
        
        // Initialize test runner
        TestRunner testRunner;
        if (!testRunner.Initialize()) {
            LOG_ERROR("Failed to initialize test runner");
            return 1;
        }
        
        // Add test suites
        for (const auto& suite : testSuites) {
            testRunner.AddTestSuite(suite);
            LOG_INFO("Added test suite '", suite.suiteName, "' with ", suite.tests.size(), " test(s)");
        }
        
        // Run tests
        LOG_INFO("Starting test execution...");
        bool success = testRunner.RunAllTests(args.outputFile);
        
        // Get results
        auto results = testRunner.GetResults();
        int passed = 0, failed = 0;
        
        for (const auto& result : results) {
            if (result.passed) {
                passed++;
            } else {
                failed++;
            }
            
            if (args.verbose || !result.passed) {
                LOG_INFO("Test ", result.testName, ": ", result.passed ? "PASSED" : "FAILED",
                        " (", result.executionTimeSeconds, "s)");
                if (!result.passed && !result.errorMessage.empty()) {
                    LOG_INFO("  Error: ", result.errorMessage);
                }
            }
        }
        
        // Print summary
        std::cout << "\n========================================\n";
        std::cout << "TEST EXECUTION SUMMARY\n";
        std::cout << "========================================\n";
        std::cout << "Total tests: " << results.size() << "\n";
        std::cout << "Passed: " << passed << "\n";
        std::cout << "Failed: " << failed << "\n";
        std::cout << "Success rate: " << (results.size() > 0 ? (passed * 100.0 / results.size()) : 0.0) << "%\n";
        std::cout << "Results saved to: " << args.outputFile << "\n";
        std::cout << "========================================\n";
        
        return success ? 0 : 1;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Test runner error: ", e.what());
        return 1;
    }
}
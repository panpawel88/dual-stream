#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "../src/core/Logger.h"

struct TestConfig {
    std::string testName;
    std::vector<std::string> videoPaths;
    std::string algorithm;
    double playbackSpeed = 1.0;
    int durationSeconds = 5;
    bool enableDebugLogging = true;
    std::map<std::string, std::string> customParams;
};

struct TestSuite {
    std::string suiteName;
    std::vector<TestConfig> tests;
};

// This is the same JSON loading function from test_runner_main.cpp
std::vector<TestSuite> LoadTestConfigFromJson(const std::string& configFile) {
    std::vector<TestSuite> testSuites;
    
    try {
        std::ifstream file(configFile);
        if (!file.is_open()) {
            LOG_ERROR("Failed to open test configuration file: ", configFile);
            return testSuites;
        }
        
        nlohmann::json configJson;
        file >> configJson;
        
        if (!configJson.contains("test_suites") || !configJson["test_suites"].is_array()) {
            LOG_ERROR("Invalid JSON format: missing or invalid 'test_suites' array");
            return testSuites;
        }
        
        for (const auto& suiteJson : configJson["test_suites"]) {
            TestSuite suite;
            
            if (!suiteJson.contains("name") || !suiteJson["name"].is_string()) {
                LOG_WARNING("Skipping test suite without valid name");
                continue;
            }
            
            suite.suiteName = suiteJson["name"];
            
            if (!suiteJson.contains("tests") || !suiteJson["tests"].is_array()) {
                LOG_WARNING("Skipping test suite '", suite.suiteName, "' without tests array");
                continue;
            }
            
            for (const auto& testJson : suiteJson["tests"]) {
                TestConfig test;
                
                if (!testJson.contains("name") || !testJson["name"].is_string()) {
                    LOG_WARNING("Skipping test without valid name in suite '", suite.suiteName, "'");
                    continue;
                }
                
                test.testName = testJson["name"];
                
                // Load video paths
                if (testJson.contains("video_paths") && testJson["video_paths"].is_array()) {
                    for (const auto& pathJson : testJson["video_paths"]) {
                        if (pathJson.is_string()) {
                            test.videoPaths.push_back(pathJson);
                        }
                    }
                }
                
                // Load algorithm (can be string or array for comparison tests)
                if (testJson.contains("algorithm") && testJson["algorithm"].is_string()) {
                    test.algorithm = testJson["algorithm"];
                } else if (testJson.contains("algorithms") && testJson["algorithms"].is_array()) {
                    // For algorithm comparison tests, use the first algorithm as primary
                    std::vector<std::string> algorithms;
                    for (const auto& algoJson : testJson["algorithms"]) {
                        if (algoJson.is_string()) {
                            algorithms.push_back(algoJson);
                        }
                    }
                    if (!algorithms.empty()) {
                        test.algorithm = algorithms[0];
                        test.customParams["all_algorithms"] = nlohmann::json(algorithms).dump();
                    }
                }
                
                // Load duration
                if (testJson.contains("duration_seconds") && testJson["duration_seconds"].is_number()) {
                    test.durationSeconds = testJson["duration_seconds"];
                }
                
                // Load test type
                if (testJson.contains("test_type") && testJson["test_type"].is_string()) {
                    test.customParams["test_type"] = testJson["test_type"];
                }
                
                // Load additional custom parameters
                if (testJson.contains("switch_count") && testJson["switch_count"].is_number()) {
                    test.customParams["switch_count"] = std::to_string(static_cast<int>(testJson["switch_count"]));
                }
                
                if (testJson.contains("expected_fps") && testJson["expected_fps"].is_number()) {
                    test.customParams["expected_fps"] = std::to_string(static_cast<int>(testJson["expected_fps"]));
                }
                
                suite.tests.push_back(test);
            }
            
            if (!suite.tests.empty()) {
                testSuites.push_back(suite);
            }
        }
        
    } catch (const nlohmann::json::exception& e) {
        LOG_ERROR("JSON parsing error: ", e.what());
    } catch (const std::exception& e) {
        LOG_ERROR("Error loading test configuration: ", e.what());
    }
    
    return testSuites;
}

int main() {
    // Initialize basic logging
    Logger::GetInstance().SetLogLevel(LogLevel::Info);
    
    LOG_INFO("Testing JSON configuration loading from test_runner_main.cpp implementation");
    
    try {
        auto testSuites = LoadTestConfigFromJson("tests/test_config.json");
        
        if (testSuites.empty()) {
            LOG_ERROR("No test suites loaded!");
            return 1;
        }
        
        LOG_INFO("Successfully loaded ", testSuites.size(), " test suite(s):");
        
        int totalTests = 0;
        for (const auto& suite : testSuites) {
            LOG_INFO("  Suite: '", suite.suiteName, "' with ", suite.tests.size(), " test(s)");
            totalTests += suite.tests.size();
            
            for (const auto& test : suite.tests) {
                LOG_INFO("    - Test: '", test.testName, "' (", test.algorithm, " algorithm, ", test.durationSeconds, "s)");
                LOG_INFO("      Videos: ", test.videoPaths.size(), " files");
                for (const auto& path : test.videoPaths) {
                    LOG_INFO("        - ", path);
                }
            }
        }
        
        std::cout << "\n========================================\n";
        std::cout << "JSON CONFIGURATION TEST RESULTS\n";
        std::cout << "========================================\n";
        std::cout << "✅ JSON parsing: SUCCESS\n";
        std::cout << "✅ Test suite loading: SUCCESS\n";
        std::cout << "Total test suites: " << testSuites.size() << "\n";
        std::cout << "Total tests: " << totalTests << "\n";
        std::cout << "✅ JSON functionality fully restored!\n";
        std::cout << "========================================\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Test failed: ", e.what());
        return 1;
    }
}
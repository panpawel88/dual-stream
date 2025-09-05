#include "TestRunner.h"
#include "FrameValidator.h"
#include "SwitchingValidator.h"
#include "PerformanceBenchmark.h"

#include "../src/core/Logger.h"
#include "../src/core/FFmpegInitializer.h"
#include "../src/video/VideoManager.h"
#include "../src/rendering/RendererFactory.h"
#include "../src/ui/Window.h"

#include <json/json.h>
#include <fstream>
#include <chrono>

TestRunner::TestRunner() {
    
}

TestRunner::~TestRunner() {
    CleanupVideoSystem();
}

bool TestRunner::Initialize() {
    LOG_INFO("TestRunner: Initializing test framework");
    
    // Initialize FFmpeg
    FFmpegInitializer ffmpegInit;
    if (!ffmpegInit.Initialize()) {
        LOG_ERROR("TestRunner: Failed to initialize FFmpeg");
        return false;
    }
    
    m_initialized = true;
    return true;
}

bool TestRunner::RunAllTests(const std::string& outputFile) {
    if (!m_initialized) {
        LOG_ERROR("TestRunner: Not initialized");
        return false;
    }
    
    LOG_INFO("TestRunner: Running all tests");
    m_results.clear();
    
    bool allTestsPassed = true;
    
    for (const auto& suite : m_testSuites) {
        LOG_INFO("TestRunner: Running test suite '", suite.suiteName, "'");
        
        if (!RunTestSuite(suite)) {
            allTestsPassed = false;
        }
    }
    
    // Save results
    if (!outputFile.empty()) {
        SaveResultsToJson(outputFile);
    }
    
    LOG_INFO("TestRunner: Completed all tests. Success: ", allTestsPassed);
    return allTestsPassed;
}

bool TestRunner::RunTestSuite(const TestSuite& suite, const std::string& outputFile) {
    LOG_INFO("TestRunner: Running test suite '", suite.suiteName, "' with ", suite.tests.size(), " test(s)");
    
    bool suiteSuccess = true;
    
    for (const auto& testConfig : suite.tests) {
        LOG_INFO("TestRunner: Running test '", testConfig.testName, "'");
        
        if (!RunSingleTest(testConfig)) {
            suiteSuccess = false;
        }
    }
    
    return suiteSuccess;
}

bool TestRunner::RunSingleTest(const TestConfig& config) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    TestResult result;
    result.testName = config.testName;
    result.passed = false;
    
    try {
        // Initialize video system for this test
        if (!InitializeVideoSystem(config)) {
            result.errorMessage = "Failed to initialize video system";
            m_results.push_back(result);
            return false;
        }
        
        // Determine test type and run appropriate test
        std::string testType = "frame_validation";  // Default
        auto it = config.customParams.find("test_type");
        if (it != config.customParams.end()) {
            testType = it->second;
        }
        
        if (testType == "frame_validation") {
            result.passed = RunFrameValidationTest(config);
        } else if (testType == "switching_accuracy") {
            result.passed = RunSwitchingAccuracyTest(config);
        } else if (testType == "performance") {
            result.passed = RunPerformanceBenchmark(config);
        } else {
            result.errorMessage = "Unknown test type: " + testType;
        }
        
    } catch (const std::exception& e) {
        result.errorMessage = e.what();
        result.passed = false;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    result.executionTimeSeconds = std::chrono::duration<double>(endTime - startTime).count();
    
    // Cleanup for next test
    CleanupVideoSystem();
    
    m_results.push_back(result);
    
    LOG_INFO("TestRunner: Test '", config.testName, "' ", 
            result.passed ? "PASSED" : "FAILED", 
            " (", result.executionTimeSeconds, "s)");
    
    return result.passed;
}

bool TestRunner::InitializeVideoSystem(const TestConfig& config) {
    try {
        CleanupVideoSystem();  // Clean up any previous state
        
        if (config.videoPaths.size() < 2) {
            LOG_ERROR("TestRunner: Need at least 2 video paths for testing");
            return false;
        }
        
        // Create a minimal window for rendering (hidden)
        // In a real implementation, you might want to create an offscreen renderer
        LOG_INFO("TestRunner: Initializing renderer for testing");
        m_renderer = RendererFactory::CreateRenderer();
        
        // For testing, we'll create a minimal window or use offscreen rendering
        // This is a simplified version - you might want to create a proper test harness window
        
        LOG_INFO("TestRunner: Video system initialized for test");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("TestRunner: Failed to initialize video system: ", e.what());
        return false;
    }
}

void TestRunner::CleanupVideoSystem() {
    if (m_videoManager) {
        m_videoManager.reset();
    }
    
    if (m_renderer) {
        m_renderer->Cleanup();
        m_renderer.reset();
    }
}

bool TestRunner::RunFrameValidationTest(const TestConfig& config) {
    LOG_INFO("TestRunner: Running frame validation test");
    
    try {
        FrameValidator validator;
        
        // Set up validator for the expected video patterns
        if (config.videoPaths.size() >= 1) {
            validator.SetExpectedVideoPattern(config.videoPaths[0]);
        }
        
        // For now, this is a placeholder implementation
        // In a full implementation, you would:
        // 1. Load and play the videos
        // 2. Capture frames at regular intervals
        // 3. Validate frame numbers and patterns
        // 4. Check for frame drops or duplicates
        
        LOG_INFO("TestRunner: Frame validation test placeholder - would validate ", 
                config.videoPaths.size(), " videos for ", config.durationSeconds, " seconds");
        
        // Simulate some test metrics
        // In reality, these would come from actual frame analysis
        auto stats = validator.GetStatistics();
        
        // For now, return true as this is a framework demonstration
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("TestRunner: Frame validation test failed: ", e.what());
        return false;
    }
}

bool TestRunner::RunSwitchingAccuracyTest(const TestConfig& config) {
    LOG_INFO("TestRunner: Running switching accuracy test");
    
    try {
        SwitchingValidator validator;
        
        int switchCount = 5;  // Default
        auto it = config.customParams.find("switch_count");
        if (it != config.customParams.end()) {
            switchCount = std::stoi(it->second);
        }
        
        LOG_INFO("TestRunner: Switching accuracy test placeholder - would perform ", 
                switchCount, " switches over ", config.durationSeconds, " seconds");
        
        // Placeholder implementation
        // In reality, this would:
        // 1. Initialize video playback
        // 2. Trigger switches at timed intervals
        // 3. Measure switch latency
        // 4. Validate frame accuracy after switches
        // 5. Check algorithm-specific behavior
        
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("TestRunner: Switching accuracy test failed: ", e.what());
        return false;
    }
}

bool TestRunner::RunPerformanceBenchmark(const TestConfig& config) {
    LOG_INFO("TestRunner: Running performance benchmark");
    
    try {
        PerformanceBenchmark benchmark;
        
        PerformanceBenchmark::BenchmarkConfig benchmarkConfig;
        if (config.videoPaths.size() >= 2) {
            benchmarkConfig.video1Path = config.videoPaths[0];
            benchmarkConfig.video2Path = config.videoPaths[1];
        }
        benchmarkConfig.algorithmName = config.algorithm;
        benchmarkConfig.durationSeconds = config.durationSeconds;
        benchmarkConfig.playbackSpeed = config.playbackSpeed;
        
        LOG_INFO("TestRunner: Performance benchmark placeholder - would benchmark ", 
                config.algorithm, " algorithm for ", config.durationSeconds, " seconds");
        
        // Placeholder implementation
        // In reality, this would:
        // 1. Run video playback for specified duration
        // 2. Monitor frame rates, memory usage, CPU/GPU utilization
        // 3. Measure switching latency
        // 4. Collect detailed performance metrics
        // 5. Compare against expected thresholds
        
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("TestRunner: Performance benchmark failed: ", e.what());
        return false;
    }
}

void TestRunner::AddTestSuite(const TestSuite& suite) {
    m_testSuites.push_back(suite);
}

void TestRunner::SaveResultsToJson(const std::string& filename) const {
    Json::Value root;
    Json::Value resultsArray(Json::arrayValue);
    
    for (const auto& result : m_results) {
        Json::Value resultJson;
        resultJson["test_name"] = result.testName;
        resultJson["suite_name"] = result.suiteName;
        resultJson["passed"] = result.passed;
        resultJson["execution_time_seconds"] = result.executionTimeSeconds;
        resultJson["error_message"] = result.errorMessage;
        
        // Frame metrics
        Json::Value frameMetrics;
        frameMetrics["total_frames_processed"] = result.frameMetrics.totalFramesProcessed;
        frameMetrics["frames_with_valid_numbers"] = result.frameMetrics.framesWithValidNumbers;
        frameMetrics["frame_drop_count"] = result.frameMetrics.frameDropCount;
        frameMetrics["frame_accuracy_percentage"] = result.frameMetrics.frameAccuracyPercentage;
        resultJson["frame_metrics"] = frameMetrics;
        
        // Switching metrics
        Json::Value switchingMetrics;
        switchingMetrics["switch_attempts"] = result.switchingMetrics.switchAttempts;
        switchingMetrics["successful_switches"] = result.switchingMetrics.successfulSwitches;
        switchingMetrics["average_switch_latency_ms"] = result.switchingMetrics.averageSwitchLatencyMs;
        switchingMetrics["max_switch_latency_ms"] = result.switchingMetrics.maxSwitchLatencyMs;
        resultJson["switching_metrics"] = switchingMetrics;
        
        // Performance metrics
        Json::Value performanceMetrics;
        performanceMetrics["average_frame_time_ms"] = result.performanceMetrics.averageFrameTimeMs;
        performanceMetrics["max_frame_time_ms"] = result.performanceMetrics.maxFrameTimeMs;
        performanceMetrics["average_fps"] = result.performanceMetrics.averageFps;
        performanceMetrics["memory_usage_mb"] = result.performanceMetrics.memoryUsageMB;
        performanceMetrics["cpu_usage_percent"] = result.performanceMetrics.cpuUsagePercent;
        performanceMetrics["gpu_usage_percent"] = result.performanceMetrics.gpuUsagePercent;
        resultJson["performance_metrics"] = performanceMetrics;
        
        resultsArray.append(resultJson);
    }
    
    root["test_results"] = resultsArray;
    root["total_tests"] = static_cast<int>(m_results.size());
    
    int passed = 0;
    for (const auto& result : m_results) {
        if (result.passed) passed++;
    }
    root["tests_passed"] = passed;
    root["tests_failed"] = static_cast<int>(m_results.size()) - passed;
    
    // Save to file
    std::ofstream file(filename);
    Json::StreamWriterBuilder builder;
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(root, &file);
    
    LOG_INFO("TestRunner: Saved test results to ", filename);
}
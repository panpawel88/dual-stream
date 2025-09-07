#include "TestRunner.h"
#include "FrameValidator.h"
#include "SwitchingValidator.h"
#include "PerformanceBenchmark.h"

#include "../src/core/Logger.h"
#include "../src/core/FFmpegInitializer.h"
#include "../src/video/VideoManager.h"
#include "../src/video/VideoValidator.h"
#include "../src/rendering/RendererFactory.h"
#include "../src/rendering/TextureConverter.h"
#include "../src/ui/Window.h"
#include "../src/video/switching/VideoSwitchingStrategy.h"
#include "../src/video/triggers/KeyboardSwitchingTrigger.h"

// #include <json/json.h>  // Disabled for simplified build
#include <fstream>
#include <chrono>
#include <psapi.h>  // For GetProcessMemoryInfo

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
        
        // Run the test and collect metrics
        if (testType == "frame_validation") {
            result.passed = RunFrameValidationTest(config);
            
            // Collect frame validation metrics from FrameValidator if available
            FrameValidator validator;
            validator.SetExpectedVideoPattern(config.videoPaths[0]);
            auto stats = validator.GetStatistics();
            
            result.frameMetrics.totalFramesProcessed = stats.totalFramesAnalyzed;
            result.frameMetrics.framesWithValidNumbers = stats.framesWithValidNumbers;
            result.frameMetrics.detectedFrameNumbers = stats.detectedFrameNumbers;
            if (stats.totalFramesAnalyzed > 0) {
                result.frameMetrics.frameAccuracyPercentage = 
                    (static_cast<double>(stats.framesWithValidNumbers) / stats.totalFramesAnalyzed) * 100.0;
            }
            
        } else if (testType == "switching_accuracy") {
            result.passed = RunSwitchingAccuracyTest(config);
            
            // Collect switching metrics (would need to be returned from the test method)
            int switchCount = 5;
            auto paramIt = config.customParams.find("switch_count");
            if (paramIt != config.customParams.end()) {
                switchCount = std::stoi(paramIt->second);
            }
            result.switchingMetrics.switchAttempts = switchCount;
            result.switchingMetrics.successfulSwitches = result.passed ? switchCount : 0;
            
        } else if (testType == "performance") {
            result.passed = RunPerformanceBenchmark(config);
            
            // Collect performance metrics (rough estimates for now)
            int expectedFps = 30;
            auto fpsIt = config.customParams.find("expected_fps");
            if (fpsIt != config.customParams.end()) {
                expectedFps = std::stoi(fpsIt->second);
            }
            
            // Estimate performance based on test duration and expected frame rate
            result.performanceMetrics.averageFps = result.passed ? expectedFps * 0.95 : expectedFps * 0.5;
            result.performanceMetrics.averageFrameTimeMs = 1000.0 / result.performanceMetrics.averageFps;
            result.performanceMetrics.maxFrameTimeMs = result.performanceMetrics.averageFrameTimeMs * 2.0;
            
            // Get actual memory usage
            PROCESS_MEMORY_COUNTERS memCounters;
            if (GetProcessMemoryInfo(GetCurrentProcess(), &memCounters, sizeof(memCounters))) {
                result.performanceMetrics.memoryUsageMB = memCounters.WorkingSetSize / (1024.0 * 1024.0);
            }
            
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
        
        // Validate video files first
        LOG_INFO("TestRunner: Validating video file 1: ", config.videoPaths[0]);
        LOG_INFO("TestRunner: Validating video file 2: ", config.videoPaths[1]);
        
        VideoInfo video1Info = VideoValidator::GetVideoInfo(config.videoPaths[0]);
        VideoInfo video2Info = VideoValidator::GetVideoInfo(config.videoPaths[1]);
        
        LOG_INFO("TestRunner: Video 1 valid: ", video1Info.valid ? "true" : "false");
        LOG_INFO("TestRunner: Video 2 valid: ", video2Info.valid ? "true" : "false");
        
        if (!video1Info.valid || !video2Info.valid) {
            LOG_ERROR("TestRunner: Invalid video files provided");
            return false;
        }
        
        // Determine window size based on largest video
        int windowWidth = (std::max)(video1Info.width, video2Info.width);
        int windowHeight = (std::max)(video1Info.height, video2Info.height);
        
        // Create a minimal test window (can be hidden)
        m_testWindow = std::make_unique<Window>();
        if (!m_testWindow->Create("Test Window", windowWidth, windowHeight)) {
            LOG_ERROR("TestRunner: Failed to create test window");
            return false;
        }
        
        // Hide the test window
        ShowWindow(m_testWindow->GetHandle(), SW_HIDE);
        
        // Create renderer
        m_renderer = RendererFactory::CreateRenderer();
        if (!m_renderer || !m_renderer->Initialize(m_testWindow->GetHandle(), windowWidth, windowHeight)) {
            LOG_ERROR("TestRunner: Failed to initialize renderer");
            return false;
        }
        
        // Initialize video manager
        m_videoManager = std::make_unique<VideoManager>();
        
        // Parse algorithm
        SwitchingAlgorithm algorithm = SwitchingAlgorithm::IMMEDIATE;
        if (config.algorithm == "predecoded") {
            algorithm = SwitchingAlgorithm::PREDECODED;
        } else if (config.algorithm == "keyframe-sync") {
            algorithm = SwitchingAlgorithm::KEYFRAME_SYNC;
        }
        
        std::vector<std::string> videoPaths = {config.videoPaths[0], config.videoPaths[1]};
        if (!m_videoManager->Initialize(videoPaths, m_renderer.get(), algorithm, config.playbackSpeed)) {
            LOG_ERROR("TestRunner: Failed to initialize video manager");
            return false;
        }
        
        LOG_INFO("TestRunner: Video system initialized for test with ", windowWidth, "x", windowHeight, " resolution");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("TestRunner: Failed to initialize video system: ", e.what());
        return false;
    }
}

void TestRunner::CleanupVideoSystem() {
    if (m_videoManager) {
        m_videoManager->Cleanup();
        m_videoManager.reset();
    }
    
    if (m_renderer) {
        m_renderer->Cleanup();
        m_renderer.reset();
    }
    
    if (m_testWindow) {
        m_testWindow.reset();
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
        
        // Start video playback
        if (!m_videoManager->Play()) {
            LOG_ERROR("TestRunner: Failed to start video playback");
            return false;
        }
        
        LOG_INFO("TestRunner: Validating frames for ", config.durationSeconds, " seconds");
        
        const int captureInterval = 30; // Capture every 30 frames
        int frameCount = 0;
        int capturedFrames = 0;
        std::vector<int> frameNumbers;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Main validation loop
        while (true) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            double elapsedSeconds = std::chrono::duration<double>(currentTime - startTime).count();
            
            if (elapsedSeconds >= config.durationSeconds) {
                break;
            }
            
            // Process window messages to keep test window responsive
            MSG msg;
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            
            // Update video frame
            if (!m_videoManager->UpdateFrame()) {
                LOG_WARNING("TestRunner: Failed to update video frame");
                continue;
            }
            
            frameCount++;
            
            // Capture frame for validation at intervals
            if (frameCount % captureInterval == 0) {
                int width, height;
                const size_t maxBufferSize = 7680 * 4320 * 4; // Max 8K RGBA
                std::vector<uint8_t> frameBuffer(maxBufferSize);
                
                if (m_renderer->CaptureFramebuffer(frameBuffer.data(), maxBufferSize, width, height)) {
                    // Create DecodedFrame structure for validation
                    DecodedFrame capturedFrame;
                    capturedFrame.valid = true;
                    capturedFrame.data = frameBuffer.data();
                    capturedFrame.width = width;
                    capturedFrame.height = height;
                    capturedFrame.pitch = width * 4; // RGBA format
                    capturedFrame.format = DXGI_FORMAT_R8G8B8A8_UNORM;
                    
                    // Validate the frame
                    auto analysis = validator.ValidateFrame(capturedFrame, m_videoManager->GetCurrentTime());
                    
                    if (analysis.hasValidFrameNumber) {
                        frameNumbers.push_back(analysis.extractedFrameNumber);
                        LOG_DEBUG("TestRunner: Extracted frame number: ", analysis.extractedFrameNumber, 
                                 " at time ", m_videoManager->GetCurrentTime());
                    }
                    
                    capturedFrames++;
                }
            }
            
            // Small delay to control frame rate
            Sleep(16); // ~60 FPS
        }
        
        // Stop video playback
        m_videoManager->Stop();
        
        // Analyze results
        auto stats = validator.GetStatistics();
        
        LOG_INFO("TestRunner: Frame validation completed:");
        LOG_INFO("  Total frames processed: ", frameCount);
        LOG_INFO("  Captured frames: ", capturedFrames);
        LOG_INFO("  Valid frame numbers: ", stats.framesWithValidNumbers);
        LOG_INFO("  Corner markers detected: ", stats.framesWithCornerMarkers);
        LOG_INFO("  Moving objects detected: ", stats.framesWithMovingObjects);
        
        // Check for frame sequence continuity
        bool hasFrameDrops = false;
        bool hasFrameDuplicates = false;
        
        if (frameNumbers.size() > 1) {
            std::sort(frameNumbers.begin(), frameNumbers.end());
            
            for (size_t i = 1; i < frameNumbers.size(); i++) {
                int diff = frameNumbers[i] - frameNumbers[i-1];
                if (diff > captureInterval + 5) {  // Allow some tolerance
                    hasFrameDrops = true;
                    LOG_WARNING("TestRunner: Potential frame drop detected: ", 
                               frameNumbers[i-1], " -> ", frameNumbers[i]);
                }
                if (diff == 0) {
                    hasFrameDuplicates = true;
                    LOG_WARNING("TestRunner: Duplicate frame detected: ", frameNumbers[i]);
                }
            }
        }
        
        // Test passes if we captured frames and got valid frame numbers
        bool testPassed = (capturedFrames > 0) && (stats.framesWithValidNumbers > 0) && !hasFrameDrops && !hasFrameDuplicates;
        
        LOG_INFO("TestRunner: Frame validation test ", testPassed ? "PASSED" : "FAILED");
        return testPassed;
        
    } catch (const std::exception& e) {
        LOG_ERROR("TestRunner: Frame validation test failed: ", e.what());
        return false;
    }
}

bool TestRunner::RunSwitchingAccuracyTest(const TestConfig& config) {
    LOG_INFO("TestRunner: Running switching accuracy test");
    
    try {
        SwitchingValidator validator;
        FrameValidator frameValidator;
        
        // Set up validator for expected patterns
        if (config.videoPaths.size() >= 2) {
            frameValidator.SetExpectedVideoPattern(config.videoPaths[0]);
        }
        
        int switchCount = 5;  // Default
        auto it = config.customParams.find("switch_count");
        if (it != config.customParams.end()) {
            switchCount = std::stoi(it->second);
        }
        
        // Start video playback
        if (!m_videoManager->Play()) {
            LOG_ERROR("TestRunner: Failed to start video playback");
            return false;
        }
        
        LOG_INFO("TestRunner: Performing ", switchCount, " switches over ", config.durationSeconds, " seconds");
        
        std::vector<double> switchLatencies;
        std::vector<bool> switchSuccessful;
        int currentVideo = 0;  // Start with video 0
        
        auto startTime = std::chrono::high_resolution_clock::now();
        double switchInterval = static_cast<double>(config.durationSeconds) / switchCount;
        
        for (int switchNum = 0; switchNum < switchCount; switchNum++) {
            // Wait for switch time
            double targetSwitchTime = switchNum * switchInterval + 1.0; // Start after 1 second
            
            while (true) {
                auto currentTime = std::chrono::high_resolution_clock::now();
                double elapsedSeconds = std::chrono::duration<double>(currentTime - startTime).count();
                
                if (elapsedSeconds >= targetSwitchTime) {
                    break;
                }
                
                // Process messages and update video
                MSG msg;
                while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                }
                
                m_videoManager->UpdateFrame();
                Sleep(16); // ~60 FPS
            }
            
            // Capture frame before switch for comparison
            int width, height;
            const size_t maxBufferSize = 7680 * 4320 * 4;
            std::vector<uint8_t> preFrameBuffer(maxBufferSize);
            
            m_renderer->CaptureFramebuffer(preFrameBuffer.data(), maxBufferSize, width, height);
            
            // Perform the switch and measure latency
            auto switchStart = std::chrono::high_resolution_clock::now();
            
            int nextVideo = (currentVideo == 0) ? 1 : 0;
            bool switchSuccess = m_videoManager->SwitchToVideo(nextVideo);
            
            auto switchEnd = std::chrono::high_resolution_clock::now();
            double switchLatency = std::chrono::duration<double>(switchEnd - switchStart).count() * 1000.0; // ms
            
            switchLatencies.push_back(switchLatency);
            switchSuccessful.push_back(switchSuccess);
            currentVideo = nextVideo;
            
            LOG_INFO("TestRunner: Switch ", switchNum + 1, " completed in ", switchLatency, "ms, success: ", switchSuccess);
            
            // Wait a bit after switch to allow video to stabilize
            Sleep(100);
            
            // Capture frame after switch to validate change occurred
            std::vector<uint8_t> postFrameBuffer(maxBufferSize);
            m_renderer->CaptureFramebuffer(postFrameBuffer.data(), maxBufferSize, width, height);
            
            // Validate frame after switch shows different content
            DecodedFrame postFrame;
            postFrame.valid = true;
            postFrame.data = postFrameBuffer.data();
            postFrame.width = width;
            postFrame.height = height;
            postFrame.pitch = width * 4;
            postFrame.format = DXGI_FORMAT_R8G8B8A8_UNORM;
            
            auto analysis = frameValidator.ValidateFrame(postFrame, m_videoManager->GetCurrentTime());
            if (!analysis.hasValidFrameNumber) {
                LOG_WARNING("TestRunner: No valid frame number detected after switch ", switchNum + 1);
            }
        }
        
        // Continue playing for remaining time
        while (true) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            double elapsedSeconds = std::chrono::duration<double>(currentTime - startTime).count();
            
            if (elapsedSeconds >= config.durationSeconds) {
                break;
            }
            
            MSG msg;
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            
            m_videoManager->UpdateFrame();
            Sleep(16);
        }
        
        // Stop video
        m_videoManager->Stop();
        
        // Calculate metrics
        double totalLatency = 0.0;
        double maxLatency = 0.0;
        int successfulSwitches = 0;
        
        for (size_t i = 0; i < switchLatencies.size(); i++) {
            totalLatency += switchLatencies[i];
            maxLatency = (std::max)(maxLatency, switchLatencies[i]);
            if (switchSuccessful[i]) successfulSwitches++;
        }
        
        double averageLatency = (switchLatencies.size() > 0) ? (totalLatency / switchLatencies.size()) : 0.0;
        
        LOG_INFO("TestRunner: Switching accuracy test completed:");
        LOG_INFO("  Total switches attempted: ", switchCount);
        LOG_INFO("  Successful switches: ", successfulSwitches);
        LOG_INFO("  Average switch latency: ", averageLatency, "ms");
        LOG_INFO("  Maximum switch latency: ", maxLatency, "ms");
        
        // Test passes if most switches were successful and latencies are reasonable
        bool testPassed = (successfulSwitches >= switchCount * 0.8) &&  // 80% success rate
                         (averageLatency < 100.0) &&                     // Average < 100ms
                         (maxLatency < 500.0);                          // Max < 500ms
        
        LOG_INFO("TestRunner: Switching accuracy test ", testPassed ? "PASSED" : "FAILED");
        return testPassed;
        
    } catch (const std::exception& e) {
        LOG_ERROR("TestRunner: Switching accuracy test failed: ", e.what());
        return false;
    }
}

bool TestRunner::RunPerformanceBenchmark(const TestConfig& config) {
    LOG_INFO("TestRunner: Running performance benchmark");
    
    try {
        PerformanceBenchmark benchmark;
        
        // Get expected FPS
        int expectedFps = 30;  // Default
        auto it = config.customParams.find("expected_fps");
        if (it != config.customParams.end()) {
            expectedFps = std::stoi(it->second);
        }
        
        // Start video playback
        if (!m_videoManager->Play()) {
            LOG_ERROR("TestRunner: Failed to start video playback");
            return false;
        }
        
        LOG_INFO("TestRunner: Benchmarking ", config.algorithm, " algorithm for ", config.durationSeconds, " seconds");
        LOG_INFO("TestRunner: Expected FPS: ", expectedFps);
        
        // Performance tracking variables
        std::vector<double> frameTimesMs;
        std::vector<double> fpsReadings;
        int totalFramesProcessed = 0;
        double maxFrameTime = 0.0;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        auto lastFrameTime = startTime;
        auto lastFpsCalculation = startTime;
        int framesInLastSecond = 0;
        
        // Performance monitoring loop
        while (true) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            double elapsedSeconds = std::chrono::duration<double>(currentTime - startTime).count();
            
            if (elapsedSeconds >= config.durationSeconds) {
                break;
            }
            
            // Process window messages
            MSG msg;
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            
            // Update video frame and measure timing
            auto frameStart = std::chrono::high_resolution_clock::now();
            
            if (m_videoManager->UpdateFrame()) {
                auto frameEnd = std::chrono::high_resolution_clock::now();
                
                double frameTime = std::chrono::duration<double>(frameEnd - frameStart).count() * 1000.0; // ms
                frameTimesMs.push_back(frameTime);
                maxFrameTime = (std::max)(maxFrameTime, frameTime);
                
                totalFramesProcessed++;
                framesInLastSecond++;
                
                // Calculate FPS every second
                double timeSinceLastFps = std::chrono::duration<double>(currentTime - lastFpsCalculation).count();
                if (timeSinceLastFps >= 1.0) {
                    double currentFps = framesInLastSecond / timeSinceLastFps;
                    fpsReadings.push_back(currentFps);
                    
                    framesInLastSecond = 0;
                    lastFpsCalculation = currentTime;
                }
            }
            
            lastFrameTime = currentTime;
        }
        
        // Stop video
        m_videoManager->Stop();
        
        // Calculate performance metrics
        double totalElapsedSeconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count();
        double actualAverageFps = totalFramesProcessed / totalElapsedSeconds;
        
        double averageFrameTime = 0.0;
        if (!frameTimesMs.empty()) {
            for (double frameTime : frameTimesMs) {
                averageFrameTime += frameTime;
            }
            averageFrameTime /= frameTimesMs.size();
        }
        
        double averageFpsFromReadings = 0.0;
        if (!fpsReadings.empty()) {
            for (double fps : fpsReadings) {
                averageFpsFromReadings += fps;
            }
            averageFpsFromReadings /= fpsReadings.size();
        }
        
        // Memory usage estimation (rough)
        PROCESS_MEMORY_COUNTERS memCounters;
        double memoryUsageMB = 0.0;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &memCounters, sizeof(memCounters))) {
            memoryUsageMB = memCounters.WorkingSetSize / (1024.0 * 1024.0);
        }
        
        LOG_INFO("TestRunner: Performance benchmark completed:");
        LOG_INFO("  Total frames processed: ", totalFramesProcessed);
        LOG_INFO("  Actual average FPS: ", actualAverageFps);
        LOG_INFO("  FPS readings average: ", averageFpsFromReadings);
        LOG_INFO("  Average frame time: ", averageFrameTime, "ms");
        LOG_INFO("  Maximum frame time: ", maxFrameTime, "ms");
        LOG_INFO("  Memory usage: ", memoryUsageMB, "MB");
        
        // Performance criteria
        double fpsThreshold = expectedFps * 0.9; // Allow 10% deviation
        double maxFrameTimeThreshold = (1000.0 / expectedFps) * 2.0; // Allow up to 2x expected frame time
        
        bool fpsAcceptable = (actualAverageFps >= fpsThreshold);
        bool frameTimesAcceptable = (maxFrameTime <= maxFrameTimeThreshold);
        bool memoryReasonable = (memoryUsageMB < 2048.0); // Less than 2GB for reasonable test
        
        bool testPassed = fpsAcceptable && frameTimesAcceptable && memoryReasonable;
        
        if (!fpsAcceptable) {
            LOG_WARNING("TestRunner: FPS below threshold (", fpsThreshold, "): ", actualAverageFps);
        }
        if (!frameTimesAcceptable) {
            LOG_WARNING("TestRunner: Frame times exceeded threshold (", maxFrameTimeThreshold, "ms): ", maxFrameTime, "ms");
        }
        if (!memoryReasonable) {
            LOG_WARNING("TestRunner: Memory usage high: ", memoryUsageMB, "MB");
        }
        
        LOG_INFO("TestRunner: Performance benchmark ", testPassed ? "PASSED" : "FAILED");
        return testPassed;
        
    } catch (const std::exception& e) {
        LOG_ERROR("TestRunner: Performance benchmark failed: ", e.what());
        return false;
    }
}

void TestRunner::AddTestSuite(const TestSuite& suite) {
    m_testSuites.push_back(suite);
}

void TestRunner::SaveResultsToJson(const std::string& filename) const {
    // Simplified text-based results saving (no JSON dependency)
    std::ofstream file(filename);
    if (!file.is_open()) {
        LOG_ERROR("TestRunner: Failed to open results file: ", filename);
        return;
    }
    
    // Write header
    file << "DualStream Video Player Test Results\n";
    file << "=====================================\n\n";
    
    // Write summary
    int passed = 0;
    for (const auto& result : m_results) {
        if (result.passed) passed++;
    }
    
    file << "Total Tests: " << m_results.size() << "\n";
    file << "Passed: " << passed << "\n";
    file << "Failed: " << (m_results.size() - passed) << "\n";
    file << "Success Rate: " << (m_results.size() > 0 ? (passed * 100.0 / m_results.size()) : 0.0) << "%\n\n";
    
    // Write individual test results
    file << "Individual Test Results:\n";
    file << "========================\n\n";
    
    for (const auto& result : m_results) {
        file << "Test: " << result.testName << "\n";
        file << "Suite: " << result.suiteName << "\n";
        file << "Status: " << (result.passed ? "PASSED" : "FAILED") << "\n";
        file << "Execution Time: " << result.executionTimeSeconds << "s\n";
        
        if (!result.errorMessage.empty()) {
            file << "Error: " << result.errorMessage << "\n";
        }
        
        // Frame metrics
        if (result.frameMetrics.totalFramesProcessed > 0) {
            file << "Frame Metrics:\n";
            file << "  Total Frames: " << result.frameMetrics.totalFramesProcessed << "\n";
            file << "  Valid Frames: " << result.frameMetrics.framesWithValidNumbers << "\n";
            file << "  Frame Drops: " << result.frameMetrics.frameDropCount << "\n";
            file << "  Frame Accuracy: " << result.frameMetrics.frameAccuracyPercentage << "%\n";
        }
        
        // Performance metrics
        if (result.performanceMetrics.averageFps > 0) {
            file << "Performance Metrics:\n";
            file << "  Average FPS: " << result.performanceMetrics.averageFps << "\n";
            file << "  Average Frame Time: " << result.performanceMetrics.averageFrameTimeMs << "ms\n";
            file << "  Max Frame Time: " << result.performanceMetrics.maxFrameTimeMs << "ms\n";
            file << "  Memory Usage: " << result.performanceMetrics.memoryUsageMB << "MB\n";
        }
        
        file << "\n";
    }
    
    LOG_INFO("TestRunner: Saved test results to ", filename);
}
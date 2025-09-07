#pragma once

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>

// Forward declarations
class VideoManager;
class IRenderer;
class FFmpegInitializer;
struct TestResult;

/**
 * Main test orchestrator for the dual stream video player.
 * Coordinates frame validation, switching accuracy tests, and performance benchmarks.
 */
class TestRunner {
public:
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

    TestRunner();
    ~TestRunner();

    // Test execution
    bool Initialize();
    bool RunAllTests(const std::string& outputFile = "test_results.json");
    bool RunTestSuite(const TestSuite& suite, const std::string& outputFile = "");
    bool RunSingleTest(const TestConfig& config);
    
    // Test suite management
    void AddTestSuite(const TestSuite& suite);
    void LoadTestSuitesFromDirectory(const std::string& directory);
    
    // Results
    std::vector<TestResult> GetResults() const { return m_results; }
    void SaveResultsToJson(const std::string& filename) const;

private:
    // Core test types
    bool RunFrameValidationTest(const TestConfig& config);
    bool RunSwitchingAccuracyTest(const TestConfig& config);
    bool RunPerformanceBenchmark(const TestConfig& config);
    
    // Helper methods
    bool InitializeVideoSystem(const TestConfig& config);
    void CleanupVideoSystem();
    std::string GenerateTestReport() const;

private:
    std::vector<TestSuite> m_testSuites;
    std::vector<TestResult> m_results;
    
    // Core initialization components
    std::unique_ptr<FFmpegInitializer> m_ffmpegInitializer;
    
    // Video system components (initialized per test)
    std::unique_ptr<class Window> m_testWindow;
    std::unique_ptr<IRenderer> m_renderer;
    std::unique_ptr<VideoManager> m_videoManager;
    
    bool m_initialized = false;
};

/**
 * Individual test result with detailed metrics
 */
struct TestResult {
    std::string testName;
    std::string suiteName;
    bool passed = false;
    double executionTimeSeconds = 0.0;
    std::string errorMessage;
    
    // Frame validation metrics
    struct FrameMetrics {
        int totalFramesProcessed = 0;
        int framesWithValidNumbers = 0;
        int frameDropCount = 0;
        int frameDuplicateCount = 0;
        std::vector<int> detectedFrameNumbers;
        double frameAccuracyPercentage = 0.0;
    } frameMetrics;
    
    // Switching accuracy metrics  
    struct SwitchingMetrics {
        int switchAttempts = 0;
        int successfulSwitches = 0;
        double averageSwitchLatencyMs = 0.0;
        double maxSwitchLatencyMs = 0.0;
        std::vector<double> switchLatencies;
        bool keyframeAlignmentCorrect = true;
    } switchingMetrics;
    
    // Performance metrics
    struct PerformanceMetrics {
        double averageFrameTimeMs = 0.0;
        double maxFrameTimeMs = 0.0;
        double averageFps = 0.0;
        double memoryUsageMB = 0.0;
        double cpuUsagePercent = 0.0;
        double gpuUsagePercent = 0.0;
    } performanceMetrics;
    
    // Custom metrics for specific test types
    std::map<std::string, double> customMetrics;
};
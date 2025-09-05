#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <memory>

// Forward declarations
class VideoManager;
class FrameValidator;
enum class SwitchingAlgorithm;

/**
 * Validates video switching behavior and timing for all switching algorithms.
 * Tests frame accuracy, latency, and algorithm-specific behavior.
 */
class SwitchingValidator {
public:
    struct SwitchEvent {
        double triggerTime = 0.0;
        double actualSwitchTime = 0.0;  
        double latencyMs = 0.0;
        int expectedFrameNumber = -1;
        int actualFrameNumber = -1;
        bool switchedToCorrectVideo = false;
        bool frameNumberMatches = false;
        std::string algorithmUsed;
        std::string errorMessage;
    };

    struct AlgorithmTestResult {
        SwitchingAlgorithm algorithm;
        std::string algorithmName;
        
        std::vector<SwitchEvent> switchEvents;
        
        // Aggregate metrics
        double averageLatencyMs = 0.0;
        double maxLatencyMs = 0.0;
        double minLatencyMs = 0.0;
        int successfulSwitches = 0;
        int totalSwitches = 0;
        double successRate = 0.0;
        
        // Algorithm-specific metrics
        bool keyframeAlignmentCorrect = true;  // For keyframe-sync
        bool simultaneousPlayback = false;     // For predecoded
        bool immediateResponse = false;        // For immediate
    };

    SwitchingValidator();
    ~SwitchingValidator();

    // Test execution
    AlgorithmTestResult TestSwitchingAlgorithm(SwitchingAlgorithm algorithm,
                                              const std::string& video1Path,
                                              const std::string& video2Path,
                                              int numberOfSwitches = 10);
                                              
    std::vector<AlgorithmTestResult> TestAllAlgorithms(const std::string& video1Path,
                                                       const std::string& video2Path,
                                                       int numberOfSwitches = 10);

    // Manual switch testing
    bool StartSwitchingTest(SwitchingAlgorithm algorithm,
                           const std::string& video1Path, 
                           const std::string& video2Path);
    SwitchEvent TriggerSwitch();  // Returns event with timing data
    void EndSwitchingTest();

    // Validation methods
    bool ValidateKeyframeSynchronization(const std::vector<SwitchEvent>& events);
    bool ValidatePredecodedBehavior(const std::vector<SwitchEvent>& events);
    bool ValidateImmediateBehavior(const std::vector<SwitchEvent>& events);

    // Configuration
    void SetFrameValidator(std::shared_ptr<FrameValidator> validator);
    void SetToleranceMs(double toleranceMs) { m_toleranceMs = toleranceMs; }
    void SetDebugMode(bool enabled) { m_debugMode = enabled; }

private:
    // Algorithm-specific testing
    AlgorithmTestResult TestKeyframeSyncAlgorithm(const std::string& video1Path,
                                                  const std::string& video2Path,
                                                  int numberOfSwitches);
                                                  
    AlgorithmTestResult TestPredecodedAlgorithm(const std::string& video1Path,
                                                const std::string& video2Path,
                                                int numberOfSwitches);
                                                
    AlgorithmTestResult TestImmediateAlgorithm(const std::string& video1Path,
                                               const std::string& video2Path,
                                               int numberOfSwitches);

    // Switch execution and timing
    SwitchEvent ExecuteTimedSwitch(double atTimestamp);
    bool WaitForSwitchCompletion(double maxWaitTimeMs = 2000.0);
    double MeasureSwitchLatency();

    // Frame validation integration  
    bool ValidatePostSwitchFrame(double expectedTimestamp, int expectedVideo);
    int DetectActiveVideo();  // Returns 1 or 2 based on frame analysis

    // Keyframe analysis
    bool IsAtKeyframe(double timestamp, const std::string& videoPath);
    std::vector<double> GetKeyframeTimes(const std::string& videoPath);

private:
    std::shared_ptr<FrameValidator> m_frameValidator;
    std::unique_ptr<VideoManager> m_videoManager;
    
    // Test configuration
    double m_toleranceMs = 100.0;  // Acceptable latency tolerance
    bool m_debugMode = false;
    
    // Current test state
    bool m_testInProgress = false;
    SwitchingAlgorithm m_currentAlgorithm;
    int m_currentVideo = 1;  // 1 or 2
    std::chrono::high_resolution_clock::time_point m_testStartTime;
    
    // Timing utilities
    std::chrono::high_resolution_clock::time_point m_switchStartTime;
    std::chrono::high_resolution_clock::time_point m_switchEndTime;
};
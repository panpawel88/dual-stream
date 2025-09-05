#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <memory>

// Forward declarations
class VideoManager;
class IRenderer;

/**
 * Performance benchmarking system for different video resolutions, frame rates, and algorithms.
 * Measures frame delivery consistency, memory usage, and system resource utilization.
 */
class PerformanceBenchmark {
public:
    struct BenchmarkConfig {
        std::string video1Path;
        std::string video2Path;
        std::string algorithmName;
        double playbackSpeed = 1.0;
        int durationSeconds = 30;  // Benchmark duration
        int switchFrequencySeconds = 2;  // How often to switch videos
        bool enableDebugOutput = false;
    };

    struct FrameTimingData {
        double timestamp = 0.0;
        double frameTimeMs = 0.0;  // Time to process this frame
        double presentTimeMs = 0.0;  // Time to present to screen
        double totalTimeMs = 0.0;  // Total frame processing time
        bool frameDropped = false;
        int videoSource = 1;  // 1 or 2
    };

    struct MemorySnapshot {
        double timestamp = 0.0;
        double systemMemoryMB = 0.0;
        double processMemoryMB = 0.0;
        double gpuMemoryMB = 0.0;
        double videoMemoryMB = 0.0;  // Estimated video-specific memory
    };

    struct BenchmarkResult {
        std::string testName;
        std::string algorithm;
        double totalDurationSeconds = 0.0;
        
        // Frame rate metrics
        int totalFramesProcessed = 0;
        int frameDropCount = 0;
        double averageFps = 0.0;
        double minFps = 0.0;
        double maxFps = 0.0;
        double frameRateStability = 0.0;  // Coefficient of variation
        
        // Timing metrics
        double averageFrameTimeMs = 0.0;
        double maxFrameTimeMs = 0.0;
        double p95FrameTimeMs = 0.0;  // 95th percentile
        double p99FrameTimeMs = 0.0;  // 99th percentile
        
        // Memory metrics
        double averageMemoryUsageMB = 0.0;
        double peakMemoryUsageMB = 0.0;
        double memoryGrowthMB = 0.0;  // Memory increase from start to end
        
        // Resource utilization
        double averageCpuPercent = 0.0;
        double peakCpuPercent = 0.0;
        double averageGpuPercent = 0.0;
        double peakGpuPercent = 0.0;
        
        // Switch-specific metrics
        int switchCount = 0;
        double averageSwitchLatencyMs = 0.0;
        double maxSwitchLatencyMs = 0.0;
        
        // Raw data
        std::vector<FrameTimingData> frameTimings;
        std::vector<MemorySnapshot> memorySnapshots;
        std::vector<double> switchLatencies;
        
        // Video-specific info
        int videoWidth = 0;
        int videoHeight = 0;
        double videoFps = 0.0;
        std::string videoCodec;
    };

    PerformanceBenchmark();
    ~PerformanceBenchmark();

    // Benchmark execution
    BenchmarkResult RunBenchmark(const BenchmarkConfig& config);
    std::vector<BenchmarkResult> RunBenchmarkSuite(const std::vector<BenchmarkConfig>& configs);

    // Resolution-specific benchmarks
    BenchmarkResult BenchmarkResolution(const std::string& resolutionCategory,  // "HD", "4K", "8K"
                                        const std::string& algorithm,
                                        int durationSeconds = 30);
                                        
    // Frame rate benchmarks
    BenchmarkResult BenchmarkFrameRate(int fps,  // 30, 60, etc.
                                       const std::string& algorithm, 
                                       int durationSeconds = 30);

    // Algorithm comparison
    std::vector<BenchmarkResult> CompareAlgorithms(const std::string& video1Path,
                                                   const std::string& video2Path,
                                                   int durationSeconds = 30);

    // Memory profiling
    struct MemoryProfile {
        std::string algorithmName;
        double baseMemoryMB = 0.0;  // Memory before video loading
        double singleStreamMB = 0.0;  // Memory with one stream
        double dualStreamMB = 0.0;  // Memory with both streams (predecoded)
        double memoryEfficiency = 0.0;  // Frames per MB
    };
    
    MemoryProfile ProfileMemoryUsage(const std::string& algorithm,
                                     const std::string& video1Path,
                                     const std::string& video2Path);

    // Stress testing
    BenchmarkResult StressTest(const std::string& video1Path,
                              const std::string& video2Path,
                              int rapidSwitchingDurationSeconds = 60,
                              double switchIntervalSeconds = 0.1);  // Very rapid switching

    // Configuration
    void SetMemoryMonitoringInterval(double intervalSeconds) { m_memoryMonitorInterval = intervalSeconds; }
    void SetResourceMonitoringEnabled(bool enabled) { m_resourceMonitoring = enabled; }

    // Results export
    void SaveBenchmarkResults(const std::vector<BenchmarkResult>& results, 
                             const std::string& filename);
    std::string GeneratePerformanceReport(const std::vector<BenchmarkResult>& results);

private:
    // Monitoring methods
    void StartPerformanceMonitoring();
    void StopPerformanceMonitoring();
    void RecordFrameTiming(double frameTime, double presentTime, int videoSource);
    void RecordMemorySnapshot();
    void RecordSwitchLatency(double latency);

    // System resource monitoring
    double GetCurrentCpuUsage();
    double GetCurrentGpuUsage();
    double GetCurrentMemoryUsage();
    double GetGpuMemoryUsage();

    // Statistics calculation
    void CalculateFrameRateMetrics(BenchmarkResult& result);
    void CalculateTimingMetrics(BenchmarkResult& result);
    void CalculateMemoryMetrics(BenchmarkResult& result);
    double CalculatePercentile(const std::vector<double>& values, double percentile);

    // Video file analysis
    void AnalyzeVideoProperties(const std::string& videoPath, BenchmarkResult& result);

private:
    // Current benchmark state
    std::unique_ptr<VideoManager> m_videoManager;
    std::unique_ptr<IRenderer> m_renderer;
    
    // Monitoring configuration
    double m_memoryMonitorInterval = 1.0;  // seconds
    bool m_resourceMonitoring = true;
    
    // Performance data collection
    std::vector<FrameTimingData> m_frameTimings;
    std::vector<MemorySnapshot> m_memorySnapshots;
    std::vector<double> m_switchLatencies;
    
    // Timing utilities
    std::chrono::high_resolution_clock::time_point m_benchmarkStartTime;
    std::chrono::high_resolution_clock::time_point m_lastMemoryCheck;
    
    // Background monitoring thread
    bool m_monitoringActive = false;
    std::thread m_monitoringThread;
    
    // System baseline measurements
    double m_baselineCpuPercent = 0.0;
    double m_baselineMemoryMB = 0.0;
};
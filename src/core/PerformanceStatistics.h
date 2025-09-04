#pragma once
#include <chrono>
#include <string>
#include <mutex>
#include <deque>
#include <map>
#include "../ui/IUIDrawable.h"

class PerformanceStatistics : public IUIDrawable {
public:
    static PerformanceStatistics& GetInstance();
    
    // Video metrics
    void SetVideoFrameRate(double fps);
    void SetCurrentPlaybackTime(double time);
    void SetActiveVideoIndex(int index);
    void SetPlaybackSpeed(double speed);
    void SetVideoResolution(int width, int height);
    void SetVideoFormat(const std::string& format);
    
    // Decoding metrics
    void RecordDecodeTime(double milliseconds);
    void SetDecoderType(const std::string& type);
    void IncrementDroppedFrames();
    void SetDecoderQueueDepth(int depth);
    
    // Application performance
    void RecordApplicationFrameTime(double milliseconds);
    void RecordPresentationTime(double milliseconds);
    void RecordMainLoopTime(double milliseconds);
    
    // Rendering performance
    void RecordRenderTime(double milliseconds);
    void RecordTextureConversionTime(double milliseconds);
    void RecordRenderPassTime(double milliseconds);
    
    // System metrics
    void SetMemoryUsage(size_t cpuMB, size_t gpuMB);
    void SetSwitchingStrategy(const std::string& strategy);
    
    // UI Integration
    void DrawUI() override;
    std::string GetUIName() const override { return "Performance Statistics"; }
    std::string GetUICategory() const override { return "Performance"; }
    
private:
    PerformanceStatistics() = default;
    ~PerformanceStatistics() = default;
    
    mutable std::mutex m_mutex;
    
    // Video metrics
    double m_videoFrameRate = 0.0;
    double m_currentPlaybackTime = 0.0;
    int m_activeVideoIndex = 0;
    double m_playbackSpeed = 1.0;
    int m_videoWidth = 0;
    int m_videoHeight = 0;
    std::string m_videoFormat;
    
    // Decoding metrics
    std::string m_decoderType;
    int m_droppedFrames = 0;
    int m_decoderQueueDepth = 0;
    
    // System metrics
    size_t m_cpuMemoryMB = 0;
    size_t m_gpuMemoryMB = 0;
    std::string m_switchingStrategy;
    
    // Performance history for graphs (keep last 60 samples)
    static constexpr size_t MAX_SAMPLES = 60;
    std::deque<double> m_decodeTimeHistory;
    std::deque<double> m_applicationFpsHistory;
    std::deque<double> m_renderTimeHistory;
    std::deque<double> m_mainLoopTimeHistory;
    std::deque<double> m_presentationTimeHistory;
    
    // Timing helpers
    std::chrono::steady_clock::time_point m_lastFrameTime;
    double m_currentApplicationFps = 0.0;
    
    // Helper methods
    void AddToHistory(std::deque<double>& history, double value);
    double CalculateAverage(const std::deque<double>& history) const;
    double GetMinValue(const std::deque<double>& history) const;
    double GetMaxValue(const std::deque<double>& history) const;
    void DrawMetricSection(const char* title, const std::map<std::string, std::string>& metrics);
    void DrawGraphSection(const char* title, const std::deque<double>& data, const char* unit, float minVal = 0.0f, float maxVal = 100.0f);
};